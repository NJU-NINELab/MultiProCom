from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from actual_data_utils import ACTUAL_SCENARIOS
from baselines.multimodal_ablation_beam.model import MultiProComAblation, resolve_ablation_config
from baselines.single_modality_beam.common import build_model as build_single_model, model_inputs
from baselines.single_modality_beam.dataset import VehicleTrackTemporalDataset
from dataset import ActualMultimodalDataset, TemporalSequenceDataset
from multiprocom import MultiProCom


METHODS = ("multiprocom", "vision_only", "radar_only", "wo_ramf", "wo_afsp")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_dataset(args):
    load_modalities = ("radar",) if args.method != "vision_only" else ("vision",)
    base = ActualMultimodalDataset(
        root=args.root, scenarios=ACTUAL_SCENARIOS, load_modalities=load_modalities,
        precomputed_radar_root=args.precomputed_radar_root,
        radar_norm_mode="global_stats" if args.method != "vision_only" else "frame_logminmax",
        radar_norm_stats=args.radar_norm_stats if args.method != "vision_only" else "",
        cache_radar_maps=False,
    )
    labels = [int(s["label"]) for s in base.samples]
    if min(labels) < 0 or max(labels) >= args.num_beams:
        raise ValueError(f"Labels [{min(labels)}, {max(labels)}] incompatible with num_beams={args.num_beams}")
    if args.method == "vision_only":
        return VehicleTrackTemporalDataset(base, 5, 8, args.motion_tracks, max_vehicles_per_frame=8)
    return TemporalSequenceDataset(
        base, history_len=5, future_steps=8, include_future_maps=False,
        vehicle_track_json=args.motion_tracks if args.method in {"multiprocom", "wo_ramf", "wo_afsp"} else "",
        max_vehicles_per_frame=8,
    )


def build_model(method, num_beams=9):
    if method == "multiprocom":
        model = MultiProCom(feat_dim=256, num_beams=num_beams, history_len=5, pred_horizon_max=8)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.2
        return model
    if method == "vision_only":
        return build_single_model("vision", 256, num_beams, 5, 8)
    if method == "radar_only":
        return build_single_model("radar", 256, num_beams, 5, 8)
    preset = "wo_ramf_r05_v05" if method == "wo_ramf" else "wo_afsp_parallel"
    config = resolve_ablation_config(preset, "auto", 0.8, "auto")
    model = MultiProComAblation(
        feat_dim=256, num_beams=num_beams, history_len=5, pred_horizon_max=8,
        fusion_mode=config.fusion_mode, fixed_radar_weight=config.fixed_radar_weight,
        decoder_mode=config.decoder_mode,
    )
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2
    return model


def transfer_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    source = {k.removeprefix("module."): v for k, v in source.items()}
    target = model.state_dict(); loaded, skipped = [], []
    allowed = ("decoder.beam_token_embed.weight", "decoder.head.2.weight", "decoder.head.2.bias")
    merged = dict(target)
    for key, value in source.items():
        if key not in target:
            skipped.append((key, "unexpected")); continue
        if target[key].shape == value.shape:
            merged[key] = value; loaded.append(key); continue
        if not any(key.endswith(name) for name in allowed):
            raise ValueError(f"Non-class transfer mismatch: {key} source={tuple(value.shape)} target={tuple(target[key].shape)}")
        skipped.append((key, f"shape {tuple(value.shape)}->{tuple(target[key].shape)}"))
        if key.endswith("decoder.beam_token_embed.weight"):
            merged[key][-1] = value[-1]
    model.load_state_dict(merged, strict=True)
    print(f"[Transfer] loaded={len(loaded)} skipped={skipped} source={Path(path).resolve()}")
    return {"loaded": len(loaded), "skipped": skipped, "source": str(Path(path).resolve())}


def forward_model(model, method, batch):
    if method == "vision_only":
        return model(**model_inputs("vision", batch, 8))["beam_logits_seq"]
    if method == "radar_only":
        return model(**model_inputs("radar", batch, 8))["beam_logits_seq"]
    return model(
        hist_vehicle_bboxes=batch["hist_vehicle_bboxes"],
        hist_range_doppler_map=batch["hist_range_doppler_map"],
        hist_range_angle_map=batch["hist_range_angle_map"],
        hist_delay_doppler_map=batch["hist_delay_doppler_map"],
        hist_power_map=batch["hist_power_map"], pred_horizon=8,
    )["beam_logits_seq"]


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def balanced_neighbor_loss(logits, labels, class_weights=None):
    losses = []
    class_ids = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype).view(1, -1)
    for horizon in range(logits.shape[1]):
        target = labels[:, horizon]
        distance = torch.abs(class_ids - target.to(logits.dtype).view(-1, 1))
        kernel = torch.exp(-0.5 * (distance / 1.5) ** 2)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        soft = 0.85 * F.one_hot(target, num_classes=logits.shape[-1]).to(logits.dtype) + 0.15 * kernel
        per_sample = -(soft * F.log_softmax(logits[:, horizon], dim=1)).sum(dim=1)
        if class_weights is not None:
            per_sample = per_sample * class_weights[target]
        losses.append((8 - horizon) * per_sample.mean())
    return sum(losses) / 36.0


def compute_class_weights(temporal, train_indices, num_beams=9):
    counts = torch.zeros(num_beams, dtype=torch.float64)
    for wid in train_indices:
        for sample_id in temporal.windows[wid]["future_indices"]:
            counts[int(temporal.base.samples[sample_id]["label"])] += 1
    weights = torch.sqrt(counts.max() / counts.clamp_min(1.0)).clamp(max=3.0)
    weights = weights / weights.mean()
    print(f"[ClassBalance] counts={counts.int().tolist()} weights={[round(float(x),3) for x in weights]}")
    return weights.float()


def run_epoch(model, method, loader, device, optimizer=None, max_steps=0, class_weights=None):
    training = optimizer is not None; model.train(training)
    total = 0; loss_sum = 0.0
    correct = {k: [0] * 8 for k in (1, 3, 5)}; counts = [0] * 8
    scenario_stats = {}; absolute_error = 0.0; within_one = 0
    if class_weights is not None:
        class_weights = class_weights.to(device)
    for step_idx, batch in enumerate(loader):
        scenarios = list(batch["meta_scenario"]); batch = move_batch(batch, device)
        labels = batch["future_beam_labels"][:, :8]
        with torch.set_grad_enabled(training):
            logits = forward_model(model, method, batch)
            loss = balanced_neighbor_loss(logits, labels, class_weights)
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        batch_size = labels.shape[0]; total += batch_size; loss_sum += float(loss.item()) * batch_size
        for horizon in range(8):
            ranked = logits[:, horizon].topk(k=min(5, logits.shape[-1]), dim=1).indices
            prediction = ranked[:, 0]
            error = torch.abs(prediction - labels[:, horizon])
            absolute_error += float(error.sum().item())
            within_one += int((error <= 1).sum().item())
            for k in (1, 3, 5): correct[k][horizon] += int(ranked[:, :k].eq(labels[:, horizon, None]).any(1).sum())
            counts[horizon] += batch_size
            for row, scene in enumerate(scenarios):
                st = scenario_stats.setdefault(
                    scene,
                    {"n": [0]*8, "correct": {k:[0]*8 for k in (1,3,5)}, "absolute_error": 0.0, "within_one": 0},
                )
                st["n"][horizon] += 1
                row_error = int(error[row].item())
                st["absolute_error"] += row_error
                st["within_one"] += int(row_error <= 1)
                for k in (1,3,5): st["correct"][k][horizon] += int(ranked[row,:k].eq(labels[row,horizon]).any())
        if max_steps and step_idx + 1 >= max_steps: break
    metrics = {"loss": loss_sum / max(total, 1), "windows": int(total), "step_metrics": {}, "per_scenario": {},
               "beam_mae": absolute_error / max(total * 8, 1), "within_one_accuracy": within_one / max(total * 8, 1)}
    for k in (1, 3, 5):
        values = [correct[k][i] / max(counts[i], 1) for i in range(8)]
        metrics[f"top{k}_mean"] = float(sum(values) / 8)
        for i, value in enumerate(values): metrics["step_metrics"][f"Top{k}@t+{i+1}"] = value
    for scene, st in scenario_stats.items():
        scenario_predictions = max(sum(st["n"]), 1)
        out = {
            "step_metrics": {}, "windows": st["n"][0],
            "beam_mae": st["absolute_error"] / scenario_predictions,
            "within_one_accuracy": st["within_one"] / scenario_predictions,
        }
        for k in (1,3,5):
            values = [st["correct"][k][i] / max(st["n"][i],1) for i in range(8)]
            out[f"top{k}_mean"] = sum(values)/8
            for i,v in enumerate(values): out["step_metrics"][f"Top{k}@t+{i+1}"] = v
        metrics["per_scenario"][scene] = out
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MultiProCom and its paper baselines on all contiguous windows.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--root", default="/home/ybpeng/Data/ActualMulData/dataset_multimodal_data")
    parser.add_argument("--precomputed-radar-root", default="experiments/multiprocom/assets/precomputed_radar_maps")
    parser.add_argument("--radar-norm-stats", default="experiments/multiprocom/assets/radar_normalization.json")
    parser.add_argument("--motion-tracks", default="experiments/multiprocom/assets/motion_components.json")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--output-dir", default="experiments/multiprocom/training_runs")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5); parser.add_argument("--enc-lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=3e-4); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-beams", type=int, default=9); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing an accidental CPU training run")
    device = torch.device(args.device)
    temporal = build_dataset(args)
    indices = {"train": list(range(len(temporal)))}
    print(f"[FullTrain] using all {len(indices['train'])} contiguous windows")
    class_weights = compute_class_weights(temporal, indices["train"], args.num_beams)
    model = build_model(args.method, args.num_beams).to(device)
    transfer = transfer_checkpoint(model, args.init_checkpoint) if args.init_checkpoint else None
    encoder_params, other_params = [], []
    for name, parameter in model.named_parameters():
        (encoder_params if "encoder" in name else other_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.enc_lr_mult},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    run_dir = Path(args.output_dir) / args.method; run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest_checkpoint.pt"
    start_epoch = 1
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu"); model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"]); start_epoch = int(ckpt["epoch"]) + 1
    fields = ["epoch", "train_loss", "train_top1", "train_top3", "train_top5"]
    log_path = run_dir / "full_train_log.csv"
    if start_epoch == 1:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writeheader()
    for epoch in range(start_epoch, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(Subset(temporal, indices["train"]), args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers)
        train = run_epoch(model, args.method, train_loader, device, optimizer, args.max_train_steps, class_weights)
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "args": vars(args), "method": args.method, "transfer": transfer,
            "training_scope": "all_contiguous_windows", "num_train_windows": len(indices["train"]),
        }
        torch.save(checkpoint, latest)
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writerow({
                "epoch": epoch, "train_loss": train["loss"], "train_top1": train["top1_mean"],
                "train_top3": train["top3_mean"], "train_top5": train["top5_mean"],
            })
        print(f"[{args.method} {epoch:03d}] top1={train['top1_mean']:.4f}", flush=True)

    final_checkpoint = run_dir / "final_checkpoint.pt"
    final_payload = {key: value for key, value in checkpoint.items() if key != "optimizer"}
    torch.save(final_payload, final_checkpoint)
    evaluation_loader = DataLoader(
        Subset(temporal, indices["train"]), args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    full_metrics = run_epoch(model, args.method, evaluation_loader, device)
    result = {
        "method": args.method, "epoch": args.epochs,
        "training_scope": "all_contiguous_windows", "num_train_windows": len(indices["train"]),
        "metrics_scope": "training_data_resubstitution_not_generalization",
        "full_data_metrics": full_metrics, "checkpoint": str(final_checkpoint.resolve()),
    }
    result_path = run_dir / "full_data_metrics.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] top1={full_metrics['top1_mean']:.4f} checkpoint={final_checkpoint}")


if __name__ == "__main__":
    main()

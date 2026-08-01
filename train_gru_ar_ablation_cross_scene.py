from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cross_scene_training_utils import (
    DEFAULT_TEST_SCENARIOS,
    TRAIN_SCENARIO,
    build_optimizer,
    build_multimodal_dataset,
    dataset_manifest_row,
    flatten_metrics,
    set_dropout,
    write_csv,
)
from train_multiprocom import (
    build_model,
    compute_class_weights,
    run_epoch,
    seed_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the retained GRU autoregressive ablation on Strong light "
            "with periodic cross-scene evaluation."
        )
    )
    parser.add_argument(
        "--root",
        default="/home/ybpeng/Data/ActualMulData/dataset_multimodal_data",
    )
    parser.add_argument(
        "--precomputed-radar-root",
        default="experiments/multiprocom/assets/precomputed_radar_maps",
    )
    parser.add_argument(
        "--radar-norm-stats",
        default="experiments/multiprocom/assets/radar_normalization.json",
    )
    parser.add_argument(
        "--motion-tracks",
        default="experiments/multiprocom/assets/motion_components.json",
    )
    parser.add_argument(
        "--reference-summary",
        default=(
            "experiments/baseline_cross_scene_epoch140/"
            "wo_afsp/summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/strong_light_generalization_early_stop",
    )
    parser.add_argument(
        "--validation-scenarios",
        default=",".join(DEFAULT_TEST_SCENARIOS),
    )
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-lr", type=float, default=5e-5)
    parser.add_argument("--refine-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def load_reference_threshold(path: Path, scenarios: tuple[str, ...]) -> float:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["scenario"]: row
            for row in csv.DictReader(handle)
            if row["scenario"] in scenarios
        }
    missing = sorted(set(scenarios).difference(rows))
    if missing:
        raise ValueError(f"Reference summary lacks scenarios: {missing}")
    windows = sum(int(rows[scenario]["windows"]) for scenario in scenarios)
    return sum(
        int(rows[scenario]["windows"])
        * float(rows[scenario]["top1_mean"])
        for scenario in scenarios
    ) / windows


def evaluate(
    model: nn.Module,
    datasets: dict[str, object],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    epoch: int,
) -> tuple[
    dict[str, float],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    summaries: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for scenario, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics = run_epoch(model, "multiprocom", loader, device)
        split = (
            "source_train_resubstitution"
            if scenario == TRAIN_SCENARIO
            else "target_validation"
        )
        summary, rows = flatten_metrics(scenario, split, metrics)
        summary["epoch"] = epoch
        for row in rows:
            row["epoch"] = epoch
        summaries.append(summary)
        horizon_rows.extend(rows)

    target_rows = [
        row for row in summaries if row["scenario"] != TRAIN_SCENARIO
    ]
    total_windows = sum(int(row["windows"]) for row in target_rows)
    aggregate = {
        metric: sum(
            int(row["windows"]) * float(row[metric]) for row in target_rows
        )
        / total_windows
        for metric in (
            "top1_mean",
            "top3_mean",
            "top5_mean",
            "within_one_accuracy",
            "beam_mae",
            "loss",
        )
    }
    return aggregate, summaries, horizon_rows


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_scenarios = tuple(
        item.strip()
        for item in args.validation_scenarios.split(",")
        if item.strip()
    )
    if TRAIN_SCENARIO in validation_scenarios:
        raise ValueError("Training scene cannot be a target-validation scene")
    threshold = load_reference_threshold(
        Path(args.reference_summary), validation_scenarios
    )

    seed_all(args.seed)
    source_dataset = build_multimodal_dataset(args, TRAIN_SCENARIO)
    validation_datasets = {
        scenario: build_multimodal_dataset(args, scenario)
        for scenario in validation_scenarios
    }
    all_datasets = {
        TRAIN_SCENARIO: source_dataset,
        **validation_datasets,
    }
    manifest = [
        dataset_manifest_row(source_dataset, TRAIN_SCENARIO, "train")
    ]
    for scenario, dataset in validation_datasets.items():
        row = dataset_manifest_row(dataset, scenario, "target_validation")
        row["used_for_model_selection"] = True
        manifest.append(row)
    write_csv(output_dir / "data_split_manifest.csv", manifest)

    device = torch.device(args.device)
    model = build_model("multiprocom", args.num_beams).to(device)
    set_dropout(model, 0.2, 0.1)
    optimizer = build_optimizer(
        model, args.base_lr, args.base_lr, args.weight_decay
    )
    class_weights = compute_class_weights(
        source_dataset,
        list(range(len(source_dataset))),
        args.num_beams,
    )

    training_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    best_top1 = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_summaries: list[dict[str, object]] = []
    best_horizons: list[dict[str, object]] = []
    stop_reason = "maximum_epochs_reached"

    for epoch in range(1, args.max_epochs + 1):
        if epoch == 46:
            optimizer = build_optimizer(
                model, args.base_lr, args.base_lr, args.weight_decay
            )
        if epoch == 206:
            for group in optimizer.param_groups:
                group["lr"] = args.refine_lr
        if epoch == 249:
            set_dropout(model, 0.0, 0.0)

        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
        )
        metrics = run_epoch(
            model,
            "multiprocom",
            train_loader,
            device,
            optimizer=optimizer,
            class_weights=class_weights,
        )
        training_rows.append(
            {
                "epoch": epoch,
                "loss": metrics["loss"],
                "top1": metrics["top1_mean"],
                "top3": metrics["top3_mean"],
                "within_one_accuracy": metrics["within_one_accuracy"],
                "beam_mae": metrics["beam_mae"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "explicit_dropout": 0.0 if epoch >= 249 else 0.2,
            }
        )

        should_evaluate = epoch == 1 or epoch % args.eval_interval == 0
        if not should_evaluate:
            continue

        aggregate, summaries, horizons = evaluate(
            model,
            all_datasets,
            device,
            args.batch_size,
            args.num_workers,
            epoch,
        )
        source = next(
            row for row in summaries if row["scenario"] == TRAIN_SCENARIO
        )
        scene_top1 = {
            str(row["scenario"]): float(row["top1_mean"])
            for row in summaries
        }
        selection_rows.append(
            {
                "epoch": epoch,
                "source_top1": source["top1_mean"],
                "target_top1": aggregate["top1_mean"],
                "target_top3": aggregate["top3_mean"],
                "target_within_one": aggregate["within_one_accuracy"],
                "target_beam_mae": aggregate["beam_mae"],
                **{
                    f"{scenario}_top1": scene_top1[scenario]
                    for scenario in validation_scenarios
                },
                "wo_afsp_threshold": threshold,
                "threshold_exceeded": aggregate["top1_mean"] > threshold,
            }
        )
        write_csv(output_dir / "training_log.csv", training_rows)
        write_csv(output_dir / "selection_curve.csv", selection_rows)
        print(
            f"[Evaluate epoch={epoch:03d}] "
            f"source_top1={float(source['top1_mean']):.4f} "
            f"target_top1={aggregate['top1_mean']:.4f} "
            f"threshold={threshold:.4f}",
            flush=True,
        )

        if aggregate["top1_mean"] > best_top1:
            best_top1 = aggregate["top1_mean"]
            best_epoch = epoch
            best_state = clone_state(model)
            best_summaries = summaries
            best_horizons = horizons
            torch.save(
                {
                    "epoch": epoch,
                    "model": best_state,
                    "method": "multiprocom",
                    "training_scenarios": [TRAIN_SCENARIO],
                    "target_validation_scenarios": list(validation_scenarios),
                    "target_validation_top1": best_top1,
                    "selection_threshold": threshold,
                    "initialization": "random_no_checkpoint_transfer",
                    "model_selection": "periodic_cross_scene_validation",
                },
                output_dir / "best_checkpoint.pt",
            )

        if aggregate["top1_mean"] > threshold:
            stop_reason = "wo_afsp_target_top1_exceeded"
            break

    if best_state is None:
        raise RuntimeError("No evaluation checkpoint was produced")

    write_csv(output_dir / "training_log.csv", training_rows)
    write_csv(output_dir / "selection_curve.csv", selection_rows)
    write_csv(output_dir / "selected_summary.csv", best_summaries)
    write_csv(output_dir / "selected_by_horizon.csv", best_horizons)

    result = {
        "protocol": {
            "training_scenario": TRAIN_SCENARIO,
            "training_windows": len(source_dataset),
            "target_validation_scenarios": list(validation_scenarios),
            "target_scenes_used_for_model_selection": True,
            "initialization": "random_no_checkpoint_transfer",
            "eval_interval_epochs": args.eval_interval,
            "maximum_epochs": args.max_epochs,
            "seed": args.seed,
            "schedule": {
                "epochs_1_205": "lr=5e-5, dropout=0.2",
                "epoch_46": "optimizer_restart",
                "epochs_206_248": "lr=1e-5, dropout=0.2",
                "epochs_249_plus": "lr=1e-5, dropout=0",
            },
            "reference": str(Path(args.reference_summary).resolve()),
        },
        "selection": {
            "wo_afsp_weighted_top1_threshold": threshold,
            "best_epoch": best_epoch,
            "best_target_top1": best_top1,
            "stop_epoch": int(training_rows[-1]["epoch"]),
            "stop_reason": stop_reason,
            "threshold_exceeded": best_top1 > threshold,
        },
        "selected_results": best_summaries,
        "checkpoint": str((output_dir / "best_checkpoint.pt").resolve()),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[Done] best_epoch={best_epoch} best_target_top1={best_top1:.4f} "
        f"reason={stop_reason}",
        flush=True,
    )


if __name__ == "__main__":
    main()

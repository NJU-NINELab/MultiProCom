from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cross_scene_training_utils import (
    BASELINE_METHODS,
    DEFAULT_TEST_SCENARIOS,
    TRAIN_SCENARIO,
    build_baseline_dataset,
    build_optimizer,
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
            "Train one baseline from random initialization on Strong light for "
            "the fixed MultiProCom-selected 140-epoch schedule."
        )
    )
    parser.add_argument("--method", choices=BASELINE_METHODS, required=True)
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
        "--output-root",
        default="experiments/baseline_cross_scene_epoch140",
    )
    parser.add_argument(
        "--test-scenarios",
        default=",".join(DEFAULT_TEST_SCENARIOS),
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_root) / args.method
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = torch.device(args.device)
    train_dataset = build_baseline_dataset(args, TRAIN_SCENARIO)
    class_weights = compute_class_weights(
        train_dataset,
        list(range(len(train_dataset))),
        args.num_beams,
    )
    model = build_model(args.method, args.num_beams).to(device)
    set_dropout(model, 0.2, 0.1)
    optimizer = build_optimizer(
        model, args.lr, args.lr, args.weight_decay
    )

    training_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        if epoch == 46:
            optimizer = build_optimizer(
                model, args.lr, args.lr, args.weight_decay
            )
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
        )
        metrics = run_epoch(
            model,
            args.method,
            loader,
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
                "explicit_dropout": 0.2,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch in {45, 46}:
            print(
                f"[{args.method} {epoch:03d}/{args.epochs}] "
                f"loss={metrics['loss']:.4f} "
                f"top1={metrics['top1_mean']:.4f}",
                flush=True,
            )
    write_csv(output_dir / "training_log.csv", training_rows)

    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    checkpoint_path = output_dir / "checkpoint_epoch140.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model": state,
            "method": args.method,
            "training_scenarios": [TRAIN_SCENARIO],
            "training_windows": len(train_dataset),
            "initialization": "random_no_checkpoint_transfer",
            "schedule": (
                "epochs 1-45 lr5e-5 dropout0.2; optimizer restart at "
                "epoch 46; epochs 46-140 lr5e-5 dropout0.2"
            ),
            "fixed_epoch_source": (
                "MultiProCom cross-scene validation first exceeded w/o AFSP "
                "at epoch 140"
            ),
        },
        checkpoint_path,
    )

    test_scenarios = tuple(
        item.strip()
        for item in args.test_scenarios.split(",")
        if item.strip()
    )
    print(
        f"[{args.method}] training complete; constructing target scenes",
        flush=True,
    )
    datasets = {
        TRAIN_SCENARIO: train_dataset,
        **{
            scenario: build_baseline_dataset(args, scenario)
            for scenario in test_scenarios
        },
    }
    summaries: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for scenario, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        metrics = run_epoch(model, args.method, loader, device)
        split = (
            "source_train_resubstitution"
            if scenario == TRAIN_SCENARIO
            else "target_validation_fixed_epoch"
        )
        summary, rows = flatten_metrics(scenario, split, metrics)
        summaries.append(summary)
        horizon_rows.extend(rows)
        print(
            f"[{args.method} evaluate] {scenario} "
            f"top1={summary['top1_mean']:.4f} "
            f"top3={summary['top3_mean']:.4f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "by_horizon.csv", horizon_rows)
    target_rows = [
        row for row in summaries if row["scenario"] != TRAIN_SCENARIO
    ]
    target_windows = sum(int(row["windows"]) for row in target_rows)
    aggregate = {
        metric: sum(
            int(row["windows"]) * float(row[metric])
            for row in target_rows
        )
        / target_windows
        for metric in (
            "top1_mean",
            "top3_mean",
            "top5_mean",
            "within_one_accuracy",
            "beam_mae",
            "loss",
        )
    }
    result = {
        "protocol": {
            "method": args.method,
            "training_scenario": TRAIN_SCENARIO,
            "training_windows": len(train_dataset),
            "target_scenarios": list(test_scenarios),
            "epochs": args.epochs,
            "seed": args.seed,
            "initialization": "random_no_checkpoint_transfer",
            "target_scenes_used_for_this_method_selection": False,
            "fixed_epoch_selected_by": (
                "MultiProCom cross-scene validation"
            ),
        },
        "target_weighted": aggregate,
        "results": summaries,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[Done {args.method}] target_top1={aggregate['top1_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

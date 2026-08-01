"""Shared data and optimization helpers for retained cross-scene experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from baselines.single_modality_beam.dataset import (
    VehicleTrackTemporalDataset,
)
from dataset import ActualMultimodalDataset, TemporalSequenceDataset


TRAIN_SCENARIO = "strong_light"
DEFAULT_TEST_SCENARIOS = ("Obstruction", "dim_light", "multiTarget")
BASELINE_METHODS = ("radar_only", "vision_only", "wo_ramf", "wo_afsp")


def build_multimodal_dataset(
    args: argparse.Namespace,
    scenario: str,
) -> TemporalSequenceDataset:
    base = ActualMultimodalDataset(
        root=args.root,
        scenarios=[scenario],
        load_modalities=("radar",),
        precomputed_radar_root=args.precomputed_radar_root,
        radar_norm_mode="global_stats",
        radar_norm_stats=args.radar_norm_stats,
        cache_radar_maps=False,
    )
    labels = [int(sample["label"]) for sample in base.samples]
    if min(labels) < 0 or max(labels) >= args.num_beams:
        raise ValueError(
            f"{scenario}: labels [{min(labels)}, {max(labels)}] are "
            f"incompatible with num_beams={args.num_beams}"
        )
    return TemporalSequenceDataset(
        base,
        history_len=5,
        future_steps=8,
        include_future_maps=False,
        vehicle_track_json=args.motion_tracks,
        max_vehicles_per_frame=8,
    )


def build_baseline_dataset(
    args: argparse.Namespace,
    scenario: str,
):
    vision_only = args.method == "vision_only"
    base = ActualMultimodalDataset(
        root=args.root,
        scenarios=[scenario],
        load_modalities=("vision",) if vision_only else ("radar",),
        precomputed_radar_root=args.precomputed_radar_root,
        radar_norm_mode="frame_logminmax" if vision_only else "global_stats",
        radar_norm_stats="" if vision_only else args.radar_norm_stats,
        cache_radar_maps=False,
    )
    if vision_only:
        return VehicleTrackTemporalDataset(
            base,
            history_len=5,
            future_steps=8,
            vehicle_track_json=args.motion_tracks,
            max_vehicles_per_frame=8,
        )
    return TemporalSequenceDataset(
        base,
        history_len=5,
        future_steps=8,
        include_future_maps=False,
        vehicle_track_json=(
            args.motion_tracks
            if args.method in {"wo_ramf", "wo_afsp"}
            else ""
        ),
        max_vehicles_per_frame=8,
    )


def dataset_manifest_row(
    temporal: TemporalSequenceDataset,
    scenario: str,
    split: str,
) -> dict[str, object]:
    labels = sorted(
        {
            int(temporal.base.samples[sample_id]["label"])
            for window in temporal.windows
            for sample_id in window["future_indices"]
        }
    )
    return {
        "scenario": scenario,
        "split": split,
        "raw_rows": len(temporal.base),
        "contiguous_windows": len(temporal),
        "history_length": temporal.history_len,
        "prediction_horizon": temporal.future_steps,
        "beam_labels": ",".join(str(label) for label in labels),
        "used_for_gradient_updates": split == "train",
        "used_for_model_selection": False,
    }


def flatten_metrics(
    scenario: str,
    split: str,
    metrics: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = {
        "scenario": scenario,
        "split": split,
        "windows": int(metrics["windows"]),
        "top1_mean": float(metrics["top1_mean"]),
        "top3_mean": float(metrics["top3_mean"]),
        "top5_mean": float(metrics["top5_mean"]),
        "within_one_accuracy": float(metrics["within_one_accuracy"]),
        "beam_mae": float(metrics["beam_mae"]),
        "loss": float(metrics["loss"]),
    }
    horizon_rows = [
        {
            "scenario": scenario,
            "split": split,
            "horizon": horizon,
            "top1": float(metrics["step_metrics"][f"Top1@t+{horizon}"]),
            "top3": float(metrics["step_metrics"][f"Top3@t+{horizon}"]),
            "top5": float(metrics["step_metrics"][f"Top5@t+{horizon}"]),
        }
        for horizon in range(1, 9)
    ]
    return summary, horizon_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_dropout(
    model: nn.Module,
    explicit_probability: float,
    attention_probability: float,
) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = explicit_probability
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = attention_probability


def build_optimizer(
    model: nn.Module,
    other_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decoder, other = [], []
    for name, parameter in model.named_parameters():
        (decoder if name.startswith("decoder.") else other).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": other, "lr": other_lr},
            {"params": decoder, "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )

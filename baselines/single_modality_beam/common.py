from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.single_modality_beam.radar_model import RadarOnlyTemporalBeamModel
from baselines.single_modality_beam.vision_model import VisionOnlyTemporalBeamModel


def build_model(
    modality: str,
    feat_dim: int,
    num_beams: int,
    history_len: int,
    pred_horizon_max: int,
) -> nn.Module:
    if modality == "vision":
        return VisionOnlyTemporalBeamModel(
            feat_dim=feat_dim,
            num_beams=num_beams,
            history_len=history_len,
            pred_horizon_max=pred_horizon_max,
        )
    if modality == "radar":
        return RadarOnlyTemporalBeamModel(
            feat_dim=feat_dim,
            num_beams=num_beams,
            history_len=history_len,
            pred_horizon_max=pred_horizon_max,
        )
    raise ValueError(f"Unsupported modality: {modality}")


def model_inputs(
    modality: str,
    batch: dict,
    pred_horizon: int,
    target_tokens: torch.Tensor | None = None,
    teacher_forcing: bool = False,
) -> dict:
    shared = {
        "pred_horizon": pred_horizon,
        "target_tokens": target_tokens,
        "teacher_forcing": teacher_forcing,
    }
    if modality == "vision":
        return {"hist_vehicle_bboxes": batch["hist_vehicle_bboxes"], **shared}
    if modality == "radar":
        return {
            "hist_range_doppler_map": batch["hist_range_doppler_map"],
            "hist_range_angle_map": batch["hist_range_angle_map"],
            "hist_delay_doppler_map": batch["hist_delay_doppler_map"],
            "hist_power_map": batch["hist_power_map"],
            **shared,
        }
    raise ValueError(f"Unsupported modality: {modality}")


def parameter_count(model: nn.Module) -> dict[str, int]:
    base = model.module if isinstance(model, nn.DataParallel) else model
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}

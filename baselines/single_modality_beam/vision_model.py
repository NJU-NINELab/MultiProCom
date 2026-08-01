from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from multimodal_encoders import MotionComponentVisualEncoder
from multiprocom import AutoregressiveFutureStatePredictor


class VisionOnlyTemporalBeamModel(nn.Module):
    """Motion-component-only model using the same AFSP module as MultiProCom."""

    def __init__(
        self,
        feat_dim: int = 256,
        num_beams: int = 64,
        history_len: int = 5,
        pred_horizon_max: int = 8,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.pred_horizon_max = int(pred_horizon_max)
        self.vision_encoder = MotionComponentVisualEncoder(feat_dim=feat_dim)
        self.decoder = AutoregressiveFutureStatePredictor(
            feat_dim=feat_dim,
            num_beams=num_beams,
            pred_horizon_max=pred_horizon_max,
        )

    def forward(
        self,
        hist_vehicle_bboxes: Optional[torch.Tensor] = None,
        pred_horizon: Optional[int] = None,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> dict[str, torch.Tensor]:
        if hist_vehicle_bboxes is None:
            raise ValueError("hist_vehicle_bboxes is required.")
        _, seq_len = hist_vehicle_bboxes.shape[:2]
        hist_tokens = self.vision_encoder(hist_vehicle_bboxes)
        output_device = hist_vehicle_bboxes.device
        if seq_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {seq_len}.")
        if pred_horizon is None:
            pred_horizon = self.pred_horizon_max
        if not (1 <= int(pred_horizon) <= self.pred_horizon_max):
            raise ValueError(f"pred_horizon must be in [1, {self.pred_horizon_max}].")

        decoded = self.decoder(
            hist_tokens=hist_tokens,
            target_tokens=target_tokens,
            teacher_forcing=teacher_forcing,
        )
        logits_max = decoded["beam_logits_seq_max"]
        return {
            "beam_logits_seq": logits_max[:, : int(pred_horizon), :],
            "beam_logits_seq_max": logits_max,
            "decoder_states": decoded["decoder_states"],
            "hist_tokens": hist_tokens,
            "pred_horizon": torch.tensor(int(pred_horizon), device=output_device),
        }

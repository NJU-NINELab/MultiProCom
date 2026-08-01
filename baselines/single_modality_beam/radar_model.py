from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from multimodal_encoders import MultiViewRadarEncoder
from multiprocom import AutoregressiveFutureStatePredictor


class RadarOnlyTemporalBeamModel(nn.Module):
    """Four-map RF-only ablation using the same temporal decoder as the fusion model."""

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
        self.radar_encoder = MultiViewRadarEncoder(token_dim=feat_dim)
        self.decoder = AutoregressiveFutureStatePredictor(
            feat_dim=feat_dim,
            num_beams=num_beams,
            pred_horizon_max=pred_horizon_max,
        )

    def forward(
        self,
        hist_range_doppler_map: torch.Tensor,
        hist_range_angle_map: torch.Tensor,
        hist_delay_doppler_map: torch.Tensor,
        hist_power_map: torch.Tensor,
        pred_horizon: Optional[int] = None,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> dict[str, torch.Tensor]:
        if hist_range_doppler_map.ndim != 5:
            raise ValueError(
                "hist_range_doppler_map expected [B,N,1,H,W], "
                f"got {tuple(hist_range_doppler_map.shape)}"
            )
        bsz, seq_len = hist_range_doppler_map.shape[:2]
        if seq_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {seq_len}.")
        if pred_horizon is None:
            pred_horizon = self.pred_horizon_max
        if not (1 <= int(pred_horizon) <= self.pred_horizon_max):
            raise ValueError(f"pred_horizon must be in [1, {self.pred_horizon_max}].")

        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(bsz * seq_len, *x.shape[2:])

        hist_flat, branch_tokens = self.radar_encoder(
            rd_map=flatten(hist_range_doppler_map),
            ra_map=flatten(hist_range_angle_map),
            dd_map=flatten(hist_delay_doppler_map),
            p_map=flatten(hist_power_map),
        )
        hist_tokens = hist_flat.view(bsz, seq_len, -1)
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
            "radar_branch_tokens": branch_tokens,
            "pred_horizon": torch.tensor(int(pred_horizon), device=hist_range_doppler_map.device),
        }

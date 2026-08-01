from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from multimodal_encoders import MotionComponentVisualEncoder, MultiViewRadarEncoder
from multiprocom import AutoregressiveFutureStatePredictor, ReliabilityAwareMultimodalFusion


@dataclass(frozen=True)
class AblationConfig:
    ablation_name: str
    fusion_mode: str
    fixed_radar_weight: float
    decoder_mode: str


_ABLATION_PRESETS: dict[str, AblationConfig] = {
    "wo_ramf_r05_v05": AblationConfig("wo_ramf_r05_v05", "fixed", 0.5, "afsp"),
    "wo_afsp_parallel": AblationConfig("wo_afsp_parallel", "ramf", 0.8, "parallel_step"),
}


def resolve_ablation_config(
    ablation_name: str,
    fusion_mode: str = "auto",
    fixed_radar_weight: float = 0.8,
    decoder_mode: str = "auto",
) -> AblationConfig:
    name = ablation_name.strip()
    if name:
        if name not in _ABLATION_PRESETS:
            raise ValueError(
                f"Unknown --ablation-name {name!r}. Valid presets: {', '.join(sorted(_ABLATION_PRESETS))}"
            )
        preset = _ABLATION_PRESETS[name]
        resolved_fusion = preset.fusion_mode if fusion_mode == "auto" else fusion_mode
        resolved_decoder = preset.decoder_mode if decoder_mode == "auto" else decoder_mode
        resolved_weight = preset.fixed_radar_weight if fusion_mode == "auto" else fixed_radar_weight
        return AblationConfig(name, resolved_fusion, float(resolved_weight), resolved_decoder)

    if fusion_mode == "auto" or decoder_mode == "auto":
        raise ValueError("Either --ablation-name or explicit --fusion-mode/--decoder-mode must be provided.")
    return AblationConfig(
        ablation_name=f"custom_{fusion_mode}_{decoder_mode}",
        fusion_mode=fusion_mode,
        fixed_radar_weight=float(fixed_radar_weight),
        decoder_mode=decoder_mode,
    )


class FixedRatioFusion(nn.Module):
    """Fixed-ratio multimodal fusion used by w/o RAMF ablations."""

    def __init__(self, feat_dim: int = 256, fixed_radar_weight: float = 0.8):
        super().__init__()
        if not (0.0 <= fixed_radar_weight <= 1.0):
            raise ValueError("fixed_radar_weight must be in [0, 1].")
        self.fixed_radar_weight = float(fixed_radar_weight)
        self.fixed_vision_weight = 1.0 - self.fixed_radar_weight
        self.radar_proj = nn.Linear(feat_dim, feat_dim)
        self.vision_proj = nn.Linear(feat_dim, feat_dim)

    def forward(
        self,
        f_r: torch.Tensor,
        f_v: torch.Tensor,
        vision_availability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del vision_availability
        if f_r.shape != f_v.shape:
            raise ValueError(f"Shape mismatch: radar={tuple(f_r.shape)} vision={tuple(f_v.shape)}")
        if f_r.ndim != 3:
            raise ValueError(f"Expected [B, L, D], got {tuple(f_r.shape)}")

        w_r = torch.full(f_r.shape[:2], self.fixed_radar_weight, device=f_r.device, dtype=f_r.dtype)
        w_v = torch.full(f_r.shape[:2], self.fixed_vision_weight, device=f_r.device, dtype=f_r.dtype)
        fused = w_r.unsqueeze(-1) * self.radar_proj(f_r) + w_v.unsqueeze(-1) * self.vision_proj(f_v)
        gate = torch.stack([w_r, w_v], dim=-1)
        gate_entropy = -(gate * torch.log(gate + 1e-8)).sum(dim=-1)
        tau = torch.zeros((), device=f_r.device, dtype=f_r.dtype)
        return {
            "fused": fused,
            "w_r": w_r,
            "w_v": w_v,
            "gate_entropy": gate_entropy,
            "tau": tau.detach(),
            "w_r_gate": w_r,
            "w_v_gate": w_v,
        }


class NonAutoregressiveFutureStatePredictor(nn.Module):
    """Non-autoregressive decoder: all future steps are predicted in parallel."""

    def __init__(self, feat_dim: int = 256, num_beams: int = 64, pred_horizon_max: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pred_horizon_max = int(pred_horizon_max)
        self.num_beams = int(num_beams)
        self.hist_gru = nn.GRU(input_size=feat_dim, hidden_size=feat_dim, num_layers=1, batch_first=True)
        self.step_embed = nn.Parameter(torch.randn(self.pred_horizon_max, feat_dim) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_beams),
        )

    def forward(
        self,
        hist_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del target_tokens, teacher_forcing
        _, h_n = self.hist_gru(hist_tokens)
        hidden = h_n[-1]
        states = hidden.unsqueeze(1) + self.step_embed.unsqueeze(0)
        logits = self.head(states)
        return {"beam_logits_seq_max": logits, "decoder_states": states}


def build_decoder(
    decoder_mode: str,
    feat_dim: int,
    num_beams: int,
    pred_horizon_max: int,
) -> nn.Module:
    if decoder_mode == "afsp":
        return AutoregressiveFutureStatePredictor(
            feat_dim=feat_dim, num_beams=num_beams, pred_horizon_max=pred_horizon_max
        )
    if decoder_mode == "parallel_step":
        return NonAutoregressiveFutureStatePredictor(
            feat_dim=feat_dim, num_beams=num_beams, pred_horizon_max=pred_horizon_max
        )
    raise ValueError(f"Unsupported decoder_mode: {decoder_mode}")


class MultiProComAblation(nn.Module):
    """Multimodal temporal beam model with switchable RAMF and AFSP ablations."""

    def __init__(
        self,
        feat_dim: int = 256,
        num_beams: int = 64,
        history_len: int = 5,
        pred_horizon_max: int = 8,
        fusion_mode: str = "ramf",
        fixed_radar_weight: float = 0.8,
        decoder_mode: str = "afsp",
        max_vision_weight: float = 0.35,
    ):
        super().__init__()
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        if pred_horizon_max < 1:
            raise ValueError("pred_horizon_max must be >= 1.")
        if fusion_mode not in {"ramf", "fixed"}:
            raise ValueError("fusion_mode must be 'ramf' or 'fixed'.")

        self.history_len = int(history_len)
        self.pred_horizon_max = int(pred_horizon_max)
        self.fusion_mode = str(fusion_mode)
        self.decoder_mode = str(decoder_mode)
        self.fixed_radar_weight = float(fixed_radar_weight)

        self.vision_encoder = MotionComponentVisualEncoder(feat_dim=feat_dim)
        self.radar_encoder = MultiViewRadarEncoder(token_dim=feat_dim)

        if self.fusion_mode == "ramf":
            self.fusion = ReliabilityAwareMultimodalFusion(
                feat_dim=feat_dim,
                max_vision_weight=max_vision_weight,
            )
        else:
            self.fusion = FixedRatioFusion(feat_dim=feat_dim, fixed_radar_weight=fixed_radar_weight)
        self.decoder = build_decoder(
            decoder_mode=self.decoder_mode,
            feat_dim=feat_dim,
            num_beams=num_beams,
            pred_horizon_max=pred_horizon_max,
        )

    def _encode_histories(
        self,
        hist_vehicle_bboxes: torch.Tensor,
        hist_range_doppler_map: torch.Tensor,
        hist_range_angle_map: torch.Tensor,
        hist_delay_doppler_map: torch.Tensor,
        hist_power_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if hist_vehicle_bboxes.ndim != 4:
            raise ValueError(
                f"hist_vehicle_bboxes expected [B,N,K,6], got {tuple(hist_vehicle_bboxes.shape)}"
            )
        bsz, seq_len = hist_vehicle_bboxes.shape[:2]
        if seq_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {seq_len}.")

        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(bsz * seq_len, *x.shape[2:])

        f_v = self.vision_encoder(hist_vehicle_bboxes)
        f_r_flat, _ = self.radar_encoder(
            rd_map=flatten(hist_range_doppler_map),
            ra_map=flatten(hist_range_angle_map),
            dd_map=flatten(hist_delay_doppler_map),
            p_map=flatten(hist_power_map),
        )
        feat_dim = f_v.shape[-1]
        f_r = f_r_flat.view(bsz, seq_len, feat_dim)
        valid = (hist_vehicle_bboxes[..., -1] > 0.0).to(f_v.dtype)
        confidence = hist_vehicle_bboxes[..., 4].clamp(0.0, 1.0) * valid
        vision_availability = confidence.max(dim=-1).values
        return {"f_v": f_v, "f_r": f_r, "vision_availability": vision_availability}

    def forward(
        self,
        hist_vehicle_bboxes: torch.Tensor,
        hist_range_doppler_map: torch.Tensor,
        hist_range_angle_map: torch.Tensor,
        hist_delay_doppler_map: torch.Tensor,
        hist_power_map: torch.Tensor,
        pred_horizon: Optional[int] = None,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if pred_horizon is None:
            pred_horizon = self.pred_horizon_max
        pred_horizon = int(pred_horizon)
        if pred_horizon < 1 or pred_horizon > self.pred_horizon_max:
            raise ValueError(f"pred_horizon must be in [1, {self.pred_horizon_max}], got {pred_horizon}.")

        encoded = self._encode_histories(
            hist_vehicle_bboxes=hist_vehicle_bboxes,
            hist_range_doppler_map=hist_range_doppler_map,
            hist_range_angle_map=hist_range_angle_map,
            hist_delay_doppler_map=hist_delay_doppler_map,
            hist_power_map=hist_power_map,
        )
        fused = self.fusion(
            f_r=encoded["f_r"],
            f_v=encoded["f_v"],
            vision_availability=encoded["vision_availability"],
        )
        decoded = self.decoder(
            hist_tokens=fused["fused"],
            target_tokens=target_tokens,
            teacher_forcing=teacher_forcing,
        )
        logits_max = decoded["beam_logits_seq_max"]
        return {
            "beam_logits_seq": logits_max[:, :pred_horizon, :],
            "beam_logits_seq_max": logits_max,
            "w_v": fused["w_v"],
            "w_r": fused["w_r"],
            "hist_fused": fused["fused"],
            "hist_f_v": encoded["f_v"],
            "hist_f_r": encoded["f_r"],
            "vision_availability": encoded["vision_availability"],
            "decoder_states": decoded["decoder_states"],
            "gate_entropy": fused["gate_entropy"],
            "tau": fused["tau"],
            "pred_horizon": torch.tensor(pred_horizon, device=hist_vehicle_bboxes.device, dtype=torch.long),
        }


def parameter_count(model: nn.Module) -> dict[str, int]:
    base = model.module if isinstance(model, nn.DataParallel) else model
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}

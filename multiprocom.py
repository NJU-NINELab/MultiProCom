from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_encoders import MotionComponentVisualEncoder, MultiViewRadarEncoder


class ReliabilityAwareMultimodalFusion(nn.Module):
    """
    Radar-dominant fusion: radar is always the main stream,
    while vision contributes a bounded residual gain.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        max_vision_weight: float = 0.35,
        radar_prior_bias: float = 0.3,
        tau_init: float = 1.0,
    ):
        super().__init__()
        if tau_init <= 0.0:
            raise ValueError("tau_init must be > 0.")
        self.max_vision_weight = float(max_vision_weight)
        self.radar_prior_bias = float(radar_prior_bias)
        self.radar_proj = nn.Linear(feat_dim, feat_dim)
        self.vision_proj = nn.Linear(feat_dim, feat_dim)
        self.gate_norm_r = nn.LayerNorm(feat_dim)
        self.gate_norm_v = nn.LayerNorm(feat_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 2),
        )
        tau_raw = torch.log(torch.exp(torch.tensor(float(tau_init))) - 1.0)
        self.tau_raw = nn.Parameter(tau_raw)

    def forward(
        self,
        f_r: torch.Tensor,
        f_v: torch.Tensor,
        vision_availability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f_r: [B, L, D]
            f_v: [B, L, D]
        Returns:
            fused: [B, L, D]
            w_r: [B, L]
            w_v: [B, L]
        """
        if f_r.shape != f_v.shape:
            raise ValueError(f"Shape mismatch: radar={tuple(f_r.shape)} vision={tuple(f_v.shape)}")
        if f_r.ndim != 3:
            raise ValueError(f"Expected [B, L, D], got {tuple(f_r.shape)}")

        bsz, seq_len, feat_dim = f_r.shape
        f_r_2d = f_r.reshape(bsz * seq_len, feat_dim)
        f_v_2d = f_v.reshape(bsz * seq_len, feat_dim)

        f_r_gate = self.gate_norm_r(f_r_2d)
        f_v_gate = self.gate_norm_v(f_v_2d)
        tau = F.softplus(self.tau_raw) + 1e-4
        gate_logits = self.gate_mlp(torch.cat([f_r_gate, f_v_gate], dim=-1))
        gate_logits[:, 0] = gate_logits[:, 0] + self.radar_prior_bias
        gate = torch.softmax(gate_logits / tau, dim=-1)
        w_r_gate = gate[:, 0]
        w_v_gate = gate[:, 1]
        w_v_base = w_v_gate
        if vision_availability is not None:
            availability = vision_availability.reshape(bsz * seq_len).to(dtype=w_v_base.dtype)
            w_v_base = w_v_base * availability.clamp(0.0, 1.0)
        # Radar-dominant bounded fusion:
        # even if raw gate prefers vision, vision contribution is capped.
        w_v = self.max_vision_weight * w_v_base
        w_r = 1.0 - w_v

        f_r_main = self.radar_proj(f_r_2d)
        f_v_aux = self.vision_proj(f_v_2d)
        fused = w_r.unsqueeze(-1) * f_r_main + w_v.unsqueeze(-1) * f_v_aux
        gate_effective = torch.stack([w_r, w_v], dim=-1)
        gate_entropy = -(gate_effective * torch.log(gate_effective + 1e-8)).sum(dim=-1)

        return {
            "fused": fused.view(bsz, seq_len, feat_dim),
            "w_r": w_r.view(bsz, seq_len),
            "w_v": w_v.view(bsz, seq_len),
            "gate_entropy": gate_entropy.view(bsz, seq_len),
            "tau": tau.detach(),
            "w_v_gate": w_v_gate.view(bsz, seq_len),
            "w_r_gate": w_r_gate.view(bsz, seq_len),
        }


class AutoregressiveFutureStatePredictor(nn.Module):
    """Encodes history and decodes future multi-step beam logits."""

    def __init__(
        self,
        feat_dim: int = 256,
        num_beams: int = 64,
        pred_horizon_max: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if pred_horizon_max < 1:
            raise ValueError("pred_horizon_max must be >= 1.")
        self.pred_horizon_max = int(pred_horizon_max)
        self.num_beams = int(num_beams)
        self.sos_token_id = int(num_beams)
        self.hist_gru = nn.GRU(input_size=feat_dim, hidden_size=feat_dim, num_layers=1, batch_first=True)
        self.step_embed = nn.Parameter(torch.randn(self.pred_horizon_max, feat_dim) * 0.02)
        self.beam_token_embed = nn.Embedding(self.num_beams + 1, feat_dim)
        self.dec_cell = nn.GRUCell(input_size=feat_dim, hidden_size=feat_dim)
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
        # hist_tokens: [B, N, D]
        _, h_n = self.hist_gru(hist_tokens)
        hidden = h_n[-1]  # [B, D]

        logits_steps = []
        state_steps = []
        bsz = hist_tokens.shape[0]
        prev_tokens = torch.full(
            (bsz,),
            fill_value=self.sos_token_id,
            device=hist_tokens.device,
            dtype=torch.long,
        )
        for step in range(self.pred_horizon_max):
            if step > 0:
                if teacher_forcing and target_tokens is not None:
                    prev_tokens = target_tokens[:, step - 1].long().clamp(0, self.num_beams - 1)
                else:
                    prev_tokens = logits_steps[-1].argmax(dim=1)
            prev_emb = self.beam_token_embed(prev_tokens)
            step_in = self.step_embed[step].unsqueeze(0).expand(bsz, -1) + prev_emb
            hidden = self.dec_cell(step_in, hidden)
            state_steps.append(hidden)
            logits_steps.append(self.head(hidden))

        return {
            "beam_logits_seq_max": torch.stack(logits_steps, dim=1),  # [B, Mmax, K]
            "decoder_states": torch.stack(state_steps, dim=1),        # [B, Mmax, D]
        }


class MultiProCom(nn.Module):
    """
    Temporal multimodal model:
    history (vision/radar) -> radar-dominant fusion -> future beam sequence logits.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_beams: int = 64,
        history_len: int = 5,
        pred_horizon_max: int = 8,
        max_vision_weight: float = 0.35,
    ):
        super().__init__()
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        if pred_horizon_max < 1:
            raise ValueError("pred_horizon_max must be >= 1.")

        self.history_len = int(history_len)
        self.pred_horizon_max = int(pred_horizon_max)
        self.vision_encoder = MotionComponentVisualEncoder(feat_dim=feat_dim)
        self.radar_encoder = MultiViewRadarEncoder(token_dim=feat_dim)
        self.fusion = ReliabilityAwareMultimodalFusion(
            feat_dim=feat_dim,
            max_vision_weight=max_vision_weight,
        )
        self.decoder = AutoregressiveFutureStatePredictor(
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
            raise ValueError(
                f"Input history length mismatch: expected {self.history_len}, got {seq_len}."
            )

        rd_flat = hist_range_doppler_map.reshape(bsz * seq_len, *hist_range_doppler_map.shape[2:])
        ra_flat = hist_range_angle_map.reshape(bsz * seq_len, *hist_range_angle_map.shape[2:])
        dd_flat = hist_delay_doppler_map.reshape(bsz * seq_len, *hist_delay_doppler_map.shape[2:])
        p_flat = hist_power_map.reshape(bsz * seq_len, *hist_power_map.shape[2:])

        f_v = self.vision_encoder(hist_vehicle_bboxes)
        f_r_flat, _ = self.radar_encoder(rd_map=rd_flat, ra_map=ra_flat, dd_map=dd_flat, p_map=p_flat)

        feat_dim = f_v.shape[-1]
        f_r = f_r_flat.view(bsz, seq_len, feat_dim)
        # Use detector/tracker confidence as a continuous availability signal.  A boolean
        # flag forced noisy low-light motion boxes to contribute as much as clean tracks.
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
            raise ValueError(
                f"pred_horizon must satisfy 1 <= pred_horizon <= {self.pred_horizon_max}, got {pred_horizon}."
            )

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


if __name__ == "__main__":
    bsz, n_hist, m_max = 2, 5, 8
    model = MultiProCom(feat_dim=128, num_beams=64, history_len=n_hist, pred_horizon_max=m_max)
    out = model(
        hist_vehicle_bboxes=torch.rand(bsz, n_hist, 8, 6),
        hist_range_doppler_map=torch.rand(bsz, n_hist, 1, 32, 64),
        hist_range_angle_map=torch.rand(bsz, n_hist, 1, 64, 64),
        hist_delay_doppler_map=torch.rand(bsz, n_hist, 1, 32, 64),
        hist_power_map=torch.rand(bsz, n_hist, 1, 32, 64),
        pred_horizon=5,
    )
    print("beam_logits_seq:", tuple(out["beam_logits_seq"].shape))
    print("w_r:", tuple(out["w_r"].shape), "w_v:", tuple(out["w_v"].shape))

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ConvMapEncoder(nn.Module):
    """Small CNN encoder for one RF map branch."""

    def __init__(self, in_chans: int = 1, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.proj(x)


class MotionComponentVisualEncoder(nn.Module):
    """Encode each frame's motion-component set and adjacent-frame feature change."""

    def __init__(self, feat_dim: int = 256, input_dim: int = 6, num_heads: int = 8):
        super().__init__()
        self.input_dim = int(input_dim)
        self.box_encoder = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        self.frame_query = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.no_object_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.frame_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.motion_encoder = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        nn.init.trunc_normal_(self.frame_query, std=0.02)
        nn.init.trunc_normal_(self.no_object_token, std=0.02)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.ndim != 4 or boxes.shape[-1] != self.input_dim:
            raise ValueError(f"boxes expected [B,N,K,{self.input_dim}], got {tuple(boxes.shape)}")
        bsz, seq_len, num_boxes, _ = boxes.shape
        flat = boxes.reshape(bsz * seq_len, num_boxes, self.input_dim)
        box_tokens = self.box_encoder(flat)
        tokens = torch.cat(
            [self.no_object_token.expand(bsz * seq_len, -1, -1), box_tokens],
            dim=1,
        )
        invalid_boxes = flat[..., -1] <= 0.0
        key_padding_mask = torch.cat(
            [
                torch.zeros((bsz * seq_len, 1), dtype=torch.bool, device=boxes.device),
                invalid_boxes,
            ],
            dim=1,
        )
        frame_tokens, _ = self.frame_attention(
            query=self.frame_query.expand(bsz * seq_len, -1, -1),
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        frame_tokens = frame_tokens[:, 0, :].view(bsz, seq_len, -1)
        delta = torch.zeros_like(frame_tokens)
        delta[:, 1:, :] = frame_tokens[:, 1:, :] - frame_tokens[:, :-1, :]
        return self.motion_encoder(torch.cat([frame_tokens, delta], dim=-1))


class RadarTokenAggregator(nn.Module):
    """
    RF-token Transformer aggregator:
    input tokens = [cls, RD, RA, DD, P], output cls token as unified radar representation.
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, token_dim))

        ff_dim = int(token_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens_4: torch.Tensor) -> torch.Tensor:
        """
        tokens_4: [B, 4, D] for (RD, RA, DD, P)
        returns cls token: [B, D]
        """
        bsz = tokens_4.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, tokens_4], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        return self.norm(tokens[:, 0, :])


class MultiViewRadarEncoder(nn.Module):
    """Encode the RD, RA, DD and power maps and aggregate their view tokens."""

    def __init__(
        self,
        token_dim: int = 256,
        agg_layers: int = 2,
        agg_heads: int = 8,
        agg_dropout: float = 0.1,
    ):
        super().__init__()
        self.rd_encoder = ConvMapEncoder(in_chans=1, embed_dim=token_dim)
        self.ra_encoder = ConvMapEncoder(in_chans=1, embed_dim=token_dim)
        self.dd_encoder = ConvMapEncoder(in_chans=1, embed_dim=token_dim)
        self.p_encoder = ConvMapEncoder(in_chans=1, embed_dim=token_dim)
        self.aggregator = RadarTokenAggregator(
            token_dim=token_dim,
            num_layers=agg_layers,
            num_heads=agg_heads,
            dropout=agg_dropout,
        )

    def forward(
        self,
        rd_map: torch.Tensor,
        ra_map: torch.Tensor,
        dd_map: torch.Tensor,
        p_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_rd = self.rd_encoder(rd_map)
        z_ra = self.ra_encoder(ra_map)
        z_dd = self.dd_encoder(dd_map)
        z_p = self.p_encoder(p_map)

        rf_tokens = torch.stack([z_rd, z_ra, z_dd, z_p], dim=1)
        z_r = self.aggregator(rf_tokens)

        branch_tokens = {
            "z_rd": z_rd,
            "z_ra": z_ra,
            "z_dd": z_dd,
            "z_p": z_p,
        }
        return z_r, branch_tokens

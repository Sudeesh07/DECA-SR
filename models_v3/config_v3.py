"""Configuration helpers for models_v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from models_v3.network_v3 import DSRPSNRNet


@dataclass
class DSRPSNRConfig:
    scale: int = 4
    in_channels: int = 3
    embed_dim: int = 64
    enc_blocks: Tuple[int, int, int] = (4, 6, 6)
    depa_k: int = 8
    num_heads: int = 4

    depa_max_disparity_x4: int = 32
    depa_max_disparity_x2: int = 64
    corr_max_disp_x4: int = 12
    corr_max_disp_x2: int = 16

    use_bottleneck_stereo: bool = True
    sample_span: float = 4.0
    residual_range: float = 3.0

    depa_max_disparity: int | None = field(default=None)
    corr_max_disp: int | None = field(default=None)

    def resolve_stereo_ranges(self) -> tuple[int, int]:
        depa = self.depa_max_disparity
        corr = self.corr_max_disp
        if depa is None:
            depa = self.depa_max_disparity_x2 if self.scale == 2 else self.depa_max_disparity_x4
        if corr is None:
            corr = self.corr_max_disp_x2 if self.scale == 2 else self.corr_max_disp_x4
        return depa, corr


def build_model_v3_from_config(cfg: DSRPSNRConfig) -> DSRPSNRNet:
    """Build DSRPSNRNet from DSRPSNRConfig."""
    depa_max_disparity, corr_max_disp = cfg.resolve_stereo_ranges()
    return DSRPSNRNet(
        scale=cfg.scale,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        enc_blocks=cfg.enc_blocks,
        depa_k=cfg.depa_k,
        depa_max_disparity=depa_max_disparity,
        corr_max_disp=corr_max_disp,
        use_bottleneck_stereo=cfg.use_bottleneck_stereo,
        num_heads=cfg.num_heads,
        sample_span=cfg.sample_span,
        residual_range=cfg.residual_range,
    )

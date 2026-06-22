"""StereoSR V9 — DECA-SR (Deformable Epipolar Cost-view-Aware stereo SR).

Novelty: DECVF, a cross-view fusion operator that builds a lightweight cost
volume along each epipolar row to obtain a soft disparity prior + match
confidence, uses the prior to initialize and bound learned deformable offsets,
sub-pixel-samples the other view (no softmax-averaging blur), and fuses through
a confidence + SFAM gate with a zero-init residual. Paired with a
reparameterizable large-kernel NAF trunk and the project's validated SCAM.
"""
from __future__ import annotations

import argparse

from .config_v9 import SIZE_PRESETS, get_config_v9
from .deformable_fusion import DeformableEpipolarFusion
from .ema import ModelEMA
from .epipolar_attention import StereoCrossAttention
from .losses_v9 import (
    CharbonnierLoss,
    FocalFrequencyLoss,
    StereoPhotometricConsistency,
    StereoSRV9Loss,
)
from .network_v9 import DECABlock, StereoSRV9, StereoSRV9Output


def build_model_v9(args: argparse.Namespace) -> StereoSRV9:
    return StereoSRV9(
        scale=args.scale,
        in_channels=args.in_channels,
        channels=args.channels,
        num_blocks=args.num_blocks,
        dw_expansion=args.dw_expansion,
        ffn_expansion=args.ffn_expansion,
        large_kernel=args.large_kernel,
        use_scam=not getattr(args, "no_scam", False),
        use_decvf=not getattr(args, "no_decvf", False),
        decvf_every=args.decvf_every,
        cost_volume_d=getattr(args, "cost_volume_d", 24),
        decvf_k=getattr(args, "decvf_k", 8),
        decvf_y_range=getattr(args, "decvf_y_range", 1.5),
    )


def build_loss_v9(args: argparse.Namespace) -> StereoSRV9Loss:
    return StereoSRV9Loss(
        charbonnier_eps=args.charbonnier_eps,
        stereo_weight=args.stereo_weight,
        stereo_max_shift=args.stereo_max_shift,
        ffl_weight=getattr(args, "ffl_weight", 0.1),
    )


__all__ = [
    "SIZE_PRESETS",
    "StereoSRV9",
    "StereoSRV9Output",
    "DECABlock",
    "StereoSRV9Loss",
    "CharbonnierLoss",
    "FocalFrequencyLoss",
    "StereoPhotometricConsistency",
    "DeformableEpipolarFusion",
    "StereoCrossAttention",
    "ModelEMA",
    "build_model_v9",
    "build_loss_v9",
    "get_config_v9",
]

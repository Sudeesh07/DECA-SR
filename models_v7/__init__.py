"""StereoSR V7 — multi-scale SCAM trunk with disparity-prior bias.

Three novel contributions over V6:
1. Disparity-prior-conditioned SCAM (soft-argmax cost-volume bias).
2. Multi-scale trunk (parallel LR/2 coarse branch, zero-init fusion).
3. Engineering bundle (EMA, AdamW(0.9, 0.9), patches 48×144, self-ensemble,
   RGB-shuffle augmentation).
"""
from __future__ import annotations

import argparse

from .config_v7 import SIZE_PRESETS, get_config_v7
from .losses_v7 import CharbonnierLoss, StereoPhotometricConsistency, StereoSRV7Loss
from .network_v7 import StereoSpatialBlock, StereoSRV7, StereoSRV7Output
from .ema import ModelEMA


def build_model_v7(args: argparse.Namespace) -> StereoSRV7:
    """Factory constructing StereoSRV7 from argparse-style config."""
    return StereoSRV7(
        scale=args.scale,
        in_channels=args.in_channels,
        channels=args.channels,
        num_blocks=args.num_blocks,
        dw_expansion=args.dw_expansion,
        ffn_expansion=args.ffn_expansion,
        use_disparity_prior=not getattr(args, "no_disparity_prior", False),
        sigma_init=getattr(args, "sigma_init", 4.0),
        use_multiscale=not getattr(args, "no_multiscale", False),
        num_blocks_coarse=args.num_blocks_coarse,
    )


def build_loss_v7(args: argparse.Namespace) -> StereoSRV7Loss:
    return StereoSRV7Loss(
        charbonnier_eps=args.charbonnier_eps,
        stereo_weight=args.stereo_weight,
        stereo_max_shift=args.stereo_max_shift,
    )


__all__ = [
    "SIZE_PRESETS",
    "StereoSRV7",
    "StereoSRV7Output",
    "StereoSpatialBlock",
    "StereoSRV7Loss",
    "CharbonnierLoss",
    "StereoPhotometricConsistency",
    "ModelEMA",
    "build_model_v7",
    "build_loss_v7",
    "get_config_v7",
]

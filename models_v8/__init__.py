"""StereoSR V8 — dual-axis cross-view attention for stereo SR.

Novelty: channel-transposed cross-view attention (CT-CVA) complementing the
spatial SCAM. Architecturally simple, scale-invariant by construction,
sidesteps every V1–V7 failure mode (no spectral magnitude mixing, no bounded
tanh, no pixel-disparity priors, no pool-area dependence, no multi-scale).
"""
from __future__ import annotations

import argparse

from .channel_cross_view import ChannelTransposedCrossView
from .config_v8 import SIZE_PRESETS, get_config_v8
from .ema import ModelEMA
from .epipolar_attention import StereoCrossAttention
from .losses_v8 import CharbonnierLoss, StereoPhotometricConsistency, StereoSRV8Loss
from .network_v8 import StereoSpatialChannelBlock, StereoSRV8, StereoSRV8Output


def build_model_v8(args: argparse.Namespace) -> StereoSRV8:
    """Factory constructing StereoSRV8 from argparse-style config."""
    return StereoSRV8(
        scale=args.scale,
        in_channels=args.in_channels,
        channels=args.channels,
        num_blocks=args.num_blocks,
        dw_expansion=args.dw_expansion,
        ffn_expansion=args.ffn_expansion,
        use_scam=not getattr(args, "no_scam", False),
        use_ct_cva=not getattr(args, "no_ct_cva", False),
        ct_cva_heads=getattr(args, "ct_cva_heads", 4),
        ct_cva_temperature_init=getattr(args, "ct_cva_temperature_init", 1.0),
    )


def build_loss_v8(args: argparse.Namespace) -> StereoSRV8Loss:
    return StereoSRV8Loss(
        charbonnier_eps=args.charbonnier_eps,
        stereo_weight=args.stereo_weight,
        stereo_max_shift=args.stereo_max_shift,
    )


__all__ = [
    "SIZE_PRESETS",
    "StereoSRV8",
    "StereoSRV8Output",
    "StereoSpatialChannelBlock",
    "StereoSRV8Loss",
    "CharbonnierLoss",
    "StereoPhotometricConsistency",
    "ChannelTransposedCrossView",
    "StereoCrossAttention",
    "ModelEMA",
    "build_model_v8",
    "build_loss_v8",
    "get_config_v8",
]

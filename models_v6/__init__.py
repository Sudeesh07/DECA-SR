"""StereoSR V6: SpecFuse — spectral cross-view fusion for stereo SR."""
from __future__ import annotations

import argparse

from .config_v6 import SIZE_PRESETS, get_config_v6
from .losses_v6 import (
    CharbonnierLoss,
    SpectralConsistencyLoss,
    StereoPhotometricConsistency,
    StereoSRV6Loss,
)
from .network_v6 import SpectralStereoBlock, StereoSRV6, StereoSRV6Output


def build_model_v6(args: argparse.Namespace) -> StereoSRV6:
    """Factory constructing StereoSRV6 from argparse-style config."""
    return StereoSRV6(
        scale=args.scale,
        in_channels=args.in_channels,
        channels=args.channels,
        num_blocks=args.num_blocks,
        dw_expansion=args.dw_expansion,
        ffn_expansion=args.ffn_expansion,
        use_scam=not args.no_scam,
        use_cvsfm=not args.no_cvsfm,
    )


def build_loss_v6(args: argparse.Namespace) -> StereoSRV6Loss:
    return StereoSRV6Loss(
        charbonnier_eps=args.charbonnier_eps,
        stereo_weight=args.stereo_weight,
        spectral_weight=args.spectral_weight,
        stereo_max_shift=args.stereo_max_shift,
    )


__all__ = [
    "SIZE_PRESETS",
    "StereoSRV6",
    "StereoSRV6Output",
    "SpectralStereoBlock",
    "StereoSRV6Loss",
    "CharbonnierLoss",
    "StereoPhotometricConsistency",
    "SpectralConsistencyLoss",
    "build_model_v6",
    "build_loss_v6",
    "get_config_v6",
]

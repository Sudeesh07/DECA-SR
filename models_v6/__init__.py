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
        cvsfm_use_gate=not getattr(args, "cvsfm_no_gate", False),
        cvsfm_use_1d_fft=getattr(args, "cvsfm_1d_fft", False),
        cvsfm_version=(
            "v3" if getattr(args, "cvsfm_v3", False)
            else "v2" if getattr(args, "cvsfm_v2", False)
            else "v1"
        ),
        cvsfm_v3_window=getattr(args, "cvsfm_v3_window", 0),
        cvsfm_v3_hf_only=getattr(args, "cvsfm_v3_hf_only", False),
        cvsfm_v3_gain_scale=getattr(args, "cvsfm_v3_gain_scale", 0.1),
        cvsfm_first=getattr(args, "cvsfm_first", False),
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

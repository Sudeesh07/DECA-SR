"""StereoSR V6 (SpecFuse): clean, shared-weight trunk + single tail upsample.

Macro-architecture
------------------
    LR_L, LR_R  ──► stem ──► [SpectralStereoBlock] × N ──► tail ──► SR_L, SR_R
                                                                         ▲
                             all at LR resolution                        │
                                                        + Bicubic(LR) ───┘

Design principles
-----------------
* **Trunk operates entirely at LR resolution.** No encoder-decoder; no cascaded
  PixelShuffle. The HR representation is constructed only at the tail via a
  single sub-pixel convolution, avoiding the compounding checkerboard artefacts
  present in U-Net SR designs.
* **Shared weights across views.** Both L and R pass through the same trunk,
  enforcing view symmetry and halving the parameter budget (vs. separate towers).
* **Zero-init every residual.** Each SpectralStereoBlock starts as the identity
  (NAFBlock β,γ=0; SCAM β,γ=0; CVSFM γ=0). The full network at step 0 produces
  `bicubic(LR) + 0` ≡ pure bicubic upscale.
* **Cross-view fusion at every block.** Each block runs NAFBlock per-view, then
  SCAM (spatial), then CVSFM (spectral). SCAM and CVSFM are complementary:
  SCAM aligns pixels along epipolar rows; CVSFM exchanges shared appearance in
  the Fourier magnitude domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, NAFBlock, TailUpsample
from .epipolar_attention import StereoCrossAttention
from .spectral_fusion import (
    CrossViewSpectralFusion,
    CrossViewSpectralFusionV2,
    CrossViewSpectralFusionV3,
)


@dataclass
class StereoSRV6Output:
    sr_left: torch.Tensor
    sr_right: torch.Tensor
    aux: dict[str, torch.Tensor]


class SpectralStereoBlock(nn.Module):
    """One trunk block: per-view NAFBlock → SCAM → CVSFM."""

    def __init__(
        self,
        channels: int,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        use_scam: bool = True,
        use_cvsfm: bool = True,
        cvsfm_use_gate: bool = True,
        cvsfm_use_1d_fft: bool = False,
        cvsfm_version: str = "v1",
        cvsfm_v3_window: int = 0,
        cvsfm_v3_hf_only: bool = False,
        cvsfm_v3_gain_scale: float = 0.1,
        cvsfm_first: bool = False,
    ):
        super().__init__()
        if cvsfm_version not in ("v1", "v2", "v3"):
            raise ValueError(f"cvsfm_version must be 'v1', 'v2', or 'v3' (got {cvsfm_version!r})")
        self.cvsfm_first = cvsfm_first
        # Shared per-view spatial processor (single weight set for both L and R).
        self.naf = NAFBlock(channels, dw_expansion, ffn_expansion)
        self.scam = StereoCrossAttention(channels) if use_scam else None
        if not use_cvsfm:
            self.cvsfm = None
        elif cvsfm_version == "v3":
            self.cvsfm = CrossViewSpectralFusionV3(
                channels,
                window=cvsfm_v3_window,
                hf_only=cvsfm_v3_hf_only,
                gain_scale=cvsfm_v3_gain_scale,
            )
        elif cvsfm_version == "v2":
            # v2 ignores use_gate/use_1d_fft (it is 1D by construction and uses
            # a spatial gate, not a per-frequency one).
            self.cvsfm = CrossViewSpectralFusionV2(channels)
        else:
            self.cvsfm = CrossViewSpectralFusion(
                channels,
                use_gate=cvsfm_use_gate,
                use_1d_fft=cvsfm_use_1d_fft,
            )

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-view spatial processing (shared weights).
        x_l = self.naf(x_l)
        x_r = self.naf(x_r)
        # Spectral-first ordering puts CVSFM (v3 = phase-correlation alignment)
        # before SCAM, demoting SCAM to a refinement / fallback step. Use this
        # when CVSFM v3 is intended as the primary cross-view module. Default
        # ordering (SCAM → CVSFM) is preserved for v1/v2 ablations.
        if self.cvsfm_first:
            if self.cvsfm is not None:
                x_l, x_r = self.cvsfm(x_l, x_r)
            if self.scam is not None:
                x_l, x_r = self.scam(x_l, x_r)
        else:
            if self.scam is not None:
                x_l, x_r = self.scam(x_l, x_r)
            if self.cvsfm is not None:
                x_l, x_r = self.cvsfm(x_l, x_r)
        return x_l, x_r


class StereoSRV6(nn.Module):
    """SpecFuse: spectral cross-view fusion for stereo super-resolution."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 20,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        use_scam: bool = True,
        use_cvsfm: bool = True,
        cvsfm_use_gate: bool = True,
        cvsfm_use_1d_fft: bool = False,
        cvsfm_version: str = "v1",
        cvsfm_v3_window: int = 0,
        cvsfm_v3_hf_only: bool = False,
        cvsfm_v3_gain_scale: float = 0.1,
        cvsfm_first: bool = False,
    ):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"scale must be 2, 3, or 4 (got {scale})")
        self.scale = scale

        # Shared stem (same Conv3x3 applied to both views).
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Shared trunk (each block processes both views).
        self.trunk = nn.ModuleList([
            SpectralStereoBlock(
                channels=channels,
                dw_expansion=dw_expansion,
                ffn_expansion=ffn_expansion,
                use_scam=use_scam,
                use_cvsfm=use_cvsfm,
                cvsfm_use_gate=cvsfm_use_gate,
                cvsfm_use_1d_fft=cvsfm_use_1d_fft,
                cvsfm_version=cvsfm_version,
                cvsfm_v3_window=cvsfm_v3_window,
                cvsfm_v3_hf_only=cvsfm_v3_hf_only,
                cvsfm_v3_gain_scale=cvsfm_v3_gain_scale,
                cvsfm_first=cvsfm_first,
            )
            for _ in range(num_blocks)
        ])

        # Optional: a trunk-level LayerNorm to stabilise deep stacks.
        self.trunk_norm = LayerNorm2d(channels)

        # Single tail upsample: Conv3x3 → PixelShuffle(scale). ICNR-initialised.
        self.tail = TailUpsample(channels, in_channels, scale)

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        """Stem for a single view (to be called on L and R separately)."""
        return self.stem(x)

    def enable_tlsc(self, base_size) -> int:
        """Enable inference-time Test-time Local Statistics Correction.

        Swaps each NAFBlock's SCA global pool for a windowed local pool of
        size ``base_size`` (use the training LR patch shape, e.g.
        ``(patch_h, patch_w)``). Returns the number of pool layers replaced.
        Idempotent — calling twice is a no-op. Call AFTER loading weights and
        BEFORE evaluation.
        """
        from .tlsc import enable_tlsc as _enable_tlsc

        return _enable_tlsc(self, base_size)

    def disable_tlsc(self) -> int:
        """Revert TLSC: restore global ``AdaptiveAvgPool2d(1)`` in every SCA."""
        from .tlsc import disable_tlsc as _disable_tlsc

        return _disable_tlsc(self)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> StereoSRV6Output:
        if left.shape != right.shape:
            raise ValueError(f"Left/right shapes must match: {left.shape} vs {right.shape}")
        lr_l_in, lr_r_in = left, right

        # Shared stem.
        x_l = self.stem(left)
        x_r = self.stem(right)

        # Shared trunk — cross-view fusion happens inside each block.
        for block in self.trunk:
            x_l, x_r = block(x_l, x_r)

        x_l = self.trunk_norm(x_l)
        x_r = self.trunk_norm(x_r)

        # Tail: single PixelShuffle upsample.
        pred_l = self.tail(x_l)
        pred_r = self.tail(x_r)

        # Global bicubic residual — network learns the HR residual only.
        up_l = F.interpolate(lr_l_in, scale_factor=self.scale, mode="bicubic", align_corners=False)
        up_r = F.interpolate(lr_r_in, scale_factor=self.scale, mode="bicubic", align_corners=False)

        sr_l = pred_l + up_l
        sr_r = pred_r + up_r

        return StereoSRV6Output(sr_left=sr_l, sr_right=sr_r, aux={})

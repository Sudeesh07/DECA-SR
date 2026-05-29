"""StereoSR V7 — main model with multi-scale SCAM trunk + disparity-prior bias.

Architecture
------------
```
LR_L, LR_R  (H × W)
   │
   ├─ Stem (shared Conv3×3): channels
   │
   ├─ MAIN trunk × N blocks, all at LR resolution:
   │     SpectralStereoBlock(use_disparity_prior=True):
   │        ├─ NAFBlock(x)                                [per-view, shared]
   │        └─ SCAM_with_disparity_prior(x_L, x_R)        [epipolar attention]
   │
   ├─ (optional) COARSE trunk × M blocks, at LR/2 resolution:
   │   AvgPool2d(2) → [same block structure] → Upsample(bilinear)
   │   Output blended into main trunk via zero-init per-channel scale.
   │
   ├─ trunk_norm (LayerNorm2d)
   │
   ├─ Tail (shared): Conv3×3(c → 3·s²) + PixelShuffle(s)
   │                 [ICNR init, init_scale = 0.01]
   │
   └─ + Bicubic(LR_L) / Bicubic(LR_R)
         → SR_L, SR_R  (sH × sW)
```

Multi-scale trunk
-----------------
The coarse branch runs a smaller number of identical blocks at half spatial
resolution (avg-pooled stem features), giving each position a 2× receptive
field in input space. Its output is bilinearly upsampled and added to the
main-trunk output via a per-channel zero-init scale ``coarse_alpha``. At
step 0 the coarse path contributes nothing (alpha = 0); training can lift
alpha if the long-range signal helps.

Design principles inherited from V6 (validated by ablations)
------------------------------------------------------------
1. **Trunk operates entirely at LR resolution.** No encoder/decoder, no
   cascaded PixelShuffle. The HR representation is built only at the tail
   via a single sub-pixel convolution. The coarse branch above downsamples
   then upsamples, but the *main* path stays at LR.
2. **Shared weights across views.** Both L and R pass through the same
   stem, same trunk block list, and same tail.
3. **Zero-init every residual.** β, γ in NAFBlock and SCAM all zero;
   bias_scale in disparity prior zero; coarse_alpha zero. At step 0 the
   network output is exactly ``bicubic(LR)``.
4. **Bicubic global residual.** Trunk learns the HR residual over bicubic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, NAFBlock, TailUpsample
from .epipolar_attention import StereoCrossAttention


@dataclass
class StereoSRV7Output:
    sr_left: torch.Tensor
    sr_right: torch.Tensor
    aux: dict


class StereoSpatialBlock(nn.Module):
    """Per-block unit: shared NAFBlock per view → SCAM with optional disp prior."""

    def __init__(
        self,
        channels: int,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        use_disparity_prior: bool = True,
        sigma_init: float = 4.0,
    ):
        super().__init__()
        # Shared per-view spatial processor (one weight set for both L and R).
        self.naf = NAFBlock(channels, dw_expansion, ffn_expansion)
        self.scam = StereoCrossAttention(
            channels,
            use_disparity_prior=use_disparity_prior,
            sigma_init=sigma_init,
        )

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_l = self.naf(x_l)
        x_r = self.naf(x_r)
        x_l, x_r = self.scam(x_l, x_r)
        return x_l, x_r


class StereoSRV7(nn.Module):
    """V7 — multi-scale SCAM trunk with disparity-prior bias."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        use_disparity_prior: bool = True,
        sigma_init: float = 4.0,
        use_multiscale: bool = True,
        num_blocks_coarse: int = 8,
    ):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"scale must be 2, 3, or 4 (got {scale})")
        self.scale = scale
        self.use_multiscale = use_multiscale

        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Main trunk at LR resolution.
        self.trunk = nn.ModuleList([
            StereoSpatialBlock(
                channels=channels,
                dw_expansion=dw_expansion,
                ffn_expansion=ffn_expansion,
                use_disparity_prior=use_disparity_prior,
                sigma_init=sigma_init,
            )
            for _ in range(num_blocks)
        ])

        # Optional coarse trunk at LR/2 resolution.
        if use_multiscale:
            self.coarse_trunk = nn.ModuleList([
                StereoSpatialBlock(
                    channels=channels,
                    dw_expansion=dw_expansion,
                    ffn_expansion=ffn_expansion,
                    use_disparity_prior=use_disparity_prior,
                    sigma_init=sigma_init,
                )
                for _ in range(num_blocks_coarse)
            ])
            # Per-channel zero-init scale on the coarse output. The block list
            # above is identity at step 0 (zero-init β, γ, bias_scale), but
            # avg_pool + upsample is not exactly identity, so we keep the
            # zero-init scale here to prevent any coarse-path leakage at init.
            self.coarse_alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.coarse_trunk = None
            self.coarse_alpha = None

        # Trunk-level LayerNorm to stabilise deep stacks.
        self.trunk_norm = LayerNorm2d(channels)

        # Single tail upsample: Conv3×3 → PixelShuffle(scale). ICNR-initialised
        # then × init_scale=0.01 so the initial residual is small vs bicubic.
        self.tail = TailUpsample(channels, in_channels, scale)

    def enable_tlsc(self, base_size) -> int:
        """Enable TLSC inference-time correction (see ``tlsc.py``)."""
        from .tlsc import enable_tlsc as _enable_tlsc
        return _enable_tlsc(self, base_size)

    def disable_tlsc(self) -> int:
        from .tlsc import disable_tlsc as _disable_tlsc
        return _disable_tlsc(self)

    def _run_trunk(self, trunk: nn.ModuleList, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in trunk:
            x_l, x_r = block(x_l, x_r)
        return x_l, x_r

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> StereoSRV7Output:
        if left.shape != right.shape:
            raise ValueError(f"Left/right shapes must match: {left.shape} vs {right.shape}")
        lr_l_in, lr_r_in = left, right

        # Shared stem.
        x_l = self.stem(left)
        x_r = self.stem(right)

        # Main trunk at LR.
        main_l, main_r = self._run_trunk(self.trunk, x_l, x_r)

        # Optional coarse trunk at LR/2, blended in via zero-init coarse_alpha.
        if self.use_multiscale and self.coarse_trunk is not None:
            # Pool the *post-stem* features (NOT post-trunk: we want the
            # coarse branch to operate in parallel, not sequentially).
            coarse_l_in = F.avg_pool2d(x_l, kernel_size=2)
            coarse_r_in = F.avg_pool2d(x_r, kernel_size=2)
            coarse_l, coarse_r = self._run_trunk(self.coarse_trunk, coarse_l_in, coarse_r_in)
            # Upsample back to LR resolution via bilinear interpolation.
            coarse_l_up = F.interpolate(coarse_l, size=main_l.shape[-2:], mode="bilinear", align_corners=False)
            coarse_r_up = F.interpolate(coarse_r, size=main_r.shape[-2:], mode="bilinear", align_corners=False)
            # Per-channel zero-init scale; at step 0 contributes nothing.
            main_l = main_l + self.coarse_alpha * coarse_l_up
            main_r = main_r + self.coarse_alpha * coarse_r_up

        main_l = self.trunk_norm(main_l)
        main_r = self.trunk_norm(main_r)

        # Tail: single PixelShuffle upsample (× init_scale=0.01 → SR_init ≈ 0).
        pred_l = self.tail(main_l)
        pred_r = self.tail(main_r)

        # Global bicubic residual — network learns the HR residual only.
        up_l = F.interpolate(lr_l_in, scale_factor=self.scale, mode="bicubic", align_corners=False)
        up_r = F.interpolate(lr_r_in, scale_factor=self.scale, mode="bicubic", align_corners=False)

        return StereoSRV7Output(sr_left=pred_l + up_l, sr_right=pred_r + up_r, aux={})

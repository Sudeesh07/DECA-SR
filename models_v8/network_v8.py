"""StereoSR V8 — main model with dual-axis cross-view attention.

Architecture
------------
```
LR_L, LR_R  (H × W)
   │
   ├─ Stem (shared Conv3×3): channels
   │
   ├─ Trunk × N blocks, all at LR resolution:
   │     StereoSpatialChannelBlock:
   │        ├─ NAFBlock(x_L); NAFBlock(x_R)          [per-view spatial, shared]
   │        ├─ SCAM(x_L, x_R)                        [spatial cross-view]
   │        └─ CT-CVA(x_L, x_R)                      [channel cross-view, NEW]
   │
   ├─ trunk_norm (LayerNorm2d)
   │
   ├─ Tail (shared): Conv3×3(c → 3·s²) + PixelShuffle(s)
   │                 [ICNR init, init_scale = 0.01]
   │
   └─ + Bicubic(LR_L) / Bicubic(LR_R)
         → SR_L, SR_R  (sH × sW)
```

Design principles inherited from V6 (validated by ablations)
------------------------------------------------------------
1. **Trunk operates entirely at LR resolution.** No encoder/decoder, no
   cascaded PixelShuffle. The HR representation is built only at the tail.
2. **Shared weights across views.** L and R share the stem, trunk, and tail.
3. **Zero-init every residual.** NAFBlock β,γ = 0; SCAM β,γ = 0; CT-CVA γ = 0.
   At step 0 the network output is exactly ``bicubic(LR)``.
4. **Bicubic global residual.** Trunk learns the HR residual only.

V8 contribution
---------------
The trunk block stacks SCAM (spatial cross-view, V6's only mechanism that
beat the baseline) and CT-CVA (channel cross-view, new) in series. Each
block can be configured with `use_scam` and `use_ct_cva` flags for ablations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, NAFBlock, TailUpsample
from .epipolar_attention import StereoCrossAttention
from .channel_cross_view import ChannelTransposedCrossView


@dataclass
class StereoSRV8Output:
    sr_left: torch.Tensor
    sr_right: torch.Tensor
    aux: dict


class StereoSpatialChannelBlock(nn.Module):
    """Dual-axis cross-view trunk block: NAFBlock → SCAM → CT-CVA.

    SCAM does spatial (row-axis) cross-view; CT-CVA does channel-axis
    cross-view. The two are complementary — they answer "*where* is the match"
    and "*which* feature-channels correspond" respectively.

    Both cross-view paths have zero-init residual scales so the block is the
    exact identity at step 0. Training lifts SCAM's β/γ first (they receive
    the larger gradient signal because their attention is image-content-scale);
    CT-CVA's γ_L/γ_R lift once SCAM has trained enough to produce non-trivial
    features. This is the same warm-up pattern V6/V7's NAFBlock β/γ exhibits.
    """

    def __init__(
        self,
        channels: int,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        use_scam: bool = True,
        use_ct_cva: bool = True,
        ct_cva_heads: int = 4,
        ct_cva_temperature_init: float = 1.0,
    ):
        super().__init__()
        # Shared per-view spatial processor (single weight set for both L and R).
        self.naf = NAFBlock(channels, dw_expansion, ffn_expansion)
        self.scam = StereoCrossAttention(channels) if use_scam else None
        self.ct_cva = (
            ChannelTransposedCrossView(
                channels,
                num_heads=ct_cva_heads,
                temperature_init=ct_cva_temperature_init,
            )
            if use_ct_cva
            else None
        )

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_l = self.naf(x_l)
        x_r = self.naf(x_r)
        if self.scam is not None:
            x_l, x_r = self.scam(x_l, x_r)
        if self.ct_cva is not None:
            x_l, x_r = self.ct_cva(x_l, x_r)
        return x_l, x_r


class StereoSRV8(nn.Module):
    """V8 — dual-axis cross-view stereo SR. SCAM + CT-CVA in each trunk block."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        use_scam: bool = True,
        use_ct_cva: bool = True,
        ct_cva_heads: int = 4,
        ct_cva_temperature_init: float = 1.0,
    ):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"scale must be 2, 3, or 4 (got {scale})")
        if channels % ct_cva_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by ct_cva_heads ({ct_cva_heads})"
            )
        self.scale = scale

        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)

        self.trunk = nn.ModuleList([
            StereoSpatialChannelBlock(
                channels=channels,
                dw_expansion=dw_expansion,
                ffn_expansion=ffn_expansion,
                use_scam=use_scam,
                use_ct_cva=use_ct_cva,
                ct_cva_heads=ct_cva_heads,
                ct_cva_temperature_init=ct_cva_temperature_init,
            )
            for _ in range(num_blocks)
        ])

        self.trunk_norm = LayerNorm2d(channels)
        self.tail = TailUpsample(channels, in_channels, scale)

    def enable_tlsc(self, base_size) -> int:
        from .tlsc import enable_tlsc as _enable_tlsc
        return _enable_tlsc(self, base_size)

    def disable_tlsc(self) -> int:
        from .tlsc import disable_tlsc as _disable_tlsc
        return _disable_tlsc(self)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> StereoSRV8Output:
        if left.shape != right.shape:
            raise ValueError(f"Left/right shapes must match: {left.shape} vs {right.shape}")
        lr_l_in, lr_r_in = left, right

        x_l = self.stem(left)
        x_r = self.stem(right)

        for block in self.trunk:
            x_l, x_r = block(x_l, x_r)

        x_l = self.trunk_norm(x_l)
        x_r = self.trunk_norm(x_r)

        pred_l = self.tail(x_l)
        pred_r = self.tail(x_r)

        up_l = F.interpolate(lr_l_in, scale_factor=self.scale, mode="bicubic", align_corners=False)
        up_r = F.interpolate(lr_r_in, scale_factor=self.scale, mode="bicubic", align_corners=False)

        return StereoSRV8Output(sr_left=pred_l + up_l, sr_right=pred_r + up_r, aux={})

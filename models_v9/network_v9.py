"""StereoSR V9 — DECA-SR (Deformable Epipolar Cost-view-Aware stereo SR).

Trunk block = RepLKNAFBlock (per-view, shared weights) -> SCAM (every block,
cheap dense epipolar row attention) -> DECVF (every ``decvf_every``-th block, the
novel sub-pixel deformable cost-view fusion). Single PixelShuffle tail over a
global bicubic residual (network learns the HR residual; ~bicubic at init).

Identity-at-init across the whole net: NAF beta/gamma=0, SCAM beta/gamma=0,
DECVF gamma_l/gamma_r=0, tail init_scale=0.01 -> SR_init ~= bicubic(LR).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, RepLargeKernelDW, RepLKNAFBlock, TailUpsample
from .deformable_fusion import DeformableEpipolarFusion
from .epipolar_attention import StereoCrossAttention


@dataclass
class StereoSRV9Output:
    sr_left: torch.Tensor
    sr_right: torch.Tensor
    aux: dict


class DECABlock(nn.Module):
    """One trunk block: shared-weight RepLKNAFBlock per view, then optional
    SCAM, then optional DECVF cross-view fusion."""

    def __init__(
        self,
        channels: int,
        dw_expansion: float,
        ffn_expansion: float,
        large_kernel: int,
        use_scam: bool,
        use_decvf: bool,
        cost_volume_d: int,
        decvf_k: int,
        decvf_y_range: float,
    ):
        super().__init__()
        self.naf = RepLKNAFBlock(channels, dw_expansion, ffn_expansion, large_kernel)
        self.scam = StereoCrossAttention(channels) if use_scam else None
        self.decvf = (
            DeformableEpipolarFusion(channels, cost_volume_d, decvf_k, decvf_y_range)
            if use_decvf else None
        )

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor):
        x_l = self.naf(x_l)
        x_r = self.naf(x_r)
        if self.scam is not None:
            x_l, x_r = self.scam(x_l, x_r)
        if self.decvf is not None:
            x_l, x_r = self.decvf(x_l, x_r)
        return x_l, x_r


class StereoSRV9(nn.Module):
    """DECA-SR network."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        large_kernel: int = 9,
        use_scam: bool = True,
        use_decvf: bool = True,
        decvf_every: int = 2,
        cost_volume_d: int = 24,
        decvf_k: int = 8,
        decvf_y_range: float = 1.5,
    ):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"scale must be 2, 3, or 4 (got {scale})")
        self.scale = scale

        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)

        blocks = []
        for i in range(num_blocks):
            place_decvf = use_decvf and decvf_every > 0 and ((i % decvf_every) == (decvf_every - 1))
            blocks.append(
                DECABlock(
                    channels=channels,
                    dw_expansion=dw_expansion,
                    ffn_expansion=ffn_expansion,
                    large_kernel=large_kernel,
                    use_scam=use_scam,
                    use_decvf=place_decvf,
                    cost_volume_d=cost_volume_d,
                    decvf_k=decvf_k,
                    decvf_y_range=decvf_y_range,
                )
            )
        self.trunk = nn.ModuleList(blocks)

        self.trunk_norm = LayerNorm2d(channels)
        self.tail = TailUpsample(channels, in_channels, scale)

    # -- TLSC hooks -------------------------------------------------------
    def enable_tlsc(self, base_size) -> int:
        from .tlsc import enable_tlsc as _enable
        return _enable(self, base_size)

    def disable_tlsc(self) -> int:
        from .tlsc import disable_tlsc as _disable
        return _disable(self)

    # -- Reparameterization (RepLK -> single DW) for inference -----------
    @torch.no_grad()
    def reparametrize(self) -> int:
        n = 0
        for m in self.modules():
            if isinstance(m, RepLargeKernelDW):
                m.reparametrize()
                n += 1
        return n

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> StereoSRV9Output:
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
        return StereoSRV9Output(sr_left=pred_l + up_l, sr_right=pred_r + up_r, aux={})

"""
StereoSRNet V2: Clean architecture following the DEPA design.

Architecture:
    Shared shallow encoder
    -> Encoder stages (NAFBlocks + BidirectionalDEPA at each scale)
    -> Bottleneck (correlation prior + single DEPA)
    -> U-Net decoder with skip connections
    -> Lightweight frequency refinement
    -> Streamlined occlusion-aware fusion
    -> PixelShuffle reconstruction + global residual

Key differences from v1:
- DEPA (ESA + ECA) at every encoder stage instead of DECA-only / TCVA-only split
- No HCPF — replaced by lightweight correlation-prior bottleneck
- Bottleneck receives pre-DEPA features to avoid redundant cross-view processing
- Proper U-Net decoder with NAFBlock refinement at each level
- Lightweight frequency refinement (spatial + fixed-Laplacian HF, no FFT cross-view)
- Streamlined occlusion fusion without double-residual bug
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .scam import LightweightSCAM, InterleavedEncoderStage
from models.nafblock import NAFBlock, LayerNorm2d

from .depa import DEPA, BidirectionalDEPA
from .frequency import LightweightFrequencyRefinement
from .occlusion import HybridOcclusionFusion


# ── Encoder Stage ───────────────────────────────────────────────────────


class EncoderStage(nn.Module):
    """
    NAFBlocks (shared L/R) -> BidirectionalDEPA -> optional downsample.

    Returns both post-DEPA features (for skip connections) and pre-DEPA
    features (for bottleneck input at the deepest stage).
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        depa_k: int = 8,
        depa_max_disparity: int = 32,
        num_heads: int = 4,
        downsample: bool = True,
        drop_path_rates: Optional[List[float]] = None,
    ):
        super().__init__()

        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks

        self.blocks = nn.Sequential(
            *[
                NAFBlock(channels, expansion=2, drop_path=drop_path_rates[i])
                for i in range(num_blocks)
            ]
        )

        self.cross_view = BidirectionalDEPA(
            channels, k=depa_k, num_heads=num_heads, max_disparity=depa_max_disparity,
        )

        if downsample:
            self.down = nn.Sequential(
                LayerNorm2d(channels),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            )
        else:
            self.down = None

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor,  # post-DEPA (skip connections)
        torch.Tensor, torch.Tensor,  # next-stage input (possibly downsampled)
        torch.Tensor, torch.Tensor,  # pre-DEPA (for bottleneck)
    ]:
        fl = self.blocks(f_left)
        fr = self.blocks(f_right)

        fl_pre, fr_pre = fl, fr

        fl, fr = self.cross_view(fl, fr)

        fl_skip, fr_skip = fl, fr

        if self.down is not None:
            fl_next = self.down(fl)
            fr_next = self.down(fr)
        else:
            fl_next, fr_next = fl, fr

        return fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre


# ── Correlation Prior ───────────────────────────────────────────────────


class CorrelationPrior(nn.Module):
    """
    Group-wise correlation volume as a stereo matching prior.

    Computes normalized dot-product correlation between left and right
    features at multiple disparity candidates, then aggregates into
    a compact feature representation. Provides the DEPA bottleneck
    with explicit matching information.
    """

    def __init__(self, channels: int, num_groups: int = 8, max_disp: int = 12):
        super().__init__()
        self.num_groups = num_groups
        self.max_disp = max_disp
        self.group_channels = channels // num_groups
        assert channels % num_groups == 0

        self.aggregate = nn.Sequential(
            nn.Conv2d(num_groups * max_disp, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1)*0.05)

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_left.shape
        G, Cg = self.num_groups, self.group_channels
        max_d = min(self.max_disp, W)

        fl = F.normalize(f_left.view(B, G, Cg, H, W), dim=2)
        fr = F.normalize(f_right.view(B, G, Cg, H, W), dim=2)

        corr_vol = f_left.new_zeros(B, G * self.max_disp, H, W)

        for d in range(max_d):
            if d == 0:
                corr = (fl * fr).mean(dim=2)
            else:
                corr = (fl[:, :, :, :, d:] * fr[:, :, :, :, :W - d]).mean(dim=2)
                corr = F.pad(corr, (d, 0), mode="constant", value=0)
            corr_vol[:, d * G : (d + 1) * G] = corr

        out = self.aggregate(corr_vol)
        return f_left + out * self.beta


# ── Bottleneck ──────────────────────────────────────────────────────────


class Bottleneck(nn.Module):
    """
    Correlation prior -> single DEPA for cross-view fusion.

    Receives PRE-DEPA features from the deepest encoder stage so the
    correlation prior operates on features that haven't already been
    cross-view processed (avoids redundant processing).
    """

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 16,
        corr_max_disp: int = 12,
    ):
        super().__init__()
        self.corr_prior = CorrelationPrior(channels, max_disp=corr_max_disp)
        self.depa = DEPA(channels, k, num_heads, max_disparity)

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        fl = self.corr_prior(f_left, f_right)
        return self.depa(fl, f_right)


# ── Decoder Stage ───────────────────────────────────────────────────────


class DecoderStage(nn.Module):
    """PixelShuffle upsample -> add skip -> NAFBlock refinement."""

    def __init__(self, channels: int, num_blocks: int = 4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.blocks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(x + skip)


# ── Upsampler ───────────────────────────────────────────────────────────


class Upsampler(nn.Module):
    """PixelShuffle-based upsampler for x2 or x4 SR."""

    def __init__(self, channels: int, scale: int, out_channels: int = 3):
        super().__init__()
        # if scale == 4:
        #     self.pre = nn.Sequential(
        #         NAFBlock(channels),
        #         NAFBlock(channels),
        #     )
        #     self.up = nn.Sequential(
        #         nn.Conv2d(channels, channels * 4, 3, padding=1),
        #         nn.PixelShuffle(2),
        #         LayerNorm2d(channels),
        #         nn.GELU(),

        #         nn.Conv2d(channels, channels * 4, 3, padding=1),
        #         nn.PixelShuffle(2),
        #     )
        # elif scale == 2:
        #     self.up = nn.Sequential(
        #         nn.Conv2d(channels, channels * 4, 3, padding=1),
        #         nn.PixelShuffle(2),
        #     )
        # else:
        #     raise ValueError(f"Unsupported scale: {scale}")

        self.scale = scale
        self.up = nn.Sequential(
            nn.Conv2d(channels, out_channels * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
        )


        # self.final = nn.Sequential(
        #     LayerNorm2d(channels),
        #     nn.Conv2d(channels, out_channels, 3, padding=1),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
        # return self.final(self.up(x))


# ── Main Network ────────────────────────────────────────────────────────


class StereoSRNetV2(nn.Module):
    """
    Stereo Image Super-Resolution Network V2.

    Architecture follows the DEPA design: every encoder stage has both
    ESA (spatial deformable) + ECA (channel) attention. Bottleneck uses
    a correlation prior for explicit stereo matching. Occlusion-aware
    fusion applied once at the end.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        embed_dim: int = 48,
        enc_blocks: Optional[List[int]] = None,
        dec_blocks: Optional[List[int]] = None,
        depa_k: int = 8,
        depa_max_disparity: int = 32,
        num_heads: int = 4,
        corr_max_disp: int = 12,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]

        self.scale = scale
        num_enc = len(enc_blocks)
        num_down = num_enc - 1  # last encoder stage has no downsample

        assert len(dec_blocks) == num_down, (
            f"dec_blocks length ({len(dec_blocks)}) must equal "
            f"number of downsamples ({num_down})"
        )

        # Shallow feature extraction (shared weights for L/R)
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )

        # Drop-path rates linearly increasing across all encoder blocks
        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            downsample = i < num_down
            # Scale max_disparity with resolution: i downsamples happened before this stage
            stage_disp = depa_max_disparity // (2 ** i)
            self.encoder_stages.append(
                # EncoderStage(
                #     embed_dim, nb, depa_k, stage_disp, num_heads, downsample, rates,
                # )
                InterleavedEncoderStage(
                    embed_dim, nb,
                    scam_interval=2,
                    depa_k=depa_k,
                    depa_max_disparity=stage_disp,
                    num_heads=num_heads,
                    downsample=downsample,
                    drop_path_rates=rates,
                )
            )
            block_idx += nb

        # Bottleneck (operates at deepest resolution)
        bottleneck_disp = depa_max_disparity // (2 ** num_down)
        self.bottleneck = Bottleneck(
            embed_dim, depa_k, num_heads, bottleneck_disp, corr_max_disp,
        )

        # Decoder stages (reverse order, skipping the deepest encoder)
        self.decoder_stages = nn.ModuleList(
            [DecoderStage(embed_dim, nb) for nb in dec_blocks]
        )

        # Frequency refinement
        self.freq_refine = LightweightFrequencyRefinement(embed_dim)

        # Occlusion-aware fusion
        self.occlusion = HybridOcclusionFusion(embed_dim)

        # Reconstruction head
        self.upsampler = Upsampler(embed_dim, scale, in_channels)

    def _encode(
        self, fl: torch.Tensor, fr: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor],  # skips_l, skips_r
        torch.Tensor, torch.Tensor,               # pre_deca_l, pre_deca_r
    ]:
        """
        Run encoder ONCE for both views.

        BidirectionalDEPA always receives (left, right) in the correct order,
        preserving l2r/r2l directional specialization.
        """
        skips_l: List[torch.Tensor] = []
        skips_r: List[torch.Tensor] = []
        pre_l = fl  # will be overwritten at deepest stage
        pre_r = fr

        for i, stage in enumerate(self.encoder_stages):
            fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre = stage(fl, fr)
            skips_l.append(fl_skip)
            skips_r.append(fr_skip)
            if i == len(self.encoder_stages) - 1:
                pre_l, pre_r = fl_pre, fr_pre
            fl, fr = fl_next, fr_next

        return skips_l, skips_r, pre_l, pre_r

    def _decode(
        self,
        pre_target: torch.Tensor,
        pre_ref: torch.Tensor,
        skips_target: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Bottleneck + decoder + frequency refinement. No occlusion fusion.

        Returns decoded features at full resolution, ready for occlusion.
        """
        x = self.bottleneck(pre_target, pre_ref)

        for i, dec_stage in enumerate(self.decoder_stages):
            skip_idx = len(self.encoder_stages) - 2 - i
            x = dec_stage(x, skips_target[skip_idx])

        return self.freq_refine(x)

    def forward(
        self, img_left: torch.Tensor, img_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full stereo SR forward pass.

        Encoder runs ONCE (BidirectionalDEPA sees correct L/R order).
        Both views are decoded BEFORE occlusion fusion, so the fusion
        module receives features at the same processing depth.

        Returns: (sr_left, sr_right, conf_left, conf_right)
        """
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)

        # Encode once
        skips_l, skips_r, pre_l, pre_r = self._encode(fl, fr)

        # Decode both views first (same processing depth)
        xl = self._decode(pre_l, pre_r, skips_l)
        xr = self._decode(pre_r, pre_l, skips_r)

        # Occlusion fusion with matched-depth features
        fused_l, conf_l = self.occlusion(xl, xr)
        fused_r, conf_r = self.occlusion(xr, xl)

        sr_left = self.upsampler(fused_l)
        sr_right = self.upsampler(fused_r)

        # Global residual learning
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, mode="bilinear", align_corners=False)
        sr_left = sr_left + lr_up_l
        sr_right = sr_right + lr_up_r

        # Upsample confidence to SR resolution for downstream losses
        conf_l = F.interpolate(conf_l, size=sr_left.shape[-2:], mode="bilinear", align_corners=False)
        conf_r = F.interpolate(conf_r, size=sr_right.shape[-2:], mode="bilinear", align_corners=False)

        return sr_left, sr_right, conf_l, conf_r


# ── Model Variants ──────────────────────────────────────────────────────


def stereo_sr_v2_tiny(scale: int = 4, **kw) -> StereoSRNetV2:
    return StereoSRNetV2(
        scale=scale, embed_dim=32, enc_blocks=[2, 4, 4], dec_blocks=[2, 2],
        depa_k=4, num_heads=2, corr_max_disp=8, **kw,
    )


def stereo_sr_v2_base(scale: int = 4, **kw) -> StereoSRNetV2:
    return StereoSRNetV2(
        scale=scale, embed_dim=48, enc_blocks=[4, 6, 6], dec_blocks=[4, 4],
        depa_k=8, num_heads=4, corr_max_disp=12, **kw,
    )


def stereo_sr_v2_large(scale: int = 4, **kw) -> StereoSRNetV2:
    return StereoSRNetV2(
        scale=scale, embed_dim=64, enc_blocks=[6, 8, 8, 8], dec_blocks=[4, 4, 4],
        depa_k=8, num_heads=4, corr_max_disp=16, drop_path_rate=0.2, **kw,
    )


def build_model_v2(args) -> StereoSRNetV2:
    """Build model from args namespace (compatible with train_fixed.py)."""
    return StereoSRNetV2(
        scale=getattr(args, "scale", 4),
        in_channels=getattr(args, "in_channels", 3),
        embed_dim=getattr(args, "embed_dim", 48),
        enc_blocks=getattr(args, "enc_blocks", [4, 6, 6]),
        dec_blocks=getattr(args, "dec_blocks", [4, 4]),
        depa_k=getattr(args, "depa_k", 8),
        depa_max_disparity=getattr(args, "depa_max_disparity", 32),
        num_heads=getattr(args, "num_heads", 4),
        corr_max_disp=getattr(args, "corr_max_disp", 12),
        drop_path_rate=getattr(args, "stochastic_depth", 0.1),
    )


# ── Quick smoke test ────────────────────────────────────────────────────


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = stereo_sr_v2_tiny(scale=4).to(device)

    left = torch.randn(1, 3, 30, 90, device=device)
    right = torch.randn(1, 3, 30, 90, device=device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")

    with torch.no_grad():
        sr_l, sr_r, c_l, c_r = model(left, right)

    print(f"Input:  {tuple(left.shape)}")
    print(f"SR_L:   {tuple(sr_l.shape)}")
    print(f"SR_R:   {tuple(sr_r.shape)}")
    print(f"Conf_L: {tuple(c_l.shape)}")
    print(f"Conf_R: {tuple(c_r.shape)}")
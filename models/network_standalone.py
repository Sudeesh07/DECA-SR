"""
StereoSRNet using standalone block implementations.

This version wires together modules from:
- nafblock.py (NAFBlock, LayerNorm2d)
- deca.py (DECA)
- tcva.py (TCVA)
- hcpf.py (HCPF)
- frequency.py (FrequencyAwareRefinement)
- occlusion.py (OcclusionAwareFusion)

It is intentionally separate from models/network.py so you can compare behavior
without touching the inline implementation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafblock import NAFBlock, LayerNorm2d
from .deca import DECA, BidirectionalDECA
from .tcva import TCVA
from .hcpf import HCPF
from .frequency import FrequencyAwareRefinement
from .occlusion import OcclusionAwareFusion


class EncoderStage(nn.Module):
    """Encoder stage with shared NAFBlocks and optional cross-view interaction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        cross_view_type: Optional[str] = "deca",
        deca_k: int = 8,
        deca_max_disparity: int = 32,
        num_heads: int = 1,
        downsample: bool = False,
        drop_path_rates: Optional[List[float]] = None,
    ):
        super().__init__()

        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks

        self.blocks = nn.Sequential(
            *[
                NAFBlock(
                    in_channels,
                    expansion=2,
                    drop_path=drop_path_rates[i],
                    use_checkpoint=False,
                )
                for i in range(num_blocks)
            ]
        )

        if cross_view_type == "deca":
            self.cross_view = BidirectionalDECA(
                channels=in_channels,
                k=deca_k,
                num_heads=max(1, num_heads),
                max_disparity=deca_max_disparity,
            )
        elif cross_view_type == "tcva":
            self.cross_view = TCVA(
                channels=in_channels,
                num_heads=max(1, num_heads),
            )
        else:
            self.cross_view = None

        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            )
            self.channel_proj = None
        else:
            self.downsample = None
            self.channel_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl = self.blocks(f_left)
        fr = self.blocks(f_right)

        if self.cross_view is not None:
            if isinstance(self.cross_view, BidirectionalDECA):
                fl, fr = self.cross_view(fl, fr)
            else:
                fl_new = self.cross_view(fl, fr)
                fr_new = self.cross_view(fr, fl)
                fl, fr = fl_new, fr_new

        fl_out, fr_out = fl, fr

        if self.downsample is not None:
            fl_next = self.downsample(fl)
            fr_next = self.downsample(fr)
        elif self.channel_proj is not None:
            fl_next = self.channel_proj(fl)
            fr_next = self.channel_proj(fr)
        else:
            fl_next, fr_next = fl, fr

        return fl_out, fr_out, fl_next, fr_next


class Upsampler(nn.Module):
    """PixelShuffle upsampler for x2/x4 SR."""

    def __init__(self, channels: int, scale: int, out_channels: int = 3):
        super().__init__()
        if scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                LayerNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        elif scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            raise ValueError(f"Unsupported scale: {scale}. Use 2 or 4.")

        self.final = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.up(x))


class StereoSRNetStandalone(nn.Module):
    """Stereo SR network assembled only from standalone block files."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        embed_dim: int = 48,
        enc_blocks: Optional[List[int]] = None,
        enc_channels: Optional[List[int]] = None,
        deca_k: int = 8,
        deca_max_disparity: int = 32,
        num_heads: int = 1,
        spatial_blocks: int = 4,
        freq_blocks: int = 2,
        drop_path_rate: float = 0.1,
        coarse_max_disp: int = 4,
        use_vectorized_corr: bool = False,
        use_multiscale_conf: bool = True,
    ):
        super().__init__()

        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if enc_channels is None:
            enc_channels = [embed_dim] * (len(enc_blocks) + 1)
        if len(enc_channels) != len(enc_blocks) + 1:
            raise ValueError("enc_channels must have len(enc_blocks) + 1 values")

        self.scale = scale
        self.num_stages = len(enc_blocks)

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )

        total_blocks = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, num_blocks in enumerate(enc_blocks):
            in_ch = enc_channels[i]
            out_ch = enc_channels[i + 1]
            cross_view_type = "tcva" if i == len(enc_blocks) - 1 else "deca"
            downsample = i < len(enc_blocks) - 1

            stage_dpr = dpr[block_idx:block_idx + num_blocks]
            self.encoder_stages.append(
                EncoderStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_blocks=num_blocks,
                    cross_view_type=cross_view_type,
                    deca_k=deca_k,
                    deca_max_disparity=deca_max_disparity,
                    num_heads=num_heads,
                    downsample=downsample,
                    drop_path_rates=stage_dpr,
                )
            )
            block_idx += num_blocks

        self.hcpf = HCPF(
            channels=enc_channels[-1],
            k=deca_k,
            coarse_max_disp=coarse_max_disp,
            use_vectorized_corr=use_vectorized_corr,
        )

        self.skip_projs = nn.ModuleList(
            [nn.Conv2d(enc_channels[i], enc_channels[-1], 1) for i in range(self.num_stages - 1)]
        )

        self.decoder_ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(enc_channels[-1], enc_channels[-1] * 4, 1),
                    nn.PixelShuffle(2),
                )
                for _ in range(self.num_stages - 1)
            ]
        )

        self.freq_refinement = FrequencyAwareRefinement(
            channels=enc_channels[-1],
            spatial_blocks=spatial_blocks,
            freq_blocks=freq_blocks,
            use_hf_branch=True,
            use_bidirectional=False,
        )
        self.occlusion_fusion = OcclusionAwareFusion(
            channels=enc_channels[-1],
            use_multiscale_conf=use_multiscale_conf,
            num_self_blocks=2,
        )

        self.upsampler = Upsampler(enc_channels[-1], scale, in_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _align_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward_single_view(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        skip_features = []
        fl, fr = f_left, f_right

        for stage in self.encoder_stages:
            fl_out, fr_out, fl_next, fr_next = stage(fl, fr)
            skip_features.append((fl_out, fr_out))
            fl, fr = fl_next, fr_next

        x = self.hcpf(skip_features[-1][0], skip_features[-1][1])

        for i in range(self.num_stages - 2, -1, -1):
            x = self.decoder_ups[i](x)
            skip_l = self.skip_projs[i](skip_features[i][0])
            x = self._align_like(x, skip_l) + skip_l

        right_ref = skip_features[0][1]
        x = self._align_like(x, right_ref)
        refined = self.freq_refinement(x, right_ref)
        fused, confidence = self.occlusion_fusion(refined, right_ref)

        return fused, confidence

    def forward(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)

        fused_l, conf_l = self.forward_single_view(fl, fr)
        fused_r, conf_r = self.forward_single_view(fr, fl)

        sr_left = self.upsampler(fused_l)
        sr_right = self.upsampler(fused_r)

        lr_up_left = F.interpolate(img_left, scale_factor=self.scale, mode="bilinear", align_corners=False)
        lr_up_right = F.interpolate(img_right, scale_factor=self.scale, mode="bilinear", align_corners=False)

        sr_left = sr_left + lr_up_left
        sr_right = sr_right + lr_up_right

        # Return SR-sized confidence maps so downstream losses can consume them directly.
        conf_l = F.interpolate(conf_l, size=sr_left.shape[-2:], mode="bilinear", align_corners=False)
        conf_r = F.interpolate(conf_r, size=sr_right.shape[-2:], mode="bilinear", align_corners=False)

        return sr_left, sr_right, conf_l, conf_r


def stereo_sr_tiny_standalone(scale: int = 4, **kwargs) -> StereoSRNetStandalone:
    return StereoSRNetStandalone(
        scale=scale,
        embed_dim=32,
        enc_blocks=[2, 4, 4],
        deca_k=4,
        num_heads=1,
        coarse_max_disp=4,
        spatial_blocks=2,
        freq_blocks=1,
        **kwargs,
    )


def stereo_sr_base_standalone(scale: int = 4, **kwargs) -> StereoSRNetStandalone:
    return StereoSRNetStandalone(
        scale=scale,
        embed_dim=48,
        enc_blocks=[4, 6, 6],
        deca_k=8,
        num_heads=1,
        coarse_max_disp=4,
        spatial_blocks=4,
        freq_blocks=2,
        **kwargs,
    )


def stereo_sr_large_standalone(scale: int = 4, **kwargs) -> StereoSRNetStandalone:
    return StereoSRNetStandalone(
        scale=scale,
        embed_dim=64,
        enc_blocks=[6, 8, 8, 8],
        enc_channels=[64, 64, 128, 128, 128],
        deca_k=8,
        num_heads=4,
        coarse_max_disp=6,
        spatial_blocks=6,
        freq_blocks=3,
        drop_path_rate=0.2,
        **kwargs,
    )


def build_model_standalone(args) -> StereoSRNetStandalone:
    return StereoSRNetStandalone(
        scale=getattr(args, "scale", 4),
        in_channels=getattr(args, "in_channels", 3),
        embed_dim=getattr(args, "embed_dim", 48),
        enc_blocks=getattr(args, "enc_blocks", [4, 6, 6]),
        enc_channels=getattr(args, "enc_channels", None),
        deca_k=getattr(args, "deca_k", 8),
        deca_max_disparity=getattr(args, "deca_max_disparity", 32),
        num_heads=getattr(args, "num_heads", 1),
        coarse_max_disp=getattr(args, "coarse_max_disp", 4),
        spatial_blocks=getattr(args, "spatial_blocks", 4),
        freq_blocks=getattr(args, "freq_blocks", 2),
        drop_path_rate=getattr(args, "stochastic_depth", 0.1),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = stereo_sr_tiny_standalone(scale=4).to(device)

    left = torch.randn(1, 3, 64, 64, device=device)
    right = torch.randn(1, 3, 64, 64, device=device)

    with torch.no_grad():
        sr_l, sr_r, conf_l, conf_r = model(left, right)

    print("sr_l", tuple(sr_l.shape))
    print("sr_r", tuple(sr_r.shape))
    print("conf_l", tuple(conf_l.shape))
    print("conf_r", tuple(conf_r.shape))

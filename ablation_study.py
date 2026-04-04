"""
Ablation study for StereoSRNet V2.

Sequentially tests model configurations:
1. Baseline (simple PixelShuffle - encoder + decoder + upsampler only)
2. + DEPA (bidirectional cross-view attention)
3. + Correlation Prior (matching prior in bottleneck)
4. + Frequency Refine (lightweight frequency refinement)
5. + Occlusion Fusion (occlusion-aware fusion)
6. Full Model (all components)

Each ablation runs for 1000 iterations with fresh initialization.
Records: Params, FLOPs, PSNR (L, R, Avg), SSIM.

Usage:
    python ablation_study.py --scale 2 --batch_size 4
"""

import os
import time
from typing import Dict, Tuple, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import build_dataset
from utils_fixed import (
    AverageMeter, compute_psnr, compute_ssim, setup_logger,
)
from models_v2.config import get_config
try:
    from models_v2.network import NAFBlock, LayerNorm2d
except ImportError:
    from models.nafblock import NAFBlock, LayerNorm2d
from models_v2.depa import DEPA, BidirectionalDEPA
from models_v2.frequency import LightweightFrequencyRefinement
from models_v2.occlusion import HybridOcclusionFusion


# ─── Baseline: Simple Encoder + Decoder + Upsampler ─────────────────────


class BaselineEncoder(nn.Module):
    """Shared encoder without cross-view processing."""
    def __init__(self, embed_dim, enc_blocks, depa_k=8, depa_max_disparity=32, 
                 num_heads=4, drop_path_rate=0.1):
        super().__init__()
        self.shallow = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )
        
        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]
        
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        num_down = len(enc_blocks) - 1
        
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            downsample = i < num_down
            
            blocks = nn.Sequential(
                *[NAFBlock(embed_dim, expansion=2, drop_path=rates[j]) 
                  for j in range(nb)]
            )
            
            if downsample:
                down = nn.Sequential(
                    LayerNorm2d(embed_dim),
                    nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
                )
            else:
                down = None
            
            self.encoder_stages.append(nn.ModuleDict({
                'blocks': blocks,
                'down': down if down else nn.Identity(),
            }))
            block_idx += nb
    
    def forward(self, fl, fr):
        fl = self.shallow(fl)
        fr = self.shallow(fr)
        
        skips_l, skips_r = [], []
        for i, stage in enumerate(self.encoder_stages):
            fl = stage['blocks'](fl)
            fr = stage['blocks'](fr)
            skips_l.append(fl)
            skips_r.append(fr)
            if stage['down'] is not None and not isinstance(stage['down'], nn.Identity):
                fl = stage['down'](fl)
                fr = stage['down'](fr)
        
        return skips_l, skips_r, fl, fr


class BaselineDecoder(nn.Module):
    """Simple decoder without cross-view or sophisticated components."""
    def __init__(self, embed_dim, dec_blocks):
        super().__init__()
        self.decoder_stages = nn.ModuleList()
        for nb in dec_blocks:
            decoder = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                *[NAFBlock(embed_dim) for _ in range(nb)]
            )
            self.decoder_stages.append(decoder)
    
    def forward(self, x, skips):
        num_skips = len(skips)
        num_decoder_stages = len(self.decoder_stages)
        
        for i, dec in enumerate(self.decoder_stages):
            # Compute which skip to use: for encoder with N stages and decoder with M stages,
            # decoder stage i uses skip from encoder stage (N - M + i)
            skip_idx = num_skips - num_decoder_stages + i
            x = dec(x)
            if x.shape[-2:] != skips[skip_idx].shape[-2:]:
                x = F.interpolate(x, size=skips[skip_idx].shape[-2:], 
                                mode="bilinear", align_corners=False)
            x = x + skips[skip_idx]
        return x


class Upsampler(nn.Module):
    """PixelShuffle upsampler."""
    def __init__(self, channels, scale, out_channels=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, out_channels * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
        )
    
    def forward(self, x):
        return self.up(x)


class BaselineModel(nn.Module):
    """Baseline: encoder + decoder + upsampler only."""
    def __init__(self, scale=4, embed_dim=48, enc_blocks=None, dec_blocks=None, 
                 depa_k=8, depa_max_disparity=32, num_heads=4, corr_max_disp=12,
                 drop_path_rate=0.1, **kwargs):
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]
        
        self.scale = scale
        self.encoder = BaselineEncoder(embed_dim, enc_blocks, drop_path_rate=drop_path_rate)
        self.decoder = BaselineDecoder(embed_dim, dec_blocks)
        # Reconstruction head: feature maps to RGB
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 3, 3, padding=1),
        )
    
    def forward(self, img_left, img_right):
        B, C, H, W = img_left.shape
        target_h, target_w = H * self.scale, W * self.scale
        
        skips_l, skips_r, fl, fr = self.encoder(img_left, img_right)
        
        # Decode left view
        xl = self.decoder(fl, skips_l)
        sr_left = self.head(xl)
        # Upsample to target size
        sr_left = F.interpolate(sr_left, size=(target_h, target_w), 
                               mode="bilinear", align_corners=False)
        
        # Global residual
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_left = sr_left + lr_up_l
        
        # Decode right view
        xr = self.decoder(fr, skips_r)
        sr_right = self.head(xr)
        sr_right = F.interpolate(sr_right, size=(target_h, target_w), 
                                mode="bilinear", align_corners=False)
        
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_right = sr_right + lr_up_r
        
        # Dummy confidence maps
        conf_l = torch.ones(sr_left.shape[0], 1, sr_left.shape[2], sr_left.shape[3],
                           device=sr_left.device)
        conf_r = torch.ones(sr_right.shape[0], 1, sr_right.shape[2], sr_right.shape[3],
                           device=sr_right.device)
        
        return sr_left, sr_right, conf_l, conf_r


# ─── Ablation Models ─────────────────────────────────────────────────────


class WithDEPAModel(nn.Module):
    """Baseline + DEPA in encoder stages."""
    def __init__(self, scale=4, embed_dim=48, enc_blocks=None, dec_blocks=None, 
                 depa_k=8, depa_max_disparity=32, num_heads=4, drop_path_rate=0.1, **kwargs):
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]
        
        self.scale = scale
        self.shallow = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )
        
        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]
        
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        num_down = len(enc_blocks) - 1
        
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            downsample = i < num_down
            stage_disp = depa_max_disparity // (2 ** i)
            
            blocks = nn.Sequential(
                *[NAFBlock(embed_dim, expansion=2, drop_path=rates[j]) 
                  for j in range(nb)]
            )
            
            cross_view = BidirectionalDEPA(
                embed_dim, k=depa_k, num_heads=num_heads, max_disparity=stage_disp
            )
            
            if downsample:
                down = nn.Sequential(
                    LayerNorm2d(embed_dim),
                    nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
                )
            else:
                down = None
            
            self.encoder_stages.append(nn.ModuleDict({
                'blocks': blocks,
                'cross_view': cross_view,
                'down': down if down else nn.Identity(),
            }))
            block_idx += nb
        
        self.decoder = BaselineDecoder(embed_dim, dec_blocks)
        self.upsampler = Upsampler(embed_dim, scale, 3)
    
    def forward(self, img_left, img_right):
        B, C, H, W = img_left.shape
        target_h, target_w = H * self.scale, W * self.scale
        
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)
        
        skips_l, skips_r = [], []
        for stage in self.encoder_stages:
            fl = stage['blocks'](fl)
            fr = stage['blocks'](fr)
            fl, fr = stage['cross_view'](fl, fr)
            skips_l.append(fl)
            skips_r.append(fr)
            if stage['down'] is not None and not isinstance(stage['down'], nn.Identity):
                fl = stage['down'](fl)
                fr = stage['down'](fr)
        
        xl = self.decoder(fl, skips_l)
        sr_left = self.upsampler(xl)
        sr_left = F.interpolate(sr_left, size=(target_h, target_w), 
                               mode="bilinear", align_corners=False)
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_left = sr_left[:, :, :lr_up_l.shape[2], :lr_up_l.shape[3]]
        sr_left = sr_left + lr_up_l
        
        xr = self.decoder(fr, skips_r)
        sr_right = self.upsampler(xr)
        sr_right = F.interpolate(sr_right, size=(target_h, target_w), 
                                mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_right = sr_right[:, :, :lr_up_r.shape[2], :lr_up_r.shape[3]]
        sr_right = sr_right + lr_up_r
        
        conf_l = torch.ones(sr_left.shape[0], 1, sr_left.shape[2], sr_left.shape[3],
                           device=sr_left.device)
        conf_r = torch.ones(sr_right.shape[0], 1, sr_right.shape[2], sr_right.shape[3],
                           device=sr_right.device)
        
        return sr_left, sr_right, conf_l, conf_r


class CorrelationPrior(nn.Module):
    """Group-wise correlation volume."""
    def __init__(self, channels, num_groups=8, max_disp=12):
        super().__init__()
        self.num_groups = num_groups
        self.max_disp = max_disp
        self.group_channels = channels // num_groups
        
        self.aggregate = nn.Sequential(
            nn.Conv2d(num_groups * max_disp, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.05)
    
    def forward(self, f_left, f_right):
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


class Bottleneck(nn.Module):
    """Correlation prior + DEPA."""
    def __init__(self, channels, k=8, num_heads=4, max_disparity=16, corr_max_disp=12):
        super().__init__()
        self.corr_prior = CorrelationPrior(channels, max_disp=corr_max_disp)
        self.depa = DEPA(channels, k, num_heads, max_disparity)
    
    def forward(self, f_left, f_right):
        fl = self.corr_prior(f_left, f_right)
        return self.depa(fl, f_right)


class WithCorrPriorModel(nn.Module):
    """Baseline + DEPA + Correlation Prior."""
    def __init__(self, scale=4, embed_dim=48, enc_blocks=None, dec_blocks=None,
                 depa_k=8, depa_max_disparity=32, num_heads=4, corr_max_disp=12,
                 drop_path_rate=0.1, **kwargs):
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]
        
        self.scale = scale
        self.shallow = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )
        
        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]
        num_down = len(enc_blocks) - 1
        
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            downsample = i < num_down
            stage_disp = depa_max_disparity // (2 ** i)
            
            blocks = nn.Sequential(
                *[NAFBlock(embed_dim, expansion=2, drop_path=rates[j]) 
                  for j in range(nb)]
            )
            cross_view = BidirectionalDEPA(
                embed_dim, k=depa_k, num_heads=num_heads, max_disparity=stage_disp
            )
            
            if downsample:
                down = nn.Sequential(
                    LayerNorm2d(embed_dim),
                    nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
                )
            else:
                down = None
            
            self.encoder_stages.append(nn.ModuleDict({
                'blocks': blocks,
                'cross_view': cross_view,
                'down': down if down else nn.Identity(),
            }))
            block_idx += nb
        
        bottleneck_disp = depa_max_disparity // (2 ** num_down)
        self.bottleneck = Bottleneck(embed_dim, depa_k, num_heads, bottleneck_disp, corr_max_disp)
        
        self.decoder = BaselineDecoder(embed_dim, dec_blocks)
        self.upsampler = Upsampler(embed_dim, scale, 3)
    
    def forward(self, img_left, img_right):
        B, C, H, W = img_left.shape
        target_h, target_w = H * self.scale, W * self.scale
        
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)
        
        skips_l, skips_r = [], []
        pre_fl, pre_fr = fl, fr
        for i, stage in enumerate(self.encoder_stages):
            fl = stage['blocks'](fl)
            fr = stage['blocks'](fr)
            pre_fl, pre_fr = fl, fr
            fl, fr = stage['cross_view'](fl, fr)
            skips_l.append(fl)
            skips_r.append(fr)
            if stage['down'] is not None and not isinstance(stage['down'], nn.Identity):
                fl = stage['down'](fl)
                fr = stage['down'](fr)
        
        fl = self.bottleneck(pre_fl, pre_fr)
        xl = self.decoder(fl, skips_l)
        sr_left = self.upsampler(xl)
        sr_left = F.interpolate(sr_left, size=(target_h, target_w), 
                               mode="bilinear", align_corners=False)
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_left = sr_left[:, :, :lr_up_l.shape[2], :lr_up_l.shape[3]]
        sr_left = sr_left + lr_up_l
        
        fr = self.bottleneck(pre_fr, pre_fl)
        xr = self.decoder(fr, skips_r)
        sr_right = self.upsampler(xr)
        sr_right = F.interpolate(sr_right, size=(target_h, target_w), 
                                mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_right = sr_right[:, :, :lr_up_r.shape[2], :lr_up_r.shape[3]]
        sr_right = sr_right + lr_up_r
        
        conf_l = torch.ones(sr_left.shape[0], 1, sr_left.shape[2], sr_left.shape[3],
                           device=sr_left.device)
        conf_r = torch.ones(sr_right.shape[0], 1, sr_right.shape[2], sr_right.shape[3],
                           device=sr_right.device)
        
        return sr_left, sr_right, conf_l, conf_r


class WithFreqRefineModel(nn.Module):
    """Baseline + DEPA + Correlation Prior + Frequency Refinement."""
    def __init__(self, scale=4, embed_dim=48, enc_blocks=None, dec_blocks=None,
                 depa_k=8, depa_max_disparity=32, num_heads=4, corr_max_disp=12,
                 drop_path_rate=0.1, **kwargs):
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]
        
        self.scale = scale
        self.shallow = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )
        
        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]
        num_down = len(enc_blocks) - 1
        
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            downsample = i < num_down
            stage_disp = depa_max_disparity // (2 ** i)
            
            blocks = nn.Sequential(
                *[NAFBlock(embed_dim, expansion=2, drop_path=rates[j]) 
                  for j in range(nb)]
            )
            cross_view = BidirectionalDEPA(
                embed_dim, k=depa_k, num_heads=num_heads, max_disparity=stage_disp
            )
            
            if downsample:
                down = nn.Sequential(
                    LayerNorm2d(embed_dim),
                    nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
                )
            else:
                down = None
            
            self.encoder_stages.append(nn.ModuleDict({
                'blocks': blocks,
                'cross_view': cross_view,
                'down': down if down else nn.Identity(),
            }))
            block_idx += nb
        
        bottleneck_disp = depa_max_disparity // (2 ** num_down)
        self.bottleneck = Bottleneck(embed_dim, depa_k, num_heads, bottleneck_disp, corr_max_disp)
        
        self.decoder = BaselineDecoder(embed_dim, dec_blocks)
        self.freq_refine = LightweightFrequencyRefinement(embed_dim)
        self.upsampler = Upsampler(embed_dim, scale, 3)
    
    def forward(self, img_left, img_right):
        B, C, H, W = img_left.shape
        target_h, target_w = H * self.scale, W * self.scale
        
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)
        
        skips_l, skips_r = [], []
        pre_fl, pre_fr = fl, fr
        for i, stage in enumerate(self.encoder_stages):
            fl = stage['blocks'](fl)
            fr = stage['blocks'](fr)
            pre_fl, pre_fr = fl, fr
            fl, fr = stage['cross_view'](fl, fr)
            skips_l.append(fl)
            skips_r.append(fr)
            if stage['down'] is not None and not isinstance(stage['down'], nn.Identity):
                fl = stage['down'](fl)
                fr = stage['down'](fr)
        
        fl = self.bottleneck(pre_fl, pre_fr)
        xl = self.decoder(fl, skips_l)
        xl = self.freq_refine(xl)
        sr_left = self.upsampler(xl)
        sr_left = F.interpolate(sr_left, size=(target_h, target_w), 
                               mode="bilinear", align_corners=False)
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_left = sr_left[:, :, :lr_up_l.shape[2], :lr_up_l.shape[3]]
        sr_left = sr_left + lr_up_l
        
        fr = self.bottleneck(pre_fr, pre_fl)
        xr = self.decoder(fr, skips_r)
        xr = self.freq_refine(xr)
        sr_right = self.upsampler(xr)
        sr_right = F.interpolate(sr_right, size=(target_h, target_w), 
                                mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, 
                               mode="bilinear", align_corners=False)
        sr_right = sr_right[:, :, :lr_up_r.shape[2], :lr_up_r.shape[3]]
        sr_right = sr_right + lr_up_r
        
        conf_l = torch.ones(sr_left.shape[0], 1, sr_left.shape[2], sr_left.shape[3],
                           device=sr_left.device)
        conf_r = torch.ones(sr_right.shape[0], 1, sr_right.shape[2], sr_right.shape[3],
                           device=sr_right.device)
        
        return sr_left, sr_right, conf_l, conf_r


class FullModel(nn.Module):
    """Full model from models_v2.network."""
    def __init__(self, scale=4, embed_dim=48, enc_blocks=None, dec_blocks=None,
                 depa_k=8, depa_max_disparity=32, num_heads=4, corr_max_disp=12,
                 drop_path_rate=0.1, **kwargs):
        super().__init__()
        from models_v2.network import StereoSRNetV2
        
        self.model = StereoSRNetV2(
            scale=scale,
            in_channels=3,
            embed_dim=embed_dim,
            enc_blocks=enc_blocks or [4, 6, 6],
            dec_blocks=dec_blocks or [4, 4],
            depa_k=depa_k,
            depa_max_disparity=depa_max_disparity,
            num_heads=num_heads,
            corr_max_disp=corr_max_disp,
            drop_path_rate=drop_path_rate,
        )
    
    def forward(self, img_left, img_right):
        return self.model(img_left, img_right)


# ─── Utils ──────────────────────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, img_size: Tuple[int, int] = (60, 180)) -> int:
    """Rough FLOPs estimate using fvcore (if available, otherwise approximation)."""
    try:
        from fvcore.nn import FlopCounterMode
        with FlopCounterMode(model) as flop_counter:
            dummy_left = torch.randn(1, 3, img_size[0], img_size[1])
            dummy_right = torch.randn(1, 3, img_size[0], img_size[1])
            _ = model(dummy_left, dummy_right)
        return flop_counter.flop_counts['']
    except:
        # Fallback: rough estimate based on params and img size
        params = count_parameters(model)
        ops_per_param = 2  # Conservative estimate
        return params * ops_per_param * (img_size[0] * img_size[1])


def evaluate_model(model: nn.Module, dataloader: DataLoader,
                  device: torch.device, num_batches: int = -1) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    
    psnr_l_meter = AverageMeter()
    psnr_r_meter = AverageMeter()
    psnr_avg_meter = AverageMeter()
    ssim_l_meter = AverageMeter()
    ssim_r_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if num_batches > 0 and idx >= num_batches:
                break
            
            lr_l = batch["lr_left"].to(device)
            lr_r = batch["lr_right"].to(device)
            hr_l = batch["hr_left"].to(device)
            hr_r = batch["hr_right"].to(device)
            
            sr_l, sr_r, conf_l, conf_r = model(lr_l, lr_r)
            sr_l = sr_l.clamp(0, 1)
            sr_r = sr_r.clamp(0, 1)
            
            # Crop to match HR
            min_h = min(sr_l.shape[2], hr_l.shape[2])
            min_w = min(sr_l.shape[3], hr_l.shape[3])
            sr_l = sr_l[:, :, :min_h, :min_w]
            sr_r = sr_r[:, :, :min_h, :min_w]
            hr_l = hr_l[:, :, :min_h, :min_w]
            hr_r = hr_r[:, :, :min_h, :min_w]
            
            psnr_l = compute_psnr(sr_l, hr_l)
            psnr_r = compute_psnr(sr_r, hr_r)
            psnr_avg = 0.5 * (psnr_l + psnr_r)
            
            ssim_l = compute_ssim(sr_l, hr_l)
            ssim_r = compute_ssim(sr_r, hr_r)
            ssim_avg = 0.5 * (ssim_l + ssim_r)
            
            psnr_l_meter.update(psnr_l)
            psnr_r_meter.update(psnr_r)
            psnr_avg_meter.update(psnr_avg)
            ssim_l_meter.update(ssim_l)
            ssim_r_meter.update(ssim_r)
            ssim_avg_meter.update(ssim_avg)
    
    return {
        'psnr_left': psnr_l_meter.avg,
        'psnr_right': psnr_r_meter.avg,
        'psnr_avg': psnr_avg_meter.avg,
        'ssim_left': ssim_l_meter.avg,
        'ssim_right': ssim_r_meter.avg,
        'ssim_avg': ssim_avg_meter.avg,
    }


def train_config(model: nn.Module, train_dataloader: DataLoader, device: torch.device,
                num_iters: int = 1000, logger=None) -> Dict[str, float]:
    """Train model for num_iters with proper loss optimization."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0)
    
    # Simple L1 loss for reconstruction
    loss_fn = nn.L1Loss()
    
    loss_meter = AverageMeter()
    
    # Create infinite iterator
    dataloader_iter = iter(train_dataloader)
    
    for step in range(num_iters):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader if exhausted
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)
        
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)
        
        optimizer.zero_grad()
        sr_l, sr_r, conf_l, conf_r = model(lr_l, lr_r)
        
        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)
        
        # Crop to match HR
        min_h = min(sr_l.shape[2], hr_l.shape[2])
        min_w = min(sr_l.shape[3], hr_l.shape[3])
        sr_l = sr_l[:, :, :min_h, :min_w]
        sr_r = sr_r[:, :, :min_h, :min_w]
        hr_l = hr_l[:, :, :min_h, :min_w]
        hr_r = hr_r[:, :, :min_h, :min_w]
        
        # Reconstruction loss (L1)
        loss_recon = loss_fn(sr_l, hr_l) + loss_fn(sr_r, hr_r)
        
        loss_recon.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_meter.update(loss_recon.item())
        
        if logger and (step + 1) % 200 == 0:
            logger.info(f"  Step {step+1}/{num_iters} | Loss: {loss_meter.avg:.6f}")
    
    return {'training_loss': loss_meter.avg}


# ─── Ablation Study ─────────────────────────────────────────────────────


def run_ablation_study():
    """Run full ablation study."""
    args = get_config()
    args.test_only = False
    args.save_images = False
    args.tlsc = False
    args.num_workers = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="ablation_study")
    
    # Build train and test datasets
    train_dataset = build_dataset(args, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    logger.info(f"Train set: {len(train_dataset)} samples")
    
    test_dataset = build_dataset(args, split="test")
    # Test samples can have different resolutions, so use batch_size=1 to avoid
    # tensor stacking errors during collation.
    test_loader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    logger.info(f"Test set: {len(test_dataset)} samples, scale={args.scale}")
    
    # Ablation configurations: (id, display_name, model_class)
    all_configs = [
        ("baseline", "Baseline (PixelShuffle only)", BaselineModel),
        ("depa", "+ DEPA", WithDEPAModel),
        ("corr", "+ Correlation Prior", WithCorrPriorModel),
        ("freq", "+ Frequency Refine", WithFreqRefineModel),
        ("full", "Full Model", FullModel),
    ]

    if args.ablation_configs:
        requested = [x.lower().strip() for x in args.ablation_configs]
        valid_ids = {cfg_id for cfg_id, _, _ in all_configs}
        invalid = [x for x in requested if x not in valid_ids]
        if invalid:
            raise ValueError(
                f"Unknown --ablation_configs value(s): {invalid}. "
                f"Valid options: {sorted(valid_ids)}"
            )
        configs = [c for c in all_configs if c[0] in requested]
    else:
        configs = all_configs

    logger.info(
        "Ablation run list: " + ", ".join([cfg_id for cfg_id, _, _ in configs])
    )
    
    ablation_results = []
    
    for config_idx, (config_id, config_name, ModelClass) in enumerate(configs):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{config_idx+1}/{len(configs)}] Testing: {config_name}")
        logger.info(f"{'='*70}")
        
        # Create fresh model with random initialization
        try:
            model = ModelClass(
                scale=args.scale,
                embed_dim=args.embed_dim,
                enc_blocks=args.enc_blocks,
                dec_blocks=args.dec_blocks,
                depa_k=args.depa_k,
                depa_max_disparity=args.depa_max_disparity,
                num_heads=args.num_heads,
                corr_max_disp=args.corr_max_disp,
            )
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        model = model.to(device)
        
        # Count params and FLOPs
        num_params = count_parameters(model)
        num_flops = estimate_flops(model, 
                                   img_size=(args.patch_h * args.scale, 
                                            args.patch_w * args.scale))
        
        logger.info(f"Parameters: {num_params:,}")
        logger.info(f"FLOPs (approx): {num_flops:,}")
        
        # Train for 1000 iterations
        logger.info(f"Training for 1000 iterations...")
        train_start = time.time()
        train_metrics = train_config(model, train_loader, device, 
                                    num_iters=1000, logger=logger)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.1f}s")
        logger.info(f"Training loss: {train_metrics.get('training_loss', 0):.6f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device)
        
        result = {
            'config_id': config_id,
            'config': config_name,
            'params': num_params,
            'flops': num_flops,
            'training_time_sec': train_time,
            'training_loss': train_metrics.get('training_loss', 0),
            **test_metrics,
        }
        ablation_results.append(result)
        
        logger.info(f"\nTest Results:")
        logger.info(f"  PSNR (Left):  {test_metrics['psnr_left']:.4f} dB")
        logger.info(f"  PSNR (Right): {test_metrics['psnr_right']:.4f} dB")
        logger.info(f"  PSNR (Avg):   {test_metrics['psnr_avg']:.4f} dB")
        logger.info(f"  SSIM (Left):  {test_metrics['ssim_left']:.4f}")
        logger.info(f"  SSIM (Right): {test_metrics['ssim_right']:.4f}")
        logger.info(f"  SSIM (Avg):   {test_metrics['ssim_avg']:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Save results
    results_file = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_file, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Print summary table
    logger.info(f"\n{'='*150}")
    logger.info("ABLATION STUDY SUMMARY (1000 iterations per config)")
    logger.info(f"{'='*150}")
    logger.info(
        f"{'Config':<28} {'Params':>12} {'FLOPs':>15} {'Time(s)':>8} "
        f"{'Loss':>10} {'PSNR-L':>10} {'PSNR-R':>10} {'PSNR-Avg':>10} {'SSIM-Avg':>10}"
    )
    logger.info(f"{'-'*150}")
    
    for result in ablation_results:
        logger.info(
            f"{result['config']:<28} "
            f"{result['params']:>12,} {result['flops']:>15,} {result['training_time_sec']:>8.1f} "
            f"{result['training_loss']:>10.6f} "
            f"{result['psnr_left']:>10.4f} {result['psnr_right']:>10.4f} "
            f"{result['psnr_avg']:>10.4f} {result['ssim_avg']:>10.4f}"
        )


if __name__ == "__main__":
    run_ablation_study()

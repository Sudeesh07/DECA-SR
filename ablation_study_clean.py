"""
Clean, fair, publication-quality ablation study for StereoSRNetV2-style stereo SR.

Run:
    python ablation_study_clean.py
    python ablation_study_clean.py --scale 4 --batch_size 8 --train_iters 1000
    python ablation_study_clean.py --variants baseline depa corr

Fairness guarantees:
- All variants share the SAME shallow encoder, encoder depth, decoder depth,
  decoder implementation, upsampling head, global residual path, optimizer,
  loss, and training schedule.
- Only optional modules are toggled on/off.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from models_v2.depa import DEPA, BidirectionalDEPA
from models_v2.frequency import LightweightFrequencyRefinement
from models_v2.network import NAFBlock, LayerNorm2d
from models_v2.occlusion import HybridOcclusionFusion
from utils_fixed import AverageMeter, compute_psnr, compute_ssim, setup_logger


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for one ablation variant."""

    variant_id: str
    display_name: str
    use_depa_encoder: bool
    use_corr_prior: bool
    use_freq_refine: bool
    use_occlusion: bool
    use_bottleneck_depa: bool


class EncoderStage(nn.Module):
    """
    Shared encoder stage with optional encoder DEPA.

    Structure (fixed across variants):
    NAFBlocks -> optional BidirectionalDEPA -> optional downsample
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        depa_k: int,
        depa_max_disparity: int,
        num_heads: int,
        downsample: bool,
        drop_path_rates: Optional[List[float]],
        use_depa_encoder: bool,
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

        self.use_depa_encoder = use_depa_encoder
        if use_depa_encoder:
            self.cross_view = BidirectionalDEPA(
                channels,
                k=depa_k,
                num_heads=num_heads,
                max_disparity=depa_max_disparity,
            )
        else:
            self.cross_view = None

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
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fl = self.blocks(f_left)
        fr = self.blocks(f_right)

        fl_pre, fr_pre = fl, fr

        if self.cross_view is not None:
            fl, fr = self.cross_view(fl, fr)

        fl_skip, fr_skip = fl, fr

        if self.down is not None:
            fl_next = self.down(fl)
            fr_next = self.down(fr)
        else:
            fl_next, fr_next = fl, fr

        return fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre


class CorrelationPrior(nn.Module):
    """Group-wise stereo correlation prior used at bottleneck."""

    def __init__(self, channels: int, num_groups: int = 8, max_disp: int = 12):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels={channels} must be divisible by num_groups={num_groups}")

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

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_left.shape
        g, cg = self.num_groups, self.group_channels
        max_d = min(self.max_disp, w)

        fl = F.normalize(f_left.view(b, g, cg, h, w), dim=2)
        fr = F.normalize(f_right.view(b, g, cg, h, w), dim=2)

        corr_vol = f_left.new_zeros(b, g * self.max_disp, h, w)
        for d in range(max_d):
            if d == 0:
                corr = (fl * fr).mean(dim=2)
            else:
                corr = (fl[:, :, :, :, d:] * fr[:, :, :, :, : w - d]).mean(dim=2)
                corr = F.pad(corr, (d, 0), mode="constant", value=0)
            corr_vol[:, d * g : (d + 1) * g] = corr

        return f_left + self.aggregate(corr_vol) * self.beta


class Bottleneck(nn.Module):
    """Optional correlation prior + optional bottleneck DEPA."""

    def __init__(
        self,
        channels: int,
        depa_k: int,
        num_heads: int,
        max_disparity: int,
        corr_max_disp: int,
        use_corr_prior: bool,
        use_bottleneck_depa: bool,
    ):
        super().__init__()

        self.use_corr_prior = use_corr_prior
        self.use_bottleneck_depa = use_bottleneck_depa

        self.corr_prior = CorrelationPrior(channels, max_disp=corr_max_disp)
        self.depa = DEPA(channels, depa_k, num_heads, max_disparity)

    def forward(self, f_target: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        x = f_target
        if self.use_corr_prior:
            x = self.corr_prior(x, f_ref)
        if self.use_bottleneck_depa:
            x = self.depa(x, f_ref)
        return x


class DecoderStage(nn.Module):
    """Fixed decoder stage shared by all variants."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.blocks = nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(x + skip)


class SimpleUpsampler(nn.Module):
    """Fixed final PixelShuffle upsampler used by all variants."""

    def __init__(self, channels: int, scale: int, out_channels: int = 3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, out_channels * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AblationStereoSRNet(nn.Module):
    """
    One configurable model for all ablation variants.

    Fairness: architecture skeleton is fixed; only optional modules are toggled.
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
        use_depa_encoder: bool = True,
        use_corr_prior: bool = True,
        use_freq_refine: bool = True,
        use_occlusion: bool = True,
        use_bottleneck_depa: bool = True,
    ):
        super().__init__()

        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]

        self.scale = scale
        self.use_freq_refine = use_freq_refine
        self.use_occlusion = use_occlusion

        num_enc = len(enc_blocks)
        num_down = num_enc - 1
        if len(dec_blocks) != num_down:
            raise ValueError(
                f"dec_blocks length ({len(dec_blocks)}) must equal number of downsamples ({num_down})"
            )

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )

        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]

        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            block_idx += nb
            stage_disp = depa_max_disparity // (2**i)
            self.encoder_stages.append(
                EncoderStage(
                    channels=embed_dim,
                    num_blocks=nb,
                    depa_k=depa_k,
                    depa_max_disparity=stage_disp,
                    num_heads=num_heads,
                    downsample=(i < num_down),
                    drop_path_rates=rates,
                    use_depa_encoder=use_depa_encoder,
                )
            )

        bottleneck_disp = depa_max_disparity // (2**num_down)
        self.bottleneck = Bottleneck(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=bottleneck_disp,
            corr_max_disp=corr_max_disp,
            use_corr_prior=use_corr_prior,
            use_bottleneck_depa=use_bottleneck_depa,
        )

        self.decoder_stages = nn.ModuleList([DecoderStage(embed_dim, nb) for nb in dec_blocks])

        self.freq_refine = LightweightFrequencyRefinement(embed_dim)
        self.occlusion = HybridOcclusionFusion(embed_dim)
        self.upsampler = SimpleUpsampler(embed_dim, scale, in_channels)

    def _encode(
        self, fl: torch.Tensor, fr: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        skips_l: List[torch.Tensor] = []
        skips_r: List[torch.Tensor] = []
        pre_l, pre_r = fl, fr

        for i, stage in enumerate(self.encoder_stages):
            fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre = stage(fl, fr)
            skips_l.append(fl_skip)
            skips_r.append(fr_skip)
            if i == len(self.encoder_stages) - 1:
                pre_l, pre_r = fl_pre, fr_pre
            fl, fr = fl_next, fr_next

        return skips_l, skips_r, pre_l, pre_r

    def _decode(self, pre_target: torch.Tensor, pre_ref: torch.Tensor, skips_target: List[torch.Tensor]) -> torch.Tensor:
        x = self.bottleneck(pre_target, pre_ref)
        for i, dec_stage in enumerate(self.decoder_stages):
            skip_idx = len(self.encoder_stages) - 2 - i
            x = dec_stage(x, skips_target[skip_idx])

        if self.use_freq_refine:
            x = self.freq_refine(x)
        return x

    def forward(
        self, img_left: torch.Tensor, img_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)

        skips_l, skips_r, pre_l, pre_r = self._encode(fl, fr)

        xl = self._decode(pre_l, pre_r, skips_l)
        xr = self._decode(pre_r, pre_l, skips_r)

        if self.use_occlusion:
            fused_l, conf_l = self.occlusion(xl, xr)
            fused_r, conf_r = self.occlusion(xr, xl)
        else:
            fused_l, fused_r = xl, xr
            conf_l = xl.new_ones(xl.shape[0], 1, xl.shape[2], xl.shape[3])
            conf_r = xr.new_ones(xr.shape[0], 1, xr.shape[2], xr.shape[3])

        sr_left = self.upsampler(fused_l)
        sr_right = self.upsampler(fused_r)

        # Global residual learning (same for all variants)
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, mode="bilinear", align_corners=False)
        sr_left = sr_left + lr_up_l
        sr_right = sr_right + lr_up_r

        conf_l = F.interpolate(conf_l, size=sr_left.shape[-2:], mode="bilinear", align_corners=False)
        conf_r = F.interpolate(conf_r, size=sr_right.shape[-2:], mode="bilinear", align_corners=False)

        return sr_left, sr_right, conf_l, conf_r


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, img_size: Tuple[int, int] = (60, 180)) -> int:
    """Estimate FLOPs with fvcore if available; fallback to rough approximation."""
    try:
        from fvcore.nn import FlopCountAnalysis

        model_eval = model.eval().cpu()
        left = torch.randn(1, 3, img_size[0], img_size[1])
        right = torch.randn(1, 3, img_size[0], img_size[1])
        flops = FlopCountAnalysis(model_eval, (left, right)).total()
        return int(flops)
    except Exception:
        params = count_parameters(model)
        return int(params * 2 * img_size[0] * img_size[1])


def train_variant(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_iters: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
    logger,
) -> Dict[str, float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
    loss_meter = AverageMeter()

    data_iter = iter(train_loader)

    for step in range(num_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)

        optimizer.zero_grad()
        sr_l, sr_r, _, _ = model(lr_l, lr_r)

        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)

        min_h = min(sr_l.shape[2], hr_l.shape[2])
        min_w = min(sr_l.shape[3], hr_l.shape[3])
        sr_l = sr_l[:, :, :min_h, :min_w]
        sr_r = sr_r[:, :, :min_h, :min_w]
        hr_l = hr_l[:, :, :min_h, :min_w]
        hr_r = hr_r[:, :, :min_h, :min_w]

        loss = loss_fn(sr_l, hr_l) + loss_fn(sr_r, hr_r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_meter.update(loss.item())

        if (step + 1) % log_every == 0:
            logger.info(f"  Step {step + 1}/{num_iters} | L1: {loss_meter.avg:.6f}")

    return {"training_l1": loss_meter.avg}


def evaluate_variant(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    psnr_l_meter = AverageMeter()
    psnr_r_meter = AverageMeter()
    psnr_avg_meter = AverageMeter()
    ssim_l_meter = AverageMeter()
    ssim_r_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()

    with torch.no_grad():
        for batch in test_loader:
            lr_l = batch["lr_left"].to(device)
            lr_r = batch["lr_right"].to(device)
            hr_l = batch["hr_left"].to(device)
            hr_r = batch["hr_right"].to(device)

            sr_l, sr_r, _, _ = model(lr_l, lr_r)
            sr_l = sr_l.clamp(0, 1)
            sr_r = sr_r.clamp(0, 1)

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
        "psnr_left": psnr_l_meter.avg,
        "psnr_right": psnr_r_meter.avg,
        "psnr_avg": psnr_avg_meter.avg,
        "ssim_left": ssim_l_meter.avg,
        "ssim_right": ssim_r_meter.avg,
        "ssim_avg": ssim_avg_meter.avg,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean StereoSR ablation study")

    # Data / run
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--num_workers", type=int, default=0)

    # Training fairness settings
    parser.add_argument("--train_iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=200)

    # Architecture (shared across all variants)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--enc_blocks", nargs="+", type=int, default=[4, 6, 6])
    parser.add_argument("--dec_blocks", nargs="+", type=int, default=[4, 4])
    parser.add_argument("--depa_k", type=int, default=8)
    parser.add_argument("--depa_max_disparity", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--corr_max_disp", type=int, default=12)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)

    # Reproducibility / output
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="ablation_clean")

    # Optional subset run
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional variant IDs: baseline depa corr freq occ full",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir, name="ablation_study_clean")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)

    # Build dataset args namespace expected by build_dataset.
    args.patch_h = 30
    args.patch_w = 90
    args.augment = True

    train_dataset = build_dataset(args, split="train")
    test_dataset = build_dataset(args, split="test")

    # NOTE: Test loader is batch_size=1 since test scenes can have variable size.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples, scale={args.scale}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")

    all_variants: List[AblationConfig] = [
        AblationConfig("baseline", "Baseline", False, False, False, False, False),
        AblationConfig("depa", "+ DEPA", True, False, False, False, False),
        AblationConfig("corr", "+ Correlation Prior", True, True, False, False, False),
        AblationConfig("freq", "+ Frequency Refinement", True, True, True, False, False),
        AblationConfig("occ", "+ Occlusion Fusion", True, True, True, True, True),
        # IMPORTANT: In this clean ablation implementation, `occ` and `full` are
        # intentionally identical. Keep both labels for reporting continuity.
        AblationConfig("full", "Full Model", True, True, True, True, True),
    ]

    if args.variants:
        requested = [x.lower().strip() for x in args.variants]
        valid = {v.variant_id for v in all_variants}
        invalid = [x for x in requested if x not in valid]
        if invalid:
            raise ValueError(f"Unknown --variants {invalid}. Valid: {sorted(valid)}")
        variants = [v for v in all_variants if v.variant_id in requested]
    else:
        variants = all_variants

    logger.info("Run variants: " + ", ".join(v.variant_id for v in variants))

    results: List[Dict[str, float]] = []

    for idx, vcfg in enumerate(variants, start=1):
        logger.info("\n" + "=" * 80)
        logger.info(f"[{idx}/{len(variants)}] {vcfg.display_name} ({vcfg.variant_id})")
        logger.info("=" * 80)

        # Re-seed so each variant starts from deterministic initialization and
        # deterministic data ordering for fair comparison.
        seed_everything(args.seed)

        # Deterministic train shuffling for each variant.
        train_gen = torch.Generator()
        train_gen.manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            generator=train_gen,
        )

        model = AblationStereoSRNet(
            scale=args.scale,
            in_channels=args.in_channels,
            embed_dim=args.embed_dim,
            enc_blocks=args.enc_blocks,
            dec_blocks=args.dec_blocks,
            depa_k=args.depa_k,
            depa_max_disparity=args.depa_max_disparity,
            num_heads=args.num_heads,
            corr_max_disp=args.corr_max_disp,
            drop_path_rate=args.drop_path_rate,
            use_depa_encoder=vcfg.use_depa_encoder,
            use_corr_prior=vcfg.use_corr_prior,
            use_freq_refine=vcfg.use_freq_refine,
            use_occlusion=vcfg.use_occlusion,
            use_bottleneck_depa=vcfg.use_bottleneck_depa,
        ).to(device)

        params = count_parameters(model)
        flops = estimate_flops(model, img_size=(args.patch_h * args.scale, args.patch_w * args.scale))

        logger.info(f"Params: {params:,}")
        logger.info(f"FLOPs:  {flops:,}")

        t0 = time.time()
        train_metrics = train_variant(
            model=model,
            train_loader=train_loader,
            device=device,
            num_iters=args.train_iters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            logger=logger,
        )
        train_time = time.time() - t0

        logger.info(f"Training done in {train_time:.1f}s | L1: {train_metrics['training_l1']:.6f}")

        eval_metrics = evaluate_variant(model, test_loader, device)

        logger.info(
            f"PSNR L/R/Avg: {eval_metrics['psnr_left']:.4f} / "
            f"{eval_metrics['psnr_right']:.4f} / {eval_metrics['psnr_avg']:.4f}"
        )
        logger.info(
            f"SSIM L/R/Avg: {eval_metrics['ssim_left']:.4f} / "
            f"{eval_metrics['ssim_right']:.4f} / {eval_metrics['ssim_avg']:.4f}"
        )

        row: Dict[str, float] = {
            "variant_id": vcfg.variant_id,
            "variant": vcfg.display_name,
            "use_depa_encoder": int(vcfg.use_depa_encoder),
            "use_corr_prior": int(vcfg.use_corr_prior),
            "use_freq_refine": int(vcfg.use_freq_refine),
            "use_occlusion": int(vcfg.use_occlusion),
            "use_bottleneck_depa": int(vcfg.use_bottleneck_depa),
            "params": int(params),
            "flops": int(flops),
            "train_time_sec": float(train_time),
            "train_l1": float(train_metrics["training_l1"]),
            **{k: float(v) for k, v in eval_metrics.items()},
        }
        results.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path = os.path.join(output_dir, "ablation_results_clean.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 140)
    logger.info("CLEAN ABLATION SUMMARY")
    logger.info("=" * 140)
    logger.info(
        f"{'Variant':<24} {'Params':>12} {'FLOPs':>16} {'L1':>10} "
        f"{'PSNR-L':>10} {'PSNR-R':>10} {'PSNR-Avg':>10} {'SSIM-Avg':>10}"
    )
    logger.info("-" * 140)
    for r in results:
        logger.info(
            f"{r['variant']:<24} {int(r['params']):>12,} {int(r['flops']):>16,} {r['train_l1']:>10.6f} "
            f"{r['psnr_left']:>10.4f} {r['psnr_right']:>10.4f} {r['psnr_avg']:>10.4f} {r['ssim_avg']:>10.4f}"
        )

    logger.info(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()

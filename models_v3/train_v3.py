"""Training script for DSRPSNRNet (models_v3).

This script keeps models_v2 behavior untouched by providing a separate entrypoint.

Usage:
    python -m models_v3.train_v3 --scale 4 --model_size base
    python -m models_v3.train_v3 --scale 2 --model_size tiny --batch_size 16
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from utils_fixed import (
    AverageMeter,
    compute_psnr,
    compute_ssim,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
)

from models_v2.config import get_config
from models_v3.network_v3 import build_model_v3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_phase(iteration: int, args) -> int:
    if iteration < args.phase1_iters:
        return 1
    if iteration < args.phase1_iters + args.phase2_iters:
        return 2
    if iteration < args.phase1_iters + args.phase2_iters + args.phase3_iters:
        return 3
    return 4


def get_patch_size(phase: int, scale: int) -> tuple:
    """Returns (patch_h, patch_w) in LR space for each phase."""
    sizes = {1: (30, 90), 2: (40, 120), 3: (48, 144), 4: (60, 180)}
    return sizes.get(phase, (30, 90))


def build_optimizer(model, args):
    return optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    def lr_lambda(step):
        if step < args.warmup_iters:
            return step / max(args.warmup_iters, 1)
        progress = (step - args.warmup_iters) / max(args.total_iters - args.warmup_iters, 1)
        return max(0.5 * (1 + np.cos(np.pi * progress)), 1e-6)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_v3_from_args(args):
    return build_model_v3(
        scale=getattr(args, "scale", 4),
        in_channels=getattr(args, "in_channels", 3),
        embed_dim=getattr(args, "embed_dim", 64),
        enc_blocks=tuple(getattr(args, "enc_blocks", [4, 6, 6])),
        depa_k=getattr(args, "depa_k", 8),
        depa_max_disparity=getattr(args, "depa_max_disparity", None),
        corr_max_disp=getattr(args, "corr_max_disp", None),
        use_bottleneck_stereo=True,
        num_heads=getattr(args, "num_heads", 4),
    )


@torch.no_grad()
def validate(model, val_loader, device, logger):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in val_loader:
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)

        sr_l, sr_r = model(lr_l, lr_r)
        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)

        psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
        ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
        psnr_meter.update(psnr, lr_l.size(0))
        ssim_meter.update(ssim, lr_l.size(0))

    logger.info(f"Validation - PSNR: {psnr_meter.avg:.4f} dB, SSIM: {ssim_meter.avg:.4f}")
    model.train()
    return psnr_meter.avg, ssim_meter.avg


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Keep logs/checkpoints isolated from v2 by default.
    if args.exp_name == "stereo_sr_v2":
        args.exp_name = f"stereo_sr_v3_x{args.scale}"
        args.output_dir = os.path.join(args.log_dir, args.exp_name)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    logger = setup_logger(args.output_dir, name="stereo_sr_v3")
    logger.info(f"Config: {args}")
    set_seed(args.seed)

    model = _build_v3_from_args(args).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params / 1e6:.2f}M")

    if args.pretrain:
        logger.info(f"Loading pre-trained weights from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    criterion_l1 = nn.L1Loss().to(device)
    criterion_mse = nn.MSELoss().to(device)

    optimizer_g = build_optimizer(model, args)
    scheduler_g = build_scheduler(optimizer_g, args)

    scaler_g = torch.amp.GradScaler("cuda", enabled=args.fp16)
    start_iter = 0
    best_psnr = 0.0

    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, model, optimizer_g, scheduler_g, str(device), logger,
        )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
    except Exception as e:
        logger.warning(f"TensorBoard not available: {e}")

    val_dataset = build_dataset(args, split="val")
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=min(2, args.num_workers),
        )

    train_dataset = build_dataset(args, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(train_loader)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    model.train()
    for iteration in range(start_iter, args.total_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lr_l = batch["lr_left"].to(device, non_blocking=True)
        lr_r = batch["lr_right"].to(device, non_blocking=True)
        hr_l = batch["hr_left"].to(device, non_blocking=True)
        hr_r = batch["hr_right"].to(device, non_blocking=True)

        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.fp16):
            sr_l, sr_r = model(lr_l, lr_r)
            loss_l1 = 0.5 * (criterion_l1(sr_l, hr_l) + criterion_l1(sr_r, hr_r))
            loss_mse = 0.5 * (criterion_mse(sr_l, hr_l) + criterion_mse(sr_r, hr_r))
            loss_g = loss_l1 + 0.1 * loss_mse
            loss_dict = {
                "l1": loss_l1,
                "mse": loss_mse,
                "loss": loss_g,
            }

            with torch.no_grad():
                sr_l_clamped = sr_l.clamp(0, 1)
                sr_r_clamped = sr_r.clamp(0, 1)

                psnr = 0.5 * (compute_psnr(sr_l_clamped, hr_l) + compute_psnr(sr_r_clamped, hr_r))
                ssim = 0.5 * (compute_ssim(sr_l_clamped, hr_l) + compute_ssim(sr_r_clamped, hr_r))
                psnr_meter.update(psnr, lr_l.size(0))
                ssim_meter.update(ssim, lr_l.size(0))
                loss_dict["psnr"] = psnr
                loss_dict["ssim"] = ssim

        scaler_g.scale(loss_g).backward()
        if args.grad_clip > 0:
            scaler_g.unscale_(optimizer_g)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler_g.step(optimizer_g)
        scaler_g.update()
        scheduler_g.step()

        if iteration % args.log_interval == 0:
            lr = optimizer_g.param_groups[0]["lr"]
            msg = (
                f"Iter {iteration}/{args.total_iters} | "
                f"lr={lr:.2e} | "
                f"PSNR={psnr_meter.avg:.2f} | SSIM={ssim_meter.avg:.4f} | "
            )
            msg += " | ".join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
            )
            logger.info(msg)

            if writer is not None:
                writer.add_scalar("train/psnr", psnr_meter.avg, iteration)
                writer.add_scalar("train/ssim", ssim_meter.avg, iteration)
                writer.add_scalar("train/lr", lr, iteration)
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"train/{k}", v.item(), iteration)

            psnr_meter.reset()
            ssim_meter.reset()

        if iteration > 0 and iteration % args.val_interval == 0 and val_loader is not None:
            psnr, ssim = validate(model, val_loader, device, logger)
            if writer is not None:
                writer.add_scalar("val/psnr", psnr, iteration)
                writer.add_scalar("val/ssim", ssim, iteration)
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoints", "best.pth"),
                    model, optimizer_g, scheduler_g, iteration, best_psnr, logger,
                )

        if iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                model, optimizer_g, scheduler_g, iteration, best_psnr, logger,
            )

    save_checkpoint(
        os.path.join(args.output_dir, "checkpoints", "final.pth"),
        model, optimizer_g, scheduler_g, args.total_iters, best_psnr, logger,
    )
    logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = get_config()
    if args.test_only:
        raise SystemExit("Use a dedicated test script for v3 evaluation.")
    train(args)

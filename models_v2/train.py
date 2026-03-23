"""
Training script for StereoSRNet V2.

Usage:
    python -m models_v2.train --scale 4 --model_size base
    python -m models_v2.train --scale 2 --model_size tiny --batch_size 16
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import StereoMixup, build_dataset
from utils_fixed import (
    AverageMeter, compute_psnr, compute_ssim,
    load_checkpoint, save_checkpoint, setup_logger,
)

from .config import get_config
from .losses import CombinedLoss, Discriminator
from .network import build_model_v2


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
    """
    Returns (patch_h, patch_w) in LR space for each phase.

    Source LR patches are 30x90 (x2). Curriculum increases crop size
    but stays within what the source data supports to avoid excessive padding.
    For x4, patches are halved in LR space (same HR coverage).
    """
    x2_sizes = {1: (24, 72), 2: (28, 84), 3: (30, 90), 4: (30, 90)}
    h, w = x2_sizes.get(phase, (30, 90))
    if scale == 4:
        h, w = h // 2, w // 2
    return h, w


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


def _active_losses(criterion):
    active = ["pixel"]
    if criterion.use_freq:
        active.append("freq")
    if criterion.use_stereo:
        active.append("stereo")
    if criterion.use_percep:
        active.append("percep")
    if criterion.use_conf_reg:
        active.append("conf_reg")
    if criterion.use_adv:
        active.append("adv")
    return active


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

        sr_l, sr_r, _, _ = model(lr_l, lr_r)
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
    logger = setup_logger(args.output_dir, name="stereo_sr_v2")
    logger.info(f"Config: {args}")
    set_seed(args.seed)

    model = build_model_v2(args).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params / 1e6:.2f}M")

    if args.pretrain:
        logger.info(f"Loading pre-trained weights from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    criterion = CombinedLoss(
        w_pixel=args.w_pixel, w_percep=args.w_percep, w_adv=args.w_adv,
        w_freq=args.w_freq, w_stereo=args.w_stereo, device=str(device),
    ).to(device)

    discriminator = Discriminator().to(device)

    optimizer_g = build_optimizer(model, args)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.9, 0.999))
    scheduler_g = build_scheduler(optimizer_g, args)
    # Discriminator scheduler: only active during phase 4, so use a flat LR
    scheduler_d = optim.lr_scheduler.ConstantLR(optimizer_d, factor=1.0)

    scaler_g = torch.amp.GradScaler("cuda", enabled=args.fp16)
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.fp16)
    start_iter = 0
    best_psnr = 0.0

    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, model, optimizer_g, scheduler_g, str(device), logger,
        )

    mixup = StereoMixup(alpha=0.2)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
    except ImportError:
        logger.warning("TensorBoard not available")

    # Build validation dataset/loader once (doesn't change with phase)
    val_dataset = build_dataset(args, split="val")
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=min(2, args.num_workers),
        )

    current_phase = 0
    train_loader = None
    data_iter = None

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    model.train()
    for iteration in range(start_iter, args.total_iters):
        phase = get_phase(iteration, args)

        # Phase transition: rebuild dataloader with new patch size
        if phase != current_phase:
            current_phase = phase
            criterion.set_phase(phase)
            args.patch_h, args.patch_w = get_patch_size(phase, args.scale)
            logger.info(
                f"=== Phase {phase} @ iter {iteration}: "
                f"patch=({args.patch_h},{args.patch_w}), "
                f"losses={_active_losses(criterion)} ==="
            )
            train_dataset = build_dataset(args, split="train")
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
            )
            data_iter = iter(train_loader)

        # Fetch batch (restart iterator at epoch boundary)
        try:
            if data_iter is None:
                raise TypeError
            batch = next(data_iter)
        except (StopIteration, TypeError):
            if train_loader is None:
                train_dataset = build_dataset(args, split="train")
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                )
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Mixup: phase 2 only. Phase 1 needs clean stereo pairs for offset
        # learning; phases 3-4 need realistic images for perceptual/GAN losses.
        if current_phase == 2 and random.random() > 0.5:
            batch = mixup(batch)

        lr_l = batch["lr_left"].to(device, non_blocking=True)
        lr_r = batch["lr_right"].to(device, non_blocking=True)
        hr_l = batch["hr_left"].to(device, non_blocking=True)
        hr_r = batch["hr_right"].to(device, non_blocking=True)

        # ── Generator step ──────────────────────────────────────────
        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.fp16):
            sr_l, sr_r, conf_l, conf_r = model(lr_l, lr_r)
            loss_g, loss_dict = criterion(sr_l, sr_r, hr_l, hr_r, conf_l, conf_r)

            with torch.no_grad():
                sr_l_clamped = sr_l.clamp(0, 1)
                sr_r_clamped = sr_r.clamp(0, 1)

                psnr = 0.5 * (compute_psnr(sr_l_clamped, hr_l) + compute_psnr(sr_r_clamped, hr_r))
                ssim = 0.5 * (compute_ssim(sr_l_clamped, hr_l) + compute_ssim(sr_r_clamped, hr_r))
                psnr_meter.update(psnr, lr_l.size(0))
                ssim_meter.update(ssim, lr_l.size(0))
                loss_dict["psnr"] = psnr
                loss_dict["ssim"] = ssim

            # GAN generator loss: both views through the CombinedLoss helper
            if criterion.use_adv:
                adv_g_l = criterion.compute_gan_g_loss(discriminator, sr_l, hr_l)
                adv_g_r = criterion.compute_gan_g_loss(discriminator, sr_r, hr_r)
                loss_adv_g = 0.5 * (adv_g_l + adv_g_r)
                loss_g = loss_g + loss_adv_g
                loss_dict["adv_g"] = loss_adv_g

        scaler_g.scale(loss_g).backward()
        if args.grad_clip > 0:
            scaler_g.unscale_(optimizer_g)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler_g.step(optimizer_g)
        scaler_g.update()
        scheduler_g.step()

        # ── Discriminator step (phase 4 only) ───────────────────────
        if criterion.use_adv:
            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.fp16):
                # Both views through the CombinedLoss helper
                loss_d_l = criterion.compute_gan_d_loss(discriminator, sr_l, hr_l)
                loss_d_r = criterion.compute_gan_d_loss(discriminator, sr_r, hr_r)
                loss_d = 0.5 * (loss_d_l + loss_d_r)
                loss_dict["adv_d"] = loss_d

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            scheduler_d.step()

        # ── Logging ─────────────────────────────────────────────────
        if iteration % args.log_interval == 0:
            lr = optimizer_g.param_groups[0]["lr"]
            msg = (
                f"[Phase {current_phase}] Iter {iteration}/{args.total_iters} | "
                f"lr={lr:.2e} | "
                f"PSNR={psnr_meter.avg:.2f} | SSIM={ssim_meter.avg:.4f} | "
            )            
            msg += " | ".join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
            )
            logger.info(msg)
            psnr_meter.reset()
            ssim_meter.reset()

            if writer is not None:
                writer.add_scalar("train/psnr", psnr_meter.avg, iteration)
                writer.add_scalar("train/ssim", ssim_meter.avg, iteration)
                writer.add_scalar("train/lr", current_phase, iteration)
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"train/{k}", v.item(), iteration)

        # ── Validation ──────────────────────────────────────────────
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

        # ── Periodic checkpoint ─────────────────────────────────────
        if iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                model, optimizer_g, scheduler_g, iteration, best_psnr, logger,
            )

    # Final checkpoint
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
        raise SystemExit("Use models_v2.test for testing.")
    train(args)

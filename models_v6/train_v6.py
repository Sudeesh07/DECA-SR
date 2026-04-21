"""Training script for StereoSR V6 (SpecFuse)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Avoid Docker /dev/shm bus-error failures in DataLoader workers.
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from data.dataset import build_dataset
from models_v6 import build_loss_v6, build_model_v6, get_config_v6
from utils_fixed import (
    AverageMeter,
    compute_psnr,
    compute_ssim,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer: optim.Optimizer, args):
    def lr_lambda(step: int) -> float:
        if step < args.warmup_iters:
            return step / max(1, args.warmup_iters)
        progress = (step - args.warmup_iters) / max(1, args.total_iters - args.warmup_iters)
        return max(0.5 * (1.0 + np.cos(np.pi * progress)), 1e-6)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        output = model(lr_l, lr_r)
        sr_l = output.sr_left.clamp(0, 1)
        sr_r = output.sr_right.clamp(0, 1)
        psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
        ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
        psnr_meter.update(psnr, lr_l.size(0))
        ssim_meter.update(ssim, lr_l.size(0))
    logger.info(f"Validation - PSNR: {psnr_meter.avg:.4f} dB, SSIM: {ssim_meter.avg:.4f}")
    model.train()
    return psnr_meter.avg, ssim_meter.avg


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    logger = setup_logger(args.output_dir, name="stereo_sr_v6")
    logger.info(f"Config: {args}")
    set_seed(args.seed)

    model = build_model_v6(args).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params / 1e6:.2f}M")

    if args.pretrain:
        logger.info(f"Loading pretrain from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    criterion = build_loss_v6(args).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(autocast_device, enabled=args.fp16 and device.type == "cuda")

    start_iter = 0
    best_psnr = 0.0
    if args.resume:
        start_iter, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler, str(device), logger)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
    except Exception as exc:
        logger.warning(f"TensorBoard unavailable: {exc}")

    train_dataset = build_dataset(args, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = build_dataset(args, split="val")
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(2, args.num_workers),
        )

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    data_iter = iter(train_loader)

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

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(autocast_device, enabled=args.fp16 and device.type == "cuda"):
            output = model(lr_l, lr_r)
            loss, loss_dict = criterion(output.sr_left, output.sr_right, hr_l, hr_r, aux=output.aux)

            with torch.no_grad():
                sr_l = output.sr_left.clamp(0, 1)
                sr_r = output.sr_right.clamp(0, 1)
                psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
                ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
                psnr_meter.update(psnr, lr_l.size(0))
                ssim_meter.update(ssim, lr_l.size(0))

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if iteration % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"Iter {iteration}/{args.total_iters} | lr={lr:.2e} | "
                f"PSNR={psnr_meter.avg:.2f} | SSIM={ssim_meter.avg:.4f} | "
            )
            msg += " | ".join(
                f"{key}={value.item():.4f}"
                for key, value in loss_dict.items()
                if isinstance(value, torch.Tensor)
            )
            logger.info(msg)

            if writer is not None:
                writer.add_scalar("train/psnr", psnr_meter.avg, iteration)
                writer.add_scalar("train/ssim", ssim_meter.avg, iteration)
                writer.add_scalar("train/lr", lr, iteration)
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        writer.add_scalar(f"train/{key}", value.item(), iteration)

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
                    model, optimizer, scheduler, iteration, best_psnr, logger,
                )

        if iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                model, optimizer, scheduler, iteration, best_psnr, logger,
            )

    save_checkpoint(
        os.path.join(args.output_dir, "checkpoints", "final.pth"),
        model, optimizer, scheduler, args.total_iters, best_psnr, logger,
    )
    logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train(get_config_v6())

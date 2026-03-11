import os
import sys
import time
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import get_config
from models.network import build_model
from losses.losses import CombinedLoss, Discriminator, RelativisticAverageGANLoss
from data.dataset import build_dataset, StereoMixup
from utils import (
    AverageMeter, compute_psnr, compute_ssim,
    setup_logger, save_checkpoint, load_checkpoint,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_phase(iteration, args):
    if iteration < args.phase1_iters:
        return 1
    elif iteration < args.phase1_iters + args.phase2_iters:
        return 2
    elif iteration < args.phase1_iters + args.phase2_iters + args.phase3_iters:
        return 3
    else:
        return 4


def get_patch_size(phase, args):
    sizes = {1: 64, 2: 96, 3: 128, 4: 160}
    return sizes.get(phase, 128)


def build_optimizer(model, args):
    return optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    def lr_lambda(step):
        if step < args.warmup_iters:
            return step / max(args.warmup_iters, 1)
        progress = (step - args.warmup_iters) / max(args.total_iters - args.warmup_iters, 1)
        return max(0.5 * (1 + np.cos(np.pi * progress)), 1e-6)

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

        sr_l, sr_r, _, _ = model(lr_l, lr_r)

        sr_l = sr_l.clamp(0, 1)
        psnr = compute_psnr(sr_l, hr_l)
        ssim = compute_ssim(sr_l, hr_l)
        psnr_meter.update(psnr, lr_l.size(0))
        ssim_meter.update(ssim, lr_l.size(0))

    logger.info(f"Validation — PSNR: {psnr_meter.avg:.4f} dB, SSIM: {ssim_meter.avg:.4f}")
    model.train()
    return psnr_meter.avg, ssim_meter.avg


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir)
    logger.info(f"Config: {args}")
    set_seed(args.seed)
    model = build_model(args).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count / 1e6:.2f}M")
    if args.pretrain:
        logger.info(f"Loading pre-trained weights from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    criterion = CombinedLoss(
        w_pixel=args.w_pixel, w_percep=args.w_percep,
        w_adv=args.w_adv, w_freq=args.w_freq, w_stereo=args.w_stereo,
        device=device,
    ).to(device)
    discriminator = Discriminator().to(device)
    gan_loss = RelativisticAverageGANLoss()
    optimizer_g = build_optimizer(model, args)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5,
                               betas=(0.9, 0.999))
    scheduler_g = build_scheduler(optimizer_g, args)
    scheduler_d = build_scheduler(optimizer_d, args)
    scaler = GradScaler(enabled=args.fp16)
    start_iter = 0
    best_psnr = 0.0
    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, model, optimizer_g, scheduler_g, device, logger)
    mixup = StereoMixup(alpha=0.2)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
    except ImportError:
        writer = None
        logger.warning("TensorBoard not available, skipping logging")
    current_phase = 0
    train_dataset = None
    train_loader = None
    data_iter = None

    model.train()
    for iteration in range(start_iter, args.total_iters):
        phase = get_phase(iteration, args)
        if phase != current_phase:
            current_phase = phase
            criterion.set_phase(phase)
            new_patch = get_patch_size(phase, args)
            logger.info(f"═══ Phase {phase} @ iter {iteration}: "
                        f"patch={new_patch}, losses={_active_losses(criterion)} ═══")
            args.patch_size = new_patch
            train_dataset = build_dataset(args, split="train")
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                pin_memory=True, drop_last=True,
            )
            data_iter = iter(train_loader)
        try:
            batch = next(data_iter)
        except (StopIteration, TypeError):
            if train_loader is None:
                train_dataset = build_dataset(args, split="train")
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True,
                )
            data_iter = iter(train_loader)
            batch = next(data_iter)
        if random.random() > 0.5:
            batch = mixup(batch)

        lr_l = batch["lr_left"].to(device, non_blocking=True)
        lr_r = batch["lr_right"].to(device, non_blocking=True)
        hr_l = batch["hr_left"].to(device, non_blocking=True)
        hr_r = batch["hr_right"].to(device, non_blocking=True)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=args.fp16):
            sr_l, sr_r, conf_l, conf_r = model(lr_l, lr_r)
            loss_g, loss_dict = criterion(sr_l, sr_r, hr_l, hr_r, conf_l, conf_r)
            if criterion.use_adv and current_phase >= 4:
                with torch.no_grad():
                    d_real = discriminator(hr_l)
                d_fake = discriminator(sr_l)
                loss_adv = gan_loss.forward_g(d_real, d_fake)
                loss_g = loss_g + args.w_adv * loss_adv
                loss_dict["adv_g"] = loss_adv

        scaler.scale(loss_g).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer_g)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer_g)
        scaler.update()
        scheduler_g.step()
        if criterion.use_adv and current_phase >= 4:
            optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=args.fp16):
                d_real = discriminator(hr_l.detach())
                d_fake = discriminator(sr_l.detach())
                loss_d = gan_loss.forward_d(d_real, d_fake)
                loss_dict["adv_d"] = loss_d

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()
            scheduler_d.step()
        if iteration % args.log_interval == 0:
            lr = optimizer_g.param_groups[0]["lr"]
            msg = (f"[Phase {current_phase}] Iter {iteration}/{args.total_iters} | "
                   f"lr={lr:.2e} | ")
            msg += " | ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items()
                              if isinstance(v, torch.Tensor))
            logger.info(msg)

            if writer:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"train/{k}", v.item(), iteration)
                writer.add_scalar("train/lr", lr, iteration)
        if iteration > 0 and iteration % args.val_interval == 0:
            val_dataset = build_dataset(args, split="val")
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=1,
                                        shuffle=False, num_workers=2)
                psnr, ssim = validate(model, val_loader, device, logger)
                if writer:
                    writer.add_scalar("val/psnr", psnr, iteration)
                    writer.add_scalar("val/ssim", ssim, iteration)

                if psnr > best_psnr:
                    best_psnr = psnr
                    save_checkpoint(
                        os.path.join(args.output_dir, "checkpoints", "best.pth"),
                        model, optimizer_g, scheduler_g, iteration, best_psnr, logger
                    )
        if iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                model, optimizer_g, scheduler_g, iteration, best_psnr, logger
            )
    save_checkpoint(
        os.path.join(args.output_dir, "checkpoints", "final.pth"),
        model, optimizer_g, scheduler_g, args.total_iters, best_psnr, logger
    )
    logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")

    if writer:
        writer.close()


def _active_losses(criterion):
    active = ["pixel"]
    if criterion.use_freq:
        active.append("freq")
    if criterion.use_stereo:
        active.append("stereo")
    if criterion.use_percep:
        active.append("percep")
    if criterion.use_adv:
        active.append("adv")
    return active


if __name__ == "__main__":
    args = get_config()
    if args.test_only:
        print("Use test.py for testing.")
        sys.exit(1)
    train(args)

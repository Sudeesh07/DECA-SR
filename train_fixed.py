import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from configs.config_fixed import get_config
from data.dataset import StereoMixup, build_dataset
from losses.losses_fixed import CombinedLoss, Discriminator, RelativisticAverageGANLoss
from models.network_standalone import build_model_standalone
from utils_fixed import AverageMeter, compute_psnr, compute_ssim, load_checkpoint, save_checkpoint, setup_logger


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
    if iteration < args.phase1_iters + args.phase2_iters:
        return 2
    if iteration < args.phase1_iters + args.phase2_iters + args.phase3_iters:
        return 3
    return 4


def get_patch_size(phase):
    """Returns (patch_h, patch_w) per phase. Width > height for stereo."""
    sizes = {1: (30, 90), 2: (40, 120), 3: (48, 144), 4: (60, 180)}
    return sizes.get(phase, (48, 144))


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
        sr_r = sr_r.clamp(0, 1)

        psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
        ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
        psnr_meter.update(psnr, lr_l.size(0))
        ssim_meter.update(ssim, lr_l.size(0))

    logger.info(f"Validation - PSNR: {psnr_meter.avg:.4f} dB, SSIM: {ssim_meter.avg:.4f}")
    model.train()
    return psnr_meter.avg, ssim_meter.avg


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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="stereo_sr_fixed")
    logger.info(f"Config: {args}")
    set_seed(args.seed)

    model = build_model_standalone(args).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params / 1e6:.2f}M")

    if args.pretrain:
        logger.info(f"Loading pre-trained weights from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    criterion = CombinedLoss(
        w_pixel=args.w_pixel,
        w_percep=args.w_percep,
        w_adv=args.w_adv,
        w_freq=args.w_freq,
        w_stereo=args.w_stereo,
        device=str(device),
    ).to(device)

    discriminator = Discriminator().to(device)
    gan_loss = RelativisticAverageGANLoss()

    optimizer_g = build_optimizer(model, args)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.9, 0.999))
    scheduler_g = build_scheduler(optimizer_g, args)
    scheduler_d = build_scheduler(optimizer_d, args)

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    start_iter = 0
    best_psnr = 0.0

    if args.resume:
        start_iter, best_psnr = load_checkpoint(args.resume, model, optimizer_g, scheduler_g, str(device), logger)

    mixup = StereoMixup(alpha=0.2)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
    except ImportError:
        logger.warning("TensorBoard not available, skipping logging")

    current_phase = 0
    train_loader = None
    data_iter = None

    model.train()
    for iteration in range(start_iter, args.total_iters):
        phase = get_phase(iteration, args)
        if phase != current_phase:
            current_phase = phase
            criterion.set_phase(phase)
            args.patch_h, args.patch_w = get_patch_size(phase)
            logger.info(
                f"=== Phase {phase} @ iter {iteration}: patch=({args.patch_h},{args.patch_w}), losses={_active_losses(criterion)} ==="
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

        try:
            if data_iter is None:
                raise TypeError("data_iter is not initialized")
            batch = next(data_iter)
        except (StopIteration, TypeError):
            if train_loader is None:
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
            msg = f"[Phase {current_phase}] Iter {iteration}/{args.total_iters} | lr={lr:.2e} | "
            msg += " | ".join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
            )
            logger.info(msg)

            if writer is not None:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"train/{k}", v.item(), iteration)
                writer.add_scalar("train/lr", lr, iteration)

        if iteration > 0 and iteration % args.val_interval == 0:
            val_dataset = build_dataset(args, split="val")
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=min(2, args.num_workers))
                psnr, ssim = validate(model, val_loader, device, logger)
                if writer is not None:
                    writer.add_scalar("val/psnr", psnr, iteration)
                    writer.add_scalar("val/ssim", ssim, iteration)
                if psnr > best_psnr:
                    best_psnr = psnr
                    save_checkpoint(
                        os.path.join(args.output_dir, "checkpoints", "best.pth"),
                        model,
                        optimizer_g,
                        scheduler_g,
                        iteration,
                        best_psnr,
                        logger,
                    )

        if iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                model,
                optimizer_g,
                scheduler_g,
                iteration,
                best_psnr,
                logger,
            )

    save_checkpoint(
        os.path.join(args.output_dir, "checkpoints", "final.pth"),
        model,
        optimizer_g,
        scheduler_g,
        args.total_iters,
        best_psnr,
        logger,
    )
    logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = get_config()
    if args.test_only:
        raise SystemExit("Use test.py for testing.")
    train(args)

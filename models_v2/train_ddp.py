"""
Distributed training for StereoSRNet V2 on 2x NVIDIA A5000.

Launch:
    torchrun --nproc_per_node=2 -m models_v2.train_ddp --scale 2 --model_size base
    torchrun --nproc_per_node=2 -m models_v2.train_ddp --scale 4 --model_size tiny --batch_size 8

Each A5000 has 24GB VRAM. With fp16 enabled:
    - tiny  (0.4M): batch_size=16 per GPU  (32 effective)
    - base  (2.5M): batch_size=8 per GPU   (16 effective)
    - large (8M+):  batch_size=4 per GPU    (8 effective)

The script handles:
    - DDP with NCCL backend (fastest for multi-GPU NVIDIA)
    - Per-GPU grad scaler (AMP)
    - DistributedSampler for data sharding
    - Only rank 0 logs, validates, and saves checkpoints
    - Automatic port selection to avoid collisions
    - Graceful cleanup on error
"""

import os
import random
import signal
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataset import StereoMixup, build_dataset
from utils_fixed import (
    AverageMeter,
    compute_psnr,
    compute_ssim,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
)

from .config import get_config
from .losses import CombinedLoss, Discriminator
from .network import build_model_v2


# ── Helpers ─────────────────────────────────────────────────────────────


def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_phase(iteration: int, args) -> int:
    if iteration < args.phase1_iters:
        return 1
    if iteration < args.phase1_iters + args.phase2_iters:
        return 2
    if iteration < args.phase1_iters + args.phase2_iters + args.phase3_iters:
        return 3
    return 4


def get_patch_size(phase: int, scale: int) -> tuple:
    sizes = {1: (24, 72), 2: (28, 84), 3: (30, 90), 4: (30, 90)}
    return sizes.get(phase, (30, 90))


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
        progress = (step - args.warmup_iters) / max(
            args.total_iters - args.warmup_iters, 1
        )
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


def build_train_loader(args, rank, world_size):
    dataset = build_dataset(args, split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


# ── Validation (rank 0 only) ───────────────────────────────────────────


@torch.no_grad()
def validate(model, val_loader, device, logger):
    # Unwrap DDP for validation
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in val_loader:
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)

        sr_l, sr_r, _, _ = raw_model(lr_l, lr_r)
        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)

        psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
        ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
        psnr_meter.update(psnr, lr_l.size(0))
        ssim_meter.update(ssim, lr_l.size(0))

    logger.info(
        f"Validation - PSNR: {psnr_meter.avg:.4f} dB, SSIM: {ssim_meter.avg:.4f}"
    )
    raw_model.train()
    return psnr_meter.avg, ssim_meter.avg


# ── Main training loop ─────────────────────────────────────────────────


def train(args):
    # ── DDP setup ────────────────────────────────────────────────────
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    set_seed(args.seed, rank)

    logger = None
    writer = None
    if is_main():
        logger = setup_logger(args.output_dir, name="stereo_sr_v2")
        logger.info(f"Config: {args}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
        except ImportError:
            logger.warning("TensorBoard not available")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model_v2(args).to(device)

    if is_main():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {params / 1e6:.2f}M")

    if args.pretrain:
        if is_main():
            logger.info(f"Loading pre-trained weights from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # Wrap in DDP (find_unused_parameters needed because bottleneck uses
    # pre-DEPA features while post-DEPA skips are also stored)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )

    raw_model = model.module if hasattr(model, "module") else model

    # ── Losses ───────────────────────────────────────────────────────
    criterion = CombinedLoss(
        w_pixel=args.w_pixel,
        w_percep=args.w_percep,
        w_adv=args.w_adv,
        w_freq=args.w_freq,
        w_stereo=args.w_stereo,
        device=str(device),
    ).to(device)

    discriminator = Discriminator().to(device)
    if world_size > 1:
        discriminator = DDP(discriminator, device_ids=[rank], output_device=rank)

    # ── Optimizers / schedulers ──────────────────────────────────────
    # Scale LR linearly with effective batch size (base LR is for batch=8)
    effective_bs = args.batch_size * world_size
    lr_scale = effective_bs / 8.0
    scaled_lr = args.lr * lr_scale

    optimizer_g = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    optimizer_d = optim.AdamW(
        discriminator.parameters(), lr=scaled_lr * 0.5, betas=(0.9, 0.999)
    )
    scheduler_g = build_scheduler(optimizer_g, args)
    scheduler_d = optim.lr_scheduler.ConstantLR(optimizer_d, factor=1.0)

    scaler_g = torch.amp.GradScaler("cuda", enabled=args.fp16)
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.fp16)

    start_iter = 0
    best_psnr = 0.0

    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, raw_model, optimizer_g, scheduler_g, str(device),
            logger if is_main() else None,
        )

    mixup = StereoMixup(alpha=0.2)

    # ── Validation loader (rank 0 only) ──────────────────────────────
    val_loader = None
    if is_main():
        val_dataset = build_dataset(args, split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=min(2, args.num_workers),
            )

    # ── Training loop ────────────────────────────────────────────────
    current_phase = 0
    train_loader = None
    train_sampler = None
    data_iter = None
    epoch = 0

    psnr_meter = AverageMeter()

    model.train()
    for iteration in range(start_iter, args.total_iters):
        phase = get_phase(iteration, args)

        # Phase transition
        if phase != current_phase:
            current_phase = phase
            criterion.set_phase(phase)
            args.patch_h, args.patch_w = get_patch_size(phase, args.scale)
            if is_main():
                logger.info(
                    f"=== Phase {phase} @ iter {iteration}: "
                    f"patch=({args.patch_h},{args.patch_w}), "
                    f"losses={_active_losses(criterion)} ==="
                )
            train_loader, train_sampler = build_train_loader(args, rank, world_size)
            epoch = 0
            train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)

        # Fetch batch
        try:
            if data_iter is None:
                raise TypeError
            batch = next(data_iter)
        except (StopIteration, TypeError):
            epoch += 1
            if train_loader is None:
                train_loader, train_sampler = build_train_loader(args, rank, world_size)
            train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Mixup: phase 2 only
        if current_phase == 2 and random.random() > 0.5:
            batch = mixup(batch)

        lr_l = batch["lr_left"].to(device, non_blocking=True)
        lr_r = batch["lr_right"].to(device, non_blocking=True)
        hr_l = batch["hr_left"].to(device, non_blocking=True)
        hr_r = batch["hr_right"].to(device, non_blocking=True)

        # ── Generator step ───────────────────────────────────────────
        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.fp16):
            sr_l, sr_r, conf_l, conf_r = model(lr_l, lr_r)
            loss_g, loss_dict = criterion(sr_l, sr_r, hr_l, hr_r, conf_l, conf_r)

            with torch.no_grad():
                psnr = 0.5 * (
                    compute_psnr(sr_l.clamp(0, 1), hr_l)
                    + compute_psnr(sr_r.clamp(0, 1), hr_r)
                )
                psnr_meter.update(psnr, lr_l.size(0))

            if criterion.use_adv:
                disc_raw = discriminator.module if hasattr(discriminator, "module") else discriminator
                adv_g_l = criterion.compute_gan_g_loss(disc_raw, sr_l, hr_l)
                adv_g_r = criterion.compute_gan_g_loss(disc_raw, sr_r, hr_r)
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

        # ── Discriminator step ───────────────────────────────────────
        if criterion.use_adv:
            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.fp16):
                disc_raw = discriminator.module if hasattr(discriminator, "module") else discriminator
                loss_d_l = criterion.compute_gan_d_loss(disc_raw, sr_l, hr_l)
                loss_d_r = criterion.compute_gan_d_loss(disc_raw, sr_r, hr_r)
                loss_d = 0.5 * (loss_d_l + loss_d_r)
                loss_dict["adv_d"] = loss_d
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            scheduler_d.step()

        # ── Logging (rank 0 only) ────────────────────────────────────
        if is_main() and iteration % args.log_interval == 0:
            lr = optimizer_g.param_groups[0]["lr"]
            msg = (
                f"[Phase {current_phase}] Iter {iteration}/{args.total_iters} | "
                f"lr={lr:.2e} | PSNR={psnr_meter.avg:.2f} | "
            )
            msg += " | ".join(
                f"{k}={v.item():.4f}"
                for k, v in loss_dict.items()
                if isinstance(v, torch.Tensor)
            )
            logger.info(msg)
            psnr_meter.reset()

            if writer is not None:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"train/{k}", v.item(), iteration)
                writer.add_scalar("train/lr", lr, iteration)
                writer.add_scalar("train/phase", current_phase, iteration)

        # ── Validation (rank 0 only) ─────────────────────────────────
        if (
            is_main()
            and iteration > 0
            and iteration % args.val_interval == 0
            and val_loader is not None
        ):
            psnr_val, ssim_val = validate(model, val_loader, device, logger)
            if writer is not None:
                writer.add_scalar("val/psnr", psnr_val, iteration)
                writer.add_scalar("val/ssim", ssim_val, iteration)
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoints", "best.pth"),
                    raw_model, optimizer_g, scheduler_g,
                    iteration, best_psnr, logger,
                )

        # ── Periodic checkpoint (rank 0 only) ────────────────────────
        if is_main() and iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                raw_model, optimizer_g, scheduler_g,
                iteration, best_psnr, logger,
            )

        # Sync all ranks at checkpoint boundaries
        if world_size > 1 and iteration > 0 and iteration % args.save_interval == 0:
            dist.barrier()

    # ── Final ────────────────────────────────────────────────────────
    if is_main():
        save_checkpoint(
            os.path.join(args.output_dir, "checkpoints", "final.pth"),
            raw_model, optimizer_g, scheduler_g,
            args.total_iters, best_psnr, logger,
        )
        logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = get_config()
    if args.test_only:
        raise SystemExit("Use models_v2.test for testing.")
    train(args)

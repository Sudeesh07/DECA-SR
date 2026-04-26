"""Distributed training for StereoSR V6 (SpecFuse).

Launch examples
---------------
    # 2× H100 80GB (recommended)
    torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \\
        --model_size small --scale 4 --batch_size 32 \\
        --exp_name stereo_sr_v6_small_x4

    # 4× GPU
    torchrun --nproc_per_node=4 -m models_v6.train_ddp_v6 \\
        --model_size base --scale 4 --batch_size 16

    # Single GPU fallback (works without torchrun)
    python -m models_v6.train_ddp_v6 --model_size small --batch_size 32

Per-GPU batch size guidance (fp16)
----------------------------------
V6 is ~10× smaller than V5 (1.5M vs 12.1M params), so much larger batches fit.

    Preset   | H100 80GB | A100 80GB | A100 40GB | A5000 24GB
    ---------|-----------|-----------|-----------|-----------
    tiny     |   128     |    96     |    48     |    24
    small    |    64     |    48     |    32     |    16
    base     |    48     |    32     |    24     |    12
    large    |    24     |    16     |    12     |     8

Defaults below target 2× H100 with --batch_size 32 (effective batch 64).

LR scaling
----------
Base LR in config_v6.py is calibrated for effective batch = 8. We apply linear
scaling: scaled_lr = base_lr × (effective_batch / 8). For 2×H100 batch=32 this
gives 2e-4 × 8 = 1.6e-3, matching common SR training recipes.
"""
from __future__ import annotations

import os
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _shm_free_mb() -> float:
    """Return free space on /dev/shm in MB (used by DataLoader IPC on Linux)."""
    try:
        return shutil.disk_usage("/dev/shm").free / (1024 * 1024)
    except (FileNotFoundError, OSError):
        return float("inf")  # non-Linux or /dev/shm missing — don't downgrade


def _maybe_force_inprocess_loading(args) -> None:
    """Force num_workers=0 when /dev/shm is too small to host IPC buffers.

    Docker containers default to 64 MB /dev/shm which cannot hold even one
    batch of stereo patches. Rather than SIGBUS'ing mid-training, we detect
    the shortage up-front and fall back to in-process (zero-worker) loading.
    """
    if getattr(args, "num_workers", 0) <= 0:
        return
    free_mb = _shm_free_mb()
    # Rough per-worker budget: one batch of float32 tensors.
    # (lr + hr) × 2 views × batch × patch_h × patch_w × scale² × 4 bytes each.
    per_worker_mb = (
        2 * 2 * args.batch_size
        * args.patch_h * args.patch_w
        * (1 + args.scale ** 2) * 4
    ) / (1024 * 1024)
    needed_mb = per_worker_mb * args.num_workers * 2  # ×2 for double-buffer
    if free_mb < max(needed_mb, 512):
        print(
            f"[shm-guard] /dev/shm has only {free_mb:.0f} MB free "
            f"(need ~{needed_mb:.0f} MB for {args.num_workers} workers). "
            f"Forcing num_workers=0 to avoid SIGBUS / ENOSPC. "
            f"Remount /dev/shm or restart container with --shm-size=16g to re-enable workers.",
            flush=True,
        )
        args.num_workers = 0


# Prefer file_system strategy — more resilient to fd limits; same /dev/shm
# backing on Linux but safer under abrupt worker death.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer: optim.Optimizer, args):
    def lr_lambda(step: int) -> float:
        if step < args.warmup_iters:
            return step / max(1, args.warmup_iters)
        progress = (step - args.warmup_iters) / max(1, args.total_iters - args.warmup_iters)
        return max(0.5 * (1.0 + np.cos(np.pi * progress)), 1e-6)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_train_loader(args, rank: int, world_size: int):
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


# ---------------------------------------------------------------------------
# Validation (rank 0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, logger) -> tuple[float, float]:
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in val_loader:
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)

        output = raw_model(lr_l, lr_r)
        sr_l = output.sr_left.clamp(0, 1)
        sr_r = output.sr_right.clamp(0, 1)

        psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
        ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
        psnr_meter.update(psnr, lr_l.size(0))
        ssim_meter.update(ssim, lr_l.size(0))

    logger.info(f"Validation - PSNR: {psnr_meter.avg:.4f} dB, SSIM: {ssim_meter.avg:.4f}")
    raw_model.train()
    return psnr_meter.avg, ssim_meter.avg


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ── DDP setup ────────────────────────────────────────────────────
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    # Auto-fallback to in-process loading when /dev/shm is too small.
    _maybe_force_inprocess_loading(args)

    set_seed(args.seed, rank)

    logger = None
    writer = None
    if is_main():
        logger = setup_logger(args.output_dir, name="stereo_sr_v6_ddp")
        logger.info(f"Config: {args}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Per-GPU batch size: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
        except ImportError:
            logger.warning("TensorBoard not available")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model_v6(args).to(device)

    if is_main():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {params / 1e6:.3f}M")

    if args.pretrain:
        if is_main():
            logger.info(f"Loading pretrain from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # Wrap in DDP. Empirically the zero-init γ scales in CVSFM and NAFBlock
    # still produce non-zero gradients at step 0 (because the gate output is
    # not exactly zero — sigmoid(-5) ≈ 0.007 — so every param remains on the
    # backward graph). DDP therefore does not need find_unused_parameters,
    # which avoids the extra autograd-graph traversal per iteration.
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )

    raw_model = model.module if hasattr(model, "module") else model

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = build_loss_v6(args).to(device)

    # ── Optimizer / Scheduler ────────────────────────────────────────
    # Linear LR scaling: base LR calibrated for batch_size=8.
    effective_bs = args.batch_size * world_size
    lr_scale = effective_bs / 8.0
    scaled_lr = args.lr * lr_scale

    if is_main():
        logger.info(f"LR scaling: base={args.lr:.2e} x {lr_scale:.1f} = {scaled_lr:.2e}")

    optimizer = build_optimizer(model, scaled_lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    start_iter = 0
    best_psnr = 0.0
    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, raw_model, optimizer, scheduler, str(device),
            logger if is_main() else None,
        )

    # ── Data loaders ─────────────────────────────────────────────────
    train_loader, train_sampler = build_train_loader(args, rank, world_size)

    val_loader = None
    if is_main():
        val_dataset = build_dataset(args, split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=min(2, args.num_workers),
            )

    # ── Training loop ────────────────────────────────────────────────
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    epoch = 0
    train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)

    model.train()
    for iteration in range(start_iter, args.total_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lr_l = batch["lr_left"].to(device, non_blocking=True)
        lr_r = batch["lr_right"].to(device, non_blocking=True)
        hr_l = batch["hr_left"].to(device, non_blocking=True)
        hr_r = batch["hr_right"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.fp16):
            output = model(lr_l, lr_r)
            loss, loss_dict = criterion(
                output.sr_left, output.sr_right, hr_l, hr_r, aux=output.aux
            )

            with torch.no_grad():
                sr_l = output.sr_left.clamp(0, 1)
                sr_r = output.sr_right.clamp(0, 1)
                psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
                ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
                psnr_meter.update(psnr, lr_l.size(0))
                ssim_meter.update(ssim, lr_l.size(0))

        # NaN/Inf guard: skip this iteration if the forward produced a
        # non-finite loss. Otherwise calling backward()+step() would propagate
        # the bad value into the weights and corrupt the model permanently.
        # All DDP ranks must agree on the skip decision, so we all-reduce a
        # finite-flag (min: 0 if any rank has NaN/Inf, 1 if all are finite).
        finite_flag = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
        if world_size > 1:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

        if finite_flag.item() < 1.0:
            if is_main() and iteration % args.log_interval == 0:
                logger.warning(
                    f"Iter {iteration}: non-finite loss detected, skipping step"
                )
            # Don't touch scaler — scaler.scale()/backward() never ran, so
            # there's no inf-check state to consume. Advance the LR schedule
            # to keep the cosine plan on track.
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            # Note: nn.utils.clip_grad_norm_ may return inf when AMP produces
            # non-finite gradients; that's OK because scaler.step(optimizer)
            # below detects inf gradients via its own inf-check and silently
            # skips the optimizer step (and scaler.update() then backs off
            # the loss scale). No manual skip needed.
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Logging (rank 0 only) ────────────────────────────────────
        if is_main() and iteration % args.log_interval == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            msg = (
                f"Iter {iteration}/{args.total_iters} | lr={lr_current:.2e} | "
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
                writer.add_scalar("train/lr", lr_current, iteration)
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        writer.add_scalar(f"train/{key}", value.item(), iteration)

            psnr_meter.reset()
            ssim_meter.reset()

        # ── Validation (rank 0 only) ─────────────────────────────────
        if is_main() and iteration > 0 and iteration % args.val_interval == 0 and val_loader is not None:
            psnr_val, ssim_val = validate(model, val_loader, device, logger)
            if writer is not None:
                writer.add_scalar("val/psnr", psnr_val, iteration)
                writer.add_scalar("val/ssim", ssim_val, iteration)
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoints", "best.pth"),
                    raw_model, optimizer, scheduler, iteration, best_psnr, logger,
                )

        # ── Periodic checkpoint (rank 0 only) ────────────────────────
        if is_main() and iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                raw_model, optimizer, scheduler, iteration, best_psnr, logger,
            )

        # Sync all ranks at checkpoint boundaries
        if world_size > 1 and iteration > 0 and iteration % args.save_interval == 0:
            dist.barrier()

    # ── Final ────────────────────────────────────────────────────────
    if is_main():
        save_checkpoint(
            os.path.join(args.output_dir, "checkpoints", "final.pth"),
            raw_model, optimizer, scheduler, args.total_iters, best_psnr, logger,
        )
        logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(get_config_v6())

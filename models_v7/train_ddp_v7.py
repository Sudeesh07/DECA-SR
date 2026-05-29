"""Distributed training for StereoSR V7.

What's new vs V6
----------------
* EMA of model weights (validates on the EMA copy).
* AdamW with NAFSSR's β₂ = 0.9 (V6 used 0.99).
* Default patch size 48×144 (V6 used 30×90).
* RGB-shuffle augmentation (random channel permutation, same on L and R).
* SCAM disparity-prior diagnostics (sigma, bias_scale, predicted disparity).

Launch examples
---------------
    # 2× H100 80GB
    torchrun --nproc_per_node=2 -m models_v7.train_ddp_v7 \\
        --model_size small --scale 4 --batch_size 16 --bf16 \\
        --exp_name stereo_sr_v7_small_x4
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
    try:
        return shutil.disk_usage("/dev/shm").free / (1024 * 1024)
    except (FileNotFoundError, OSError):
        return float("inf")


def _maybe_force_inprocess_loading(args) -> None:
    """Force num_workers=0 when /dev/shm is too small (Docker default 64MB)."""
    if getattr(args, "num_workers", 0) <= 0:
        return
    free_mb = _shm_free_mb()
    per_worker_mb = (
        2 * 2 * args.batch_size
        * args.patch_h * args.patch_w
        * (1 + args.scale ** 2) * 4
    ) / (1024 * 1024)
    needed_mb = per_worker_mb * args.num_workers * 2
    if free_mb < max(needed_mb, 512):
        print(
            f"[shm-guard] /dev/shm has only {free_mb:.0f} MB free "
            f"(need ~{needed_mb:.0f} MB for {args.num_workers} workers). "
            f"Forcing num_workers=0 to avoid SIGBUS / ENOSPC. "
            f"Remount /dev/shm or restart with --shm-size=16g to re-enable workers.",
            flush=True,
        )
        args.num_workers = 0


try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


from data.dataset import build_dataset
from models_v7 import build_loss_v7, build_model_v7, get_config_v7, ModelEMA
from models_v7.epipolar_attention import StereoCrossAttention
from utils_fixed import (
    AverageMeter,
    compute_psnr,
    compute_ssim,
    load_checkpoint,
    prune_checkpoints,
    setup_logger,
)

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_TIMEOUT", "3600")


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


def save_checkpoint_with_ema(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    iteration: int,
    best_psnr: float,
    logger,
    ema: ModelEMA | None,
) -> None:
    """Atomic checkpoint save including optional EMA state.

    Adapted from utils_fixed.save_checkpoint, with an "ema" entry. The dict
    layout is BACKWARD-COMPATIBLE with V6 — if "ema" is missing the rest of
    the dict loads normally via load_checkpoint.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "best_psnr": best_psnr,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    tmp_path = path + ".tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        if logger is not None:
            logger.info(f"Checkpoint saved: {path}")
    except (OSError, RuntimeError) as exc:
        if logger is not None:
            logger.error(f"Checkpoint save FAILED at {path}: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
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
# Augmentation (applied AFTER the dataloader's crop/flip)
# ---------------------------------------------------------------------------

def rgb_shuffle_augment(lr_l, lr_r, hr_l, hr_r):
    """Random RGB channel permutation, same permutation on L and R and on
    LR and HR. NAFSSR augmentation; trivially cheap, free regularisation."""
    if lr_l.shape[1] != 3:
        return lr_l, lr_r, hr_l, hr_r  # only meaningful for RGB
    perms = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    perm = perms[random.randint(0, 5)]
    idx = torch.tensor(perm, device=lr_l.device, dtype=torch.long)
    return lr_l[:, idx], lr_r[:, idx], hr_l[:, idx], hr_r[:, idx]


def vertical_flip_augment(lr_l, lr_r, hr_l, hr_r):
    """Random vertical flip; preserves stereo geometry (epipolar lines stay
    horizontal). Same flip applied to both views and both LR/HR."""
    if random.random() < 0.5:
        return (
            torch.flip(lr_l, dims=[-2]),
            torch.flip(lr_r, dims=[-2]),
            torch.flip(hr_l, dims=[-2]),
            torch.flip(hr_r, dims=[-2]),
        )
    return lr_l, lr_r, hr_l, hr_r


# ---------------------------------------------------------------------------
# Diagnostics — gather SCAM disparity-prior stats
# ---------------------------------------------------------------------------

def gather_scam_diagnostics(model: nn.Module) -> list[dict]:
    """Walk the trunks and read each SCAM block's diagnostic captures.

    For each SCAM block, returns σ (Gaussian width), bias_scale (how active
    the prior is), and the mean/max absolute disparity (|x_L - expected x_R|)
    predicted at the most-recent forward pass.

    Empty list if the model has no disparity prior (e.g. --no_disparity_prior).
    """
    raw_model = model.module if hasattr(model, "module") else model
    out: list[dict] = []
    block_idx = 0
    for module in raw_model.modules():
        if not isinstance(module, StereoCrossAttention):
            continue
        d: dict = {"block": block_idx}
        if module._diag_sigma is not None:
            d["sigma"] = module._diag_sigma.item()
            d["bias_scale"] = module._diag_bias_scale.item()
            d["disp_mean"] = module._diag_disp_mean_abs.item()
            d["disp_max"] = module._diag_disp_max_abs.item()
        out.append(d)
        block_idx += 1
    return out


def log_scam_diagnostics(diags: list[dict], iteration: int, logger, writer):
    if not diags:
        return
    sigmas = [d["sigma"] for d in diags if "sigma" in d]
    bsc = [d["bias_scale"] for d in diags if "bias_scale" in d]
    disp_mean = [d["disp_mean"] for d in diags if "disp_mean" in d]
    disp_max = [d["disp_max"] for d in diags if "disp_max" in d]
    if not sigmas:
        return
    msg = (
        f"  diag SCAM: sigma={sum(sigmas)/len(sigmas):.2f} "
        f"(min={min(sigmas):.2f} max={max(sigmas):.2f}) | "
        f"bias_scale={sum(bsc)/len(bsc):.3f} "
        f"(min={min(bsc):.3f} max={max(bsc):.3f}) | "
        f"disp mean={sum(disp_mean)/len(disp_mean):.1f}px "
        f"max={max(disp_max):.1f}px"
    )
    logger.info(msg)
    if writer is not None:
        for d in diags:
            blk = d["block"]
            for k in ("sigma", "bias_scale", "disp_mean", "disp_max"):
                if k in d:
                    writer.add_scalar(f"diag/scam/block{blk:02d}/{k}", d[k], iteration)
        writer.add_scalar("diag/scam_summary/sigma_mean", sum(sigmas)/len(sigmas), iteration)
        writer.add_scalar("diag/scam_summary/bias_scale_mean", sum(bsc)/len(bsc), iteration)
        writer.add_scalar("diag/scam_summary/disp_mean", sum(disp_mean)/len(disp_mean), iteration)


# ---------------------------------------------------------------------------
# Validation (rank 0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, logger) -> tuple[float, float]:
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    skipped = 0

    for batch in val_loader:
        try:
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
        except torch.cuda.OutOfMemoryError as exc:
            skipped += 1
            torch.cuda.empty_cache()
            logger.warning(f"Validation: skipped one sample (OOM): {exc}")
        except Exception as exc:
            skipped += 1
            logger.warning(f"Validation: skipped one sample (error): {exc}")

    if skipped:
        logger.warning(f"Validation: skipped {skipped} samples")
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

    _maybe_force_inprocess_loading(args)
    set_seed(args.seed, rank)

    logger = None
    writer = None
    if is_main():
        logger = setup_logger(args.output_dir, name="stereo_sr_v7_ddp")
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
    model = build_model_v7(args).to(device)
    if is_main():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {params / 1e6:.3f}M")
        logger.info(
            f"V7 config: disparity_prior={not args.no_disparity_prior} "
            f"multiscale={not args.no_multiscale} sigma_init={args.sigma_init} "
            f"betas=({args.adam_beta1}, {args.adam_beta2}) ema_decay={args.ema_decay} "
            f"patch={args.patch_h}x{args.patch_w} rgb_shuffle={args.rgb_shuffle}"
        )

    if args.pretrain:
        if is_main():
            logger.info(f"Loading pretrain from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # EMA copy — held on every rank for resume robustness; only rank 0
    # actually uses it for validation. The EMA update is deterministic
    # given synchronised DDP weights, so all ranks will agree.
    ema = None
    if args.ema_decay > 0:
        raw_for_ema = model  # unwrapped yet
        ema = ModelEMA(raw_for_ema, decay=args.ema_decay, warmup_steps=args.ema_warmup, device=device)
        if is_main():
            logger.info(f"EMA enabled: decay={args.ema_decay} warmup={args.ema_warmup}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )
    raw_model = model.module if hasattr(model, "module") else model

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = build_loss_v7(args).to(device)

    # ── Optimizer / Scheduler ────────────────────────────────────────
    effective_bs = args.batch_size * world_size
    lr_scale = effective_bs / 8.0
    scaled_lr = args.lr * lr_scale
    if is_main():
        logger.info(f"LR scaling: base={args.lr:.2e} x {lr_scale:.1f} = {scaled_lr:.2e}")

    optimizer = build_optimizer(
        model, scaled_lr, args.weight_decay, betas=(args.adam_beta1, args.adam_beta2)
    )
    scheduler = build_scheduler(optimizer, args)

    use_bf16 = bool(getattr(args, "bf16", False))
    use_fp16 = bool(args.fp16) and not use_bf16
    amp_enabled = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    if is_main():
        logger.info(
            f"AMP dtype: {'bf16' if use_bf16 else ('fp16' if use_fp16 else 'fp32')} "
            f"(scaler {'enabled' if use_fp16 else 'bypassed'})"
        )

    start_iter = 0
    best_psnr = 0.0
    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, raw_model, optimizer, scheduler, str(device),
            logger if is_main() else None,
        )
        if ema is not None:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
                if is_main():
                    logger.info(f"Loaded EMA state ({ema.num_updates} updates)")

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
    consecutive_loader_failures = 0
    for iteration in range(start_iter, args.total_iters):
        batch = None
        for _ in range(5):
            try:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                consecutive_loader_failures = 0
                break
            except Exception as exc:
                consecutive_loader_failures += 1
                if is_main():
                    logger.warning(
                        f"Iter {iteration}: dataloader failure "
                        f"({type(exc).__name__}: {exc}); retrying"
                    )
                if consecutive_loader_failures > 50:
                    if is_main():
                        logger.error("Too many consecutive dataloader failures, aborting")
                    raise
        if batch is None:
            continue

        lr_l = batch["lr_left"].to(device, non_blocking=True)
        lr_r = batch["lr_right"].to(device, non_blocking=True)
        hr_l = batch["hr_left"].to(device, non_blocking=True)
        hr_r = batch["hr_right"].to(device, non_blocking=True)

        # Tensor-side augmentation (after dataloader's spatial crop/flip).
        if args.augment and args.rgb_shuffle:
            lr_l, lr_r, hr_l, hr_r = rgb_shuffle_augment(lr_l, lr_r, hr_l, hr_r)
        if args.augment and args.vertical_flip:
            lr_l, lr_r, hr_l, hr_r = vertical_flip_augment(lr_l, lr_r, hr_l, hr_r)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
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

        # NaN/Inf guard.
        finite_flag = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
        if world_size > 1:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        if finite_flag.item() < 1.0:
            if is_main() and iteration % args.log_interval == 0:
                logger.warning(f"Iter {iteration}: non-finite loss detected, skipping step")
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        if use_fp16:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()

        # EMA update — AFTER optimizer.step().
        if ema is not None:
            ema.update(raw_model)

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
                if ema is not None:
                    writer.add_scalar("train/ema_decay_eff", ema._effective_decay(), iteration)

            diags = gather_scam_diagnostics(model)
            log_scam_diagnostics(diags, iteration, logger, writer)

            psnr_meter.reset()
            ssim_meter.reset()

        # ── Validation (rank 0 only, barrier on ALL ranks) ───────────
        if iteration > 0 and iteration % args.val_interval == 0:
            if is_main() and val_loader is not None:
                # Validate on the EMA copy if available — that's the
                # publication-quality number. Falls back to live model
                # if EMA is disabled.
                val_target = ema.module if ema is not None else raw_model
                psnr_val, ssim_val = validate(val_target, val_loader, device, logger)
                if writer is not None:
                    writer.add_scalar("val/psnr", psnr_val, iteration)
                    writer.add_scalar("val/ssim", ssim_val, iteration)
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    save_checkpoint_with_ema(
                        os.path.join(args.output_dir, "checkpoints", "best.pth"),
                        raw_model, optimizer, scheduler, iteration, best_psnr, logger, ema,
                    )
            if world_size > 1:
                dist.barrier()

        # ── Periodic checkpoint ──────────────────────────────────────
        if iteration > 0 and iteration % args.save_interval == 0:
            if is_main():
                save_checkpoint_with_ema(
                    os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                    raw_model, optimizer, scheduler, iteration, best_psnr, logger, ema,
                )
                prune_checkpoints(
                    os.path.join(args.output_dir, "checkpoints"),
                    keep_last=args.keep_ckpts,
                    logger=logger,
                )
            if world_size > 1:
                dist.barrier()

        if iteration > 0 and iteration % args.save_interval == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Final ────────────────────────────────────────────────────────
    if is_main():
        save_checkpoint_with_ema(
            os.path.join(args.output_dir, "checkpoints", "final.pth"),
            raw_model, optimizer, scheduler, args.total_iters, best_psnr, logger, ema,
        )
        logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(get_config_v7())

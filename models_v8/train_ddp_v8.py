"""Distributed training for StereoSR V8.

What's in this file
-------------------
* DDP setup with NCCL, NaN-finite guard, /dev/shm dataloader fallback,
  validation barrier — all carried over from V6/V7 (well-tested).
* EMA weight averaging with warm-up — validation runs on EMA copy.
* AdamW(0.9, 0.9) optimizer with linear LR scaling × cosine decay.
* NTIRE-2024 augmentation stack — applied on batched tensors after the
  dataloader (avoids per-worker reimplementation).
* CT-CVA + SCAM diagnostics, logged every ``--log_interval``, with a
  silent-collapse warning if CT-CVA γ stays below 1e-3 past iter 5000.
* Atomic checkpoint save including EMA state.

Launch
------
    torchrun --nproc_per_node=2 -m models_v8.train_ddp_v8 \\
        --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \\
        --total_iters 200000 \\
        --exp_name stereo_sr_v8_small_x4
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
            f"(need ~{needed_mb:.0f} MB). Forcing num_workers=0.",
            flush=True,
        )
        args.num_workers = 0


try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


from data.dataset import build_dataset
from models_v8 import build_loss_v8, build_model_v8, get_config_v8, ModelEMA
from models_v8.augmentations import apply_ntire_stack
from models_v8.channel_cross_view import ChannelTransposedCrossView
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
    """Atomic checkpoint save including optional EMA state."""
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
# CT-CVA diagnostics — silent-collapse health check
# ---------------------------------------------------------------------------

def gather_ct_cva_diagnostics(model: nn.Module) -> list[dict]:
    """Walk the trunk; for every CT-CVA module, read its diagnostic captures."""
    raw_model = model.module if hasattr(model, "module") else model
    out: list[dict] = []
    block_idx = 0
    for module in raw_model.modules():
        if not isinstance(module, ChannelTransposedCrossView):
            continue
        d: dict = {"block": block_idx}
        if module._diag_gamma_l_abs is not None:
            d["gamma_l_abs"] = module._diag_gamma_l_abs.item()
            d["gamma_r_abs"] = module._diag_gamma_r_abs.item()
            d["temperature"] = module._diag_temperature.item()
            d["attn_entropy"] = module._diag_attn_entropy.item()
            d["attn_max"] = module._diag_attn_max.item()
        out.append(d)
        block_idx += 1
    return out


def log_ct_cva_diagnostics(
    diags: list[dict],
    iteration: int,
    logger,
    writer,
    silent_collapse_threshold: float = 1e-3,
    silent_collapse_after: int = 5000,
):
    """One-line summary + per-block TensorBoard scalars + silent-collapse warning."""
    if not diags:
        return
    rows_with_data = [d for d in diags if "gamma_l_abs" in d]
    if not rows_with_data:
        return

    gammas = [0.5 * (d["gamma_l_abs"] + d["gamma_r_abs"]) for d in rows_with_data]
    temps = [d["temperature"] for d in rows_with_data]
    ents = [d["attn_entropy"] for d in rows_with_data]
    maxs = [d["attn_max"] for d in rows_with_data]

    msg = (
        f"  diag CT-CVA: γ={sum(gammas)/len(gammas):.3e} "
        f"(min={min(gammas):.3e} max={max(gammas):.3e}) | "
        f"temp={sum(temps)/len(temps):.3f} | "
        f"attn_entropy={sum(ents)/len(ents):.3f} (1.0 = uniform) | "
        f"attn_max={sum(maxs)/len(maxs):.3f}"
    )
    logger.info(msg)

    if writer is not None:
        for d in rows_with_data:
            blk = d["block"]
            for k in ("gamma_l_abs", "gamma_r_abs", "temperature", "attn_entropy", "attn_max"):
                writer.add_scalar(f"diag/ct_cva/block{blk:02d}/{k}", d[k], iteration)
        writer.add_scalar("diag/ct_cva_summary/gamma_mean", sum(gammas) / len(gammas), iteration)
        writer.add_scalar("diag/ct_cva_summary/gamma_max", max(gammas), iteration)
        writer.add_scalar("diag/ct_cva_summary/temperature_mean", sum(temps) / len(temps), iteration)
        writer.add_scalar("diag/ct_cva_summary/attn_entropy_mean", sum(ents) / len(ents), iteration)

    # Silent-collapse warning: if past warm-up, γ has not lifted, CT-CVA is inert.
    if iteration >= silent_collapse_after and max(gammas) < silent_collapse_threshold:
        logger.warning(
            f"  [CT-CVA SILENT-COLLAPSE WARNING] iter {iteration}: max block-mean |γ| "
            f"= {max(gammas):.2e} < {silent_collapse_threshold:.0e}. "
            f"The channel cross-view branch is functionally inert. Consider "
            f"killing this run — V8 has degenerated to V6 A1 + engbundle."
        )


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
        logger = setup_logger(args.output_dir, name="stereo_sr_v8_ddp")
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
    model = build_model_v8(args).to(device)
    if is_main():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {params / 1e6:.3f}M")
        logger.info(
            f"V8 config: scam={not args.no_scam} ct_cva={not args.no_ct_cva} "
            f"ct_cva_heads={args.ct_cva_heads} temperature_init={args.ct_cva_temperature_init} "
            f"betas=({args.adam_beta1}, {args.adam_beta2}) ema_decay={args.ema_decay} "
            f"patch={args.patch_h}x{args.patch_w}"
        )
        logger.info(
            f"NTIRE aug: rgb_shuffle={args.aug_rgb_shuffle} hflip={args.aug_hflip} "
            f"vflip={args.aug_vflip} hshift={args.aug_hshift} "
            f"hshift_max={args.aug_hshift_max}"
        )

    if args.pretrain:
        if is_main():
            logger.info(f"Loading pretrain from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # EMA copy: held on every rank (deterministic given synchronised DDP weights).
    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay, warmup_steps=args.ema_warmup, device=device)
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

    criterion = build_loss_v8(args).to(device)

    # LR scaling: base LR is calibrated for effective batch 8.
    effective_bs = args.batch_size * world_size
    lr_scale = effective_bs / 8.0
    scaled_lr = args.lr * lr_scale
    max_lr = getattr(args, "max_lr", 0.0)
    capped = max_lr and max_lr > 0 and scaled_lr > max_lr
    if capped:
        scaled_lr = max_lr
    if is_main():
        logger.info(
            f"LR scaling: base={args.lr:.2e} × {lr_scale:.1f} = "
            f"{args.lr * lr_scale:.2e}"
            + (f" -> capped to max_lr={max_lr:.2e}" if capped else "")
        )

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

    train_loader, train_sampler = build_train_loader(args, rank, world_size)
    val_loader = None
    if is_main():
        val_dataset = build_dataset(args, split="val")
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=min(2, args.num_workers),
            )

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    epoch = 0
    train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)

    model.train()
    consecutive_loader_failures = 0
    for iteration in range(start_iter, args.total_iters):
        # Robust batch fetch.
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

        # NTIRE-2024 augmentation stack (after dataloader's spatial crop).
        if args.augment:
            lr_l, lr_r, hr_l, hr_r = apply_ntire_stack(
                lr_l, lr_r, hr_l, hr_r,
                scale=args.scale,
                rgb_shuffle_on=args.aug_rgb_shuffle,
                hflip_on=args.aug_hflip,
                vflip_on=args.aug_vflip,
                hshift_on=args.aug_hshift,
                max_lr_shift=args.aug_hshift_max,
            )

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

        # NaN/Inf guard with all-reduce so DDP ranks agree on skip.
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

            diags = gather_ct_cva_diagnostics(model)
            log_ct_cva_diagnostics(diags, iteration, logger, writer)

            psnr_meter.reset()
            ssim_meter.reset()

        # ── Validation (rank 0 only, barrier on ALL ranks) ───────────
        if iteration > 0 and iteration % args.val_interval == 0:
            if is_main() and val_loader is not None:
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
    train(get_config_v8())

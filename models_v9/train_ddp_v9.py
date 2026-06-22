"""Distributed training for StereoSR V9 (DECA-SR).

Carries over the well-tested V6/V8 DDP pipeline (NCCL, NaN-finite guard,
/dev/shm dataloader fallback, EMA, AdamW(0.9,0.9), linear-scaled cosine LR with a
``--max_lr`` ceiling) and adds the near-free lever stack the V8 post-mortem
prescribed: full NTIRE aug (+ optional Mixup), per-view focal-frequency loss,
a terminal pure-MSE phase, and top-K best-checkpoint weight averaging (SWA).
DECVF diagnostics are logged every ``--log_interval`` with a silent-collapse
warning if the fusion gate stays inert.

Launch
------
    torchrun --nproc_per_node=4 -m models_v9.train_ddp_v9 \\
        --model_size small --scale 4 --batch_size 32 --bf16 --lr 5e-5 \\
        --total_iters 400000 --exp_name stereo_sr_v9_small_x4
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
        2 * 2 * args.batch_size * args.patch_h * args.patch_w * (1 + args.scale ** 2) * 4
    ) / (1024 * 1024)
    needed_mb = per_worker_mb * args.num_workers * 2
    if free_mb < max(needed_mb, 512):
        print(f"[shm-guard] /dev/shm {free_mb:.0f} MB free (need ~{needed_mb:.0f}); forcing num_workers=0.", flush=True)
        args.num_workers = 0


try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


from data.dataset import build_dataset
from models_v9 import build_loss_v9, build_model_v9, get_config_v9, ModelEMA
from models_v9.augmentations import apply_ntire_stack
from models_v9.deformable_fusion import DeformableEpipolarFusion
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


# Exactly the StereoSRV9 constructor kwargs (so StereoSRV9(**model_args) rebuilds it).
ARCH_KEYS = (
    "scale", "in_channels", "channels", "num_blocks", "dw_expansion", "ffn_expansion",
    "large_kernel", "decvf_every", "cost_volume_d", "decvf_k", "decvf_y_range",
)


def model_args_from(args) -> dict:
    """Resolved architecture dict persisted with every checkpoint so eval can
    rebuild the EXACT network (incl. non-shape flags like cost_volume_d /
    decvf_y_range that strict=False cannot catch)."""
    d = {k: getattr(args, k, None) for k in ARCH_KEYS}
    d["use_scam"] = not getattr(args, "no_scam", False)
    d["use_decvf"] = not getattr(args, "no_decvf", False)
    return d


def save_checkpoint_with_ema(path, model, optimizer, scheduler, iteration, best_psnr, logger, ema, model_args=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "best_psnr": best_psnr,
    }
    if model_args is not None:
        state["model_args"] = model_args
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


def build_optimizer(model, lr, weight_decay, betas):
    return optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def build_scheduler(optimizer, args):
    def lr_lambda(step: int) -> float:
        if step < args.warmup_iters:
            return step / max(1, args.warmup_iters)
        progress = (step - args.warmup_iters) / max(1, args.total_iters - args.warmup_iters)
        return max(0.5 * (1.0 + np.cos(np.pi * progress)), 1e-6)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# DECVF diagnostics
# ---------------------------------------------------------------------------
def gather_decvf_diagnostics(model: nn.Module) -> list[dict]:
    raw = model.module if hasattr(model, "module") else model
    out = []
    blk = 0
    for module in raw.modules():
        if not isinstance(module, DeformableEpipolarFusion):
            continue
        d = {"block": blk}
        if module._diag_gamma_abs is not None:
            d["gamma_abs"] = module._diag_gamma_abs.item()
            d["conf"] = module._diag_conf.item()
            d["disp"] = module._diag_disp.item()
            d["dy_abs"] = module._diag_dy_abs.item()
        out.append(d)
        blk += 1
    return out


def log_decvf_diagnostics(diags, logger, writer, iteration, silent_after=5000, silent_thresh=1e-3):
    rows = [d for d in diags if "gamma_abs" in d]
    if not rows:
        return
    gammas = [d["gamma_abs"] for d in rows]
    confs = [d["conf"] for d in rows]
    disps = [d["disp"] for d in rows]
    dys = [d["dy_abs"] for d in rows]
    logger.info(
        f"  diag DECVF: |gamma|={sum(gammas)/len(gammas):.3e} "
        f"(min={min(gammas):.3e} max={max(gammas):.3e}) | "
        f"conf={sum(confs)/len(confs):.3f} | disp={sum(disps)/len(disps):.2f}px | "
        f"|dy|={sum(dys)/len(dys):.3f}px"
    )
    if writer is not None:
        writer.add_scalar("diag/decvf/gamma_mean", sum(gammas) / len(gammas), iteration)
        writer.add_scalar("diag/decvf/gamma_max", max(gammas), iteration)
        writer.add_scalar("diag/decvf/conf_mean", sum(confs) / len(confs), iteration)
        writer.add_scalar("diag/decvf/disp_mean", sum(disps) / len(disps), iteration)
    if iteration >= silent_after and max(gammas) < silent_thresh:
        logger.warning(
            f"  [DECVF SILENT-COLLAPSE WARNING] iter {iteration}: max |gamma| "
            f"= {max(gammas):.2e} < {silent_thresh:.0e}. The deformable cross-view "
            f"branch is inert; V9 has degenerated to RepLK-NAF + SCAM."
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(model: nn.Module, val_loader: DataLoader, device, logger) -> tuple[float, float]:
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


def _save_swa(swa_pool, output_dir, logger, model_args=None):
    """Average the top-K pooled (CPU) state dicts and save as swa.pth."""
    if not swa_pool:
        return
    keys = list(swa_pool[0][1].keys())
    avg = {}
    for k in keys:
        stack = torch.stack([sd[k].float() for _, sd in swa_pool], dim=0)
        avg[k] = stack.mean(dim=0)
    path = os.path.join(output_dir, "checkpoints", "swa.pth")
    state = {"model": avg, "swa_psnrs": [p for p, _ in swa_pool]}
    if model_args is not None:
        state["model_args"] = model_args
    torch.save(state, path)
    if logger is not None:
        logger.info(f"SWA: averaged top-{len(swa_pool)} checkpoints "
                    f"(val {min(p for p,_ in swa_pool):.3f}-{max(p for p,_ in swa_pool):.3f}) -> {path}")


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
        logger = setup_logger(args.output_dir, name="stereo_sr_v9_ddp")
        logger.info(f"Config: {args}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
        except ImportError:
            logger.warning("TensorBoard not available")

    model = build_model_v9(args).to(device)
    model_args = model_args_from(args)
    if is_main():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters (train, multi-branch): {params / 1e6:.3f}M")
        logger.info(
            f"V9 (DECA-SR): scam={not args.no_scam} decvf={not args.no_decvf} "
            f"decvf_every={args.decvf_every} D={args.cost_volume_d} K={args.decvf_k} "
            f"y_range={args.decvf_y_range} LK={args.large_kernel} "
            f"patch={args.patch_h}x{args.patch_w} betas=({args.adam_beta1},{args.adam_beta2})"
        )

    if args.pretrain:
        if is_main():
            logger.info(f"Loading pretrain from {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay, warmup_steps=args.ema_warmup, device=device)
        if is_main():
            logger.info(f"EMA enabled: decay={args.ema_decay} warmup={args.ema_warmup}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    raw_model = model.module if hasattr(model, "module") else model

    criterion = build_loss_v9(args).to(device)

    # LR scaling (base calibrated for effective batch 8) with a hard ceiling.
    effective_bs = args.batch_size * world_size
    lr_scale = effective_bs / 8.0
    scaled_lr = args.lr * lr_scale
    max_lr = getattr(args, "max_lr", 0.0)
    capped = max_lr and max_lr > 0 and scaled_lr > max_lr
    if capped:
        scaled_lr = max_lr
    if is_main():
        logger.info(
            f"LR scaling: base={args.lr:.2e} x {lr_scale:.1f} = {args.lr * lr_scale:.2e}"
            + (f" -> capped to max_lr={max_lr:.2e}" if capped else "")
        )

    optimizer = build_optimizer(model, scaled_lr, args.weight_decay, betas=(args.adam_beta1, args.adam_beta2))
    scheduler = build_scheduler(optimizer, args)

    use_bf16 = bool(getattr(args, "bf16", False))
    use_fp16 = bool(args.fp16) and not use_bf16
    amp_enabled = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    if is_main():
        logger.info(f"AMP dtype: {'bf16' if use_bf16 else ('fp16' if use_fp16 else 'fp32')} "
                    f"(scaler {'enabled' if use_fp16 else 'bypassed'})")

    start_iter = 0
    best_psnr = 0.0
    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume, raw_model, optimizer, scheduler, str(device), logger if is_main() else None,
        )
        if ema is not None:
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            if "ema" in ck:
                ema.load_state_dict(ck["ema"])

    # ── Data ─────────────────────────────────────────────────────────
    train_dataset = build_dataset(args, split="train")
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = None
    if is_main():
        val_dataset = build_dataset(args, split="val")
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=min(2, args.num_workers))

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)

    terminal_start = args.total_iters - getattr(args, "terminal_mse_iters", 0)
    in_terminal = False
    swa_pool: list = []  # [(psnr, cpu_state_dict)]

    model.train()
    consecutive_loader_failures = 0
    for iteration in range(start_iter, args.total_iters):
        # Switch to terminal MSE phase once.
        if not in_terminal and getattr(args, "terminal_mse_iters", 0) > 0 and iteration >= terminal_start:
            criterion.set_terminal(True)
            in_terminal = True
            if is_main():
                logger.info(f"Iter {iteration}: entering terminal MSE phase (aux losses dropped).")

        batch = None
        for _ in range(5):
            try:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                consecutive_loader_failures = 0
                break
            except Exception as exc:
                consecutive_loader_failures += 1
                if is_main():
                    logger.warning(f"Iter {iteration}: dataloader failure ({type(exc).__name__}: {exc}); retrying")
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

        if args.augment:
            lr_l, lr_r, hr_l, hr_r = apply_ntire_stack(
                lr_l, lr_r, hr_l, hr_r,
                scale=args.scale,
                rgb_shuffle_on=args.aug_rgb_shuffle,
                hflip_on=args.aug_hflip,
                vflip_on=args.aug_vflip,
                hshift_on=args.aug_hshift,
                max_lr_shift=args.aug_hshift_max,
                mixup_on=getattr(args, "aug_mixup", False),
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            output = model(lr_l, lr_r)
            loss, loss_dict = criterion(output.sr_left, output.sr_right, hr_l, hr_r, aux=output.aux)
            with torch.no_grad():
                sr_l = output.sr_left.clamp(0, 1)
                sr_r = output.sr_right.clamp(0, 1)
                psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
                ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
                psnr_meter.update(psnr, lr_l.size(0))
                ssim_meter.update(ssim, lr_l.size(0))

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

        if is_main() and iteration % args.log_interval == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            msg = (
                f"Iter {iteration}/{args.total_iters} | lr={lr_current:.2e} | "
                f"PSNR={psnr_meter.avg:.2f} | SSIM={ssim_meter.avg:.4f} | "
            )
            msg += " | ".join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
            )
            logger.info(msg)
            if not args.no_decvf:
                log_decvf_diagnostics(gather_decvf_diagnostics(model), logger, writer, iteration)
            if writer is not None:
                writer.add_scalar("train/psnr", psnr_meter.avg, iteration)
                writer.add_scalar("train/ssim", ssim_meter.avg, iteration)
                writer.add_scalar("train/lr", lr_current, iteration)
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"train/{k}", v.item(), iteration)
            psnr_meter.reset()
            ssim_meter.reset()

        # ── Validation ───────────────────────────────────────────────
        if iteration > 0 and iteration % args.val_interval == 0:
            if is_main():
                val_target = ema.module if ema is not None else raw_model
                psnr_val, ssim_val = validate(val_target, val_loader, device, logger)
                if writer is not None:
                    writer.add_scalar("val/psnr", psnr_val, iteration)
                    writer.add_scalar("val/ssim", ssim_val, iteration)
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    save_checkpoint_with_ema(
                        os.path.join(args.output_dir, "checkpoints", "best.pth"),
                        raw_model, optimizer, scheduler, iteration, best_psnr, logger, ema, model_args,
                    )
                if getattr(args, "swa_topk", 0) > 0:
                    src = ema.module if ema is not None else raw_model
                    sd = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
                    swa_pool.append((psnr_val, sd))
                    swa_pool.sort(key=lambda x: x[0], reverse=True)
                    del swa_pool[args.swa_topk:]
            if world_size > 1:
                dist.barrier()

        # ── Periodic checkpoint ──────────────────────────────────────
        if iteration > 0 and iteration % args.save_interval == 0:
            if is_main():
                save_checkpoint_with_ema(
                    os.path.join(args.output_dir, "checkpoints", f"iter_{iteration}.pth"),
                    raw_model, optimizer, scheduler, iteration, best_psnr, logger, ema, model_args,
                )
                prune_checkpoints(os.path.join(args.output_dir, "checkpoints"), keep_last=args.keep_ckpts, logger=logger)
            if world_size > 1:
                dist.barrier()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if is_main():
        save_checkpoint_with_ema(
            os.path.join(args.output_dir, "checkpoints", "final.pth"),
            raw_model, optimizer, scheduler, args.total_iters, best_psnr, logger, ema, model_args,
        )
        if getattr(args, "swa_topk", 0) > 0:
            _save_swa(swa_pool, args.output_dir, logger, model_args)
        logger.info(f"Training complete. Best PSNR: {best_psnr:.4f} dB")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(get_config_v9())

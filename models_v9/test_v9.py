"""Evaluation for StereoSR V9 (DECA-SR).

* Loads EMA weights if available (``--use_ema_eval``, default on).
* Optional RepLK reparameterization to a single DW conv (``--reparam_eval``).
* Optional TLSC (``--tlsc``, base size synced to the training patch).
* Optional 4-transform epipolar-preserving self-ensemble (``--ensemble``).
"""
from __future__ import annotations

import os
import shutil
import time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


def _shm_constrained() -> bool:
    try:
        free_mb = shutil.disk_usage("/dev/shm").free / (1024 * 1024)
    except (FileNotFoundError, OSError):
        return False
    return free_mb < 512


from data.dataset import build_dataset
from models_v9 import StereoSRV9, build_model_v9, get_config_v9
from utils_fixed import AverageMeter, compute_psnr, compute_ssim, save_image, setup_logger


def _forward_with_transform(model, lr_l, lr_r, transform_id: int):
    """Apply one of 4 epipolar-preserving transforms, forward, invert it."""
    h_flip = transform_id in (1, 3)
    v_flip = transform_id in (2, 3)
    swap_views = h_flip
    l, r = lr_l, lr_r
    if h_flip:
        l = torch.flip(l, dims=[-1]); r = torch.flip(r, dims=[-1])
    if v_flip:
        l = torch.flip(l, dims=[-2]); r = torch.flip(r, dims=[-2])
    if swap_views:
        l, r = r, l
    out = model(l, r)
    sr_l, sr_r = out.sr_left, out.sr_right
    if swap_views:
        sr_l, sr_r = sr_r, sr_l
    if v_flip:
        sr_l = torch.flip(sr_l, dims=[-2]); sr_r = torch.flip(sr_r, dims=[-2])
    if h_flip:
        sr_l = torch.flip(sr_l, dims=[-1]); sr_r = torch.flip(sr_r, dims=[-1])
    return sr_l, sr_r


@torch.no_grad()
def self_ensemble(model, lr_l, lr_r):
    sr_l_acc, sr_r_acc = None, None
    for tid in range(4):
        sl, sr = _forward_with_transform(model, lr_l, lr_r, tid)
        sr_l_acc = sl if sr_l_acc is None else sr_l_acc + sl
        sr_r_acc = sr if sr_r_acc is None else sr_r_acc + sr
    return sr_l_acc / 4.0, sr_r_acc / 4.0


def load_weights_from_ckpt(model, ckpt, ckpt_path, use_ema, allow_partial, logger):
    if use_ema and "ema" in ckpt:
        sd = ckpt["ema"]["module"]
        logger.info(f"Loading EMA weights from {ckpt_path}")
    else:
        sd = ckpt.get("model", ckpt)
        if use_ema:
            logger.warning("Requested --use_ema_eval but checkpoint has no 'ema' key — using live weights.")
        else:
            logger.info(f"Loading live model weights from {ckpt_path}")
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        msg = (
            f"State-dict mismatch (architecture differs from checkpoint). "
            f"Missing={result.missing_keys} Unexpected={result.unexpected_keys}."
        )
        if allow_partial:
            logger.warning(msg + " (continuing because --allow_partial_load is set)")
        else:
            raise RuntimeError(msg + " Pass --allow_partial_load to override.")
    else:
        logger.info("State dict loaded cleanly (no missing/unexpected keys).")
    return ckpt.get("iteration", 0), ckpt.get("best_psnr", 0.0)


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="stereo_sr_v9_test")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False) if args.ckpt else None
    if ckpt is not None and "model_args" in ckpt:
        model = StereoSRV9(**ckpt["model_args"]).to(device)
        logger.info(f"Built model from checkpoint arch: {ckpt['model_args']}")
    else:
        model = build_model_v9(args).to(device)
        if ckpt is not None:
            logger.warning("Checkpoint has no 'model_args' — built from CLI/preset; "
                           "VERIFY eval flags match the training config.")
    if ckpt is not None:
        load_weights_from_ckpt(
            model, ckpt, args.ckpt, args.use_ema_eval,
            getattr(args, "allow_partial_load", False), logger,
        )
    else:
        logger.warning("No checkpoint specified; evaluating random weights.")

    if getattr(args, "reparam_eval", False):
        n = model.reparametrize()
        logger.info(f"Reparameterized {n} RepLK depthwise convs to single LK conv.")

    if args.tlsc:
        base_h = getattr(args, "tlsc_base_h", None) or args.patch_h
        base_w = getattr(args, "tlsc_base_w", None) or args.patch_w
        n_replaced = model.enable_tlsc((base_h, base_w))
        logger.info(f"TLSC enabled: replaced {n_replaced} global pools with LocalAvgPool2d(({base_h}, {base_w}))")

    model.eval()
    test_dataset = build_dataset(args, split="test")
    test_workers = min(2, args.num_workers)
    if test_workers > 0 and _shm_constrained():
        logger.warning("[shm-guard] /dev/shm < 512 MB free; forcing num_workers=0 for eval.")
        test_workers = 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=test_workers)
    logger.info(f"Test set: {len(test_dataset)} samples, scale={args.scale}, ensemble={args.ensemble}")

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    time_meter = AverageMeter()

    for index, batch in enumerate(test_loader):
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)
        filename = batch["filename"][0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        if args.ensemble:
            sr_l, sr_r = self_ensemble(model, lr_l, lr_r)
        else:
            output = model(lr_l, lr_r)
            sr_l, sr_r = output.sr_left, output.sr_right
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)
        min_h = min(sr_l.shape[2], hr_l.shape[2])
        min_w = min(sr_l.shape[3], hr_l.shape[3])
        sr_l = sr_l[:, :, :min_h, :min_w]
        sr_r = sr_r[:, :, :min_h, :min_w]
        hr_l = hr_l[:, :, :min_h, :min_w]
        hr_r = hr_r[:, :, :min_h, :min_w]

        psnr = 0.5 * (compute_psnr(sr_l, hr_l) + compute_psnr(sr_r, hr_r))
        ssim = 0.5 * (compute_ssim(sr_l, hr_l) + compute_ssim(sr_r, hr_r))
        psnr_meter.update(psnr)
        ssim_meter.update(ssim)
        time_meter.update(elapsed)

        logger.info(f"[{index + 1}/{len(test_dataset)}] {filename} | "
                    f"PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f} | Time: {elapsed:.3f}s")

        if args.save_images:
            stem = os.path.splitext(filename)[0]
            img_dir = os.path.join(args.output_dir, "images")
            save_image(sr_l, os.path.join(img_dir, f"{stem}_sr_left.png"))
            save_image(sr_r, os.path.join(img_dir, f"{stem}_sr_right.png"))

    logger.info("=" * 60)
    logger.info(f"Average - PSNR: {psnr_meter.avg:.4f} dB | SSIM: {ssim_meter.avg:.4f} | "
                f"Time: {time_meter.avg:.3f}s/image")
    logger.info("=" * 60)
    return psnr_meter.avg, ssim_meter.avg


if __name__ == "__main__":
    test(get_config_v9())

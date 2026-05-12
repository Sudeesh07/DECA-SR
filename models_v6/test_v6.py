"""Evaluation for StereoSR V6 (SpecFuse)."""
from __future__ import annotations

import os
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Avoid Docker /dev/shm bus-error failures in DataLoader workers.
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


def _shm_constrained() -> bool:
    """True when /dev/shm is too small for DataLoader IPC of HR test samples."""
    try:
        free_mb = shutil.disk_usage("/dev/shm").free / (1024 * 1024)
    except (FileNotFoundError, OSError):
        return False
    # Test images are full-resolution; one HR sample can be 10-30 MB. With
    # any workers, a few prefetched batches easily exceed a 64-128 MB shm cap.
    return free_mb < 512

from data.dataset import build_dataset
from models_v6 import build_model_v6, get_config_v6
from utils_fixed import AverageMeter, TLSC, compute_psnr, compute_ssim, save_image, setup_logger


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="stereo_sr_v6_test")

    model = build_model_v6(args).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        result = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {args.ckpt}")
        logger.info(f"Missing keys: {result.missing_keys}")
        logger.info(f"Unexpected keys: {result.unexpected_keys}")
    else:
        logger.warning("No checkpoint specified; evaluating random weights.")

    model.eval()
    test_dataset = build_dataset(args, split="test")
    # Eval streams full-resolution HR samples. Auto-fallback to 0 workers if
    # /dev/shm is too small to host inter-process tensor IPC (Docker 64 MB).
    test_workers = min(2, args.num_workers)
    if test_workers > 0 and _shm_constrained():
        logger.warning(
            "[shm-guard] /dev/shm < 512 MB free; forcing num_workers=0 for eval."
        )
        test_workers = 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=test_workers)
    logger.info(f"Test set: {len(test_dataset)} samples, scale={args.scale}")

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    time_meter = AverageMeter()
    tlsc = TLSC() if args.tlsc else None

    for index, batch in enumerate(test_loader):
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)
        filename = batch["filename"][0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        output = model(lr_l, lr_r)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        sr_l = output.sr_left.clamp(0, 1)
        sr_r = output.sr_right.clamp(0, 1)

        if tlsc is not None:
            lr_up_l = F.interpolate(lr_l, scale_factor=args.scale, mode="bicubic", align_corners=False)
            lr_up_r = F.interpolate(lr_r, scale_factor=args.scale, mode="bicubic", align_corners=False)
            sr_l = tlsc(sr_l, lr_up_l)
            sr_r = tlsc(sr_r, lr_up_r)

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

        logger.info(
            f"[{index + 1}/{len(test_dataset)}] {filename} | "
            f"PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f} | Time: {elapsed:.3f}s"
        )

        if args.save_images:
            stem = os.path.splitext(filename)[0]
            img_dir = os.path.join(args.output_dir, "images")
            save_image(sr_l, os.path.join(img_dir, f"{stem}_sr_left.png"))
            save_image(sr_r, os.path.join(img_dir, f"{stem}_sr_right.png"))

    logger.info("=" * 60)
    logger.info(
        f"Average - PSNR: {psnr_meter.avg:.4f} dB | "
        f"SSIM: {ssim_meter.avg:.4f} | "
        f"Time: {time_meter.avg:.3f}s/image"
    )
    logger.info("=" * 60)
    return psnr_meter.avg, ssim_meter.avg


if __name__ == "__main__":
    test(get_config_v6())

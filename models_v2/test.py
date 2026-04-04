"""
Testing / evaluation script for StereoSRNet V2.

Usage:
    python -m models_v2.test --ckpt experiments/stereo_sr_v2/checkpoints/best.pth --scale 4
    python -m models_v2.test --ckpt best.pth --scale 2 --save_images --tlsc
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from utils_fixed import (
    AverageMeter, compute_psnr, compute_ssim,
    setup_logger, save_image, TLSC,
)

from .config import get_config
from .network import build_model_v2


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="stereo_sr_v2_test")

    model = build_model_v2(args).to(device)

    if args.ckpt:
        # PyTorch 2.6+ defaults to weights_only=True, which breaks loading older checkpoints with pickled objects.
        # Setting weights_only=False restores the old behavior and fixes UnpicklingError for numpy scalars, etc.
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {args.ckpt}")
    else:
        logger.warning("No checkpoint specified — evaluating random weights")

    model.eval()

    test_dataset = build_dataset(args, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    logger.info(f"Test set: {len(test_dataset)} samples, scale={args.scale}")

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    time_meter = AverageMeter()

    tlsc = TLSC() if args.tlsc else None

    img_dir = os.path.join(args.output_dir, "images")

    for idx, batch in enumerate(test_loader):
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)
        filename = batch["filename"][0]

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()

        sr_l, sr_r, conf_l, conf_r = model(lr_l, lr_r)

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0

        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)

        # Optional test-time local statistics correction
        if tlsc is not None:
            lr_up_l = F.interpolate(lr_l, scale_factor=args.scale, mode="bilinear", align_corners=False)
            lr_up_r = F.interpolate(lr_r, scale_factor=args.scale, mode="bilinear", align_corners=False)
            sr_l = tlsc(sr_l, lr_up_l)
            sr_r = tlsc(sr_r, lr_up_r)

        # Crop to match HR size (in case of rounding)
        min_h = min(sr_l.shape[2], hr_l.shape[2])
        min_w = min(sr_l.shape[3], hr_l.shape[3])
        sr_l = sr_l[:, :, :min_h, :min_w]
        sr_r = sr_r[:, :, :min_h, :min_w]
        hr_l = hr_l[:, :, :min_h, :min_w]
        hr_r = hr_r[:, :, :min_h, :min_w]

        psnr_l = compute_psnr(sr_l, hr_l)
        psnr_r = compute_psnr(sr_r, hr_r)
        ssim_l = compute_ssim(sr_l, hr_l)
        ssim_r = compute_ssim(sr_r, hr_r)

        psnr_avg = 0.5 * (psnr_l + psnr_r)
        ssim_avg = 0.5 * (ssim_l + ssim_r)

        psnr_meter.update(psnr_avg)
        ssim_meter.update(ssim_avg)
        time_meter.update(elapsed)

        logger.info(
            f"[{idx + 1}/{len(test_dataset)}] {filename} | "
            f"PSNR: {psnr_avg:.2f} dB | SSIM: {ssim_avg:.4f} | Time: {elapsed:.3f}s"
        )

        if args.save_images:
            base = os.path.splitext(filename)[0]
            save_image(sr_l, os.path.join(img_dir, f"{base}_sr_left.png"))
            save_image(sr_r, os.path.join(img_dir, f"{base}_sr_right.png"))
            save_image(conf_l[:, :, :min_h, :min_w], os.path.join(img_dir, f"{base}_conf_left.png"))

    logger.info("=" * 60)
    logger.info(
        f"Average - PSNR: {psnr_meter.avg:.4f} dB | "
        f"SSIM: {ssim_meter.avg:.4f} | "
        f"Time: {time_meter.avg:.3f}s/image"
    )
    logger.info("=" * 60)

    return psnr_meter.avg, ssim_meter.avg


if __name__ == "__main__":
    args = get_config()
    test(args)

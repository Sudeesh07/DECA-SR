from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from models_v4 import build_model_v4, get_config_v4
from utils_fixed import AverageMeter, TLSC, compute_psnr, compute_ssim, save_image, setup_logger


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="stereo_sr_v4_test")

    model = build_model_v4(args).to(device)
    if args.ckpt:
        # PyTorch 2.6+ defaults to weights_only=True, which breaks loading older checkpoints with pickled objects.
        # Setting weights_only=False restores the old behavior and fixes UnpicklingError for numpy scalars, etc.
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        result = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {args.ckpt}")
        logger.info(f"Missing keys: {result.missing_keys}")
        logger.info(f"Unexpected keys: {result.unexpected_keys}")
        if hasattr(result, 'mismatched_keys'):
            logger.info(f"Mismatched keys: {result.mismatched_keys}")
    else:
        logger.warning("No checkpoint specified; evaluating random weights.")

    model.eval()
    test_dataset = build_dataset(args, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=min(2, args.num_workers))
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
    test(get_config_v4())

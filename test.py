import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import get_config
from models.network import build_model
from data.dataset import build_dataset
from utils import (
    compute_psnr, compute_ssim, AverageMeter,
    TLSC, tensor_to_image, save_image,
)

try:
    import lpips    
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

def pad_to_multiple(img, multiple=16):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, [0, pad_w, 0, pad_h], mode="reflect")
    return img, h, w


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = build_model(args).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.2f}M")
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("WARNING: No checkpoint specified, using random weights!")
    model.eval()
    test_dataset = build_dataset(args, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_dataset)}")
    tlsc = TLSC(kernel_size=31) if args.tlsc else None
    if tlsc:
        print("TLSC enabled (kernel=31)")
    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()
    save_dir = os.path.join(args.output_dir, "test_results")
    if args.save_images:
        os.makedirs(os.path.join(save_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "right"), exist_ok=True)
    psnr_left = AverageMeter()
    psnr_right = AverageMeter()
    ssim_left = AverageMeter()
    ssim_right = AverageMeter()
    lpips_left = AverageMeter()
    time_meter = AverageMeter()

    results = []

    for batch in tqdm(test_loader, desc="Testing"):
        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)
        filename = batch["filename"][0]

        lr_l_pad, orig_h, orig_w = pad_to_multiple(lr_l, 16)
        lr_r_pad, _, _ = pad_to_multiple(lr_r, 16)
        start_time = time.time()
        sr_l, sr_r, conf_l, conf_r = model(lr_l_pad, lr_r_pad)
        elapsed = time.time() - start_time


        h_sr = orig_h * args.scale
        w_sr = orig_w * args.scale
        sr_l = sr_l[:, :, :h_sr, :w_sr]
        sr_r = sr_r[:, :, :h_sr, :w_sr]
        if tlsc is not None:
            lr_l_up = F.interpolate(lr_l, scale_factor=args.scale,
                                     mode="bicubic", align_corners=False)
            lr_r_up = F.interpolate(lr_r, scale_factor=args.scale,
                                     mode="bicubic", align_corners=False)
            sr_l = tlsc(sr_l, lr_l_up)
            sr_r = tlsc(sr_r, lr_r_up)

        sr_l = sr_l.clamp(0, 1)
        sr_r = sr_r.clamp(0, 1)
        p_l = compute_psnr(sr_l, hr_l)
        s_l = compute_ssim(sr_l, hr_l)
        psnr_left.update(p_l)
        ssim_left.update(s_l)
        p_r = compute_psnr(sr_r, hr_r)
        s_r = compute_ssim(sr_r, hr_r)
        psnr_right.update(p_r)
        ssim_right.update(s_r)
        if lpips_fn is not None:
            lp = lpips_fn(sr_l * 2 - 1, hr_l * 2 - 1).item()
            lpips_left.update(lp)

        time_meter.update(elapsed)
        results.append({
            "file": filename,
            "psnr_l": p_l, "psnr_r": p_r,
            "ssim_l": s_l, "ssim_r": s_r,
            "time": elapsed,
        })
        if args.save_images:
            save_image(sr_l, os.path.join(save_dir, "left", filename))
            save_image(sr_r, os.path.join(save_dir, "right", filename))
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Left':<15} {'Right':<15} {'Average':<15}")
    print("-" * 60)
    print(f"{'PSNR (dB)':<20} {psnr_left.avg:<15.4f} {psnr_right.avg:<15.4f} "
          f"{(psnr_left.avg + psnr_right.avg) / 2:<15.4f}")
    print(f"{'SSIM':<20} {ssim_left.avg:<15.4f} {ssim_right.avg:<15.4f} "
          f"{(ssim_left.avg + ssim_right.avg) / 2:<15.4f}")
    if lpips_fn is not None:
        print(f"{'LPIPS':<20} {lpips_left.avg:<15.4f}")
    print(f"{'Avg Time (s)':<20} {time_meter.avg:<15.4f}")
    print(f"{'Total Images':<20} {len(test_dataset)}")
    print("=" * 60)
    results_path = os.path.join(save_dir, "results.txt")
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Average PSNR (L/R): {psnr_left.avg:.4f} / {psnr_right.avg:.4f}\n")
        f.write(f"Average SSIM (L/R): {ssim_left.avg:.4f} / {ssim_right.avg:.4f}\n")
        f.write(f"Average Time: {time_meter.avg:.4f}s\n\n")
        for r in results:
            f.write(f"{r['file']}: PSNR={r['psnr_l']:.4f}/{r['psnr_r']:.4f} "
                    f"SSIM={r['ssim_l']:.4f}/{r['ssim_r']:.4f} "
                    f"Time={r['time']:.3f}s\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    args = get_config()
    args.test_only = True
    test(args)

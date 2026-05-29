"""Evaluation for StereoSR V7.

Features
--------
* Loads either the live or EMA weights from the checkpoint (``--use_ema_eval``).
* Optional TLSC inference-time correction (``--tlsc``).
* Optional self-ensemble (``--ensemble``): 8 forwards with rotations & flips,
  averaged. Critical detail: horizontal flip swaps L/R views because the
  flipped left image becomes the right image of the flipped scene. 180°
  rotation also swaps views.

Self-ensemble correctness
-------------------------
For a horizontal flip:
* ``L'(x, y) = L(W-1-x, y)`` and ``R'(x, y) = R(W-1-x, y)``.
* The flipped scene now has the *right* image on the left and vice versa.
  So we feed ``(L', R') = (flip(R), flip(L))`` to the model.
* The model produces ``SR_L'`` for what is now the "left" view; this is
  actually the SR of ``R'``, i.e. ``flip(SR_R)``. Inverse: flip again and
  swap.

For 180° rotation:
* ``L''(x, y) = L(W-1-x, H-1-y)`` — both axes flipped.
* Same view swap as horizontal flip, since the rotation maps the right edge
  to the left edge.

For 90° and 270° rotations: NOT applied. Rotating by 90° makes the epipolar
direction vertical, which breaks SCAM's row-wise attention assumption.
We only ensemble over symmetries that preserve epipolar geometry: identity,
180° rotation, horizontal flip, and "horizontal flip + 180° rotation" =
vertical flip. That's 4 transforms — not 8 — to keep the model in distribution.
"""
from __future__ import annotations

import os
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
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
from models_v7 import build_model_v7, get_config_v7
from utils_fixed import AverageMeter, compute_psnr, compute_ssim, save_image, setup_logger


def _forward_with_transform(model, lr_l, lr_r, transform_id: int):
    """Apply one of 4 symmetry transforms, forward, invert the transform.

    transform_id meanings (all preserve epipolar geometry):
        0: identity
        1: horizontal flip (swaps L/R views)
        2: vertical flip
        3: 180° rotation = horizontal + vertical flip (swaps L/R views)
    """
    h_flip = transform_id in (1, 3)
    v_flip = transform_id in (2, 3)
    swap_views = h_flip  # h-flip and 180-rot both swap views; v-flip alone does not

    l, r = lr_l, lr_r
    if h_flip:
        l = torch.flip(l, dims=[-1])
        r = torch.flip(r, dims=[-1])
    if v_flip:
        l = torch.flip(l, dims=[-2])
        r = torch.flip(r, dims=[-2])
    if swap_views:
        l, r = r, l

    out = model(l, r)
    sr_l, sr_r = out.sr_left, out.sr_right

    # Invert in reverse order.
    if swap_views:
        sr_l, sr_r = sr_r, sr_l
    if v_flip:
        sr_l = torch.flip(sr_l, dims=[-2])
        sr_r = torch.flip(sr_r, dims=[-2])
    if h_flip:
        sr_l = torch.flip(sr_l, dims=[-1])
        sr_r = torch.flip(sr_r, dims=[-1])

    return sr_l, sr_r


@torch.no_grad()
def self_ensemble(model, lr_l, lr_r) -> tuple[torch.Tensor, torch.Tensor]:
    """Average outputs across 4 epipolar-preserving symmetry transforms."""
    sr_l_acc, sr_r_acc = None, None
    for tid in range(4):
        sl, sr = _forward_with_transform(model, lr_l, lr_r, tid)
        sr_l_acc = sl if sr_l_acc is None else sr_l_acc + sl
        sr_r_acc = sr if sr_r_acc is None else sr_r_acc + sr
    return sr_l_acc / 4.0, sr_r_acc / 4.0


def load_model_from_ckpt(model, ckpt_path: str, device, use_ema: bool, logger):
    """Load weights into ``model`` from a V7 checkpoint, preferring EMA copy
    if available and requested."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if use_ema and "ema" in ckpt:
        sd = ckpt["ema"]["module"]
        logger.info(f"Loading EMA weights from {ckpt_path}")
    else:
        sd = ckpt.get("model", ckpt)
        if use_ema:
            logger.warning(
                f"Requested --use_ema_eval but checkpoint has no 'ema' key — "
                f"falling back to live model weights."
            )
        else:
            logger.info(f"Loading live model weights from {ckpt_path}")
    result = model.load_state_dict(sd, strict=False)
    logger.info(f"Missing keys: {result.missing_keys}")
    logger.info(f"Unexpected keys: {result.unexpected_keys}")
    return ckpt.get("iteration", 0), ckpt.get("best_psnr", 0.0)


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, name="stereo_sr_v7_test")

    model = build_model_v7(args).to(device)
    if args.ckpt:
        load_model_from_ckpt(model, args.ckpt, device, args.use_ema_eval, logger)
    else:
        logger.warning("No checkpoint specified; evaluating random weights.")

    if args.tlsc:
        base_h = getattr(args, "tlsc_base_h", None) or args.patch_h
        base_w = getattr(args, "tlsc_base_w", None) or args.patch_w
        n_replaced = model.enable_tlsc((base_h, base_w))
        logger.info(
            f"TLSC enabled: replaced {n_replaced} global pools with "
            f"LocalAvgPool2d(base_size=({base_h}, {base_w}))"
        )

    model.eval()
    test_dataset = build_dataset(args, split="test")
    test_workers = min(2, args.num_workers)
    if test_workers > 0 and _shm_constrained():
        logger.warning("[shm-guard] /dev/shm < 512 MB free; forcing num_workers=0 for eval.")
        test_workers = 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=test_workers)
    logger.info(
        f"Test set: {len(test_dataset)} samples, scale={args.scale}, ensemble={args.ensemble}"
    )

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
    test(get_config_v7())

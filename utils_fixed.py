import logging
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def _simple_ssim(img1: np.ndarray, img2: np.ndarray, k1: float = 0.01, k2: float = 0.03, l: float = 1.0) -> float:
    c1, c2 = (k1 * l) ** 2, (k2 * l) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(val)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Supports batched input by averaging per-sample SSIM.
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("Expected 4D tensors [B, C, H, W]")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    b = pred.shape[0]
    vals = []
    for i in range(b):
        pred_np = pred[i].permute(1, 2, 0).detach().cpu().numpy()
        target_np = target[i].permute(1, 2, 0).detach().cpu().numpy()
        if HAS_SKIMAGE:
            vals.append(_ssim_fn(pred_np, target_np, channel_axis=2, data_range=1.0))
        else:
            vals.append(_simple_ssim(pred_np, target_np))
    return float(np.mean(vals))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def setup_logger(output_dir: str, name: str = "stereo_sr"):
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers when called multiple times.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_checkpoint(path, model, optimizer, scheduler, iteration: int, best_psnr: float, logger):
    """Atomic checkpoint save.

    torch.save is *not* atomic: if interrupted (SIGTERM, OOM, disk-full mid-write),
    the destination .pth ends up truncated/corrupt and resumes break. We write to
    a sibling .tmp file first and rename-into-place — rename on POSIX is atomic
    on the same filesystem, so the destination is either the old or new full file
    at any instant.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "best_psnr": best_psnr,
    }
    tmp_path = path + ".tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        if logger is not None:
            logger.info(f"Checkpoint saved: {path}")
    except (OSError, RuntimeError) as exc:
        if logger is not None:
            logger.error(f"Checkpoint save FAILED at {path}: {exc}")
        # Clean up partial tmp file if we can.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        # Don't re-raise — a failed save shouldn't kill training.


def prune_checkpoints(directory: str, keep_last: int, logger=None):
    """Keep only the most-recent ``keep_last`` periodic ``iter_*.pth`` files.

    Prevents disk exhaustion over a 400k-iter run where save_interval=5000
    would otherwise produce 80 checkpoints (~2 GB total). ``best.pth`` and
    ``final.pth`` are always kept.
    """
    if keep_last <= 0:
        return
    try:
        entries = []
        for name in os.listdir(directory):
            if not name.startswith("iter_") or not name.endswith(".pth"):
                continue
            try:
                step = int(name[len("iter_"):-len(".pth")])
            except ValueError:
                continue
            entries.append((step, os.path.join(directory, name)))
        entries.sort(key=lambda x: x[0], reverse=True)  # newest first
        for _, p in entries[keep_last:]:
            try:
                os.remove(p)
                if logger is not None:
                    logger.info(f"Pruned old checkpoint: {p}")
            except OSError as exc:
                if logger is not None:
                    logger.warning(f"Failed to prune {p}: {exc}")
    except OSError as exc:
        if logger is not None:
            logger.warning(f"prune_checkpoints scan failed: {exc}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device: str = "cuda", logger=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    iteration = ckpt.get("iteration", 0)
    best_psnr = ckpt.get("best_psnr", 0.0)
    if logger is not None:
        logger.info(f"Loaded checkpoint from {path} @ iter {iteration}")
    return iteration, best_psnr


class TLSC:
    def __init__(self, kernel_size: int = 31):
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def __call__(self, sr: torch.Tensor, lr_up: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size

        sr_mean = F.avg_pool2d(F.pad(sr, [self.pad] * 4, mode="reflect"), k, stride=1)
        lr_mean = F.avg_pool2d(F.pad(lr_up, [self.pad] * 4, mode="reflect"), k, stride=1)

        sr_var = F.avg_pool2d(F.pad((sr - sr_mean) ** 2, [self.pad] * 4, mode="reflect"), k, stride=1)
        lr_var = F.avg_pool2d(F.pad((lr_up - lr_mean) ** 2, [self.pad] * 4, mode="reflect"), k, stride=1)

        sr_std = (sr_var + 1e-8).sqrt()
        lr_std = (lr_var + 1e-8).sqrt()

        corrected = (sr - sr_mean) / sr_std * lr_std + lr_mean
        return corrected.clamp(0, 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)
    # Handle single-channel tensors (e.g., confidence maps)
    if tensor.shape[0] == 1:
        img = tensor.squeeze(0).detach().cpu().numpy()
    else:
        img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255).astype(np.uint8)


def save_image(tensor: torch.Tensor, path: str):
    from PIL import Image

    img = tensor_to_image(tensor)
    Image.fromarray(img).save(path)

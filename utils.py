import os
import logging
import math

import torch
import torch.nn.functional as F
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim_fn

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(data_range ** 2 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()

    if HAS_SKIMAGE:
        return _ssim_fn(pred_np, target_np, channel_axis=2, data_range=1.0)
    return _simple_ssim(pred_np, target_np)


def _simple_ssim(img1, img2, k1=0.01, k2=0.03, L=1.0):
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_val)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(output_dir: str, name: str = "stereo_sr"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_checkpoint(path, model, optimizer, scheduler, iteration, best_psnr, logger):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "best_psnr": best_psnr,
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    device="cuda", logger=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    iteration = ckpt.get("iteration", 0)
    best_psnr = ckpt.get("best_psnr", 0.0)
    if logger:
        logger.info(f"Loaded checkpoint from {path} @ iter {iteration}")
    return iteration, best_psnr


class TLSC:
    def __init__(self, kernel_size: int = 31):
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def __call__(self, sr: torch.Tensor, lr_up: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        sr_mean = F.avg_pool2d(
            F.pad(sr, [self.pad]*4, mode="reflect"),
            k, stride=1
        )
        lr_mean = F.avg_pool2d(
            F.pad(lr_up, [self.pad]*4, mode="reflect"),
            k, stride=1
        )

        sr_var = F.avg_pool2d(
            F.pad((sr - sr_mean) ** 2, [self.pad]*4, mode="reflect"),
            k, stride=1
        )
        lr_var = F.avg_pool2d(
            F.pad((lr_up - lr_mean) ** 2, [self.pad]*4, mode="reflect"),
            k, stride=1
        )


        sr_std = (sr_var + 1e-8).sqrt()
        lr_std = (lr_var + 1e-8).sqrt()

        corrected = (sr - sr_mean) / sr_std * lr_std + lr_mean
        return corrected.clamp(0, 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def save_image(tensor: torch.Tensor, path: str):
    from PIL import Image
    img = tensor_to_image(tensor)
    Image.fromarray(img).save(path)

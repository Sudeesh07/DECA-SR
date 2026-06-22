"""NTIRE-style stereo SR augmentation stack — V9 (epipolar-safe).

All ops act on batched tensors ``(lr_l, lr_r, hr_l, hr_r)`` after the loader's
crop. Each is a stochastic no-op with explicit probability and preserves stereo
geometry (consistent transform across L, R, LR/HR; view swap where the geometry
demands it). Carried over from the fixed V8 stack (replicate pad-crop h-shift,
NOT circular roll) plus stereo Mixup.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. RGB channel shuffle
# ---------------------------------------------------------------------------
_RGB_PERMS = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
]


def rgb_shuffle(lr_l, lr_r, hr_l, hr_r):
    """Random RGB channel permutation; SAME permutation on all four tensors."""
    if lr_l.shape[1] != 3:
        return lr_l, lr_r, hr_l, hr_r
    perm = _RGB_PERMS[random.randint(0, 5)]
    idx = torch.tensor(perm, device=lr_l.device, dtype=torch.long)
    return lr_l[:, idx], lr_r[:, idx], hr_l[:, idx], hr_r[:, idx]


# ---------------------------------------------------------------------------
# 2. Horizontal flip with L/R view swap
# ---------------------------------------------------------------------------
def hflip_view_swap(lr_l, lr_r, hr_l, hr_r, prob: float = 0.5):
    """With prob: flip both views horizontally AND swap L<->R (stereo symmetry)."""
    if random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    return (
        torch.flip(lr_r, dims=[-1]),
        torch.flip(lr_l, dims=[-1]),
        torch.flip(hr_r, dims=[-1]),
        torch.flip(hr_l, dims=[-1]),
    )


# ---------------------------------------------------------------------------
# 3. Vertical flip (no view swap)
# ---------------------------------------------------------------------------
def vflip(lr_l, lr_r, hr_l, hr_r, prob: float = 0.5):
    if random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    return (
        torch.flip(lr_l, dims=[-2]),
        torch.flip(lr_r, dims=[-2]),
        torch.flip(hr_l, dims=[-2]),
        torch.flip(hr_r, dims=[-2]),
    )


# ---------------------------------------------------------------------------
# 4. Random horizontal shift (replicate pad-crop, NOT circular roll)
# ---------------------------------------------------------------------------
def _replicate_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    w = x.shape[-1]
    if shift > 0:
        return F.pad(x, (shift, 0, 0, 0), mode="replicate")[..., :w]
    return F.pad(x, (0, -shift, 0, 0), mode="replicate")[..., -w:]


def hshift_same(lr_l, lr_r, hr_l, hr_r, scale: int, max_lr_shift: int = 8, prob: float = 0.5):
    """Shift L and R by the SAME amount (preserves disparity). HR shift = LR x scale.
    Replicate pad-crop keeps the bicubic skip aligned with the HR target (no seam)."""
    if max_lr_shift <= 0 or random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    shift_lr = random.randint(-max_lr_shift, max_lr_shift)
    if shift_lr == 0:
        return lr_l, lr_r, hr_l, hr_r
    shift_hr = shift_lr * scale
    return (
        _replicate_shift(lr_l, shift_lr),
        _replicate_shift(lr_r, shift_lr),
        _replicate_shift(hr_l, shift_hr),
        _replicate_shift(hr_r, shift_hr),
    )


# ---------------------------------------------------------------------------
# 5. Stereo Mixup (same lambda + same permutation for all four tensors)
# ---------------------------------------------------------------------------
def stereo_mixup(lr_l, lr_r, hr_l, hr_r, prob: float = 0.5, alpha: float = 0.2):
    """Blend the batch with a permuted copy; identical lambda/perm across L/R/LR/HR
    so stereo geometry and LR/HR alignment are preserved."""
    b = lr_l.shape[0]
    if b < 2 or random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(b, device=lr_l.device)

    def _mix(x):
        return lam * x + (1.0 - lam) * x[perm]

    return _mix(lr_l), _mix(lr_r), _mix(hr_l), _mix(hr_r)


# ---------------------------------------------------------------------------
# Composite pipeline
# ---------------------------------------------------------------------------
def apply_ntire_stack(
    lr_l, lr_r, hr_l, hr_r,
    scale: int,
    rgb_shuffle_on: bool = True,
    hflip_on: bool = True,
    vflip_on: bool = False,
    hshift_on: bool = True,
    max_lr_shift: int = 8,
    mixup_on: bool = False,
):
    if rgb_shuffle_on:
        lr_l, lr_r, hr_l, hr_r = rgb_shuffle(lr_l, lr_r, hr_l, hr_r)
    if hflip_on:
        lr_l, lr_r, hr_l, hr_r = hflip_view_swap(lr_l, lr_r, hr_l, hr_r)
    if vflip_on:
        lr_l, lr_r, hr_l, hr_r = vflip(lr_l, lr_r, hr_l, hr_r)
    if hshift_on:
        lr_l, lr_r, hr_l, hr_r = hshift_same(lr_l, lr_r, hr_l, hr_r, scale=scale, max_lr_shift=max_lr_shift)
    if mixup_on:
        lr_l, lr_r, hr_l, hr_r = stereo_mixup(lr_l, lr_r, hr_l, hr_r)
    return lr_l, lr_r, hr_l, hr_r

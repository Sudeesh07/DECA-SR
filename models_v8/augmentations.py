"""NTIRE-2024 stereo SR augmentation stack — V8 training-time augmentations.

All augmentations operate on batched tensors ``(lr_l, lr_r, hr_l, hr_r)`` after
the dataloader's random crop and PIL→tensor conversion. Each function:
1. Preserves stereo geometry (same transform applied consistently across L, R,
   and across LR/HR; view swaps applied where the geometry demands it).
2. Preserves the LR/HR alignment (same shift applied; HR shift = LR shift × scale).
3. Is a stochastic no-op with explicit per-sample probability — when not
   applied, returns inputs unchanged.

Rationale
---------
NTIRE 2024 stereo SR winners (Davinci, HiSSR, MiVideoSR) all stack:
- RGB channel shuffle (same permutation L/R) — channel invariance regularizer
- Horizontal flip + L/R view swap — exploits stereo symmetry
- Vertical flip — preserves epipolar geometry (rows stay horizontal)
- Random horizontal shift — translation invariance, preserves disparity

These are the "free win" augmentations; CutMix/Mixup are deliberately skipped
because Yoo et al. CVPR 2020 showed they hurt SR (pixel-domain blends destroy
high-frequency content the network needs to learn).
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
    """Random RGB channel permutation; SAME permutation on all four tensors.

    NAFSSR / NTIRE 2024 standard. Trivially cheap, no architecture change,
    typically +0.05–0.10 dB. Identity-no-op if input is not RGB.
    """
    if lr_l.shape[1] != 3:
        return lr_l, lr_r, hr_l, hr_r
    perm = _RGB_PERMS[random.randint(0, 5)]
    idx = torch.tensor(perm, device=lr_l.device, dtype=torch.long)
    return lr_l[:, idx], lr_r[:, idx], hr_l[:, idx], hr_r[:, idx]


# ---------------------------------------------------------------------------
# 2. Horizontal flip with L/R view swap
# ---------------------------------------------------------------------------

def hflip_view_swap(lr_l, lr_r, hr_l, hr_r, prob: float = 0.5):
    """With probability ``prob``: horizontally flip BOTH views AND swap L↔R.

    Why the swap is necessary
    -------------------------
    Horizontal-flipping a stereo pair is equivalent to viewing the scene from
    the opposite side. After flipping, the flipped *left* view is geometrically
    the *right* view of the flipped scene (and vice versa). So we must feed
    ``(flip(R), flip(L))`` to the model with HR targets ``(flip(HR_R), flip(HR_L))``.

    This is a 2× effective dataset size augmentation that preserves epipolar
    geometry exactly.
    """
    if random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    return (
        torch.flip(lr_r, dims=[-1]),  # flipped R becomes new L
        torch.flip(lr_l, dims=[-1]),  # flipped L becomes new R
        torch.flip(hr_r, dims=[-1]),
        torch.flip(hr_l, dims=[-1]),
    )


# ---------------------------------------------------------------------------
# 3. Vertical flip (no view swap — top/bottom mirror preserves L/R identity)
# ---------------------------------------------------------------------------

def vflip(lr_l, lr_r, hr_l, hr_r, prob: float = 0.5):
    """With probability ``prob``: vertically flip all four tensors.

    Preserves epipolar geometry (rows stay horizontal, just reordered).
    Does NOT swap L↔R because vertical mirror does not swap viewing side.
    """
    if random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    return (
        torch.flip(lr_l, dims=[-2]),
        torch.flip(lr_r, dims=[-2]),
        torch.flip(hr_l, dims=[-2]),
        torch.flip(hr_r, dims=[-2]),
    )


# ---------------------------------------------------------------------------
# 4. Random horizontal shift (preserves disparity)
# ---------------------------------------------------------------------------

def hshift_same(lr_l, lr_r, hr_l, hr_r, scale: int, max_lr_shift: int = 8, prob: float = 0.5):
    """With probability ``prob``: shift L AND R by the SAME horizontal amount.

    Why same shift, not independent
    -------------------------------
    Disparity ``d = x_L - x_R``. After shifting both by ``k``: ``(x_L + k) -
    (x_R + k) = d`` — unchanged. Translation invariance regularizer.

    HR shift = LR shift × scale to preserve LR↔HR alignment.

    Implementation uses circular ``torch.roll`` (rather than pad-crop) because
    Flickr1024 patches are content-rich at edges, so a few-pixel wrap is a
    smaller perturbation than zero-padding.
    """
    if max_lr_shift <= 0 or random.random() >= prob:
        return lr_l, lr_r, hr_l, hr_r
    shift_lr = random.randint(-max_lr_shift, max_lr_shift)
    if shift_lr == 0:
        return lr_l, lr_r, hr_l, hr_r
    shift_hr = shift_lr * scale
    return (
        torch.roll(lr_l, shifts=shift_lr, dims=-1),
        torch.roll(lr_r, shifts=shift_lr, dims=-1),
        torch.roll(hr_l, shifts=shift_hr, dims=-1),
        torch.roll(hr_r, shifts=shift_hr, dims=-1),
    )


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
):
    """Apply the NTIRE-2024 augmentation stack in fixed order.

    Order matters for correctness when augmentations interact (e.g., RGB
    shuffle must come before geometric augs so the same permutation persists
    through any flips/shifts; conceptually it doesn't actually matter since
    operations commute, but fixing the order keeps the implementation easy
    to reason about).

    ``hshift_on`` includes a default ``max_lr_shift = 8`` LR pixels. For the
    default V8 patch size (48×144 LR), this is ~5% of patch width.
    """
    if rgb_shuffle_on:
        lr_l, lr_r, hr_l, hr_r = rgb_shuffle(lr_l, lr_r, hr_l, hr_r)
    if hflip_on:
        lr_l, lr_r, hr_l, hr_r = hflip_view_swap(lr_l, lr_r, hr_l, hr_r)
    if vflip_on:
        lr_l, lr_r, hr_l, hr_r = vflip(lr_l, lr_r, hr_l, hr_r)
    if hshift_on:
        lr_l, lr_r, hr_l, hr_r = hshift_same(
            lr_l, lr_r, hr_l, hr_r, scale=scale, max_lr_shift=max_lr_shift
        )
    return lr_l, lr_r, hr_l, hr_r

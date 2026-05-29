"""Soft-argmax disparity prior helpers — used by V7's biased SCAM.

Why this lives in its own file
-------------------------------
The disparity prior is computed FROM the same Q·K^T correlation that SCAM
itself uses; we don't want to recompute or duplicate that. But the prior
logic (soft-argmax → Gaussian bias) is self-contained and worth isolating
for clarity and reuse.

Theoretical motivation
----------------------
For a rectified stereo pair, the correct cross-view match for a left query at
column ``x_L`` on row ``y`` is some column ``x_R`` on the same row, with
disparity ``d = x_L - x_R ≥ 0``. SCAM's softmax attention is content-driven —
it can match arbitrary L/R columns — but it has no built-in *geometric*
prior that says "matches lie on the same epipolar row at horizontal offsets
consistent with depth."

V4's GroupwiseDisparityPrior estimated disparity via a soft-argmax over a
correlation cost volume and used it to LIMIT attention to K candidates near
the predicted match. That was too restrictive (K=5 in V4 → 22.8 dB, below
V6's dense SCAM at 23.8 dB).

V7 instead uses the soft-argmax as a SOFT BIAS on the full attention logits:
the Gaussian centered at the expected match position is added to the
pre-softmax scores. The bias has a learnable per-block scale, zero-init, so
the module starts as standard SCAM and progressively incorporates the prior
if it helps reduce loss.

Stop-gradient on the soft-argmax
--------------------------------
The expected-match position is computed from softmax(attn). Letting gradient
flow through this would create a feedback loop: the bias depends on the
soft-argmax which depends on the attention which is biased by the bias.
We stop-grad the soft-argmax so the prior is treated as a *constant* the
network sees and decides whether to follow (via the learnable bias scale).
Sigma and bias_scale remain trainable; the bias adapts even though the
soft-argmax itself does not propagate gradient backwards through attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_disparity_bias(
    attn: torch.Tensor,
    sigma: torch.Tensor,
    *,
    stop_grad: bool = True,
) -> torch.Tensor:
    """Compute a Gaussian disparity bias from a row-wise correlation tensor.

    Parameters
    ----------
    attn : torch.Tensor
        Pre-softmax correlation, shape ``[B, H, W_L, W_R]`` where ``W_L`` and
        ``W_R`` are the query (left) and key (right) widths on the same row.
        For square-width V7 this is ``[B, H, W, W]``.
    sigma : torch.Tensor
        Scalar (or broadcastable) positive sigma of the Gaussian in *pixels*.
        Larger sigma → softer prior; smaller sigma → sharper prior.
    stop_grad : bool, default True
        If True, the soft-argmax (expected match position) is detached so the
        bias does not propagate gradient back through the attention. This
        prevents a circular-feedback failure mode. ``sigma`` and any outer
        scaling parameter still receive gradient.

    Returns
    -------
    torch.Tensor
        Bias tensor of shape ``[B, H, W_L, W_R]`` to be ADDED to ``attn``
        before softmax. The bias is ≤ 0 (Gaussian penalty); 0 at the expected
        match position and falling off quadratically with distance.
    """
    B, H, W_L, W_R = attn.shape
    device, dtype = attn.device, attn.dtype

    # Soft-argmax over the key (R) dimension → expected match position per
    # (batch, row, query). Shape [B, H, W_L, 1].
    match_dist = F.softmax(attn, dim=-1)
    x_r_pos = torch.arange(W_R, device=device, dtype=dtype)
    expected_x_R = (match_dist * x_r_pos.view(1, 1, 1, W_R)).sum(dim=-1, keepdim=True)

    if stop_grad:
        expected_x_R = expected_x_R.detach()

    # Gaussian penalty centered at expected_x_R.
    # diff[b, y, x_L, x_R] = x_R - expected_x_R[b, y, x_L]
    diff = x_r_pos.view(1, 1, 1, W_R) - expected_x_R  # [B, H, W_L, W_R]

    # sigma is broadcastable. We compute 1 / (2 σ²) once; tiny floor for safety.
    inv_two_sigma_sq = 0.5 / (sigma * sigma).clamp_min(1e-6)
    return -(diff * diff) * inv_two_sigma_sq


def soft_argmax_disparity(attn: torch.Tensor) -> torch.Tensor:
    """Return the soft-argmax expected match column per (batch, row, query).

    Used for diagnostic logging only (not used in the bias path — that goes
    through :func:`gaussian_disparity_bias` directly to avoid duplicating the
    softmax). Always detaches.

    Parameters
    ----------
    attn : torch.Tensor
        Pre-softmax correlation ``[B, H, W_L, W_R]``.

    Returns
    -------
    torch.Tensor
        Expected match column index ``[B, H, W_L]``, in pixel units.
    """
    B, H, W_L, W_R = attn.shape
    with torch.no_grad():
        match_dist = F.softmax(attn, dim=-1)
        x_r_pos = torch.arange(W_R, device=attn.device, dtype=attn.dtype)
        return (match_dist * x_r_pos.view(1, 1, 1, W_R)).sum(dim=-1)

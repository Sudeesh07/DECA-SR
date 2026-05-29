"""Stereo Cross-Attention Module (SCAM) with disparity-prior bias — V7.

V6's SCAM
---------
NAFSSR-style row-wise (epipolar) cross-view softmax attention. For each row y,
Q_L attends over K_R along the full width, and symmetrically R←L via the
transposed attention. Both directions use the SAME ``attn = Q_L·K_R^T``
matrix. β, γ zero-init scalars make the block identity at step 0.

V7 extension — disparity prior bias
-----------------------------------
For a rectified stereo pair, the correct match for a query at column ``x_L``
on row ``y`` is some column ``x_R`` on the same row with disparity
``d = x_L - x_R ≥ 0``. SCAM has no built-in *geometric* prior — it's
purely content-driven. V7 adds a soft Gaussian bias on the pre-softmax
logits, centered at the soft-argmax expected match position, with a
learnable per-block scale (``bias_scale``, zero-init) and width (``sigma``,
softplus-parametrised, initialised to ``sigma_init`` pixels).

At step 0:
* ``bias_scale = 0`` → the bias contributes nothing → identical to V6 SCAM.
* ``β = γ = 0`` → SCAM's residual is zero → block is the exact identity.

As training proceeds, ``bias_scale`` can lift off zero if the geometric
prior helps reduce loss. ``sigma`` adapts to whatever spatial scale of
match-region the data demands.

Stop-gradient on the soft-argmax
--------------------------------
See ``disparity_prior.gaussian_disparity_bias`` docstring. We stop-grad the
expected match position to avoid a feedback loop (bias → attention → softmax
→ soft-argmax → bias). The bias adapts via ``sigma`` and ``bias_scale``,
both of which still receive gradient.

Symmetric application
---------------------
SCAM's two directions (R→L and L→R) both use the same ``Q_L·K_R^T`` matrix.
The bias matrix is computed once and added before BOTH softmax operations —
once in the natural orientation (R→L direction: each x_L attends over x_R),
once transposed (L→R direction: each x_R attends over x_L). This preserves
the symmetry of the stereo problem.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d
from .disparity_prior import gaussian_disparity_bias, soft_argmax_disparity


class StereoCrossAttention(nn.Module):
    """Row-wise epipolar cross-view attention with optional disparity-prior bias.

    Parameters
    ----------
    channels : int
        Feature channel count.
    use_disparity_prior : bool, default True
        Enable the Gaussian disparity bias on the pre-softmax logits.
    sigma_init : float, default 4.0
        Initial value of the Gaussian width parameter (in pixels). The actual
        sigma is ``softplus(log_sigma)``; we initialise ``log_sigma`` so that
        ``softplus(log_sigma) ≈ sigma_init``. Larger init → softer prior at
        start (less constraint on attention).
    """

    def __init__(
        self,
        channels: int,
        use_disparity_prior: bool = True,
        sigma_init: float = 4.0,
    ):
        super().__init__()
        self.scale = channels ** -0.5
        self.use_disparity_prior = use_disparity_prior

        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        self.q_l = nn.Conv2d(channels, channels, 1)
        self.q_r = nn.Conv2d(channels, channels, 1)
        self.v_l = nn.Conv2d(channels, channels, 1)
        self.v_r = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        if use_disparity_prior:
            # We parametrise sigma via softplus(log_sigma) for positivity.
            # softplus^{-1}(σ) = log(exp(σ) - 1); for σ ≈ 4 this is ≈ 3.98.
            inv_softplus = math.log(math.exp(float(sigma_init)) - 1.0)
            self.log_sigma = nn.Parameter(torch.tensor(inv_softplus))
            # Zero-init bias scale → at step 0 prior contributes nothing,
            # block is identical to V6 SCAM (which was identity itself due
            # to β = γ = 0).
            self.bias_scale = nn.Parameter(torch.zeros(1))

        # Diagnostic captures (Python attributes; not buffers — DDP-safe).
        self._diag_sigma: torch.Tensor | None = None
        self._diag_bias_scale: torch.Tensor | None = None
        self._diag_disp_mean_abs: torch.Tensor | None = None  # mean |x_L - x_R_expected|
        self._diag_disp_max_abs: torch.Tensor | None = None

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_l.shape

        # Projections.
        Ql = self.q_l(self.norm_l(x_l)).permute(0, 2, 3, 1)        # [B, H, W, C]
        Qr_T = self.q_r(self.norm_r(x_r)).permute(0, 2, 1, 3)      # [B, H, C, W]
        Vl = self.v_l(x_l).permute(0, 2, 3, 1)                    # [B, H, W, C]
        Vr = self.v_r(x_r).permute(0, 2, 3, 1)                    # [B, H, W, C]

        # Attention scores + softmax in fp32 even under AMP (long-tail Q·K^T
        # otherwise overflows fp16, → softmax NaN, → weights corrupted).
        with torch.amp.autocast("cuda", enabled=False):
            Ql_f = Ql.float()
            QrT_f = Qr_T.float()
            attn = torch.matmul(Ql_f, QrT_f) * self.scale          # [B, H, W, W]

            if self.use_disparity_prior:
                sigma = F.softplus(self.log_sigma).clamp_min(1e-3)

                # Diagnostic — read RAW soft-argmax (pre-bias). This shows
                # what disparity the model is naturally predicting before
                # the geometric prior reshapes the attention.
                with torch.no_grad():
                    self._diag_sigma = sigma.detach()
                    self._diag_bias_scale = self.bias_scale.detach()
                    expected_x_R_raw = soft_argmax_disparity(attn)  # [B, H, W]
                    x_L_pos = torch.arange(
                        W, device=attn.device, dtype=attn.dtype
                    ).view(1, 1, W)
                    disp = (x_L_pos - expected_x_R_raw).abs()
                    self._diag_disp_mean_abs = disp.mean().detach()
                    self._diag_disp_max_abs = disp.max().detach()

                # Now apply the bias for the actual attention used downstream.
                bias = gaussian_disparity_bias(attn, sigma, stop_grad=True)
                attn = attn + self.bias_scale * bias

            # Two softmaxes from the SAME (biased) logits — one per direction.
            attn_r2l = F.softmax(attn, dim=-1)
            attn_l2r = F.softmax(attn.transpose(-2, -1), dim=-1)

        attn_r2l = attn_r2l.to(Vr.dtype)
        attn_l2r = attn_l2r.to(Vl.dtype)

        F_r2l = torch.matmul(attn_r2l, Vr)                        # [B, H, W, C]
        F_l2r = torch.matmul(attn_l2r, Vl)

        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta             # [B, C, H, W]
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        return x_l + F_r2l, x_r + F_l2r

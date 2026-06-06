"""Channel-Transposed Cross-View Attention (CT-CVA) — V8 core novelty.

Motivation
----------
V6's SCAM attends across *spatial positions* on each epipolar row: it answers
*where* the match is. This captures geometric correspondence but says nothing
about *channel-statistical* cross-view correlations — what feature-channel
patterns in R should mix into which channels of L. Single-image restoration
literature (Restormer's MDTA, Zamir et al. CVPR 2022) showed channel-axis
attention is a complementary signal to pixel-axis attention; this module
transfers that idea to cross-view stereo SR for the first time.

Mechanism (per block, per view-pair)
------------------------------------
Multi-head cosine attention over the channel axis. For ``L → R`` direction:

  Q_L = qkv_l(LN(x_L))  → q_L                      [B, C, H, W]
  K_R, V_R = qkv_r(LN(x_R))  → k_R, v_R            [B, C, H, W]

  Reshape into multi-head channel tokens:
  q_L: [B, h, d, HW]   where  C = h·d
  k_R: [B, h, d, HW]
  v_R: [B, h, d, HW]

  Cosine attention (L2-normalise along the HW axis — this is the bf16 and
  *scale-invariance* trick; standard scaled-dot-product attention can overflow
  at large HW and depends on HW for proper softmax temperature):
    q_L_n = q_L / ||q_L||_HW
    k_R_n = k_R / ||k_R||_HW
    attn = q_L_n @ k_R_n^T · τ                     [B, h, d, d]    ← FIXED size!
    attn = softmax(attn, dim=-1)
    out_L = attn @ v_R                             [B, h, d, HW]
    out_L = proj_l(reshape(out_L, [B, C, H, W]))

  Residual: out_L = x_L + γ_L · proj_l(...)         (γ_L zero-init)

Symmetric for R ← L.

Scale-invariance contract
-------------------------
The attention matrix has shape ``[B, h, d, d]`` where ``d = C/h``. **It does
NOT depend on H or W.** The softmax distribution shape is determined by the
channel structure of the network, not by the input image size. So the
train→val image-size shift (e.g., 144 px training patches → ~500 px full
validation images) cannot change the attention distribution — which is the
mathematical failure mode that V4 (pixel-unit sigma) and V7 (pool-area
mismatch) both stumbled on.

Identity-at-init contract
-------------------------
γ_L = γ_R = 0 → ``out = input`` exactly at step 0. The ``proj_l(attn @ v_R)``
term is non-zero generally (Q_L · K_R cosine attention is non-degenerate at
init), so ``∂loss/∂γ = sgate-style chain · proj_output ≠ 0``: γ has a non-zero
gradient and can lift off zero. This avoids the v3 saturation failure mode
(no bounded tanh, no atan2, no FFT).

Diagnostic captures
-------------------
Like V7 and V6 spectral_fusion v2/v3, we expose Python-attribute diagnostics
(NOT buffers — DDP-safe, not in state_dict): mean γ magnitude, mean
temperature, mean attention entropy. Used by ``train_ddp_v8.gather_ct_cva_diagnostics``
to log per-block stats and emit a silent-collapse warning if γ stays near
zero past iter 5000.

Why this is novel for stereo SR
-------------------------------
Restormer's MDTA is single-image (one input, transposed attention to handle
large resolutions). PASSRnet, NAFSSR, SwinFSR, SCAM, biPAM — all stereo
cross-view modules operate on *spatial positions*. No published stereo SR
paper does channel-axis cross-view attention. CT-CVA is the first systematic
transfer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d


class ChannelTransposedCrossView(nn.Module):
    """Multi-head cosine cross-view attention along the channel axis.

    Parameters
    ----------
    channels : int
        Feature channel count. Must be divisible by ``num_heads``.
    num_heads : int, default 4
        Heads for multi-head channel attention. Default produces 16-dim
        attention matrices at channels=64.
    temperature_init : float, default 1.0
        Initial value of the per-head learnable temperature scaling the
        cosine similarity before softmax. Restormer reports 1.0 works well
        across image-restoration tasks.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        temperature_init: float = 1.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Per-head learnable temperature. Initialised to temperature_init.
        # Shape broadcasts onto [B, h, d, d].
        self.temperature = nn.Parameter(
            torch.full((num_heads, 1, 1), float(temperature_init))
        )

        # Pre-attention normalisation.
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        # Joint QKV projection per view.
        # qkv_l produces Q_L (for L→R direction) AND K_L, V_L (for R←L direction).
        # qkv_r symmetrically produces Q_R, K_R, V_R.
        self.qkv_l = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.qkv_r = nn.Conv2d(channels, channels * 3, 1, bias=True)

        # Output projections (one per view).
        self.proj_l = nn.Conv2d(channels, channels, 1, bias=True)
        self.proj_r = nn.Conv2d(channels, channels, 1, bias=True)

        # Zero-init residual scales: at step 0, CT-CVA contributes nothing.
        self.gamma_l = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma_r = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Diagnostics — Python attributes (NOT buffers): DDP-safe, no
        # checkpoint pollution. Updated under no_grad in forward.
        self._diag_gamma_l_abs: torch.Tensor | None = None
        self._diag_gamma_r_abs: torch.Tensor | None = None
        self._diag_temperature: torch.Tensor | None = None
        self._diag_attn_entropy: torch.Tensor | None = None
        self._diag_attn_max: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Internal: one direction of cross-view channel attention.
    # ------------------------------------------------------------------

    def _cross_attend_direction(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj: nn.Conv2d,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One direction of cross-view channel attention.

        Returns ``(projected_output, attn_for_diagnostics)`` where
        ``projected_output`` has shape ``[B, C, H, W]`` and ``attn_for_diagnostics``
        has shape ``[B, h, d, d]`` (used by the caller to compute entropy).
        """
        B, C, H, W = q.shape
        h, d = self.num_heads, self.head_dim
        HW = H * W

        # Reshape to multi-head channel tokens: [B, h, d, HW].
        q = q.reshape(B, h, d, HW)
        k = k.reshape(B, h, d, HW)
        v = v.reshape(B, h, d, HW)

        # Cosine attention: L2-normalise along the HW (spatial-token) axis.
        # This is the key trick that gives scale-invariance — the softmax
        # over channels is independent of HW because we normalise it out.
        q_n = F.normalize(q, dim=-1)
        k_n = F.normalize(k, dim=-1)

        # Attention in fp32 to match SCAM's bf16-stability discipline.
        with torch.amp.autocast("cuda", enabled=False):
            q_n_f = q_n.float()
            k_n_f = k_n.float()
            # attn[b, h, i, j] = <q_n[b, h, i, :], k_n[b, h, j, :]>
            attn = torch.matmul(q_n_f, k_n_f.transpose(-1, -2))  # [B, h, d, d]
            attn = attn * self.temperature                       # per-head scaling
            attn_softmax = F.softmax(attn, dim=-1)               # over channel keys

        attn_softmax_dtype = attn_softmax.to(v.dtype)

        # Apply to V: each channel-token (d_query) gets a weighted sum over the
        # d_key channel-tokens, each carrying its full HW spatial pattern.
        out = torch.matmul(attn_softmax_dtype, v)                # [B, h, d, HW]
        out = out.reshape(B, C, H, W)
        return proj(out), attn_softmax

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(
        self, x_l: torch.Tensor, x_r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # QKV projections per view.
        qkv_l = self.qkv_l(self.norm_l(x_l))                     # [B, 3C, H, W]
        qkv_r = self.qkv_r(self.norm_r(x_r))
        q_l, k_l, v_l = qkv_l.chunk(3, dim=1)
        q_r, k_r, v_r = qkv_r.chunk(3, dim=1)

        # L direction: L queries R (gets info from R).
        # R direction: R queries L (gets info from L).
        out_l, attn_lr = self._cross_attend_direction(q_l, k_r, v_r, self.proj_l)
        out_r, attn_rl = self._cross_attend_direction(q_r, k_l, v_l, self.proj_r)

        # Diagnostics: γ magnitude, temperature mean, attention entropy.
        # Entropy is averaged over [B, h, d_query]; perfect uniform = log(d).
        with torch.no_grad():
            self._diag_gamma_l_abs = self.gamma_l.data.abs().mean().detach()
            self._diag_gamma_r_abs = self.gamma_r.data.abs().mean().detach()
            self._diag_temperature = self.temperature.data.abs().mean().detach()
            # Attention entropy: -sum_j p log p, averaged over (B, h, d_q).
            # attn_lr in fp32; clamp_min to avoid log(0) NaN.
            ent_lr = -(attn_lr.clamp_min(1e-9) * attn_lr.clamp_min(1e-9).log()).sum(dim=-1)
            ent_rl = -(attn_rl.clamp_min(1e-9) * attn_rl.clamp_min(1e-9).log()).sum(dim=-1)
            # Mean entropy normalised by log(d) so it's in [0, 1].
            log_d = math.log(self.head_dim)
            self._diag_attn_entropy = (0.5 * (ent_lr.mean() + ent_rl.mean()) / log_d).detach()
            self._diag_attn_max = (0.5 * (attn_lr.max() + attn_rl.max())).detach()

        # Residual: γ zero-init → at step 0, output = input exactly.
        return x_l + self.gamma_l * out_l, x_r + self.gamma_r * out_r

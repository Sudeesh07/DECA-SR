"""Cross-View Spectral Fusion Module (CVSFM) — core novelty of V6.

Theory
------
For a rectified stereo pair (L, R) with disparity d(x, y) along the x-axis,
R(x, y) = L(x - d, y). For locally-constant d, the 2D Fourier shift theorem
gives

    F_R(u, v) = F_L(u, v) · exp(-2πi · u · d / W)

and consequently

    |F_R| = |F_L|                       (magnitude spectrum is view-invariant)
    arg(F_R) - arg(F_L) = -2π u d / W   (phase encodes disparity)

This decomposes the stereo signal into an appearance component (magnitude,
shared across views) and a geometry component (phase, view-specific).

CVSFM fuses magnitudes across views (higher-SNR joint estimate of the shared
appearance) while preserving each view's phase (view-specific geometry). A
learnable per-frequency gate handles occluded regions, where the magnitude
equality is violated, by falling back to the per-view magnitude.

Implementation notes
--------------------
* FFT is forced to fp32 regardless of AMP context for numerical stability.
* The block uses a zero-init residual scale (γ) so at initialisation it is the
  identity function; training activates the spectral path gradually.
* `torch.polar(r, θ)` constructs the complex tensor r · e^{iθ} differentiably.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossViewSpectralFusion(nn.Module):
    """Cross-view spectral magnitude fusion with view-specific phase preservation."""

    def __init__(self, channels: int, hidden_expansion: float = 0.5):
        super().__init__()
        hidden = max(8, int(channels * hidden_expansion))

        # Per-frequency gate network operating in the spectrum domain.
        # Input: concat of [mag_L, mag_R] (2C channels).
        # Output: two gates α_L, α_R (C channels each), each in (0, 1) after sigmoid.
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * channels, 1),
        )
        # Initialise final gate layer so α ≈ 0 at start → fused_mag ≈ own_mag;
        # combined with γ=0 this makes the block the exact identity at init.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -5.0)  # sigmoid(-5) ≈ 6.7e-3

        # Zero-initialised per-channel residual scale.
        self.gamma_l = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma_r = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_l.shape
        dtype = x_l.dtype

        # Spectral operations in fp32 for stability under AMP.
        with torch.amp.autocast("cuda", enabled=False):
            xl_f = x_l.float()
            xr_f = x_r.float()
            F_l = torch.fft.rfft2(xl_f, norm="ortho")
            F_r = torch.fft.rfft2(xr_f, norm="ortho")

            mag_l = F_l.abs()
            mag_r = F_r.abs()
            phase_l = torch.angle(F_l)
            phase_r = torch.angle(F_r)

            # Per-frequency cross-view gating.
            gates = self.gate(torch.cat([mag_l, mag_r], dim=1))
            alpha_l, alpha_r = gates.chunk(2, dim=1)
            alpha_l = torch.sigmoid(alpha_l)
            alpha_r = torch.sigmoid(alpha_r)

            # Fused per-view magnitudes (weighted mix of own and other-view mag).
            fused_mag_l = alpha_l * mag_r + (1.0 - alpha_l) * mag_l
            fused_mag_r = alpha_r * mag_l + (1.0 - alpha_r) * mag_r

            # Numerical safety: clamp to a small floor so torch.polar's
            # backward pass through atan2(0,0) cannot inject NaN gradients
            # in pathological underflow cases. The floor (1e-8 in fp32) is
            # well below any signal magnitude we care about.
            fused_mag_l = fused_mag_l.clamp_min(1e-8)
            fused_mag_r = fused_mag_r.clamp_min(1e-8)

            # Reconstruct complex spectrum with view-specific phase.
            F_l_new = torch.polar(fused_mag_l, phase_l)
            F_r_new = torch.polar(fused_mag_r, phase_r)

            xl_new = torch.fft.irfft2(F_l_new, s=(H, W), norm="ortho")
            xr_new = torch.fft.irfft2(F_r_new, s=(H, W), norm="ortho")

        xl_new = xl_new.to(dtype)
        xr_new = xr_new.to(dtype)

        # Zero-init residual scale: module starts as identity.
        out_l = x_l + self.gamma_l * (xl_new - x_l)
        out_r = x_r + self.gamma_r * (xr_new - x_r)
        return out_l, out_r

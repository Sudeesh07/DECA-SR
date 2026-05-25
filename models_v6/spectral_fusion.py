"""Cross-View Spectral Fusion Module — V6 spectral path.

Theory (revised; v1 strict claim was empirically falsified by ablation A6)
--------------------------------------------------------------------------
For a rectified stereo pair (L, R) with disparity d(x, y) along the x-axis,
the 1D Fourier shift theorem on a single row gives, for locally constant d,
    F_R^x(u, y) = F_L^x(u, y) · exp(-2πi · u · d(y) / W).
This implies |F_R^x| = |F_L^x| ONLY under the joint assumptions of:
    (i)   strictly constant disparity within the FFT window,
    (ii)  circular (not linear) shift — i.e. no boundary effects,
    (iii) co-visibility (no occlusion) over the entire row,
    (iv)  no photometric mismatch (illumination, specularity, …),
    (v)   strict shift-equivariance through the CNN trunk.

None of (i)–(v) hold strictly in feature space on real stereo. The DFT
boundary term B[u, v] alone is non-negligible even with all other assumptions
satisfied. The realistic statement is APPROXIMATE:

    ‖|F_R^x| − |F_L^x|‖ ≤ ε(u, σ_d, p_occ, δ_photo, RF/patch),

where ε is small but strictly positive at low/mid frequencies on co-visible
interior rows.

The module therefore does NOT enforce magnitude equality. It treats the
cross-view spectra as TWO CORRELATED EVIDENCE STREAMS for the underlying
scene structure, and learns a residual correction to each view's spectrum
conditioned on both. A per-pixel spatial gate localises this residual in the
image domain so that the correction can be down-weighted where (ii)–(iv) are
violated (occlusion, specular highlight, image boundary).

Two implementations live here:
* :class:`CrossViewSpectralFusion`        — v1, the original strict-mixing
  design retained for ablations A1/A2/A4/A5/A6.
* :class:`CrossViewSpectralFusionV2`      — v2, the theoretically-revised
  design below.

Implementation notes
--------------------
* FFT is forced to fp32 regardless of AMP context for numerical stability.
* Both modules zero-init γ so the block is approximately the identity at
  step 0; training lifts γ off zero once a non-zero residual exists.
* `torch.polar(r, θ)` constructs the complex tensor r · e^{iθ} differentiably.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossViewSpectralFusion(nn.Module):
    """Cross-view spectral magnitude fusion with view-specific phase preservation."""

    def __init__(
        self,
        channels: int,
        hidden_expansion: float = 0.5,
        use_gate: bool = True,
        use_1d_fft: bool = False,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.use_1d_fft = use_1d_fft
        hidden = max(8, int(channels * hidden_expansion))

        # Per-frequency gate network operating in the spectrum domain.
        # Input: concat of [mag_L, mag_R] (2C channels).
        # Output: two gates α_L, α_R (C channels each), each in (0, 1) after sigmoid.
        # Ablation A6: disabled — α is hard-coded to 0.5 (simple magnitude average),
        # isolating whether learnable per-frequency mixing or magnitude invariance
        # is what the spectral path actually exploits.
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(2 * channels, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, 2 * channels, 1),
            )
            # Initialise final gate layer so α ≈ 0 at start → fused_mag ≈ own_mag;
            # combined with γ=0 this makes the block the exact identity at init.
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -5.0)  # sigmoid(-5) ≈ 6.7e-3
        else:
            self.gate = None

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
            # Ablation A5: 1D rfft along the width axis only. Stereo disparity
            # is purely horizontal, so the shift theorem invoked by Proposition 1
            # is a 1D statement; the 2D variant conflates a trivial vertical
            # axis with the meaningful horizontal one.
            if self.use_1d_fft:
                F_l = torch.fft.rfft(xl_f, dim=-1, norm="ortho")
                F_r = torch.fft.rfft(xr_f, dim=-1, norm="ortho")
            else:
                F_l = torch.fft.rfft2(xl_f, norm="ortho")
                F_r = torch.fft.rfft2(xr_f, norm="ortho")

            mag_l = F_l.abs()
            mag_r = F_r.abs()
            phase_l = torch.angle(F_l)
            phase_r = torch.angle(F_r)

            # Per-frequency cross-view gating (or simple average under A6).
            if self.use_gate:
                gates = self.gate(torch.cat([mag_l, mag_r], dim=1))
                alpha_l, alpha_r = gates.chunk(2, dim=1)
                alpha_l = torch.sigmoid(alpha_l)
                alpha_r = torch.sigmoid(alpha_r)
            else:
                # A6: hard-coded α = 0.5 → fused magnitude is the arithmetic mean.
                alpha_l = torch.full_like(mag_l, 0.5)
                alpha_r = torch.full_like(mag_r, 0.5)

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

            if self.use_1d_fft:
                xl_new = torch.fft.irfft(F_l_new, n=W, dim=-1, norm="ortho")
                xr_new = torch.fft.irfft(F_r_new, n=W, dim=-1, norm="ortho")
            else:
                xl_new = torch.fft.irfft2(F_l_new, s=(H, W), norm="ortho")
                xr_new = torch.fft.irfft2(F_r_new, s=(H, W), norm="ortho")

        xl_new = xl_new.to(dtype)
        xr_new = xr_new.to(dtype)

        # Zero-init residual scale: module starts as identity.
        out_l = x_l + self.gamma_l * (xl_new - x_l)
        out_r = x_r + self.gamma_r * (xr_new - x_r)
        return out_l, out_r


class CrossViewSpectralFusionV2(nn.Module):
    """v2 spectral path: theoretically-revised cross-view residual.

    Differences from v1 (each addresses a specific peer-review critique):

    * **1D rfft along x only** — disparity is purely horizontal; the shift
      theorem we invoke is 1D. The vertical axis was added by v1 but has no
      theoretical justification (critique #4).
    * **Log-magnitude additive residual** — replaces v1's convex magnitude
      mix ``α·m_R + (1-α)·m_L`` which is pointwise bounded by
      ``[min(m_L, m_R), max(m_L, m_R)]`` and cannot synthesize energy at
      frequencies under-represented in both views (critique #3).
      Here we learn ``log m̃_L = log m_L + Δ_L(log m_L, log m_R, u/W)``,
      which is unbounded and *generative*.
    * **Per-pixel image-domain spatial gate** — v1's per-frequency gate
      pools across the whole feature-map spatially, so it cannot localise
      occlusion handling. v2 adds a depthwise 3×3 + 1×1 gate that runs in
      the image domain and multiplies the spectral residual per pixel
      (critique #5).
    * **Own-phase preserved** — phase mixing would shift content, fighting
      SCAM's role. The phase claim from v1 is dropped; v2 does NOT claim
      anything about phase semantics.
    * **Approximate, not strict** — the docstring at module level frames
      the claim as approximate from the start. No equality is enforced
      anywhere (critiques #1, #6, #8).

    What v2 still cannot do
    -----------------------
    * The DFT circular-vs-linear-shift boundary term is **not** addressed.
      Windowing would help in theory but makes the inverse FFT unstable at
      patch edges (Hann goes to 0). v2 absorbs this into the learned
      residual; honest cost is a larger ε in the approximate bound.
    * The semantic claim "magnitude = appearance, phase = geometry" from
      Oppenheim–Lim (1981) is about *images*, not feature maps. v2 makes
      no such claim. The module is justified as an information channel,
      not a semantic decomposition.

    Identity-at-init analysis
    -------------------------
    * Δ_L final layer is initialised with kaiming × ``delta_init_scale``
      (default 0.01). Output magnitude ≈ 0.01, so ``new_log_mag = log_mag +
      O(0.01)`` and ``new_mag ≈ mag · (1 ± 1%)``. After ifft, ``xl_new``
      differs from ``x_l`` by O(1%).
    * Spatial gate (sgate) final layer is zero-init for weights and bias →
      pre-sigmoid output = 0 → sigmoid = 0.5 (fully open).
    * γ_l, γ_r = 0 exactly → ``out_l = x_l + 0 · 0.5 · O(0.01) = x_l``.
      Module is the exact identity at step 0.
    * Crucially, ``∂out_l/∂γ_l = sgate_l · (xl_new − x_l) = 0.5 · O(0.01)
      ≠ 0`` so γ HAS a non-zero gradient at init. The zero-gradient
      death that would happen with delta_init_scale=0 is avoided.
    """

    def __init__(
        self,
        channels: int,
        hidden_expansion: float = 0.5,
        log_eps: float = 1e-6,
        delta_init_scale: float = 0.01,
    ):
        super().__init__()
        self.log_eps = log_eps
        hidden = max(8, int(channels * hidden_expansion))

        # Log-magnitude residual networks. 1×1 convs act per (row, freq) bin.
        # Input channels: [log|F_own|, log|F_other|, normalized freq] = 2C + 1.
        def _make_delta() -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(2 * channels + 1, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, channels, 1),
            )
            # Small but non-zero final layer so Δ ≠ 0 at init → γ has gradient.
            nn.init.kaiming_normal_(net[-1].weight, mode="fan_out", nonlinearity="linear")
            with torch.no_grad():
                net[-1].weight.mul_(delta_init_scale)
            nn.init.zeros_(net[-1].bias)
            return net

        self.delta_l = _make_delta()
        self.delta_r = _make_delta()

        # Image-domain per-pixel spatial gate. Depthwise 3×3 gives local
        # spatial context cheaply; 1×1 projects to per-channel gate values.
        def _make_sgate() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
                nn.Conv2d(2 * channels, channels, 1),
            )

        self.sgate_l = _make_sgate()
        self.sgate_r = _make_sgate()
        # Zero-init final layer → sigmoid(0) = 0.5 (gate open at init).
        nn.init.zeros_(self.sgate_l[-1].weight)
        nn.init.zeros_(self.sgate_l[-1].bias)
        nn.init.zeros_(self.sgate_r[-1].weight)
        nn.init.zeros_(self.sgate_r[-1].bias)

        # Per-channel residual scale, zero-init (identity at step 0).
        self.gamma_l = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma_r = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_l.shape
        dtype = x_l.dtype
        Wf = W // 2 + 1

        with torch.amp.autocast("cuda", enabled=False):
            xl_f = x_l.float()
            xr_f = x_r.float()

            # 1D rfft along width (disparity axis).
            F_l = torch.fft.rfft(xl_f, dim=-1, norm="ortho")  # [B, C, H, Wf]
            F_r = torch.fft.rfft(xr_f, dim=-1, norm="ortho")

            mag_l = F_l.abs()
            mag_r = F_r.abs()
            phase_l = torch.angle(F_l)
            phase_r = torch.angle(F_r)

            log_mag_l = (mag_l + self.log_eps).log()
            log_mag_r = (mag_r + self.log_eps).log()

            # Normalized frequency in [0, 0.5], broadcast to [B, 1, H, Wf].
            freq = torch.linspace(0.0, 0.5, Wf, device=xl_f.device, dtype=xl_f.dtype)
            freq = freq.view(1, 1, 1, Wf).expand(B, 1, H, Wf)

            # Log-magnitude residuals (generative, unbounded).
            delta_l = self.delta_l(torch.cat([log_mag_l, log_mag_r, freq], dim=1))
            delta_r = self.delta_r(torch.cat([log_mag_r, log_mag_l, freq], dim=1))

            new_log_mag_l = log_mag_l + delta_l
            new_log_mag_r = log_mag_r + delta_r
            # Clamp before exp to prevent fp32 overflow (exp(30) ≈ 1e13).
            new_mag_l = new_log_mag_l.clamp(min=-30.0, max=30.0).exp()
            new_mag_r = new_log_mag_r.clamp(min=-30.0, max=30.0).exp()

            # Reconstruct complex spectrum with OWN phase preserved.
            F_l_new = torch.polar(new_mag_l, phase_l)
            F_r_new = torch.polar(new_mag_r, phase_r)

            xl_new = torch.fft.irfft(F_l_new, n=W, dim=-1, norm="ortho")
            xr_new = torch.fft.irfft(F_r_new, n=W, dim=-1, norm="ortho")

        xl_new = xl_new.to(dtype)
        xr_new = xr_new.to(dtype)

        # Image-domain spatial gate (per-pixel localisation of the residual).
        sgate_l = torch.sigmoid(self.sgate_l(torch.cat([x_l, x_r], dim=1)))
        sgate_r = torch.sigmoid(self.sgate_r(torch.cat([x_r, x_l], dim=1)))

        # Two-level residual: per-channel γ × per-pixel sgate × spectral diff.
        out_l = x_l + self.gamma_l * sgate_l * (xl_new - x_l)
        out_r = x_r + self.gamma_r * sgate_r * (xr_new - x_r)
        return out_l, out_r

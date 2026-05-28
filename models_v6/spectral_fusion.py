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

Three implementations live here:
* :class:`CrossViewSpectralFusion`        — v1, the original strict-mixing
  design retained for ablations A1/A2/A4/A5/A6.
* :class:`CrossViewSpectralFusionV2`      — v2, log-magnitude residual +
  spatial gate. Retained but NOT the recommended path: v2 still operates on
  magnitudes only, which are already ~equal between views under the shift
  theorem, so it cannot transfer the cross-view geometric signal (that
  signal lives in the phase DIFFERENCE).
* :class:`CrossViewSpectralFusionV3`      — v3, phase-correlation-based
  cross-view alignment. Operates on ∠F_R − ∠F_L (the actual stereo
  signature), predicting a learned complex alignment filter per
  (channel, row, frequency) bin. This is the structurally correct
  operator for cross-view information transfer.

Implementation notes
--------------------
* FFT is forced to fp32 regardless of AMP context for numerical stability.
* Both modules zero-init γ so the block is approximately the identity at
  step 0; training lifts γ off zero once a non-zero residual exists.
* `torch.polar(r, θ)` constructs the complex tensor r · e^{iθ} differentiably.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CrossViewSpectralFusionV3(nn.Module):
    """v3 spectral path: phase-correlation cross-view alignment with bounded
    filter parametrisation and Hann/overlap-add windowing.

    Why v3 exists
    -------------
    Ablation A6 falsified v1's magnitude-mixing mechanism (−0.004 dB on top
    of SCAM). The mechanical reason: under the shift theorem,
        F_R(ω) = F_L(ω) · exp(−jωd)   ⇒   |F_R| ≈ |F_L|,
    so v1 mixes the part the views ALREADY agree on (magnitudes) while
    preserving each view's OWN phase — the part that actually distinguishes
    them. v2 inherits the flaw: log-magnitude residual is a better magnitude
    operator but still never touches the inter-view phase difference, which
    is where the cross-view signal lives.

    v3 operates on the phase DIFFERENCE — the actual stereo signature.

    Engineering safety guards (added after second-round critique)
    -------------------------------------------------------------
    Earlier v3 used an unconstrained complex filter ``A = A_real + j·A_imag``
    and non-overlapping rectangular windows. Both are unstable: the filter
    could amplify frequencies and rotate phase arbitrarily, and rectangular
    windows produce spectral leakage and boundary ringing (Gibbs phenomenon).
    This version fixes both:

    * **Bounded filter parametrisation**:
          gain  = 1 + ``gain_scale`` · tanh(g_raw)         ∈ [1 − s, 1 + s]
          theta = π · tanh(θ_raw)                          ∈ (−π, π)
          A     = gain · (cos θ + j sin θ)
      At init (network output ≈ 0) gain ≈ 1, theta ≈ 0, A ≈ 1 — identity
      filter. Worst-case gain is bounded; phase is wrapped naturally.
    * **Hann window + 50 % overlap-add (WOLA)** when ``window > 0``:
      analysis Hann tapering reduces spectral leakage; 50 % overlap with
      synthesis Hann gives weighted overlap-add reconstruction, normalised
      by sum-of-squared windows. No inverse-window division.
    * **Optional HF-only** (``hf_only=True``): scales the filter's deviation
      from identity by frequency, so DC and very-low-frequency components
      are forced to A ≈ 1 (no change). High frequencies — where SR
      reconstruction actually lives — get the full bounded range.

    Mechanism (per view-pair, per block)
    ------------------------------------
    1. **1D STFT along width** (Hann + 50 % overlap when windowed; full-row
       rfft when ``window=0``).
    2. Cross-power phase difference computed without ``atan2``:
           cp = F_L · conj(F_R);   ĉp = cp / (|cp| + ε)
           cos Δφ = ĉp.real,  sin Δφ = ĉp.imag.
    3. Predict raw filter (g_raw, θ_raw) per (channel, frame, frequency) bin
       from ``[cos Δφ, sin Δφ, log|F_L|, log|F_R|, freq]`` via 1×1 conv MLP.
    4. Apply bounded parametrisation to get complex filter A. Warp the OTHER
       view onto SELF: ``F_R→L = F_R · A_L``, ``F_L→R = F_L · A_R``.
    5. **Inverse STFT** (irfft + Hann synthesis + overlap-add + normalise).
    6. Per-pixel image-domain spatial gate from ``concat(x_self,
       x_other_warped)`` localises the residual. Occlusion handling.
    7. γ per-channel zero-init residual scale; the block is exact identity
       at step 0.

    Identity-at-init contract (verified by smoke test)
    --------------------------------------------------
    * γ_L, γ_R = 0 exactly  →  output equals input exactly.
    * Alignment final layer: weights kaiming × ``align_init_scale`` (default
      0.01), bias = 0. Output ≈ 0 ⇒ ``tanh(0) = 0`` ⇒ gain = 1, theta = 0
      ⇒ A = 1 exactly at init.
    * ``∂out/∂γ = sgate · (x_other_warped − x_self) = sgate · (x_other −
      x_self) ≠ 0`` so γ lifts off zero immediately.

    No semantic claim about "magnitude = appearance" or "phase = geometry".
    """

    def __init__(
        self,
        channels: int,
        hidden_expansion: float = 0.5,
        log_eps: float = 1e-6,
        align_init_scale: float = 0.01,
        window: int = 0,
        gain_scale: float = 0.1,
        hf_only: bool = False,
    ):
        super().__init__()
        if window < 0:
            raise ValueError(f"window must be >= 0 (got {window})")
        if window > 0 and window < 4:
            raise ValueError(f"window must be 0 (off) or >= 4 (got {window})")
        self.log_eps = log_eps
        self.window = int(window)
        self.hop = self.window // 2 if self.window > 0 else 0
        self.gain_scale = float(gain_scale)
        self.hf_only = bool(hf_only)
        hidden = max(8, int(channels * hidden_expansion))

        # Alignment networks: predict per-(spatial, freq) raw (g_raw, θ_raw)
        # pairs. Conditioning: [cos Δφ, sin Δφ, log|F_self|, log|F_other|, freq]
        # = 4C + 1 channels. Output 2C: first C is g_raw, second C is θ_raw.
        def _make_align() -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(4 * channels + 1, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, 2 * channels, 1),
            )
            nn.init.kaiming_normal_(net[-1].weight, mode="fan_out", nonlinearity="linear")
            with torch.no_grad():
                net[-1].weight.mul_(align_init_scale)
            # Bias = 0 → at init tanh(0) = 0 → gain = 1, theta = 0 → A = 1.
            nn.init.zeros_(net[-1].bias)
            return net

        self.align_l = _make_align()  # warps R onto L
        self.align_r = _make_align()  # warps L onto R

        # Image-domain spatial gates. Input: concat(x_self, x_other_warped).
        def _make_sgate() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
                nn.Conv2d(2 * channels, channels, 1),
            )

        self.sgate_l = _make_sgate()
        self.sgate_r = _make_sgate()
        nn.init.zeros_(self.sgate_l[-1].weight)
        nn.init.zeros_(self.sgate_l[-1].bias)
        nn.init.zeros_(self.sgate_r[-1].weight)
        nn.init.zeros_(self.sgate_r[-1].bias)

        # Per-channel residual scale, zero-init (identity at step 0).
        self.gamma_l = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma_r = nn.Parameter(torch.zeros(1, channels, 1, 1))

    # ------------------------------------------------------------------
    # STFT / iSTFT with Hann window + 50 % overlap-add (WOLA).
    # ------------------------------------------------------------------

    @staticmethod
    def _hann(win: int, device, dtype) -> torch.Tensor:
        """Periodic Hann window; periodic flag matches NumPy / DSP convention.

        With periodic=True and hop = win/2 the analysis/synthesis Hann pair
        satisfies the constant-overlap-add (COLA) condition: the sum of
        squared Hann windows is constant (= win/4 per bin) in the interior.
        Edges have fewer overlapping frames and are explicitly normalised
        by the per-position sum-of-squared-windows map.
        """
        return torch.hann_window(win, periodic=True, device=device, dtype=dtype)

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Frame + Hann-analyse + rfft. Returns
        ``(F: [B, C, H, NF, Wf], NF, pad_w, original_W)``.
        """
        B, C, H, W = x.shape
        win, hop = self.window, self.hop
        # Pad so (W_padded - win) is a multiple of hop.
        if W < win:
            pad_w = win - W
        else:
            rem = (W - win) % hop
            pad_w = (hop - rem) % hop
        if pad_w > 0:
            x = F.pad(x, [0, pad_w])
        Wp = x.shape[-1]
        NF = (Wp - win) // hop + 1
        # Extract overlapping frames along width.
        frames = x.unfold(-1, win, hop)  # [B, C, H, NF, win]
        # Analysis window.
        h = self._hann(win, x.device, x.dtype).view(1, 1, 1, 1, win)
        frames = frames * h
        # 1D rfft along the window axis.
        F_x = torch.fft.rfft(frames, dim=-1, norm="ortho")  # [B, C, H, NF, Wf]
        return F_x, NF, pad_w, W

    def _istft(
        self, F_x: torch.Tensor, NF: int, pad_w: int, original_W: int
    ) -> torch.Tensor:
        """Inverse: irfft → Hann-synthesise → overlap-add → normalise."""
        win, hop = self.window, self.hop
        # IFFT per frame.
        frames = torch.fft.irfft(F_x, n=win, dim=-1, norm="ortho")  # [B, C, H, NF, win]
        # Synthesis Hann.
        h = self._hann(win, frames.device, frames.dtype)  # [win]
        frames = frames * h.view(1, 1, 1, 1, win)
        B, C, H, _, _ = frames.shape
        out_len = (NF - 1) * hop + win
        # Overlap-add. NF is small (≈ 2·W/win) so a Python loop is fine on GPU.
        out = torch.zeros(B, C, H, out_len, device=frames.device, dtype=frames.dtype)
        for k in range(NF):
            out[..., k * hop : k * hop + win] = (
                out[..., k * hop : k * hop + win] + frames[..., k, :]
            )
        # COLA normalisation: divide by per-position sum of squared windows.
        h2 = h * h  # [win]
        norm_map = torch.zeros(out_len, device=frames.device, dtype=frames.dtype)
        for k in range(NF):
            norm_map[k * hop : k * hop + win] = norm_map[k * hop : k * hop + win] + h2
        out = out / norm_map.clamp_min(1e-8)
        if pad_w > 0:
            out = out[..., :original_W]
        return out

    # ------------------------------------------------------------------
    # Bounded complex filter from raw 2C output.
    # ------------------------------------------------------------------

    def _filter(self, raw: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """Build bounded complex filter from raw (g_raw, theta_raw).

        ``raw`` shape: [B, 2C, S, Wf] where S = spatial collapsed dim.
        ``freq`` shape: [B, 1, S, Wf] in [0, 0.5].
        Returns complex filter [B, C, S, Wf].
        """
        g_raw, theta_raw = raw.chunk(2, dim=1)
        gain_dev = self.gain_scale * torch.tanh(g_raw)
        theta = math.pi * torch.tanh(theta_raw)
        if self.hf_only:
            # Linear HF ramp in [0, 1]: 0 at DC, 1 at Nyquist. Suppresses
            # deviation from identity at low frequencies.
            hf = 2.0 * freq  # [B, 1, S, Wf]
            gain_dev = gain_dev * hf
            theta = theta * hf
        gain = 1.0 + gain_dev
        return torch.complex(gain * theta.cos(), gain * theta.sin())

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_l.shape
        dtype = x_l.dtype
        windowed = self.window > 0 and self.window < W

        with torch.amp.autocast("cuda", enabled=False):
            xl_f = x_l.float()
            xr_f = x_r.float()

            if windowed:
                # STFT along width: [B, C, H, NF, Wf]
                F_l, NF, pad_w, W_orig = self._stft(xl_f)
                F_r, _, _, _ = self._stft(xr_f)
                Wf = F_l.shape[-1]
                S = H * NF
                # Collapse (H, NF) into a single spatial axis for 1×1-conv processing.
                F_l_c = F_l.reshape(B, C, S, Wf)
                F_r_c = F_r.reshape(B, C, S, Wf)
            else:
                F_l = torch.fft.rfft(xl_f, dim=-1, norm="ortho")  # [B, C, H, Wf]
                F_r = torch.fft.rfft(xr_f, dim=-1, norm="ortho")
                NF = 1
                pad_w = 0
                W_orig = W
                Wf = F_l.shape[-1]
                S = H
                F_l_c, F_r_c = F_l, F_r

            # Cross-power phase difference (no atan2).
            cp = F_l_c * F_r_c.conj()
            cp_mag = cp.abs() + self.log_eps
            cos_dphi = cp.real / cp_mag
            sin_dphi = cp.imag / cp_mag

            log_mag_l = (F_l_c.abs() + self.log_eps).log()
            log_mag_r = (F_r_c.abs() + self.log_eps).log()

            # Normalised frequency embedding.
            freq = torch.linspace(0.0, 0.5, Wf, device=xl_f.device, dtype=xl_f.dtype)
            freq = freq.view(1, 1, 1, Wf).expand(B, 1, S, Wf)

            # Predict raw filter params for each direction.
            cond_l = torch.cat([cos_dphi, sin_dphi, log_mag_l, log_mag_r, freq], dim=1)
            A_l = self._filter(self.align_l(cond_l), freq)

            cond_r = torch.cat([cos_dphi, -sin_dphi, log_mag_r, log_mag_l, freq], dim=1)
            A_r = self._filter(self.align_r(cond_r), freq)

            # Warp other view onto self in the spectral domain.
            F_r_to_l_c = F_r_c * A_l
            F_l_to_r_c = F_l_c * A_r

            if windowed:
                # Reshape back to [B, C, H, NF, Wf] for iSTFT.
                F_r_to_l = F_r_to_l_c.reshape(B, C, H, NF, Wf)
                F_l_to_r = F_l_to_r_c.reshape(B, C, H, NF, Wf)
                x_r_to_l = self._istft(F_r_to_l, NF, pad_w, W_orig)
                x_l_to_r = self._istft(F_l_to_r, NF, pad_w, W_orig)
            else:
                x_r_to_l = torch.fft.irfft(F_r_to_l_c, n=W, dim=-1, norm="ortho")
                x_l_to_r = torch.fft.irfft(F_l_to_r_c, n=W, dim=-1, norm="ortho")

        x_r_to_l = x_r_to_l.to(dtype)
        x_l_to_r = x_l_to_r.to(dtype)

        # Spatial gate: per-pixel confidence that the warp is reliable.
        sgate_l = torch.sigmoid(self.sgate_l(torch.cat([x_l, x_r_to_l], dim=1)))
        sgate_r = torch.sigmoid(self.sgate_r(torch.cat([x_r, x_l_to_r], dim=1)))

        out_l = x_l + self.gamma_l * sgate_l * (x_r_to_l - x_l)
        out_r = x_r + self.gamma_r * sgate_r * (x_l_to_r - x_r)
        return out_l, out_r

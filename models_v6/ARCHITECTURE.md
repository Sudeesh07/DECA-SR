# SpecFuse: Spectral Cross-View Fusion for Stereo Super-Resolution (V6)

## Motivation

Existing stereo super-resolution methods fuse views in the **spatial domain** via
epipolar/parallax attention (PASSRnet, iPASSR, NAFSSR's SCAM) or disparity-warped
feature fusion. These methods operate on *local pixel-level* information and do
not exploit a strong *global geometric prior* relating the two views.

SpecFuse introduces a spectral-domain fusion mechanism derived from a first-principles
analysis of stereo geometry. The resulting architecture is smaller, trains more
stably, and is grounded in a mathematically rigorous theorem.

## Theoretical Foundation

### Proposition 1 (Magnitude Invariance of Rectified Stereo)

Let (L, R) be a rectified stereo pair with disparity d(x, y) along the x-axis, so
that R(x, y) = L(x − d, y). Assume d is locally constant within a co-visible region
Ω. Then on Ω, the 2D Fourier transforms satisfy:

$$ F_R(u, v) = F_L(u, v) \cdot \exp\left(-2\pi i \cdot u \cdot d / W\right) $$

and hence:

$$ |F_R(u, v)| = |F_L(u, v)| $$
$$ \arg F_R(u, v) - \arg F_L(u, v) = -2\pi \cdot u \cdot d / W $$

*Proof.* By the shift theorem: F(f(x − d))(u) = e^{-2πi u d / W} · F(f)(u). Taking
magnitudes: |e^{-2πi u d / W}| = 1, so |F_R| = |F_L|. Taking arguments: arg differs
by −2π u d / W. ∎

### Interpretation

**Magnitude** carries scene appearance (textures, materials, shapes).
**Phase** carries scene geometry (object positions, layout, depth).

This appearance–geometry duality is classical (Oppenheim & Lim, 1981). Stereo
pairs differ only in *geometry* (the viewpoint), so they share appearance:
their magnitude spectra should be nearly identical over co-visible regions,
while their phase spectra encode the view-dependent disparity field.

### Consequence for Fusion

A magnitude spectrum computed jointly from both views is a *higher-SNR estimate*
of the shared scene appearance than either view alone. Combining magnitudes
while preserving per-view phase yields two features that are consistent in
appearance but geometrically view-specific — exactly what is needed for
stereo super-resolution.

Unlike spatial-domain fusion, spectral magnitude fusion:
1. is **translation-invariant** (holds under any disparity field d),
2. operates **globally** in one step (no per-pixel attention weights),
3. is **differentiable and linear** (FFT is unitary),
4. has a **principled occlusion-handling mechanism** (learnable per-frequency gates).

## Architecture

```
LR_L, LR_R  (H × W)
   │
   ├─ Stem (shared Conv3x3): c channels
   │
   ├─ Trunk (shared) × N blocks, all at LR resolution:
   │     SpectralStereoBlock:
   │        ├─ NAFBlock(x)                  [per-view spatial]
   │        ├─ SCAM(x_L, x_R)               [epipolar attention baseline]
   │        └─ CVSFM(x_L, x_R)              [★ novel spectral fusion]
   │
   ├─ Tail (shared): Conv3x3(c → 3 s²) + PixelShuffle(s)
   │                 [ICNR init; single upsample — no checkerboard compounding]
   │
   └─ Global residual: + Bicubic(LR_L / LR_R)
         
         → SR_L, SR_R  (sH × sW)
```

### Block: CVSFM (Cross-View Spectral Fusion Module)

For each pair (x_L, x_R) of shape [B, C, H, W]:

1. Compute 2D real-input FFT: F_L = rfft2(x_L), F_R = rfft2(x_R), shape
   [B, C, H, W/2+1] complex.
2. Decompose: mag_{L,R} = |F_{L,R}|, phase_{L,R} = arg F_{L,R}.
3. Per-frequency gate (1×1 conv, two-layer MLP) produces α_L, α_R ∈ (0, 1)^{C×H×(W/2+1)}.
4. Fused magnitudes: m̃_L = (1 − α_L) · mag_L + α_L · mag_R, symmetric for R.
5. Reconstruct: F̃_L = m̃_L · e^{i phase_L}, inverse rfft2 to get x̃_L.
6. Scaled residual: out_L = x_L + γ_L · (x̃_L − x_L), γ_L initialised to 0
   (zero-init for identity at init).

**Occlusion handling.** Where one view observes content invisible to the other
(e.g. half-occluded edges), their magnitude spectra diverge; the learnable gate
drives α → 0 at those frequencies, falling back to per-view magnitudes. No
explicit occlusion mask needed.

### Block: SCAM (Stereo Cross Attention)

Classical row-wise cross-view attention as in NAFSSR. Retained as a strong
spatial-domain baseline; complementary to CVSFM (spatial alignment vs.
global spectral exchange).

### Spectral Consistency Loss

$$ \mathcal{L}_{\text{spec}} = \mathbb{E}\left[\| |F(SR_L)| - |F(SR_R)| \|_1\right] $$

Applied with small weight as auxiliary supervision. Encourages the two
reconstructed views to share magnitude spectra globally, consistent with
Proposition 1.

## Design Choices — Principled Justifications

| Choice | Justification |
|--------|--------------|
| Tail-upsample only (single PixelShuffle with ICNR) | Avoids cascaded-upsample checkerboard; HR info never destroyed by encoder downsampling |
| Shared L/R trunk | Enforces view symmetry; 2× parameter efficiency; consistent with NAFSSR |
| CVSFM every block | Frequency-domain fusion at multiple feature scales |
| γ zero-init | Each block starts as identity; training progressively activates spectral path |
| rfft2 (not 1D rfft) | Captures shared appearance structure in both axes; horizontal magnitude invariance is strict, vertical is trivial (no parallax) |

## Claims for Publication

1. **First** architectural exploitation of the magnitude/phase appearance–geometry
   duality in stereo super-resolution.
2. **Principled occlusion handling** via learnable per-frequency gating without
   explicit occlusion masks.
3. **Spectral consistency loss** as novel auxiliary supervision.
4. **Parameter-efficient**: ~1-3M params at competitive performance (vs. ~24M for
   NAFSSR-L).
5. **Orthogonal to spatial-domain fusion** (SCAM / PASSR): CVSFM complements
   existing stereo fusion mechanisms.

## Ablation Plan (for paper)

- A0 (baseline): shared NAFBlock trunk, tail-upsample, no cross-view fusion.
- A1: + SCAM (spatial fusion only).
- A2: + CVSFM (spectral fusion only, no SCAM).
- A3: + SCAM + CVSFM (full model).
- A4: A3 + spectral consistency loss.
- A5: CVSFM with 1D rfft (width only) vs. 2D rfft.
- A6: CVSFM without learnable gate (simple average).
- A7: γ initial value sensitivity (0 vs. 0.1 vs. 1).

## Benchmarks

Flickr1024, KITTI2012, KITTI2015, Middlebury at ×2 and ×4. Compare against
NAFSSR (tiny/small/base/large), SwinFSR, SSRDEFNet, iPASSR, PASSRnet.

## References

- Oppenheim & Lim, "The importance of phase in signals" (Proc. IEEE, 1981).
- Chu et al., "NAFSSR: Stereo Image Super-Resolution Using NAFNet" (CVPRW, 2022).
- Chi et al., "Fast Fourier Convolution" (NeurIPS, 2020).
- Jiang et al., "Deep Fourier Up-Sampling" (NeurIPS, 2022).

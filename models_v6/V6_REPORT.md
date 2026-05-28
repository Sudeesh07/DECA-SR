# Stereo SR V6 — Full Technical Report

**Codebase:** `models_v6/`
**Status as of:** 2026-05-25
**Primary task:** Stereo image super-resolution at ×2/×3/×4 on Flickr1024, KITTI2012, KITTI2015, Middlebury.

This document covers everything in V6: architecture (every module, every constructor argument), the full training pipeline (DDP, AMP, scheduler, robustness hooks), losses, the inference harness (including TLSC), ablation configurations actually run, measured results, and the post-mortem on which theoretical claims survived contact with data.

---

## 1 — Executive summary

V6 is a shared-weight, LR-resolution trunk for stereo SR with a single tail upsample. Each trunk block has three sub-modules: per-view NAFBlock (spatial), SCAM (row-wise epipolar cross-attention), and CVSFM (cross-view spectral fusion). The model is parameter-efficient (1.4–2.1 M params for "small") and trains stably under bf16 DDP.

**Best measured result on Flickr1024 val (×4):** **23.697 dB** (SCAM-only, 100k iters, batch 64 effective).
**Best measured result on Flickr1024 test (×4):** **23.811 dB** (same configuration).

**Empirical verdict on the original spectral path (v1):** the magnitude-invariance mechanism contributes within noise. SCAM is doing essentially all the cross-view work in current ablations. The mechanical reason — diagnosed during this work — is that v1 mixes magnitudes (which are already approximately equal between views under the shift theorem) while preserving each view's own phase (which is the part that actually distinguishes them). The module is structurally a no-op for cross-view information transfer.

**Three CVSFM variants now live in the codebase:**

| Version | Operator | Status |
|---|---|---|
| **v1** | Convex magnitude mix `α·m_R + (1-α)·m_L`, own phase preserved | empirically inert (A2 ≈ A0, A6 ≈ A1) |
| **v2** | Log-magnitude additive residual, per-pixel spatial gate, own phase preserved | better magnitude operator but still no phase signal; not yet trained |
| **v3** | Phase-correlation cross-view alignment: **bounded** complex filter `A = gain·exp(jθ)` predicted from `[cos Δφ, sin Δφ, log\|F_L\|, log\|F_R\|, freq]`, warp `F_other · A`, per-pixel spatial gate. **Hann window + 50 % overlap-add (WOLA)** when windowed. Optional **HF-only** restriction (forces `A = 1` at DC). | structurally capable of cross-view transfer; not yet trained |

v3 is the recommended path going forward. The spectral consistency loss (derived from v1's now-falsified theorem) is **disabled by default**.

---

## 2 — Architecture

### 2.1 Macro design

```
LR_L, LR_R  (H × W)
   │
   ├─ Stem (shared Conv3×3): channels
   │
   ├─ Trunk (shared) × N blocks, all at LR resolution:
   │     SpectralStereoBlock:
   │        ├─ NAFBlock(x)                                [per-view, shared]
   │        └─ (SCAM ↔ CVSFM)                             [order is configurable]
   │
   ├─ trunk_norm (LayerNorm2d)
   │
   ├─ Tail (shared): Conv3×3(c → 3·s²) + PixelShuffle(s)
   │                 [ICNR init, init_scale = 0.01]
   │
   └─ + Bicubic(LR_L) / Bicubic(LR_R)
         → SR_L, SR_R  (sH × sW)
```

Three design principles drive every choice:

1. **Trunk operates entirely at LR resolution.** No encoder/decoder, no cascaded PixelShuffle. The HR representation is constructed only at the tail via a single sub-pixel convolution. This avoids the cascaded-upsample checkerboard problem that affected V1–V5.
2. **Shared weights across views.** Both L and R pass through the same `stem` and the same trunk block list. Halves parameter count vs separate towers; enforces view symmetry.
3. **Zero-init every residual.** Each block starts as the exact identity: NAFBlock's β,γ = 0; SCAM's β,γ = 0; CVSFM's γ = 0. At step 0 the network output is exactly `bicubic(LR)`. Training progressively activates each residual path.

Source: [models_v6/network_v6.py](models_v6/network_v6.py)

### 2.2 Stem

A single `nn.Conv2d(in_channels=3, channels, kernel=3, padding=1)` applied identically to both views. No norm, no activation. ~1700 params at `channels=64`.

[network_v6.py:97](models_v6/network_v6.py#L97)

### 2.3 SpectralStereoBlock

The trunk's repeating unit. Constructor:

```python
SpectralStereoBlock(
    channels: int,
    dw_expansion: float = 2.0,
    ffn_expansion: float = 2.0,
    use_scam: bool = True,
    use_cvsfm: bool = True,
    cvsfm_use_gate: bool = True,       # A6 sets this False
    cvsfm_use_1d_fft: bool = False,    # A5 sets this True
    cvsfm_version: str = "v1",         # "v2" or "v3" select alternative spectral paths
    cvsfm_v3_window: int = 0,          # window size for v3's 1D rfft (0 = full row)
    cvsfm_first: bool = False,         # if True, run CVSFM before SCAM
)
```

Forward, conditional on `cvsfm_first`:

```python
x_l, x_r = naf(x_l), naf(x_r)
if cvsfm_first:
    x_l, x_r = cvsfm(x_l, x_r)   if cvsfm is not None
    x_l, x_r = scam(x_l, x_r)    if scam is not None
else:
    x_l, x_r = scam(x_l, x_r)    if scam is not None
    x_l, x_r = cvsfm(x_l, x_r)   if cvsfm is not None
```

The recommended configuration for v3 is `cvsfm_first=True` so v3 acts as the primary cross-view module and SCAM (if present) refines after.

[network_v6.py:48-104](models_v6/network_v6.py#L48-L104)

### 2.4 NAFBlock — per-view spatial processor

From NAFNet (Chen et al., ECCV 2022). Backbone of NAFSSR.

**Spatial path:**
```
LayerNorm2d
→ Conv1×1 (c → 2·dw_c)
→ DWConv3×3 (groups = 2·dw_c)
→ SimpleGate (split → multiply)
→ SCA: AdaptiveAvgPool2d(1) → Conv1×1 → multiply
→ Conv1×1 (dw_c → c)
→ × β (zero-init scalar per channel)
+ residual
```

**FFN path:**
```
LayerNorm2d
→ Conv1×1 (c → 2·ffn_c)
→ SimpleGate
→ Conv1×1 (ffn_c → c)
→ × γ (zero-init scalar per channel)
+ residual
```

Key tricks:
- **SimpleGate** replaces activations: split tensor along channels into (a, b), output `a * b`. No nonlinearity needed.
- **SCA (Simplified Channel Attention)**: global pool → 1×1 conv → broadcast multiply. Per-channel modulation conditioned on whole-feature-map statistics.
- **β, γ zero-init scalars**: each block starts as identity. Training lifts them off zero.

Default expansions: `dw_expansion=2.0`, `ffn_expansion=2.0`. At `channels=64`: NAFBlock has ~70K params.

[blocks.py:37-86](models_v6/blocks.py#L37-L86)

### 2.5 SCAM — Stereo Cross-Attention Module

Row-wise (epipolar) cross-view attention from NAFSSR (Chu et al., CVPRW 2022).

For each row y independently:
- Q_L = q_l(norm_l(x_L)), Q_R = q_r(norm_r(x_R))
- V_L = v_l(x_L), V_R = v_r(x_R)
- `attn = softmax(Q_L · Q_Rᵀ / sqrt(C))` along width
- `F_r2l = attn · V_R`, `F_l2r = attnᵀ · V_L`
- `out_L = x_L + β · F_r2l`, `out_R = x_R + γ · F_l2r`

β and γ are zero-init per-channel scalars → block is identity at step 0.

Implementation detail: Q·Kᵀ and softmax are forced into **fp32** under `autocast(enabled=False)` because fp16's ~65k dynamic range overflows on long-tail Q·Kᵀ values; once `exp()` produces `inf`, softmax → NaN and corrupts weights via backward.

Param cost at `channels=64`: ~17K params (4 × 1×1 convs + 2 LayerNorm2d + 2 scalar params per channel).

[epipolar_attention.py:20-64](models_v6/epipolar_attention.py#L20-L64)

### 2.6 CVSFM v1 — magnitude mixing (falsified)

Original module. Mixes magnitudes across views, preserves each view's phase. **Empirical contribution: −0.004 dB on top of SCAM** (A6 vs A1).

**Mechanical reason for failure**: under the 1D shift theorem `F_R(ω) = F_L(ω)·exp(−jωd)`, magnitudes are already approximately equal `|F_R| ≈ |F_L|`. v1 mixes the part the views *agree* on while preserving each view's *own* phase — the part where the cross-view geometric signal actually lives. Even with the gate fully open, fused_mag ≈ own_mag, so the output ≈ input. The module is structurally a no-op.

**Per-block flow** (all FFT/IFFT in fp32 under `autocast(enabled=False)`):
1. `F_L, F_R = rfft2(x_L, x_R)` (or 1D rfft along width if `use_1d_fft=True`)
2. `mag_L, mag_R = |F_L|, |F_R|`; `phase_L, phase_R = ∠F_L, ∠F_R`
3. **Gate** (if `use_gate=True`): `α_L, α_R = sigmoid(gate(concat(mag_L, mag_R)))`. Gate is a 2-layer 1×1 conv MLP `(2C → hidden → 2C)`. Final layer init: weight=0, bias=−5, so at step 0 `α ≈ sigmoid(−5) ≈ 0.007`.
4. **Mix**: `fused_mag_L = α_L · mag_R + (1−α_L) · mag_L`.
5. `F_L_new = polar(fused_mag_L, phase_L)`.
6. Reconstruct: `x_L_new = irfft2(F_L_new)`.
7. Residual: `out_L = x_L + γ_L · (x_L_new − x_L)`, `γ_L` zero-init per-channel.

**A5** sets `use_1d_fft=True`; **A6** sets `use_gate=False` (α hard-coded to 0.5).

Param cost at `channels=64`, `hidden_expansion=0.5`: ~8.4K per block. Retained for ablation comparison.

[spectral_fusion.py:50-141](models_v6/spectral_fusion.py#L50-L141)

### 2.7 CVSFM v2 — log-magnitude residual + spatial gate

A better magnitude operator: replaces v1's convex mix with an additive residual in log-magnitude space (generative; can synthesize energy outside the `[min, max]` bound), and adds a per-pixel image-domain spatial gate (localised occlusion handling). 1D rfft along width.

**But v2 still does not address the central problem**: it operates on magnitudes (already approximately equal between views) and never touches the phase difference. v2 is therefore expected to behave like a stronger version of v1 — it cannot transfer the cross-view geometric signal by construction. Retained as a stepping stone in the codebase, but **superseded by v3**.

Per-block flow:
1. `F_L, F_R = rfft(x_L, x_R, dim=-1)` (1D along width)
2. `log_mag_L = log(|F_L| + ε)`, similarly `log_mag_R`. `ε = 1e-6`.
3. `Δ_L = delta_l(concat(log_mag_L, log_mag_R, freq))`. `delta_l` is `Conv1×1(2C+1 → hidden) → GELU → Conv1×1(hidden → C)`. Final layer kaiming × 0.01 init so `Δ ≠ 0` but small.
4. `new_log_mag_L = log_mag_L + Δ_L`. Clamp to `[−30, 30]` before `exp()`.
5. `F_L_new = polar(new_log_mag_L.exp(), phase_L)`. Own phase preserved.
6. `x_L_new = irfft(F_L_new, n=W, dim=-1)`.
7. Spatial gate: `sgate_L = sigmoid(sgate_l(concat(x_L, x_R)))` — DWConv3×3 + 1×1, zero-init → `sigmoid(0) = 0.5` (gate open at init).
8. Output: `out_L = x_L + γ_L · sgate_L · (x_L_new − x_L)`.

Identity at init verified by smoke test: `max|out − x| = 0`, `max|∂loss/∂γ| = 4.3e-1`. Param cost: ~32K per block at `channels=64`.

[spectral_fusion.py:144-303](models_v6/spectral_fusion.py#L144-L303)

### 2.8 CVSFM v3 — phase-correlation alignment (current recommendation)

**v3 is the structurally correct cross-view module.** Operates on inter-view phase difference (the actual stereo signature under the shift theorem) rather than magnitudes (which carry no cross-view geometric information).

After a second-round critique flagged engineering risks in the initial v3 design (unconstrained complex filter; rectangular non-overlapping windows producing Gibbs ringing), v3 now includes three safety guards: **bounded filter parametrisation**, **Hann window + 50 % overlap-add (WOLA)**, and an optional **HF-only** restriction. All three are documented below.

**Mechanism, per view-pair, per block:**

1. **1D STFT along width.** When `window = 0`: full-row rfft, no tapering. When `window > 0`: overlapping Hann-tapered frames at hop = window/2 (analysis window), then rfft per frame.
2. **Cross-power phase difference** without `atan2`:
   ```
   cp = F_L · conj(F_R)
   ĉp = cp / (|cp| + ε)             # unit-modulus complex
   cos Δφ = ĉp.real,  sin Δφ = ĉp.imag
   ```
   No `atan2(0,0)` NaN path. Phase wrap-around handled by the `(cos, sin)` continuous embedding.
3. **Predict raw filter** `(g_raw, θ_raw)` per (channel, frame, frequency) bin from `[cos Δφ, sin Δφ, log|F_L|, log|F_R|, freq]` (4C + 1 channels). 1×1 conv MLP: `Conv1×1(4C+1 → hidden) → GELU → Conv1×1(hidden → 2C)`.
4. **Bounded complex filter** (the critical safety fix):
   ```
   gain  = 1 + gain_scale · tanh(g_raw)        ∈ [1 − s, 1 + s]
   theta = π · tanh(θ_raw)                     ∈ (−π, π)
   A     = gain · (cos θ + j sin θ)
   ```
   Default `gain_scale = 0.1` → |A| ∈ [0.9, 1.1]. Worst-case magnitude amplification is fixed; phase rotation is naturally bounded by tanh. The previous unbounded `A = real + j·imag` parametrisation could amplify frequencies arbitrarily and rotate phase arbitrarily — a real instability risk.
5. **Optional HF-only** (`hf_only = True`): scales `(gain_dev, theta)` by a linear ramp `2 · freq` so DC and low frequencies are forced to `A = 1` (no deviation from identity). High frequencies, where SR reconstruction lives, retain the full bounded range. Low-frequency structure is left to the bicubic residual + per-view processing.
6. **Warp other view onto self** in the spectral domain:
   ```
   F_R→L = F_R · A_L
   F_L→R = F_L · A_R         (conditioning has sin Δφ negated, log_mag's swapped — symmetric reflection)
   ```
7. **Inverse STFT** (windowed mode): per-frame irfft → Hann synthesis window → overlap-add → divide by per-position sum-of-squared-windows (COLA normalisation). Full-row mode: direct irfft.
8. **Per-pixel image-domain spatial gate** from `concat(x_self, x_other_warped)` — DWConv3×3 + 1×1 → sigmoid. Localises the residual; occlusion handling lives here.
9. **Residual** with γ per-channel zero-init scale: `out_L = x_L + γ_L · sgate_L · (x_R→L − x_L)`.

**Identity-at-init contract** (verified by smoke test, all 5 config variants):
- γ_L, γ_R = 0 exactly → output equals input exactly.
- Alignment final layer: weights kaiming × `align_init_scale` (default 0.01), **bias = 0**. With bounded parametrisation: `tanh(0) = 0` → `gain = 1`, `theta = 0` → `A = 1` exactly at init. No bias trick needed (the new parametrisation gives identity for free).
- `∂out/∂γ = sgate · (x_other_warped − x_self) ≠ 0` because `x_other ≠ x_self` generically. Measured: `max|∂loss/∂γ_L| ≈ 1.17–1.19e+02` at init across all configs.

**Filter-bound invariant** (verified): when the alignment net is driven to saturation (weights = 10, bias = 5), the output filter `|A|` is exactly in [0.9, 1.1] for `gain_scale = 0.1`. The unbounded earlier version had no such guarantee.

**HF-only invariant** (verified): with `hf_only = True`, even at network saturation, `|A − 1|` at DC is exactly 0 (within fp precision), while `|A − 1|` at Nyquist is the full unrestricted bounded range. The frequency mask works as designed.

**No semantic claim** about "magnitude = appearance" or "phase = geometry". v3 is justified mechanically: a bounded learned phase-correlation filter, the Fourier-domain analogue of cross-view alignment, where the cross-view signal demonstrably resides.

**Param cost**: ~44K per block at `channels=64`. The `window`, `hf_only`, and `gain_scale` flags do not change parameter count — they change FFT length, the filter constraint, or a frequency mask. Total V6-small with v3: 2.117 M params (+36.9 % vs v1's 1.546 M).

[spectral_fusion.py:306+](models_v6/spectral_fusion.py#L306) — current implementation.

**What v3 still does not handle**:
- **Hard depth discontinuities within a window**: a single phase ramp per window cannot represent foreground + background shifting by different amounts. The spatial gate is supposed to mask such regions, but it's still load on the gate. This is the main reason `windowed v3 + SCAM (cvsfm_first)` is the recommended configuration: SCAM handles disparity discontinuities natively; v3 handles sub-pixel alignment within smooth-disparity windows.
- **Vertical misalignment from imperfect rectification**: strict 1D rfft along width assumes perfect epipolar geometry. KITTI in particular has minor vertical bleed. SCAM's 2D-convolutional context absorbs this; v3 cannot. Empirical risk, more on KITTI than Flickr1024.
- **Non-translational geometry** (slanted surfaces, perspective): phase correlation excels at pure shift; foreshortening makes the model approximate. Spatial gate has to mask.

### 2.9 TailUpsample — single PixelShuffle with ICNR

```
Conv3×3(channels → 3·scale²) → PixelShuffle(scale)
```

**ICNR initialization** (Aitken et al., 2017): each of the `scale × scale` sub-pixels in a block normally receives independent random weights, producing a checkerboard pattern at init. ICNR initializes one sub-pixel and repeats its kernel `scale²` times → smooth nearest-neighbour upscale at init.

**Critical extra scaling** (`init_scale = 0.01`): after ICNR, weights are multiplied by 0.01. Without this, the tail output has std ≈ 1.4 per pixel at init, which dominates the bicubic baseline and produces nonsense initial SR (PSNR ~8 dB, pixel loss ~10). With it, the initial residual is ~0.014 per pixel, so `SR_init ≈ bicubic(LR)` and the network trains *up* from the bicubic baseline (~24 dB at step 0).

Pure zero-init would break gradient flow to the trunk. The small non-zero scale keeps gradients flowing while starting close to the identity-to-bicubic map.

[blocks.py:89-152](models_v6/blocks.py#L89-L152)

### 2.10 Bicubic global residual

```python
up = F.interpolate(lr_in, scale_factor=scale, mode="bicubic", align_corners=False)
sr = tail(trunk(stem(lr))) + up
```

The trunk learns *only the HR residual* over bicubic. Combined with the small tail `init_scale`, this means SR ≈ bicubic(LR) at step 0. Validation PSNR at iter 0 of a typical run: ~20 dB.

[network_v6.py:166-170](models_v6/network_v6.py#L166-L170)

### 2.11 LayerNorm2d

Channel-wise LN operating on `[B, C, H, W]` directly (rather than reshaping). Per-channel learned weight and bias. Used twice in NAFBlock, twice in SCAM, once at trunk output.

[blocks.py:13-26](models_v6/blocks.py#L13-L26)

### 2.12 Size presets

From [config_v6.py:8-13](models_v6/config_v6.py#L8-L13):

| Preset | channels | num_blocks | Approx params (small=default at scale=4) |
|---|---|---|---|
| tiny  | 48 | 16 | ~0.8 M |
| **small** | **64** | **16** | **1.546 M (v1 full), 1.918 M (v2 full), 2.117 M (v3 full)** |
| base  | 64 | 24 | ~2.3 M (v1) |
| large | 96 | 32 | ~6.9 M (v1) |

Measured "small" param counts across ablations:

| Config | Params |
|---|---|
| A0 (no SCAM, no CVSFM) | 1.138 M |
| A1 (SCAM only) | 1.410 M |
| A2 (CVSFM v1 only) | 1.273 M |
| A4 (full v1) | 1.546 M |
| Full v2 (SCAM + CVSFM v2) | 1.918 M |
| Full v3 (SCAM + CVSFM v3, any window) | 2.117 M |

---

## 3 — Loss functions

Composite loss in [losses_v6.py](models_v6/losses_v6.py):

```python
total = pixel + stereo_weight · stereo + spectral_weight · spectral
```

Defaults: `stereo_weight = 0.02`, **`spectral_weight = 0.0`** (disabled). Each sub-loss is skipped entirely if its weight is 0 — FFTs in `SpectralConsistencyLoss` are not computed in the default config.

### 3.1 CharbonnierLoss

`sqrt((pred − target)² + ε²)`, ε = 1e-3. Robust to outliers vs L2 while remaining smooth at 0 unlike L1.

Applied symmetrically: `pixel = 0.5 · (Charbonnier(SR_L, HR_L) + Charbonnier(SR_R, HR_R))`.

[losses_v6.py:20-27](models_v6/losses_v6.py#L20-L27)

### 3.2 StereoPhotometricConsistency

Disparity-free stereo consistency from iPASSR. For each candidate horizontal shift in `(0, max_shift/4, max_shift/2, max_shift)` (default `max_shift = 48`), compute `|SR_L − shift(SR_R)|`; take the per-pixel minimum across shifts; average. Approximates a "best-match" disparity term without an explicit disparity prediction.

This loss is compatible with v3 (it operates on SR outputs, not features). Kept on by default at small weight.

[losses_v6.py:30-55](models_v6/losses_v6.py#L30-L55)

### 3.3 SpectralConsistencyLoss — disabled by default

`L1(|F(SR_L)|, |F(SR_R)|)` with optional high-frequency emphasis. Derived from v1's strict magnitude-equality theorem, which ablations A2 and A6 falsified.

**Why disabled**: it penalises exactly the magnitude differences that real occlusion, photometric mismatch, and the DFT boundary term produce — quantities the architecture (especially v3's spatial gate) is supposed to handle gracefully. Default `--spectral_weight 0.0` means the loss is never instantiated.

Kept in the code so older experiment scripts (`--spectral_weight 0.01`) still run.

[losses_v6.py:58-82](models_v6/losses_v6.py#L58-L82)

### 3.4 StereoSRV6Loss composite

[losses_v6.py:85-126](models_v6/losses_v6.py#L85-L126). Returns `(total, loss_dict)` so each component is logged separately. Sub-losses are skipped entirely if their weight is 0.

---

## 4 — Training pipeline

Entry point: `models_v6.train_ddp_v6.train(args)` ([train_ddp_v6.py:219-505](models_v6/train_ddp_v6.py#L219-L505)).

### 4.1 DDP setup

Standard `torchrun`-based DDP with NCCL backend. Wrapped in `DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)`. This is safe for all three CVSFM versions: even with γ=0 init, the autograd graph is built through all parameters (gradient is zero, not None), so DDP's bucket allocation works without the expensive unused-param traversal.

NCCL timeouts extended to 1 hour via env vars ([train_ddp_v6.py:112-114](models_v6/train_ddp_v6.py#L112-L114)) — long validation passes on rank 0 would otherwise trip the default 10/30-min timeout while rank 1 waits at the next collective.

### 4.2 LR scaling and warmup

Linear LR scaling for the effective batch size:
```python
effective_bs = args.batch_size * world_size
scaled_lr = args.lr * (effective_bs / 8.0)
```

Base LR is calibrated for effective batch 8. With default `lr=2e-4` and 2× GPU batch_size=32, actual LR = `2e-4 × 8 = 1.6e-3`.

**Schedule**: linear warmup (default 2000 iters) then cosine decay to 1e-6 of scaled LR.

[train_ddp_v6.py:144-151](models_v6/train_ddp_v6.py#L144-L151)

### 4.3 AMP — bf16 preferred over fp16

Three modes:
- **bf16** (recommended): bf16 dtype, no GradScaler. fp32-equivalent dynamic range, eliminates fp16 inf/NaN overflows that historically killed long V4/V5 runs ~30–50k iters in.
- **fp16** (legacy): fp16 dtype, requires GradScaler with loss scaling.
- **fp32** (`--no_fp16` and no `--bf16`).

Inside the model, specific ops force fp32: FFTs in CVSFM v1/v2/v3 and Q·Kᵀ + softmax in SCAM.

[train_ddp_v6.py:293-309](models_v6/train_ddp_v6.py#L293-L309)

### 4.4 NaN/Inf guards

After loss computation:
```python
finite_flag = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
if world_size > 1:
    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
if finite_flag.item() < 1.0:
    # skip step on ALL ranks
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    continue
```

The all-reduce is critical: if rank 0 skips and rank 1 doesn't, DDP collectives desynchronize and the run hangs at the next reduce.

[train_ddp_v6.py:392-404](models_v6/train_ddp_v6.py#L392-L404)

### 4.5 Dataloader robustness

Per-batch retry loop (up to 5 attempts) absorbs single corrupt-PNG failures without killing a 400k-iter run; aborts only after 50 consecutive failures ([train_ddp_v6.py:344-368](models_v6/train_ddp_v6.py#L344-L368)). Validation has a similar OOM safety net for large Flickr1024 HR samples.

### 4.6 Shared-memory guard

Docker containers default to **64 MB /dev/shm**, too small for DataLoader IPC with workers > 0. The `_maybe_force_inprocess_loading` function ([train_ddp_v6.py:61-87](models_v6/train_ddp_v6.py#L61-L87)) estimates per-worker memory and falls back to `num_workers=0` if /dev/shm is too small. Avoids SIGBUS ~10 minutes into runs.

### 4.7 Validation

Runs on rank 0 only every `val_interval` (default 2500). Gating condition must be identical on every rank so all ranks reach the post-val barrier together — earlier bug caused rank 1 (where `val_loader = None`) to skip the barrier and deadlock the next collective. Fixed at [train_ddp_v6.py:447-467](models_v6/train_ddp_v6.py#L447-L467).

### 4.8 Checkpointing

- **Best**: saved on every val improvement at `checkpoints/best.pth`.
- **Periodic**: every `save_interval` (default 5000) at `checkpoints/iter_{N}.pth`. Pruned to the most recent `keep_ckpts` (default 5).
- **Final**: `checkpoints/final.pth`.

Each checkpoint stores model state dict, optimizer, scheduler, iteration, best PSNR.

---

## 5 — Inference pipeline

Entry point: `models_v6.test_v6.test(args)` ([test_v6.py](models_v6/test_v6.py)).

### 5.1 Test harness

Loads checkpoint, optionally enables TLSC, runs validation-style loop over the test set. SR clamped to `[0, 1]` before metric computation. Per-sample PSNR/SSIM/time logged; final averages reported.

### 5.2 TLSC — Test-time Local Statistics Correction

From Chu et al., ECCV-W 2022. NAFBlock's SCA uses `AdaptiveAvgPool2d(1)` over training patches (30×90); at inference on full images (1000×500+), that pool covers ~150× the area, and the downstream 1×1 conv breaks because it never saw those channel statistics.

**Fix**: replace each NAFBlock's `AdaptiveAvgPool2d(1)` with `LocalAvgPool2d(base_size=(patch_h, patch_w))` at inference. Local windowed average matching the training patch shape. Applied via `model.enable_tlsc((H, W))` after loading weights and before `model.eval()`. Reversible.

[tlsc.py](models_v6/tlsc.py)

---

## 6 — Configuration reference

[config_v6.py](models_v6/config_v6.py) defines all CLI flags.

### Model — selection
- `--model_size {tiny, small, base, large}` (default: small)
- `--scale {2, 3, 4}` (default: 4)
- `--channels`, `--num_blocks` — override preset
- `--dw_expansion`, `--ffn_expansion` (defaults: 2.0)

### Model — cross-view modules
- `--no_scam` — disable SCAM (ablation A2)
- `--no_cvsfm` — disable CVSFM entirely (ablation A1)
- `--cvsfm_no_gate` — v1 ablation A6: hard-code α = 0.5
- `--cvsfm_1d_fft` — v1 ablation A5: 1D rfft along width
- `--cvsfm_v2` — use CVSFM v2 (log-magnitude residual)
- `--cvsfm_v3` — use CVSFM v3 (phase-correlation alignment) **recommended**
- `--cvsfm_v3_window N` — v3 windowed STFT (0 = full row, otherwise Hann + 50 % overlap-add)
- `--cvsfm_v3_hf_only` — restrict v3 to mid/high frequencies (DC forced to A=1)
- `--cvsfm_v3_gain_scale S` — v3 magnitude gain bound: `|A| ∈ [1−S, 1+S]` (default 0.1)
- `--cvsfm_first` — run CVSFM before SCAM in each block (recommended with v3)

Precedence when multiple version flags are set: `v3 > v2 > v1`.

### Data
- `--data_root datasets/StereoSR`, `--test_source {Flickr1024, KITTI2012, KITTI2015, Middlebury}`
- `--patch_h 30 --patch_w 90` (LR patch size)
- `--num_workers 4`, `--augment`

### Training
- `--batch_size 8`, `--total_iters 400000`
- `--lr 2e-4`, `--weight_decay 0.0`, `--warmup_iters 2000`, `--grad_clip 1.0`
- `--bf16` (recommended), `--fp16`, `--no_fp16`
- `--seed 42`, `--resume`, `--pretrain`

### Losses
- `--charbonnier_eps 1e-3`
- `--stereo_weight 0.02` (photometric consistency)
- **`--spectral_weight 0.0`** (default disabled — see §3.3)
- `--stereo_max_shift 48`

### Logging
- `--log_dir ./experiments`, `--exp_name`
- `--log_interval 100`, `--val_interval 2500`, `--save_interval 5000`, `--keep_ckpts 5`

### Eval
- `--ckpt`, `--save_images`
- `--tlsc`, `--tlsc_base_h`, `--tlsc_base_w`

---

## 7 — Ablation experiments and results

### 7.1 Experimental setup

All ablations: model_size=small (channels=64, num_blocks=16), scale=4, 100k iters, batch_size=32 × 2 GPUs (effective 64), bf16, LR scaling to 1.6e-3, warmup 2000 iters, cosine decay, patch 30×90.

Data: `datasets/StereoSR` (custom layout); train+val=Flickr1024, separate test sets for Flickr1024/KITTI2012/KITTI2015/Middlebury. Note training was 100k iters, not the recipe default 400k, to keep the ablation sweep tractable.

### 7.2 Completed ablations — Flickr1024

**Validation best PSNR** (over all val checkpoints during training):

| Ablation | SCAM | CVSFM | Gate | 1D rfft | Spec loss | Params | Best val PSNR |
|---|:-:|:-:|:-:|:-:|:-:|---:|---:|
| A0 baseline | ✗ | ✗ | — | — | 0.0 | 1.138 M | **23.4125** |
| A1 SCAM-only | ✓ | ✗ | — | — | 0.0 | 1.410 M | **23.6970** |
| A2 CVSFM-v1-only | ✗ | v1 | ✓ | ✗ | 0.01 | 1.273 M | **23.4439** |
| A6 SCAM + CVSFM v1 no-gate | ✓ | v1 | ✗ (α=0.5) | ✗ | 0.0 | unmeasured | **23.6935** |

**Flickr1024 test set** (full eval, no TLSC):

| Config | Test PSNR | SSIM |
|---|---:|---:|
| A0 | 23.531 | 0.7308 |
| A1 | **23.811** | **0.7462** |
| A2 | 23.558 | 0.7314 |

### 7.3 In-progress and planned ablations

| Ablation | Description | Status |
|---|---|---|
| A3 | Full v1 (SCAM + CVSFM) without spectral loss | training, partial logs only |
| A4 | Full v1 + spectral loss | training, partial logs only |
| A5 | v1 with 1D rfft (`--cvsfm_1d_fft`) | training |
| V2-only | `--no_scam --cvsfm_v2` | not yet launched |
| V2+SCAM | `--cvsfm_v2` with SCAM | not yet launched |
| **V3-only full-row** | `--no_scam --cvsfm_v3` | **not yet launched** |
| **V3-only windowed** | `--no_scam --cvsfm_v3 --cvsfm_v3_window 30` | **not yet launched** |
| **V3 + SCAM (v3 first)** | `--cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_first` | **not yet launched (recommended)** |
| V3 + SCAM, HF-only | `--cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_v3_hf_only --cvsfm_first` | not yet launched |
| V3 + SCAM, gain_scale=0.3 | `--cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_v3_gain_scale 0.3 --cvsfm_first` | not yet launched |

### 7.4 Per-ablation analysis (completed)

- **A0 → A1: SCAM adds +0.285 dB validation, +0.280 dB test.** Clean, statistically meaningful.
- **A0 → A2: CVSFM v1 alone adds +0.031 dB validation, +0.027 dB test.** Within seed-to-seed noise on a small val set. Statistically indistinguishable from no cross-view fusion.
- **A1 vs A6: SCAM + parameter-free CVSFM averaging vs SCAM alone gives −0.004 dB.** The critical comparison. Removing the gate from CVSFM reduces it to pure magnitude averaging — the closest computational analogue to Proposition 1's claim. Adds **nothing** to SCAM.

### 7.5 Theoretical claims vs empirical reality

The original v1 paper made these claims:

| Claim | Status |
|---|---|
| Strict magnitude equality `\|F_R\| = \|F_L\|` | **Falsified.** DFT circular shift ≠ linear shift; ε term is non-zero. |
| "Magnitude = appearance, phase = geometry" | **Inapplicable to features.** Oppenheim–Lim is about images. |
| First exploitation of magnitude/phase duality in stereo SR | **Mechanism produces zero measurable benefit.** Cannot ship. |
| "Orthogonal to SCAM" | **Falsified.** A6 vs A1 shows v1 is fully subsumed by SCAM. |
| "Principled occlusion handling via per-frequency gate" | **No evidence.** Per-frequency gate cannot localize spatially. |

**Why v1 was wrong, mechanically**: under the 1D shift theorem, magnitudes are already approximately equal between views; the part that distinguishes them is the *phase difference*. v1 mixes magnitudes (the part the views agree on) and preserves each view's own phase (ignoring the part that actually carries cross-view information). Even with the gate fully open, fused_mag ≈ own_mag and the output ≈ input. Structurally a no-op.

**v2 inherits this flaw** — better magnitude operator, but still no phase signal. v2 is expected to behave similarly.

**v3 addresses the root cause** by operating on the inter-view phase difference directly via a learned complex alignment filter. It is the first CVSFM variant structurally capable of transferring cross-view information.

---

## 8 — Pre-training prediction (v3, with safety guards)

Calibrated honestly against the empirical record (four CVSFM variants failed to add anything on top of SCAM) and the safety improvements (bounded filter, Hann/WOLA, HF-only):

- **v3-only full-row** (`--no_scam --cvsfm_v3`) vs A1 (SCAM-only = 23.697): ~30 % chance v3 ≥ A1. A single phase ramp per row is too rigid for spatially-varying disparity, and the cheap full-row baseline doesn't benefit from Hann/WOLA (no windowing in this config). Mostly a sanity check.
- **v3-only windowed + Hann/WOLA** (`--no_scam --cvsfm_v3 --cvsfm_v3_window 30`) vs A1: ~50 % chance v3 ≥ A1 (revised up from 45 % after WOLA — boundary artefacts that would have hurt the prior design are now absent). The decisive single-mechanism test.
- **v3 + SCAM, cvsfm-first** (`--cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_first`) vs A1: ~55–60 % chance of ≥ +0.05 dB. v3 handles sub-pixel band-limited alignment; SCAM picks up depth discontinuities, slanted-surface foreshortening, and non-translational geometry that v3 can't represent. Genuinely complementary if the two work as designed.
- **HF-only variant**: I'd expect parity or slight win vs the unrestricted version on Flickr1024 (low frequencies are well-handled by bicubic anyway), but it could underperform on KITTI where vertical misalignment puts more weight on mid frequencies. Decide after the main v3 + SCAM result.

These are not certainties. Ablation A6 puts strong negative weight on "any spectral module helps here." But v3 is the first variant where (a) the mechanism is structurally capable of carrying the cross-view signal, and (b) the engineering safety guards (bounded filter, WOLA) eliminate the most likely silent-failure modes. Both conditions change the prior meaningfully.

---

## 9 — Launch examples

**Baseline (SCAM-only, the current best on Flickr1024):**
```bash
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 100000 \
  --no_cvsfm \
  --exp_name v6_A1_scam_only_100k
```

**v3 only, windowed STFT with Hann/WOLA (recommended decisive test):**
```bash
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 100000 \
  --no_scam --cvsfm_v3 --cvsfm_v3_window 30 \
  --exp_name v6_v3only_win30_100k
```

**v3 + SCAM with v3 as primary cross-view module (highest-upside config):**
```bash
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 100000 \
  --cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_first \
  --exp_name v6_v3_plus_scam_win30_first_100k
```

**v3 with HF-only restriction (DC reserved for bicubic residual):**
```bash
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 100000 \
  --cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_v3_hf_only --cvsfm_first \
  --exp_name v6_v3_hf_plus_scam_100k
```

**v3 with relaxed gain bound (if 10 % is too restrictive):**
```bash
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 100000 \
  --cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_v3_gain_scale 0.3 --cvsfm_first \
  --exp_name v6_v3_gs03_plus_scam_100k
```

**Curriculum (pretrain v3-only, then add SCAM):**
```bash
# Stage 1: v3-only
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 50000 \
  --no_scam --cvsfm_v3 --cvsfm_v3_window 30 \
  --exp_name v6_v3only_stage1_50k

# Stage 2: load Stage 1, enable SCAM, continue
torchrun --nproc_per_node=2 -m models_v6.train_ddp_v6 \
  --model_size small --scale 4 --batch_size 32 --bf16 --num_workers 0 \
  --total_iters 100000 \
  --cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_first \
  --pretrain experiments/v6_v3only_stage1_50k/checkpoints/best.pth \
  --exp_name v6_v3_curriculum_stage2_100k
```
Note: SCAM weights are initialized fresh in stage 2 (zero-init residual scales, so SCAM starts as identity). The Stage 1 checkpoint loads non-strictly — missing SCAM keys are tolerated.

**Eval with TLSC:**
```bash
python -m models_v6.test_v6 \
  --model_size small --scale 4 \
  --ckpt experiments/v6_v3_plus_scam_win30_first_100k/checkpoints/best.pth \
  --cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_first \
  --tlsc --tlsc_base_h 30 --tlsc_base_w 90 \
  --test_source Flickr1024 \
  --exp_name eval_v3_plus_scam_Flickr1024_tlsc
```
**Important**: eval must pass the same architecture flags (`--cvsfm_v3 --cvsfm_v3_window 30 --cvsfm_first`) as training, otherwise `build_model_v6` constructs a different module and the checkpoint load fails.

---

## 10 — Files reference

| File | Purpose | Key classes / functions |
|---|---|---|
| [`__init__.py`](models_v6/__init__.py) | Public API, factory functions | `build_model_v6`, `build_loss_v6`, `get_config_v6` |
| [`network_v6.py`](models_v6/network_v6.py) | Main model | `StereoSRV6`, `SpectralStereoBlock`, `StereoSRV6Output` |
| [`blocks.py`](models_v6/blocks.py) | Building blocks | `LayerNorm2d`, `SimpleGate`, `NAFBlock`, `TailUpsample`, `icnr_init` |
| [`epipolar_attention.py`](models_v6/epipolar_attention.py) | SCAM | `StereoCrossAttention` |
| [`spectral_fusion.py`](models_v6/spectral_fusion.py) | Spectral paths | `CrossViewSpectralFusion` (v1), `CrossViewSpectralFusionV2`, `CrossViewSpectralFusionV3` |
| [`losses_v6.py`](models_v6/losses_v6.py) | Losses | `CharbonnierLoss`, `StereoPhotometricConsistency`, `SpectralConsistencyLoss`, `StereoSRV6Loss` |
| [`tlsc.py`](models_v6/tlsc.py) | Test-time local stats | `LocalAvgPool2d`, `enable_tlsc`, `disable_tlsc` |
| [`config_v6.py`](models_v6/config_v6.py) | CLI / config | `get_config_v6`, `SIZE_PRESETS` |
| [`train_ddp_v6.py`](models_v6/train_ddp_v6.py) | DDP training entry | `train`, `validate`, `build_*` |
| [`train_v6.py`](models_v6/train_v6.py) | Single-GPU training | similar interface |
| [`test_v6.py`](models_v6/test_v6.py) | Evaluation harness | `test` |
| [`ARCHITECTURE.md`](models_v6/ARCHITECTURE.md) | High-level pitch (theory, pre-falsification) | — |
| **This file** | Full technical reference | — |

External dependencies:
- [`data/dataset.py`](data/dataset.py) — `build_dataset(args, split)`
- [`utils_fixed.py`](utils_fixed.py) — `AverageMeter`, `compute_psnr`, `compute_ssim`, `save_checkpoint`, `load_checkpoint`, `prune_checkpoints`, `setup_logger`, `save_image`

---

## 11 — Open questions and next steps

1. **v3 vs A1**: does the phase-correlation mechanism beat dense row-wise attention at the small-model scale? Three runs answer this: v3-only full-row, v3-only windowed, v3+SCAM.
2. **Window size sensitivity**: `--cvsfm_v3_window ∈ {0, 16, 30, 32, 60}` — at what point does local-window phase correlation start to behave like SCAM (local attention)?
3. **Block ordering**: does `--cvsfm_first` actually help v3, or is it neutral once both modules are active?
4. **HF-only variant**: does forcing `A = 1` at DC help (low-frequency structure cleanly delegated to bicubic) or hurt (some legitimate low-frequency cross-view signal lost)?
5. **Gain bound sensitivity**: `--cvsfm_v3_gain_scale ∈ {0.05, 0.1, 0.3, 0.5}` — how restrictive must |A| be to stay stable, and at what point does the bound start limiting useful expressiveness?
6. **Curriculum effect**: does pretraining v3-only then adding SCAM beat training the combined model from scratch?
7. **Phase-correlation auxiliary loss**: should we replace the disabled `SpectralConsistencyLoss` with a v3-consistent term (e.g. encourage `|cross_power|` to be peaked rather than diffuse)? Out of scope until v3 architectural results are in.
8. **Visualisation tooling**: spectral modules can silently collapse to identity (γ stuck near zero) without observable training-loss signature. A `--log_v3_diagnostics` flag exposing γ magnitudes, sgate activity, and filter `|A|` deviation per block would catch this. Not yet implemented.
9. **Engineering bundle** (independent of architecture): EMA + AdamW(0.9, 0.9) + larger patches + self-ensemble. Expected: +0.15 to +0.30 dB on top of whichever architecture wins.
10. **Model-size scaling**: V6 small is 1.4–2.1 M params. NAFSSR-L is 24 M at ~24.2 dB. The path to >24 dB likely runs through capacity at least as much as through architecture.

**The honest publication paths**, in priority order:

1. If v3+SCAM (windowed, cvsfm-first) gives ≥ +0.05 dB over A1: the paper has a genuine contribution — a phase-correlation cross-view module that's complementary to spatial attention, with the post-mortem on v1/v2 as supporting evidence that the *right* spectral operator matters.
2. If all v3 variants tie A1 within ±0.03 dB: reposition as a parameter-efficient NAFSSR variant with strong training discipline (zero-init, bf16, ICNR, TLSC) and report the spectral mechanism arc (v1 falsified, v3 designed correctly but no longer carrying the load) as a negative result.
3. If v3 actively hurts (< A1 − 0.05 dB): document the failure and recommend dropping spectral methods for parameter-efficient stereo SR at this resolution; suggest scaling to larger models where the mechanism might recover.

The v3 result is the next branch point in the decision tree.

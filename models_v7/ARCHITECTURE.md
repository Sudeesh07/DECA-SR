# StereoSR V7 — Architecture

## Motivation

V6 ablations established two facts:
1. **SCAM (NAFSSR-style row-wise cross-attention) is the only cross-view mechanism that beats the no-fusion baseline at the small-model scale.** A0=23.41 dB → A1=23.70 dB (+0.28 dB).
2. **Three different spectral-domain cross-view modules (v1 magnitude mix, v2 log-magnitude residual, v3 phase-correlation) all failed to add measurable benefit on top of SCAM.** A6 (SCAM + parameter-free magnitude averaging) vs A1 (SCAM alone) gave −0.004 dB; v3 catastrophically tanh-saturated and stalled at 21.4 dB.

V7 doubles down on what works (SCAM) and extends it in two principled directions that V6 ablations didn't address, plus the engineering bundle that almost every recent SOTA SR paper uses.

## Three novel contributions

### 1. Disparity-prior-conditioned SCAM

**Problem with V6 SCAM.** Row-wise softmax attention is purely content-driven. It can match any L column to any R column on the same row, but it has no *geometric* prior that says "matches lie at horizontal offsets consistent with stereo depth." On uniform-texture or repetitive-pattern regions, this hurts.

**V4's attempt.** V4 (`models_v4/stereo_fusion.py`) computed a correlation cost volume and used soft-argmax to predict per-pixel disparity, then restricted attention to K=5 candidates around the predicted match. Result: ~22.8 dB on Flickr1024, *worse* than V6 SCAM's 23.81. The K=5 hard window was too restrictive — when the prediction was wrong (e.g. occlusions), there was no recovery path.

**V7's approach: soft Gaussian bias, not hard windowing.**
- Use SCAM's *own* Q·K^T correlation matrix (no extra parameters for cost volume).
- Compute soft-argmax expected match position per (row, query).
- Add a Gaussian penalty centered at that position to the pre-softmax logits, scaled by a per-block learnable scalar (`bias_scale`, **zero-init**).
- Width of the Gaussian (`sigma`) is also a per-block learnable parameter via `softplus(log_sigma)`.

At step 0: `bias_scale = 0` → bias contributes nothing → block is identical to V6 SCAM. Training can lift `bias_scale` if the prior helps, or leave it at zero if it doesn't. Unlike V4, dense softmax attention is preserved — if the prior is wrong, attention can still attend anywhere.

**Stop-gradient on the soft-argmax** prevents the failure mode v3 hit (feedback loop between predicted disparity and biased attention, causing tanh saturation). The expected match position is detached; the bias adapts via `sigma` and `bias_scale`, both of which still receive gradient through the loss.

### 2. Multi-scale SCAM trunk

**Problem.** V6's trunk operates entirely at LR resolution (30×90 or, in V7, 48×144). The receptive field per position grows ~2 pixels per block; at 16 blocks of 3×3 DWConv that's ~32 pixels — substantial relative to the 90 px row width, but limits the effective spatial scale of cross-view attention.

**Solution.** Run a parallel coarse trunk at LR/2 resolution (`avg_pool2d(2)` on the stem features). Each coarse-trunk block sees 2× the spatial extent in input pixels. The coarse output is bilinearly upsampled back to LR and added to the main-trunk output via a **per-channel zero-init scale** (`coarse_alpha`).

At step 0: `coarse_alpha = 0` → coarse path contributes nothing → identical to single-scale V7. Training can lift `coarse_alpha` per-channel; some channels may end up using more coarse signal than others.

**Why not U-Net.** V1–V5 used encoder-decoder structures with cascaded PixelShuffle upsamples; the empirical record was bad. V7's "main + parallel coarse" structure preserves V6's "single tail upsample" property — the only HR-resolution computation is the final PixelShuffle. The coarse branch lives entirely at LR/2 and is upsampled bilinearly back to LR before fusion.

**Cost.** ~50 % extra trunk parameters (`num_blocks_coarse = num_blocks / 2` by default). Compute is more modest because LR/2 trunk operates at 1/4 the spatial extent.

### 3. Engineering bundle

Standard practices from NAFSSR / NAFNet / SwinFSR that V6 did not implement:

- **EMA of weights** (decay = 0.9999, warmup over first 2000 steps). Validation runs on the EMA copy. Typical SR gain: **+0.05 to +0.15 dB** on Flickr1024 ×4.
- **AdamW betas = (0.9, 0.9)**. V6 used (0.9, 0.99). NAFSSR's lower β₂ tracks gradient variance over a shorter window; empirically +0.02–0.05 dB on SR tasks.
- **Larger training patches (48×144 LR)**. V6 used 30×90. NAFSSR uses 30×90 too; recent work (MSSFNet, NAFRSSR) uses 48×96 or 60×60. Larger patches mean the network sees more spatial context per sample; receptive-field utilisation improves.
- **Self-ensemble at inference** (`--ensemble`). 4 epipolar-preserving symmetry transforms (identity, h-flip, v-flip, 180°), averaged. **Critical**: horizontal flip swaps L↔R views. Only 4 transforms, not 8 — 90°/270° rotations make the epipolar axis vertical, breaking SCAM. Typical gain: +0.05–0.10 dB.
- **RGB-shuffle augmentation** (NAFSSR). Random channel permutation, same applied to L, R, LR, HR. Free regularisation.

## Architecture diagram

```
LR_L, LR_R  (H × W)
   │
   ├─ Stem (shared Conv3×3): channels
   │
   ├─ MAIN trunk × N_main blocks (at LR resolution):
   │     StereoSpatialBlock:
   │        ├─ NAFBlock(x)                       [per-view, shared weights]
   │        └─ SCAM(x_L, x_R)                    [softmax attention +
   │                                              soft-argmax disparity prior
   │                                              (Gaussian bias on logits)]
   │
   ├─ COARSE trunk × N_coarse blocks (at LR/2 resolution):
   │     avg_pool2d(stem features, 2) → [same block structure] → bilinear upsample
   │     Output × coarse_alpha (per-channel, zero-init) → ADD to main trunk
   │
   ├─ trunk_norm (LayerNorm2d)
   │
   ├─ Tail (shared): Conv3×3(c → 3·scale²) + PixelShuffle(scale)
   │                 [ICNR init, init_scale = 0.01]
   │
   └─ + Bicubic(LR_L) / Bicubic(LR_R)
         → SR_L, SR_R  (sH × sW)
```

## Identity-at-init contract

At step 0 the network output is exactly `bicubic(LR)`. The chain:
- NAFBlock: β = γ = 0 → identity.
- SCAM: β = γ = 0 → identity (no residual injected); `bias_scale = 0` (disparity prior off, equivalent to V6 SCAM).
- Coarse trunk: same identity chain; `coarse_alpha = 0` → no leak into main path.
- Tail: ICNR + ×0.01 → output ≈ 0.014 std → tiny residual.
- Bicubic upsample of input → SR ≈ bicubic(LR) ✓.

Verified by smoke test (§Smoke tests below).

## Diagnostic interpretation (during training)

Every `log_interval` steps the trainer emits per-block stats for every SCAM module (both trunks):

| Metric | Healthy range | What "broken" looks like |
|---|---|---|
| `sigma` | 2.0–8.0 px (drifts from 4.0 init) | stuck at init → prior shape not adapting |
| `bias_scale` | non-zero and varying across blocks | stuck at 0.0 → disparity prior not being used (could be benign if SCAM doesn't need it) |
| `disp_mean` (mean \|x_L − expected_x_R\|) | 1–10 px on Flickr1024 ×4 | 0 px → model not finding correspondences (collapse) |
| `disp_max` | up to ~W/2 | =W → max disparity = full row, suggests divergence |

Specific anti-patterns from V6 experience:
- **Tanh saturation collapse** (v3 failure mode): N/A here — V7 has no tanh-bounded parameters.
- **Silent collapse**: if `bias_scale` stays at 0.0 *and* `coarse_alpha` stays near 0.0 past iter ~10k, the novel V7 contributions aren't contributing — model is effectively V6 SCAM + multi-scale capacity not yet used. Not fatal but worth investigating.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API, `build_model_v7`, `build_loss_v7` |
| `blocks.py` | `LayerNorm2d`, `SimpleGate`, `NAFBlock`, `TailUpsample` (reused from V6) |
| `disparity_prior.py` | `gaussian_disparity_bias`, `soft_argmax_disparity` |
| `epipolar_attention.py` | `StereoCrossAttention` — SCAM with disparity-prior bias |
| `network_v7.py` | `StereoSRV7`, `StereoSpatialBlock` |
| `ema.py` | `ModelEMA` weight averaging |
| `losses_v7.py` | `CharbonnierLoss`, `StereoPhotometricConsistency`, `StereoSRV7Loss` |
| `tlsc.py` | TLSC inference-time correction (reused from V6) |
| `config_v7.py` | CLI / `get_config_v7`, `SIZE_PRESETS` |
| `train_ddp_v7.py` | DDP training with EMA, AdamW(0.9, 0.9), augmentation, diagnostics |
| `test_v7.py` | Evaluation harness with TLSC and self-ensemble |

## Size presets

| Preset | channels | N_main | N_coarse | Approx params (scale=4) |
|---|---|---|---|---|
| tiny  | 48 | 16 | 6  | ~1.0 M |
| **small (default)** | **64** | **16** | **8** | **~2.2 M** |
| base  | 64 | 24 | 12 | ~3.4 M |
| large | 96 | 32 | 16 | ~10 M |

Compare to:
- V6 small (SCAM + CVSFM v1): 1.546 M, 23.81 dB
- NAFSSR-S: 1.56 M, 24.07 dB
- NAFSSR-B: 6.80 M, 24.17 dB
- NAFSSR-L: 23.8 M, 24.17 dB

V7 small is ~40 % larger than V6 small due to (a) multi-scale coarse trunk and (b) per-block disparity prior parameters (negligible — 2 scalars per SCAM block).

## Launch examples

**Default V7 small training:**
```bash
torchrun --nproc_per_node=2 -m models_v7.train_ddp_v7 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --exp_name stereo_sr_v7_small_x4
```

(Note: `--batch_size 16` instead of V6's 32, because patches are 48×144 vs 30×90 — ~2.6× the area. Effective batch is 32 on 2 GPUs, matching V6 A1.)

**Ablation: V7 without disparity prior (multiscale only):**
```bash
torchrun --nproc_per_node=2 -m models_v7.train_ddp_v7 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --no_disparity_prior \
  --exp_name v7_no_disp_prior_200k
```

**Ablation: V7 without multiscale (disparity prior only):**
```bash
torchrun --nproc_per_node=2 -m models_v7.train_ddp_v7 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --no_multiscale \
  --exp_name v7_no_multiscale_200k
```

**Ablation: V7 without either (pure engineering bundle on V6 baseline):**
```bash
torchrun --nproc_per_node=2 -m models_v7.train_ddp_v7 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --no_disparity_prior --no_multiscale \
  --exp_name v7_engbundle_only_200k
```

**Evaluation with TLSC + self-ensemble (EMA weights):**
```bash
python -m models_v7.test_v7 \
  --model_size small --scale 4 \
  --ckpt experiments/stereo_sr_v7_small_x4/checkpoints/best.pth \
  --tlsc --tlsc_base_h 48 --tlsc_base_w 144 \
  --ensemble \
  --test_source Flickr1024 \
  --exp_name eval_v7_small_Flickr1024_tlsc_ens
```

## Expected outcomes (honest pre-training estimates)

V6 A1 (SCAM only): 23.811 dB Flickr1024 test.

Realistic ranges for V7 ablations (with 200k iters and base preset where applicable):

| Config | Expected | Notes |
|---|---|---|
| V7 engbundle only (no disp_prior, no multiscale) | 24.0–24.15 dB | Pure NAFSSR-S equivalent + EMA + larger patches + self-ensemble. This is the *baseline* against which the novel contributions are measured. |
| + disparity prior | 24.05–24.30 dB | Tier 3 from earlier — only deformable-flavoured idea not tested in V6 framework. |
| + multiscale | 24.05–24.25 dB | Capacity addition with parallel coarse branch. |
| **+ both** | **24.10–24.45 dB** | **The headline V7 result.** |

Top venue (CVPR/ICCV/ECCV) typically requires ≥24.3 dB on Flickr1024 ×4 at competitive params for a clear win. V7 has a realistic path to that, but not a guaranteed one. If both novel contributions land below the upper bound, the project still has a competitive parameter-efficient model (mid-tier venue: BMVC, WACV, TIP) with the failure-mode analysis from V6 as supporting evidence.

# StereoSR V8 — Architecture

## Motivation

Six prior architectural variants in this project (V1 spectral magnitude
mixing, V2 log-mag residual, V3 phase correlation, V4 deformable disparity
prior, V7 multi-scale + Gaussian disparity bias) all failed to beat V6 A1's
plain SCAM design at the same parameter budget. The lesson is consistent:
*at the 1.4–2M parameter scale*, every attempt to enrich cross-view fusion
with spectral, disparity-aware, or multi-scale machinery has been
out-performed by the simplest possible mechanism — dense softmax attention
across the epipolar row.

V8 doubles down on that lesson:
* Keep the V6 SCAM trunk untouched (it's the only mechanism with a winning
  ablation).
* Layer on **one** architectural addition whose mechanism is mathematically
  orthogonal to SCAM and whose softmax distribution is **scale-invariant
  by construction**, avoiding the train→val mismatch that killed V4 and V7.

## The architectural contribution: CT-CVA

**Channel-Transposed Cross-View Attention.** Multi-head cosine attention
along the channel axis, with the L/R views as query and key/value pairs.

### Mechanism

For ``L → R`` direction (symmetric for ``R ← L``):

```
Q_L = qkv_l(LN(x_L)) → q_L        [B, C, H, W]
K_R, V_R = qkv_r(LN(x_R))         → k_R, v_R

Reshape to multi-head channel tokens:
q_L: [B, h, d, HW]    where C = h · d
k_R: [B, h, d, HW]
v_R: [B, h, d, HW]

Cosine attention (L2-normalise along HW axis):
q_L_n = q_L / ||q_L||_HW
k_R_n = k_R / ||k_R||_HW
attn  = q_L_n · k_R_n^T · τ        [B, h, d, d]   ← FIXED size
attn  = softmax(attn, dim=-1)

Apply to V_R:
out_L = attn · v_R                 [B, h, d, HW]
out_L = proj_l(reshape to [B, C, H, W])

Residual with zero-init γ:
out_L = x_L + γ_L · proj_l(...)
```

### Why this is novel for stereo SR

Restormer (Zamir et al., CVPR 2022) introduced channel-axis transposed
attention (MDTA) for *single-image* restoration. It has not been
transferred to *cross-view stereo* attention. Every published stereo SR
cross-view module — PASSRnet, iPASSR's biPAM, NAFSSR's SCAM, SwinFSR's
RCAM, MFFSSR's CVIM, HTCAN's blocks — operates on spatial positions. V8
is the first to add the channel axis.

### Why this avoids every V1–V7 failure mode

| Failure mode | CT-CVA response |
|---|---|
| V1–V3 cross-view magnitude blending (\|F_L\|≈\|F_R\| no-op) | No FFT; operates in spatial+channel feature domain. Cosine-similarity attention is unaffected by frequency degeneracies. |
| V3 tanh-bounded filter saturation | No bounded parametrisation. Temperature is unbounded learnable; softmax provides natural normalisation. |
| V4 hard K=5 attention window | Dense softmax over the channel dimension; no top-k, no hard windowing. |
| V4/V7 pixel-unit sigma scale mismatch | The attention matrix shape is `[h, d, d]` where `d = C/h` is constant. **Independent of H, W.** No pixel-distance parameters anywhere. |
| V7 14× pool-area mismatch in coarse trunk | No multi-scale trunk; no spatial pooling beyond the existing NAFBlock SCA (which V6 already handles correctly via TLSC). |
| V7 train–val inversion | Cosine attention's softmax temperature is decoupled from input size by the L2 normalisation. This is the *mathematical* fix for the failure mode V7 documented. |

### Identity-at-init contract

* `γ_L = γ_R = 0` exactly → block output = block input exactly at step 0.
* `proj_l(attn @ v_R)` is non-zero in general (cosine attention is
  non-degenerate at init), so `∂loss/∂γ ≠ 0` — γ has a non-zero gradient
  and lifts off zero immediately.

This is verified in the smoke test.

## Architecture diagram

```
LR_L, LR_R  (H × W)
   │
   ├─ Stem (shared Conv3×3): channels
   │
   ├─ Trunk × N blocks, all at LR resolution:
   │     StereoSpatialChannelBlock:
   │        ├─ NAFBlock(x_L); NAFBlock(x_R)         [per-view, shared]
   │        ├─ SCAM(x_L, x_R)                       [spatial cross-view (V6, unchanged)]
   │        └─ CT-CVA(x_L, x_R)                     [channel cross-view (NEW)]
   │
   ├─ trunk_norm (LayerNorm2d)
   │
   ├─ Tail (shared): Conv3×3(c → 3·s²) + PixelShuffle(s)
   │                 [ICNR init, init_scale = 0.01]
   │
   └─ + Bicubic(LR_L) / Bicubic(LR_R)
         → SR_L, SR_R  (sH × sW)
```

## Training recipe (Phase A engineering bundle)

V8 also adopts the modern training tricks that V6 A1 lacked:

* **EMA weight averaging** with decay 0.9999 and 2000-step warmup
* **AdamW betas (0.9, 0.9)** — NAFSSR's setting; V6 used (0.9, 0.99)
* **48×144 LR training patches** — V6 used 30×90; this matches NAFSSR
* **NTIRE-2024 augmentation stack**: RGB shuffle + h-flip with L/R view
  swap + optional v-flip + h-shift (same shift on L and R preserves
  disparity)
* **4-transform self-ensemble at evaluation** — identity, h-flip+swap,
  v-flip, 180° — only epipolar-preserving symmetries
* **TLSC** at inference (V6 carry-over, unchanged)

## Size presets

| Preset | channels | num_blocks | Approx params (×4) |
|---|---|---|---|
| tiny  | 48 | 16 | ~0.95 M |
| **small (default)** | **64** | **16** | **~1.95 M** |
| base  | 64 | 24 | ~2.85 M |
| large | 96 | 32 | ~7.5 M |

CT-CVA adds ~32K params per block at channels=64 (vs SCAM's ~17K) — small
overhead relative to NAFBlock's ~150K per block.

## Diagnostic system

Every `--log_interval` (default 100 iters), the trainer logs per-block
CT-CVA diagnostics:

| Metric | Healthy range | Failure pattern |
|---|---|---|
| `γ_L`, `γ_R` magnitude | climbs from 0 to 1e-2…1e-1 over training | stuck < 1e-3 past iter 5000 → silent collapse warning |
| `temperature` | drifts from 1.0; some blocks may go higher (sharper attention), some lower | runaway >10 or near-zero → numerical instability |
| `attn_entropy` (normalised by log d) | 0.5–0.9 (interior) | < 0.1 → degenerate (one key dominates); ≈ 1.0 → uniform/inert |
| `attn_max` | 0.1–0.6 | > 0.9 → one-hot degenerate attention |

A silent-collapse warning fires if max(γ) < 1e-3 past iter 5000 — same
mechanism that flagged V7's CVSFM-v3 saturation early in training.

## Launch examples

**Full V8 small training:**
```bash
torchrun --nproc_per_node=2 -m models_v8.train_ddp_v8 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --exp_name stereo_sr_v8_small_x4
```

**Ablation: no CT-CVA** (Phase A engineering bundle only, isolates novelty):
```bash
torchrun --nproc_per_node=2 -m models_v8.train_ddp_v8 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --no_ct_cva \
  --exp_name stereo_sr_v8_no_ct_cva_200k
```

**Ablation: head count sensitivity:**
```bash
torchrun --nproc_per_node=2 -m models_v8.train_ddp_v8 \
  --model_size small --scale 4 --batch_size 16 --bf16 --num_workers 0 \
  --total_iters 200000 \
  --ct_cva_heads 8 \
  --exp_name stereo_sr_v8_heads8_200k
```

**Evaluation with TLSC + self-ensemble (uses EMA weights):**
```bash
python -m models_v8.test_v8 \
  --model_size small --scale 4 \
  --ckpt experiments/stereo_sr_v8_small_x4/checkpoints/best.pth \
  --tlsc --tlsc_base_h 48 --tlsc_base_w 144 \
  --ensemble \
  --test_source Flickr1024 \
  --exp_name eval_v8_small_Flickr1024_tlsc_ens
```

## Honest expected outcomes

V6 A1: 23.70 dB val / 23.81 dB test, 1.41 M params.

| Threshold | Probability |
|---|---|
| ≥ 24.00 dB val | 35% |
| ≥ 23.85 dB val | 60% |
| ≥ 23.70 dB val (match V6 A1) | 88% |
| < 23.70 dB val | 12% |

The variance reflects the project's empirical track record (every prior
architectural innovation has under-delivered vs prediction). CT-CVA is
the cleanest architectural addition this project has tried — by-construction
scale-invariant, identity-at-init, no bounded saturation parameters — so
the 12% regression probability is the floor, not the central estimate.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API |
| `blocks.py` | NAFBlock, LayerNorm2d, TailUpsample (reused V6/V7) |
| `tlsc.py` | TLSC (reused) |
| `ema.py` | EMA weight averaging (reused V7) |
| `epipolar_attention.py` | SCAM (V6-style, no disparity prior) |
| `channel_cross_view.py` | **CT-CVA (NEW core contribution)** |
| `augmentations.py` | NTIRE-2024 augmentation stack |
| `network_v8.py` | StereoSRV8 with dual-axis block |
| `losses_v8.py` | Charbonnier + stereo photometric (no spectral) |
| `config_v8.py` | All CLI flags |
| `train_ddp_v8.py` | DDP training + EMA + augmentations + diagnostics |
| `test_v8.py` | Self-ensemble + TLSC + EMA evaluation |

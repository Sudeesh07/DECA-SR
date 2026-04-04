# WaveFuse (StereoSR V5) — Architecture Documentation

## Overview

WaveFuse is a stereo image super-resolution network that reconstructs high-resolution stereo pairs from low-resolution inputs. It builds on the V4 architecture with three targeted improvements: **lossless wavelet downsampling**, **sigmoid-gated stereo attention with iterative disparity refinement**, and **frequency-aware skip connections**.

**Parameters:** Base = 12.11M | Tiny = 6.49M | Large = 19.89M
**Supported scales:** x2, x4
**Input:** `(img_L, img_R)` both `[B, 3, H, W]`
**Output:** `(sr_L, sr_R)` both `[B, 3, H*s, W*s]` + auxiliary disparity/confidence maps

---

## Full Architecture Diagram

```
 img_L [B,3,H,W]                                    img_R [B,3,H,W]
       |                                                    |
       v                                                    v
 +-----------+                                        +-----------+
 | Shared    |  Conv3x3 + LayerNorm                   | Shared    |
 | Stem      |  [B,3,H,W] -> [B,C1,H,W]              | Stem      |
 +-----------+                                        +-----------+
       |                                                    |
       v                                                    v
 +==================+                              +==================+
 ||  Encoder        ||                              ||  Encoder        ||
 ||  Stage 1        ||                              ||  Stage 1        ||
 ||                 ||                              ||                 ||
 || DLFB x N1       ||                              || DLFB x N1       ||
 || (C1 channels)   ||                              || (C1 channels)   ||
 +==================+                              +==================+
       |          |                                      |          |
   skip1_L     HaarDWT                               skip1_R     HaarDWT
  [B,C1,H,W]  Pack+Proj                            [B,C1,H,W]  Pack+Proj
               [B,C2,H/2,W/2]                                  [B,C2,H/2,W/2]
                  |                                                |
                  v                                                v
          +==================+                          +==================+
          ||  Encoder        ||                          ||  Encoder        ||
          ||  Stage 2        ||                          ||  Stage 2        ||
          ||                 ||                          ||                 ||
          || DLFB x N2       ||                          || DLFB x N2       ||
          || (C2 channels)   ||                          || (C2 channels)   ||
          +==================+                          +==================+
                |          |                                  |          |
            skip2_L     HaarDWT                           skip2_R     HaarDWT
          [B,C2,H/2,W/2] Pack+Proj                     [B,C2,H/2,W/2] Pack+Proj
                       [B,C3,H/4,W/4]                               [B,C3,H/4,W/4]
                          |                                             |
                          v                                             v
                  +==================+                          +==================+
                  ||  Encoder        ||                          ||  Encoder        ||
                  ||  Stage 3        ||                          ||  Stage 3        ||
                  ||                 ||                          ||                 ||
                  || DLFB x N3       ||                          || DLFB x N3       ||
                  || (C3 channels)   ||                          || (C3 channels)   ||
                  || NO downsample   ||                          || NO downsample   ||
                  +==================+                          +==================+
                          |                                             |
                       s3_L [B,C3,H/4,W/4]                         s3_R [B,C3,H/4,W/4]
                          |                                             |
                          +----------------+    +------------------------+
                                           |    |
                                           v    v
                        +==========================================+
                        ||         DEEP STEREO FUSION              ||
                        ||                                          ||
                        ||  deep_fuse_L: StereoFusionBlock(s3_L,s3_R) -> fused_L, stats_L
                        ||  deep_fuse_R: StereoFusionBlock(s3_R,s3_L) -> fused_R, stats_R
                        ||                                          ||
                        ||  Contains:                               ||
                        ||    1. GroupwiseDisparityPrior             ||
                        ||    2. Sigmoid-Gated EpipolarAttention    ||
                        ||    3. Residual Fusion                    ||
                        +==========================================+
                                    |                |
                                 fused_L          fused_R
                                    |                |
                                    v                v
                        +==========================================+
                        ||           BOTTLENECK                    ||
                        ||                                          ||
                        ||  ResidualConvBlock x N_bn                ||
                        ||  (shared weights for L and R)            ||
                        +==========================================+
                                    |                |
                                  b_L              b_R
                                    |                |
                                    v                v
                        +==========================================+
                        ||   ITERATIVE DISPARITY REFINEMENT        ||
                        ||                                          ||
                        ||  disp_refine_L: ConvGRU (T=2 iters)    ||
                        ||    Input: deep_stats_L.disp, b_L, b_R   ||
                        ||    Output: refined_disp_L               ||
                        ||                                          ||
                        ||  disp_refine_R: ConvGRU (T=2 iters)    ||
                        ||    Input: deep_stats_R.disp, b_R, b_L   ||
                        ||    Output: refined_disp_R               ||
                        +==========================================+
                                    |                |
                          refined_disp_L      refined_disp_R
                                    |                |
                                    v                v
                        +==========================================+
                        ||      BOTTLENECK STEREO FUSION           ||
                        ||                                          ||
                        ||  bottleneck_fuse_L(b_L, b_R,            ||
                        ||      disp_init=refined_disp_L)          ||
                        ||  bottleneck_fuse_R(b_R, b_L,            ||
                        ||      disp_init=refined_disp_R)          ||
                        +==========================================+
                                    |                |
                                  b_L'             b_R'
                                    |                |
              +---------------------+                +---------------------+
              |                                                            |
              v                                                            v
  +=========================+                              +=========================+
  ||   DECODER (LEFT)       ||                              ||   DECODER (RIGHT)      ||
  ||                        ||                              ||                        ||
  ||  decode_2x:            ||                              ||  decode_2x:            ||
  ||    PixelShuffle x2     ||                              ||    PixelShuffle x2     ||
  ||    + SAG(skip2_L)      ||                              ||    + SAG(skip2_R)      ||
  ||    + ResBlocks         ||                              ||    + ResBlocks         ||
  ||                        ||                              ||                        ||
  ||  decode_1x:            ||                              ||  decode_1x:            ||
  ||    PixelShuffle x2     ||                              ||    PixelShuffle x2     ||
  ||    + SAG(skip1_L)      ||                              ||    + SAG(skip1_R)      ||
  ||    + ResBlocks         ||                              ||    + ResBlocks         ||
  ||                        ||                              ||                        ||
  ||  HR Head:              ||                              ||  HR Head:              ||
  ||    PixelShuffle x2     ||                              ||    PixelShuffle x2     ||
  ||    + RefineBlocks      ||                              ||    + RefineBlocks      ||
  ||    (+ PixelShuffle x2  ||                              ||    (+ PixelShuffle x2  ||
  ||     if scale=4)        ||                              ||     if scale=4)        ||
  ||    -> to_rgb (Conv3x3) ||                              ||    -> to_rgb (Conv3x3) ||
  +=========================+                              +=========================+
              |                                                            |
           pred_L [B,3,sH,sW]                                        pred_R [B,3,sH,sW]
              |                                                            |
              |  + Bicubic(img_L, scale)                                   |  + Bicubic(img_R, scale)
              |                                                            |
              v                                                            v
         sr_L [B,3,sH,sW]                                           sr_R [B,3,sH,sW]
```

---

## Module Details

### 1. Shared Stem

```
Input: [B, 3, H, W]
  |
  Conv2d(3 -> C1, kernel=3, padding=1)
  |
  LayerNorm2d(C1)
  |
Output: [B, C1, H, W]
```

A single `Conv3x3 + LayerNorm` projects RGB to feature space. Weights shared for both views — ensures L/R features live in the same embedding space.

---

### 2. Deformable Local Feature Block (DLFB)

The core intra-view encoder block. Uses Adaptive Sampling Attention instead of standard self-attention.

```
                        x [B, C, H, W]
                        |
             +----------+----------+
             |                     |
        LayerNorm2d           (identity)
             |                     |
    AdaptiveSamplingAttention      |
             |                     |
          * beta                   |
             |                     |
             +----------+----------+
                        | (residual add)
                        |
             +----------+----------+
             |                     |
        LayerNorm2d           (identity)
             |                     |
        GatedConvFFN               |
             |                     |
          * gamma                  |
             |                     |
             +----------+----------+
                        | (residual add)
                        |
                   Output [B, C, H, W]
```

**AdaptiveSamplingAttention detail:**
```
x -> Q_proj(1x1), K_proj(1x1), V_proj(1x1)
  -> offset_net: DWConv3x3 -> GELU -> Conv1x1 -> tanh -> * max_offset
  -> sampling positions = base_offsets + residual_offsets
  -> grid_sample K and V at K=9 positions per query
  -> Multi-head attention: softmax(Q^T K / sqrt(d)) * V
  -> out_proj(1x1)

Complexity: O(H * W * K * C)   where K=9 << HW
```

**GatedConvFFN:**
```
x -> Conv1x1(C -> 2*hidden) -> GELU -> DWConv3x3(2*hidden) -> GLU -> Conv1x1(hidden -> C)
```

**Key properties:**
- `beta`, `gamma` initialized to 0 (identity at init, stable training)
- DropPath for stochastic depth regularization
- Linear drop path schedule across all stages

---

### 3. Haar DWT Channel Packing (Lossless Downsampling)

Replaces lossy strided convolutions. Mathematically lossless — no information discarded.

```
Input: [B, C, H, W]
  |
  LayerNorm2d(C)
  |
  HaarDWT (fixed filters, stride=2, groups=C)
  |
  +-> LL [B, C, H/2, W/2]  (low-freq approximation)
  +-> LH [B, C, H/2, W/2]  (horizontal edges)
  +-> HL [B, C, H/2, W/2]  (vertical edges)
  +-> HH [B, C, H/2, W/2]  (diagonal textures)
  |
  cat along channel dim -> [B, 4C, H/2, W/2]
  |
  Conv1x1(4C -> C_out)  (learnable projection)
  |
Output: [B, C_out, H/2, W/2]
```

**Haar filter bank (fixed, non-learnable):**
```
LL = 0.5 * [+1 +1]    LH = 0.5 * [+1 +1]
           [+1 +1]                [-1 -1]

HL = 0.5 * [+1 -1]    HH = 0.5 * [+1 -1]
           [+1 -1]                [-1 +1]
```

**Why this matters:** Strided conv with stride=2 applies a learned 3x3 filter then keeps every other pixel — an irreversible, lossy operation. HaarDWT is an orthogonal transform; the 1x1 projection learns which subband combinations matter, but no information is lost in the decomposition itself.

---

### 4. Stereo Fusion Block

Applied only at the deepest encoder level (H/4 x W/4) and bottleneck. Contains three sub-modules:

```
target [B,C3,H/4,W/4]          reference [B,C3,H/4,W/4]
    |                                  |
    LayerNorm2d                   LayerNorm2d
    |                                  |
    +----------------------------------+
    |                                  |
    v                                  v
  +----------------------------------------------+
  |     GroupwiseDisparityPrior                    |
  |                                                |
  |  Split into G=8 groups, L2-normalize           |
  |  Compute correlation at d=0,1,...,D_max        |
  |  Refine cost volume with Conv network          |
  |  soft-argmax -> disparity [B,1,H/4,W/4]       |
  |  max prob   -> confidence [B,1,H/4,W/4]       |
  +----------------------------------------------+
                    |
                 (d_prior, conf)
                    |
                    v
  +----------------------------------------------+
  |     Sigmoid-Gated Epipolar Attention           |
  |                                                |
  |  Q = q_proj(target)                            |
  |  K = k_proj(reference)                         |
  |  V = v_proj(reference)                         |
  |                                                |
  |  Candidate disparities:                        |
  |    d_k = clamp(d_prior + anchor_k + delta_k)   |
  |                                                |
  |  Sample K,V at candidates via grid_sample      |
  |  (horizontal epipolar constraint only)         |
  |                                                |
  |  Attention: sigmoid(Q^T K / sqrt(d)) * m_k     |
  |  where m_k = sigmoid(mod_weights_net(...))      |
  |  Normalize: alpha / sum(alpha) + eps            |
  |                                                |
  |  out = out_proj(sum(alpha_k * V_k))            |
  +----------------------------------------------+
                    |
                transfer [B,C3,H/4,W/4]
                    |
                    v
  +----------------------------------------------+
  |     Residual Fusion                            |
  |                                                |
  |  fused = Conv1x1(GELU(Conv1x1(                |
  |            cat[target_n, transfer, conf])))     |
  |                                                |
  |  output = target + alpha * fused               |
  |  (alpha initialized to 0)                      |
  +----------------------------------------------+
                    |
         fused_target [B,C3,H/4,W/4]
```

**Sigmoid vs Softmax (DCNv4 insight):**
```
Softmax:  alpha_k = exp(s_k) / sum(exp(s_j))     -- forces sum=1
Sigmoid:  alpha_k = sigma(s_k) * m_k              -- independent per point

Softmax must distribute weight across all K points (probability distribution).
Sigmoid allows: all points zero (occluded), multiple points high (ambiguous match),
or single point dominant (clear correspondence). More expressive.
```

---

### 5. Iterative Disparity Refinement (ConvGRU)

Refines the coarse disparity from deep fusion over T=2 iterations.

```
Inputs:
  d_init [B,1,H/4,W/4]     -- from GroupwiseDisparityPrior (detached)
  feat_T [B,C3,H/4,W/4]    -- bottleneck target features
  feat_R [B,C3,H/4,W/4]    -- bottleneck reference features

Initialization:
  context = context_net(feat_T)         -- Conv3x3 -> GELU -> Conv3x3
  hidden  = tanh(context)               -- GRU hidden state
  d       = d_init.detach()             -- don't backprop through prior

For t = 1, 2:
  +-----------------------------------------------------+
  |  1. WARP reference by current disparity              |
  |     warped_R = grid_sample(feat_R, d)                |
  |     (horizontal shift only, epipolar constraint)     |
  |                                                      |
  |  2. COMPUTE correlation                              |
  |     corr = sum(normalize(feat_T) * normalize(warped_R), dim=C)
  |     -> [B, 1, H/4, W/4]                             |
  |                                                      |
  |  3. GRU UPDATE                                       |
  |     input = cat[d, corr, context]                    |
  |     z = sigmoid(W_z * cat[input, hidden])   (update) |
  |     r = sigmoid(W_r * cat[input, hidden])   (reset)  |
  |     h~ = tanh(W_h * cat[input, r*hidden])   (cand.)  |
  |     hidden = (1-z)*hidden + z*h~                     |
  |                                                      |
  |  4. PREDICT bounded delta                            |
  |     delta = tanh(delta_head(hidden)) * max_delta     |
  |     d = d + delta                                    |
  +-----------------------------------------------------+

Output: refined_d [B,1,H/4,W/4]
```

**Why detach d_init?** Prevents gradients from the refinement module flowing back through the GroupwiseDisparityPrior, which would destabilize its own learning. Standard practice in RAFT-Stereo and IGEV++.

**Why T=2?** Shared GRU weights mean extra iterations add compute but no parameters. T=2 provides a good balance; T=3 shows diminishing returns.

---

### 6. Subband Attention Gate (SAG)

Applied to skip connections in the decoder before concatenation with upsampled features.

```
skip [B, C, H, W]
  |
  HaarDWT
  |
  +-> LL [B,C,H/2,W/2] -> GAP -> Linear(C->C) -> sigmoid -> * LL
  +-> LH [B,C,H/2,W/2] -> GAP -> Linear(C->C) -> sigmoid -> * LH
  +-> HL [B,C,H/2,W/2] -> GAP -> Linear(C->C) -> sigmoid -> * HL
  +-> HH [B,C,H/2,W/2] -> GAP -> Linear(C->C) -> sigmoid -> * HH
  |
  InverseHaarDWT(LL*, LH*, HL*, HH*)
  |
  Trim to original (H, W) if input had odd dimensions
  |
Output: gated_skip [B, C, H, W]
```

Each subband gets its own independent channel attention gate. This allows the decoder to:
- Suppress noisy HH subbands in early decoder stages
- Preserve structural LL content
- Selectively enhance edge subbands (LH, HL) for texture recovery

**Cost:** Only 4 * C * C parameters per gate — negligible.

---

### 7. Progressive Reconstruction Decoder

```
bottleneck [B,C3,H/4,W/4]
  |
  DecoderStage(C3 -> C2, skip=skip2):
    PixelShuffle x2 -> [B,C2,H/2,W/2]
    SAG(skip2) -> gated_skip2
    cat[upsampled, gated_skip2] -> Conv1x1 fusion
    ResidualConvBlock x N_dec refinement
  |
  [B, C2, H/2, W/2]
  |
  DecoderStage(C2 -> C1, skip=skip1):
    PixelShuffle x2 -> [B,C1,H,W]
    SAG(skip1) -> gated_skip1
    cat[upsampled, gated_skip1] -> Conv1x1 fusion
    ResidualConvBlock x N_dec refinement
  |
  [B, C1, H, W]
  |
  HR Head:
    Conv1x1(C1 -> hr_channels)
    HRStage: PixelShuffle x2 + RefineBlocks -> [B, hr_ch, 2H, 2W]
    (if scale=4):
      HRStage: PixelShuffle x2 + RefineBlocks -> [B, hr_ch/2, 4H, 4W]
      Conv3x3 -> [B, 3, 4H, 4W]
    (if scale=2):
      Conv3x3 -> [B, 3, 2H, 2W]
  |
  + Bicubic(input, scale)    <- Global residual connection
  |
  SR Output [B, 3, sH, sW]
```

---

### 8. Loss Functions

```
L_total = L_pixel + 0.05 * L_mse + 0.05 * L_stereo + 0.01 * L_smooth

L_pixel:   Charbonnier loss = mean(sqrt((sr - hr)^2 + eps^2))
           Applied to both views, averaged.

L_mse:     Standard MSE auxiliary term.

L_stereo:  Disparity-warped stereo consistency (FIX 3+4)
           1. Upsample refined_disp from H/4 to sH resolution
           2. Warp sr_right to sr_left frame using disparity
           3. Compute L1 weighted by: validity_mask * confidence
           4. Symmetric: average L->R and R->L directions
           Automatically ignores occluded regions (low confidence)

L_smooth:  Edge-aware disparity smoothness (FIX 5)
           |grad(disp)| * exp(-|grad(image)|)
           Encourages smooth disparity except at image edges
```

---

## Model Size Presets

| Preset | C1/C2/C3 | Enc Blocks | Bottleneck | Dec Blocks | HR Blocks | Params |
|--------|----------|------------|------------|------------|-----------|--------|
| Tiny   | 48/72/96 | 3/4/6      | 3          | 3/3        | 2/2       | 6.49M  |
| Base   | 64/96/128| 4/6/8      | 4          | 4/4        | 3/3       | 12.11M |
| Large  | 80/120/160| 5/8/10    | 5          | 5/5        | 4/4       | 19.89M |

**Parameter distribution (Base x4):**
```
decoder_left          : 1.84M  (15.2%)
decoder_right         : 1.84M  (15.2%)
stage3                : 1.40M  (11.6%)
disp_refine_l         : 1.26M  (10.4%)
disp_refine_r         : 1.26M  (10.4%)
bottleneck            : 0.76M  ( 6.3%)
deep_fuse_l           : 0.72M  ( 5.9%)
deep_fuse_r           : 0.72M  ( 5.9%)
bottleneck_fuse_l     : 0.72M  ( 5.9%)
bottleneck_fuse_r     : 0.72M  ( 5.9%)
stage2                : 0.65M  ( 5.4%)
stage1                : 0.21M  ( 1.7%)
stem                  : 0.00M  ( 0.0%)
──────────────────────────────────
TOTAL                 : 12.11M
```

---

## Data Flow Summary (x4 scale, Base preset)

```
Input:  [B, 3, 30, 90]  (LR patch)

Stem:           [B, 3, 30, 90]   ->  [B, 64, 30, 90]
Stage1 + DWT:   [B, 64, 30, 90]  ->  skip1[B,64,30,90]  +  [B, 96, 15, 45]
Stage2 + DWT:   [B, 96, 15, 45]  ->  skip2[B,96,15,45]  +  [B, 128, 8, 23]
Stage3:         [B, 128, 8, 23]  ->  s3[B, 128, 8, 23]

Deep Fusion:    s3_L + s3_R  ->  fused_L[B,128,8,23] + disp_L[B,1,8,23]
Bottleneck:     fused_L      ->  b_L[B,128,8,23]
Iter Refine:    disp_L + b_L + b_R  ->  refined_disp_L[B,1,8,23]
BN Fusion:      b_L + b_R + refined_disp  ->  b_L'[B,128,8,23]

Decode 2x:      b_L'[B,128,8,23] + SAG(skip2) -> [B,96,15,45]
Decode 1x:      [B,96,15,45]     + SAG(skip1) -> [B,64,30,90]
HR x2:          [B,64,30,90]  ->  [B,64,60,180]
HR x4:          [B,64,60,180] ->  [B,32,120,360]
to_rgb:         [B,32,120,360] -> [B,3,120,360]

+ Bicubic(input, 4):  [B,3,120,360]

Output: [B, 3, 120, 360]  (SR output, 4x upscaled)
```

---

## Key Design Decisions

### Why selective deep fusion (not every stage)?
Cross-view attention at full resolution (H x W) is expensive and often unnecessary — local features at early stages don't benefit much from cross-view information. Applying fusion only at H/4 x W/4 reduces cost by 16x while capturing the most meaningful stereo correspondence at the semantic level.

### Why shared encoder weights?
Both views observe the same scene from slightly different angles. Shared weights ensure L/R features live in the same embedding space, which is required for meaningful cross-view correlation. Also halves encoder parameters.

### Why separate decoders?
After stereo fusion, L and R features diverge (different disparity, different occlusion patterns). Separate decoders allow view-specific reconstruction. The parameter cost is modest since decoders are lighter than encoders.

### Why global residual (bicubic add)?
The network only needs to learn the residual (missing high-frequency detail), not reconstruct from scratch. This dramatically stabilizes training — the network starts at bicubic quality (~20.5 dB) rather than random noise.

### Why detach initial disparity in refinement?
The GroupwiseDisparityPrior has its own learning path through the deep fusion loss. If refinement gradients flow back through it, the prior's training becomes dominated by refinement needs, destabilizing both modules. Detaching creates a clean interface: prior learns from fusion, refinement learns to correct.

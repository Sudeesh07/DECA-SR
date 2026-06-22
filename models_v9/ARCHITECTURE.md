# V9 — DECA-SR (Deformable Epipolar Cost-view-Aware Stereo SR)

## Goal
A genuinely novel, parameter-efficient stereo SR network that is competitive with
the S-class SOTA (Flickr1024 ×4 ~24.0–24.1 dB; NAFSSR-S 23.88 @1.56M, NAFRSSR-S
23.96 @1.35M, CVHSSR-S 24.08 @2.24M). Budgets (deploy, RepLK merged):
**tiny 0.68M (<0.8), small 1.82M (<2.0), base 7.0M (<10)**.

## Why DECA-SR (chosen over 2 alternatives after adversarial review)
A research workflow generated three novel designs and an adversarial critic
ranked them:
- **DECA-SR (chosen)** — deformable cost-view fusion. Plain PyTorch, stable, and
  attacks the KITTI vertical-rectification-bleed that strict-1D methods cannot.
- WaveFusion-SR (wavelet sub-band) — safe but low upside (≈ SCAM).
- PRISM-SSR (phase/frequency) — **killed**: it was a reparameterization of this
  project's own falsified V6 CVSFM v3 + a phase-aliasing stability trap.

## The novelty: DECVF (`deformable_fusion.py`)
Per pixel in the self view, for each direction (L←R and R←L, weight-shared):
1. **Cost-volume-lite** along the epipolar row: `D` shifted dot-products on a
   reduced key channel → a softmax over disparities → a **soft disparity prior
   `d0`** (sub-pixel) and a **match confidence** (peak sharpness).
2. **Disparity-initialized deformable offsets**: an offset head predicts `K`
   sample points whose x-offset is `d0 + range·tanh(·)` (centred on the
   cost-volume disparity — this fixes the cold-start that sank the earlier V4
   deformable attempt) and y-offset `y_range·tanh(·)` (clamped small, epipolar
   bias but tolerant of rectification residual).
3. **Sub-pixel gather**: `grid_sample` the other view's value map at the `K`
   points (bilinear — no softmax-averaging blur, unlike SCAM), aggregate by
   per-point weights × match confidence.
4. **Fuse**: SFAM-style global L/R channel gate × `out_proj`, added through a
   **zero-init per-channel γ** residual.

Why it should raise PSNR/SSIM: sub-pixel resampling places the correct other-view
texture sharply (vs SCAM's row-averaging low-pass), the cost-volume prior makes
the deformable offsets trainable, the 2D offsets absorb vertical bleed, and
confidence gating suppresses fusion in occluded regions. Numerically: cost-volume
softmax / soft-argmax / offset head / all `grid_sample` run in fp32 under
`autocast(enabled=False)` (the project's bf16-stability contract).

## Trunk & tail
- **RepLKNAFBlock** (`blocks.py`): NAFBlock with the 3×3 depthwise replaced by a
  reparameterizable large-kernel depthwise (train as {LK, smaller-k, identity}
  branches; **merge to a single LK conv at inference, bit-exact, zero added deploy
  params**). Large-kernel RF frees budget for fusion (CVHSSR lesson) + free PSNR
  (ECBSR/RepSR). Shared weights across L/R.
- **SCAM** (`epipolar_attention.py`) every block — the project's one validated
  cross-view win; cheap dense complement to DECVF.
- **Tail**: single PixelShuffle (ICNR, init_scale 0.01) over a global bicubic
  residual → SR ≈ bicubic(LR) at init.

**Identity-at-init everywhere**: NAF β/γ=0, SCAM β/γ=0, DECVF γ=0, offset head
zero-init (offsets start exactly at `d0`, dy=0) → the novel block cannot score
below the SCAM baseline; it only lifts γ off zero if it helps.

## Training protocol (the V8 post-mortem lesson)
The architecture was never the whole story — undertraining + missing levers were.
So: bank every near-free lever, train the FULL 400k, and judge DECVF only against
a fully-leveraged SCAM-only baseline (`--no_decvf`), not historical numbers.
- Aug: RGB-shuffle, h-flip+swap, v-flip, replicate-pad-crop h-shift (NOT roll),
  optional Mixup.
- Loss: Charbonnier + 0.02·stereo-photometric + 0.1·per-view focal-frequency;
  terminal pure-MSE phase for the last `--terminal_mse_iters`.
- AdamW(0.9,0.9), bf16, grad-clip 1.0, cosine LR with `--max_lr 1e-3` ceiling
  (the v6-small NaN-collapse guard), EMA 0.9999, top-K SWA at the end.
- Inference: `--reparam_eval`, `--tlsc` (base = training patch), `--ensemble`.

## Files
`network_v9.py` (StereoSRV9 + DECABlock), `deformable_fusion.py` (DECVF),
`blocks.py` (RepLKNAFBlock, RepLargeKernelDW, TailUpsample), `epipolar_attention.py`
(SCAM), `losses_v9.py` (+FocalFrequencyLoss, terminal MSE), `augmentations.py`
(+Mixup), `ema.py`, `tlsc.py`, `config_v9.py`, `train_ddp_v9.py`, `test_v9.py`.

## Ablations (`--no_decvf`, `--no_scam`, flags)
A0 trunk-only · A1 +SCAM (the bar) · A2 +DECVF-only · A3 full · A4 cost-volume
prior on/off (cold-start test — the load-bearing novelty) · A5 confidence/SFAM
gate on/off · A6 y-range 0 vs 1.5 (report KITTI separately) · A7 K/D sweep ·
A8 reparam on/off + merge-equality unit test · scaling tiny→S→B.

## Status
Implemented and integration-tested (param budgets, identity-at-init, reparam
bit-equality, gradient/NaN health, full train-step + eval path). Not yet trained.

# Stereo Image Super-Resolution (StereoSR)

A next-generation stereo image super-resolution method featuring five architectural innovations:

| Innovation | Description | Expected Gain |
|---|---|---|
| **DECA** | Deformable Epipolar Cross-Attention | +0.15-0.25 dB |
| **HCPF** | Hierarchical Cross-View Pyramid Fusion | +0.10-0.20 dB |
| **Frequency-Aware Dual-Stream** | FFT-domain high-frequency processing | +0.20-0.35 dB |
| **Soft Occlusion Modeling** | Continuous confidence maps | +0.10-0.15 dB |
| **TCVA** | Transposed (channel-wise) Cross-View Attention | +0.05-0.10 dB |

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
# Flickr1024 (standard benchmark)
python scripts/prepare_data.py --dataset flickr1024 --output ./data/Flickr1024

# Generate LR images from HR
python scripts/prepare_data.py --generate_lr --output ./data/Flickr1024/train

# Verify dataset
python scripts/prepare_data.py --verify --output ./data/Flickr1024
```

**Expected directory structure:**
```
data/Flickr1024/
├── train/
│   ├── hr/
│   │   ├── left/   (0001.png, 0002.png, ...)
│   │   └── right/
│   ├── lr_x2/
│   │   ├── left/
│   │   └── right/
│   └── lr_x4/
│       ├── left/
│       └── right/
├── val/
└── test/
```

## Training

**Progressive training (4 phases, 350K iterations):**

```bash
# Base model (~6M params)
python train.py \
    --train_data ./data/Flickr1024/train \
    --val_data ./data/Flickr1024/val \
    --model_size base \
    --scale 4 \
    --batch_size 8 \
    --exp_name stereo_sr_base

# Large model (~25M params)
python train.py \
    --model_size large \
    --batch_size 4 \
    --exp_name stereo_sr_large

# Lightweight model (~1M params)
python train.py \
    --model_size tiny \
    --batch_size 16 \
    --exp_name stereo_sr_tiny
```

**Training phases:**

| Phase | Iterations | Losses | Patch Size |
|-------|-----------|--------|------------|
| 1 | 100K | L_pixel | 64×64 |
| 2 | 100K | + L_freq + L_stereo | 96×96 |
| 3 | 100K | + L_percep | 128×128 |
| 4 | 50K | + L_adv | 160×160 |

## Testing

```bash
python test.py \
    --test_data ./data/Flickr1024/test \
    --ckpt ./experiments/stereo_sr_base/checkpoints/best.pth \
    --scale 4 \
    --save_images \
    --tlsc
```

**Options:**
- `--tlsc`: Test-time Local Statistics Correction (from NAFSSR)
- `--save_images`: Save super-resolved output images

## Target Metrics (4× SR, Flickr1024)

| Model | Params | PSNR (dB) |
|-------|--------|-----------|
| Tiny | ~1M | 23.9-24.0 |
| Base | ~6M | 24.5-24.7 |
| Large | ~25M | 24.8-25.0 |

## Architecture

```
Input (I_L, I_R)
     ↓
Shared Shallow Feature Extraction (3×3 Conv)
     ↓
Encoder (shared weights)
  ├─ Stage 1: NAFBlocks × 4 + DECA (1× res)
  ├─ Stage 2: NAFBlocks × 6 + DECA (1/2 res)
  └─ Stage 3: NAFBlocks × 6 + TCVA (1/4 res)
     ↓
Hierarchical Cross-View Pyramid Fusion
  ├─ Coarse (1/4): Group-wise correlation
  ├─ Mid (1/2): DECA with guided range
  └─ Fine (1×): Full epipolar attention
     ↓
Frequency-Aware Refinement
  ├─ Spatial: NAFBlocks × 4
  └─ Frequency: FFCB × 2 + FFT fusion
     ↓
Occlusion-Aware Final Fusion
  ├─ Confidence prediction
  └─ Soft-weighted cross-view aggregation
     ↓
PixelShuffle Upsampling + Reconstruction
     ↓
Output (SR_L, SR_R)
```

## Loss Function

```
L_total = L_pixel + 0.1·L_percep + 0.01·L_adv + 0.3·L_freq + 0.2·L_stereo
```

| Loss | Implementation | Purpose |
|------|---------------|---------|
| L_pixel | Charbonnier (ε=10⁻³) | Base reconstruction |
| L_percep | VGG relu4_3 | Texture quality |
| L_adv | Relativistic average GAN | Realistic details |
| L_freq | Wavelet subband-specific | High-frequency preservation |
| L_stereo | Photometric + cycle consistency | Geometric constraints |

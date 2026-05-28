"""Configuration for StereoSR V6 (SpecFuse)."""
from __future__ import annotations

import argparse
import os


SIZE_PRESETS = {
    "tiny":  {"channels": 48, "num_blocks": 16},   # ~0.8M
    "small": {"channels": 64, "num_blocks": 16},   # ~1.5M
    "base":  {"channels": 64, "num_blocks": 24},   # ~2.3M
    "large": {"channels": 96, "num_blocks": 32},   # ~6.9M
}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # Model
    parser.add_argument("--model_size", type=str, default="small", choices=tuple(SIZE_PRESETS))
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--dw_expansion", type=float, default=2.0)
    parser.add_argument("--ffn_expansion", type=float, default=2.0)
    parser.add_argument("--no_scam", action="store_true", default=False,
                        help="Disable spatial cross-view attention (ablation).")
    parser.add_argument("--no_cvsfm", action="store_true", default=False,
                        help="Disable spectral cross-view fusion (ablation).")
    parser.add_argument("--cvsfm_no_gate", action="store_true", default=False,
                        help="Ablation A6: disable CVSFM's learnable per-frequency "
                             "gate; fuse magnitudes as a simple α=0.5 average. "
                             "Tests whether learnable mixing (gate) or magnitude "
                             "invariance is what the spectral path exploits.")
    parser.add_argument("--cvsfm_1d_fft", action="store_true", default=False,
                        help="Ablation A5: use 1D rfft along the width axis only "
                             "in CVSFM (disparity is purely horizontal, so the "
                             "shift theorem invoked by Proposition 1 is 1D).")
    parser.add_argument("--cvsfm_v2", action="store_true", default=False,
                        help="Use the theoretically-revised CVSFM v2: 1D rfft, "
                             "log-magnitude additive residual (generative, not "
                             "convex-combination bounded), and per-pixel image-"
                             "domain spatial gate (localised occlusion handling). "
                             "Drops v1's strict magnitude-invariance claim. "
                             "Overrides --cvsfm_no_gate and --cvsfm_1d_fft.")
    parser.add_argument("--cvsfm_v3", action="store_true", default=False,
                        help="Use CVSFM v3: phase-correlation cross-view alignment. "
                             "Operates on inter-view phase difference (the actual "
                             "stereo signature under the shift theorem) rather than "
                             "magnitudes (which are nearly equal between views and "
                             "carry no cross-view info). Predicts a complex alignment "
                             "filter from normalised cross-power. Recommended path. "
                             "Overrides --cvsfm_v2/--cvsfm_no_gate/--cvsfm_1d_fft.")
    parser.add_argument("--cvsfm_v3_window", type=int, default=0,
                        help="Window size for v3's 1D STFT along width. 0 (default) "
                             "uses full-row FFT. Setting e.g. 30 splits each row "
                             "into Hann-windowed frames with 50%% overlap-add (WOLA) "
                             "reconstruction — locally-constant disparity, analogue "
                             "of local attention.")
    parser.add_argument("--cvsfm_v3_hf_only", action="store_true", default=False,
                        help="Restrict v3's filter deviation from identity to mid/"
                             "high frequencies (linear ramp). DC and very low "
                             "frequencies are forced to A=1 (no change). Useful "
                             "when low-frequency structure should come from the "
                             "bicubic residual and per-view processing.")
    parser.add_argument("--cvsfm_v3_gain_scale", type=float, default=0.1,
                        help="Bound on v3's magnitude gain: |gain - 1| <= "
                             "cvsfm_v3_gain_scale. Default 0.1 → gain in [0.9, 1.1]. "
                             "Phase is always bounded in (-pi, pi).")
    parser.add_argument("--cvsfm_first", action="store_true", default=False,
                        help="Swap block ordering so CVSFM runs BEFORE SCAM "
                             "(default is NAFBlock → SCAM → CVSFM). Use this when "
                             "CVSFM v3 is intended as the primary cross-view "
                             "module and SCAM as a refinement step.")

    # Data
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=30)
    parser.add_argument("--patch_w", type=int, default=90)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 AMP (recommended on H100/A100/Ampere+). "
                             "Same speed as fp16 but with fp32 dynamic range — "
                             "eliminates the inf/NaN overflows that fp16 suffers "
                             "during long training runs. GradScaler is bypassed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)

    # Losses
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--stereo_weight", type=float, default=0.02)
    parser.add_argument("--spectral_weight", type=float, default=0.0,
                        help="Weight on L1(|F(SR_L)|, |F(SR_R)|) — derived from "
                             "v1's strict magnitude-invariance theorem (now "
                             "falsified by ablation A6). Disabled by default; "
                             "the loss penalises exactly the magnitude differences "
                             "that real occlusion and photometric mismatch produce, "
                             "and the architecture (CVSFM v2/v3) already handles "
                             "the spectral information channel internally.")
    parser.add_argument("--stereo_max_shift", type=int, default=48)

    # Logging
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v6")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--keep_ckpts", type=int, default=5,
                        help="Keep the last N periodic iter_*.pth checkpoints; "
                             "older ones are pruned. best.pth and final.pth are "
                             "always kept. Set to 0 to disable pruning.")

    # Evaluation
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False,
                        help="Enable inference-time TLSC (Chu et al., ECCV-W 2022): "
                             "swap NAFBlock SCA global pool for a windowed local pool "
                             "sized to the training LR patch.")
    parser.add_argument("--tlsc_base_h", type=int, default=None,
                        help="TLSC pool window height (defaults to --patch_h).")
    parser.add_argument("--tlsc_base_w", type=int, default=None,
                        help="TLSC pool window width (defaults to --patch_w).")


def get_config_v6(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo SR v6 (SpecFuse)")
    _add_common_args(parser)
    args = parser.parse_args(argv)

    preset = SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    return args

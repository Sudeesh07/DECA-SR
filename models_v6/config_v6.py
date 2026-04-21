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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)

    # Losses
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--stereo_weight", type=float, default=0.02)
    parser.add_argument("--spectral_weight", type=float, default=0.01)
    parser.add_argument("--stereo_max_shift", type=int, default=48)

    # Logging
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v6")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)

    # Evaluation
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False)


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

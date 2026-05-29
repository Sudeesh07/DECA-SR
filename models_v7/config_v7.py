"""Configuration for StereoSR V7.

New flags vs V6
---------------
* ``--patch_h 48 --patch_w 144`` — larger training patches (NAFSSR default).
* ``--ema_decay 0.9999 --ema_warmup 2000`` — EMA weight averaging.
* ``--adam_beta2 0.9`` — NAFSSR's lower β₂ for shorter gradient-variance
  window (default V6 was 0.99).
* ``--no_disparity_prior``, ``--sigma_init 4.0`` — disparity prior toggles.
* ``--no_multiscale``, ``--num_blocks_coarse 8`` — multi-scale trunk toggles.
* ``--rgb_shuffle`` — RGB channel shuffle augmentation (NAFSSR practice).
* ``--vertical_flip`` — random vertical flip augmentation. WARNING: this is
  NOT default-on; vertical flip preserves stereo geometry but disrupts the
  TLSC base-size convention for non-square patches. Use with care.
"""
from __future__ import annotations

import argparse
import os


SIZE_PRESETS = {
    # channels, main blocks, coarse blocks
    "tiny":  {"channels": 48, "num_blocks": 16, "num_blocks_coarse": 6},
    "small": {"channels": 64, "num_blocks": 16, "num_blocks_coarse": 8},
    "base":  {"channels": 64, "num_blocks": 24, "num_blocks_coarse": 12},
    "large": {"channels": 96, "num_blocks": 32, "num_blocks_coarse": 16},
}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # Model
    parser.add_argument("--model_size", type=str, default="small", choices=tuple(SIZE_PRESETS))
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--num_blocks_coarse", type=int, default=None)
    parser.add_argument("--dw_expansion", type=float, default=2.0)
    parser.add_argument("--ffn_expansion", type=float, default=2.0)
    parser.add_argument("--no_disparity_prior", action="store_true", default=False,
                        help="Disable the soft-argmax disparity bias on SCAM. "
                             "Ablation: V7 reduces to NAFSSR + multiscale.")
    parser.add_argument("--sigma_init", type=float, default=4.0,
                        help="Initial Gaussian sigma (in pixels) for the disparity "
                             "bias. Larger = softer prior at start.")
    parser.add_argument("--no_multiscale", action="store_true", default=False,
                        help="Disable the LR/2 coarse branch. Ablation: V7 reduces "
                             "to NAFSSR + disparity prior.")

    # Data
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=48,
                        help="LR patch height. NAFSSR uses 30 (≈30×90 LR patches); "
                             "V7 defaults to 48 (48×144 LR) for ~2.6× receptive-field "
                             "utilisation, typically +0.05–0.15 dB on Flickr1024.")
    parser.add_argument("--patch_w", type=int, default=144,
                        help="LR patch width. See --patch_h.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    parser.add_argument("--rgb_shuffle", action="store_true", default=True,
                        help="Random RGB channel permutation augmentation "
                             "(NAFSSR practice). Same permutation applied to L and R.")
    parser.add_argument("--no_rgb_shuffle", action="store_false", dest="rgb_shuffle")
    parser.add_argument("--vertical_flip", action="store_true", default=False,
                        help="Random vertical flip augmentation. Preserves stereo "
                             "geometry. Disabled by default.")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base LR before linear scaling by effective batch.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.9,
                        help="NAFSSR uses (0.9, 0.9). V6 used (0.9, 0.99). "
                             "Lower β₂ tracks gradient variance over a shorter "
                             "window — empirically helps SR.")
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 AMP. Recommended on H100/A100/Ampere+.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA target decay. 0 disables EMA entirely.")
    parser.add_argument("--ema_warmup", type=int, default=2000,
                        help="EMA decay warmup: effective decay ramps from "
                             "0 to ema_decay over this many updates.")

    # Losses
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--stereo_weight", type=float, default=0.02)
    parser.add_argument("--stereo_max_shift", type=int, default=48)

    # Logging
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v7")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--keep_ckpts", type=int, default=5)

    # Evaluation
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False,
                        help="Enable inference-time TLSC.")
    parser.add_argument("--tlsc_base_h", type=int, default=None)
    parser.add_argument("--tlsc_base_w", type=int, default=None)
    parser.add_argument("--ensemble", action="store_true", default=False,
                        help="Self-ensemble at inference: average 8 forwards with "
                             "rotations & flips (with view swap under horizontal "
                             "flip). Typical gain: +0.05–0.10 dB.")
    parser.add_argument("--use_ema_eval", action="store_true", default=True,
                        help="At eval, load the EMA weights from the checkpoint "
                             "(if present) instead of the live weights.")
    parser.add_argument("--no_ema_eval", action="store_false", dest="use_ema_eval")


def get_config_v7(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo SR V7")
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

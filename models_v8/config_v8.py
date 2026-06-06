"""Configuration for StereoSR V8.

V8 = V6 A1 (proven SCAM backbone) + CT-CVA novelty + engineering bundle.

What's new vs V7
----------------
* CT-CVA (channel-transposed cross-view attention) replaces V7's failed
  disparity-prior and multi-scale-coarse-trunk.
* All V7's engineering bundle is preserved (EMA, AdamW(0.9, 0.9), patches
  48×144, self-ensemble) and extended with the full NTIRE-2024 augmentation
  stack (RGB shuffle, h-flip+swap, optional v-flip, h-shift).
"""
from __future__ import annotations

import argparse
import os


SIZE_PRESETS = {
    # channels, main blocks
    "tiny":  {"channels": 48, "num_blocks": 16},
    "small": {"channels": 64, "num_blocks": 16},
    "base":  {"channels": 64, "num_blocks": 24},
    "large": {"channels": 96, "num_blocks": 32},
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
                        help="Disable SCAM (spatial cross-view). Ablation.")
    parser.add_argument("--no_ct_cva", action="store_true", default=False,
                        help="Disable CT-CVA (channel cross-view). Ablation = "
                             "V6 A1 + engineering bundle.")
    parser.add_argument("--ct_cva_heads", type=int, default=4,
                        help="Number of heads for CT-CVA channel attention. "
                             "channels must be divisible by this.")
    parser.add_argument("--ct_cva_temperature_init", type=float, default=1.0,
                        help="Initial per-head temperature for CT-CVA softmax. "
                             "Restormer default; tune if attention is too sharp/flat.")

    # Data
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=48,
                        help="LR patch height. V6 used 30; V8 default 48 "
                             "(NAFSSR practice).")
    parser.add_argument("--patch_w", type=int, default=144,
                        help="LR patch width. V8 default 144.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    # NTIRE-2024 augmentation stack
    parser.add_argument("--aug_rgb_shuffle", action="store_true", default=True,
                        help="Random RGB channel permutation augmentation.")
    parser.add_argument("--no_aug_rgb_shuffle", action="store_false", dest="aug_rgb_shuffle")
    parser.add_argument("--aug_hflip", action="store_true", default=True,
                        help="Horizontal flip with L/R view swap augmentation.")
    parser.add_argument("--no_aug_hflip", action="store_false", dest="aug_hflip")
    parser.add_argument("--aug_vflip", action="store_true", default=False,
                        help="Vertical flip augmentation. Default OFF — "
                             "preserves epipolar geometry but is less commonly used.")
    parser.add_argument("--aug_hshift", action="store_true", default=True,
                        help="Random horizontal shift (same L/R) augmentation.")
    parser.add_argument("--no_aug_hshift", action="store_false", dest="aug_hshift")
    parser.add_argument("--aug_hshift_max", type=int, default=8,
                        help="Max LR pixel shift for hshift augmentation.")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base LR before linear scaling by effective batch.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.9,
                        help="NAFSSR uses (0.9, 0.9). Lower β₂ tracks gradient "
                             "variance over a shorter window — empirically "
                             "+0.02–0.05 dB on SR tasks.")
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
                        help="EMA decay ramp from 0 to ema_decay over this many "
                             "iterations.")

    # Losses
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--stereo_weight", type=float, default=0.02)
    parser.add_argument("--stereo_max_shift", type=int, default=48)

    # Logging
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v8")
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
                        help="Self-ensemble at inference: 4 epipolar-preserving "
                             "transforms averaged. Typical +0.05–0.10 dB.")
    parser.add_argument("--use_ema_eval", action="store_true", default=True,
                        help="At eval, prefer EMA weights from the checkpoint.")
    parser.add_argument("--no_ema_eval", action="store_false", dest="use_ema_eval")


def get_config_v8(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo SR V8")
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

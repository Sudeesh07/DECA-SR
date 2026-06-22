"""Configuration for StereoSR V9 (DECA-SR).

V9 = RepLK-NAFBlock trunk + SCAM (every block) + DECVF (deformable cost-view
fusion, every ``decvf_every``-th block) + full near-free training-lever stack.
Param budgets: tiny <0.8M, small <2M, base <10M.
"""
from __future__ import annotations

import argparse
import os


SIZE_PRESETS = {
    # channels, main blocks, large-kernel size, DECVF placement period
    "tiny":  {"channels": 40, "num_blocks": 14, "large_kernel": 9,  "decvf_every": 4},
    "small": {"channels": 64, "num_blocks": 16, "large_kernel": 9,  "decvf_every": 2},
    "base":  {"channels": 96, "num_blocks": 28, "large_kernel": 11, "decvf_every": 2},
    "large": {"channels": 128, "num_blocks": 36, "large_kernel": 13, "decvf_every": 2},
}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # Model
    parser.add_argument("--model_size", type=str, default="small", choices=tuple(SIZE_PRESETS))
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--large_kernel", type=int, default=None)
    parser.add_argument("--dw_expansion", type=float, default=2.0)
    parser.add_argument("--ffn_expansion", type=float, default=2.0)
    parser.add_argument("--no_scam", action="store_true", default=False,
                        help="Disable SCAM (dense epipolar row attention). Ablation.")
    parser.add_argument("--no_decvf", action="store_true", default=False,
                        help="Disable DECVF (the deformable cost-view fusion). Ablation "
                             "= RepLK-NAF + SCAM baseline (the bar DECVF must beat).")
    parser.add_argument("--decvf_every", type=int, default=None,
                        help="Place DECVF on every k-th trunk block.")
    parser.add_argument("--cost_volume_d", type=int, default=24,
                        help="Cost-volume disparity search range (LR pixels).")
    parser.add_argument("--decvf_k", type=int, default=8,
                        help="Number of deformable sample points per query.")
    parser.add_argument("--decvf_y_range", type=float, default=1.5,
                        help="Max |y-offset| (px) for deformable sampling (epipolar bias).")

    # Data
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=48, help="LR patch height.")
    parser.add_argument("--patch_w", type=int, default=144, help="LR patch width.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    # Augmentation stack
    parser.add_argument("--aug_rgb_shuffle", action="store_true", default=True)
    parser.add_argument("--no_aug_rgb_shuffle", action="store_false", dest="aug_rgb_shuffle")
    parser.add_argument("--aug_hflip", action="store_true", default=True)
    parser.add_argument("--no_aug_hflip", action="store_false", dest="aug_hflip")
    parser.add_argument("--aug_vflip", action="store_true", default=False)
    parser.add_argument("--aug_hshift", action="store_true", default=True)
    parser.add_argument("--no_aug_hshift", action="store_false", dest="aug_hshift")
    parser.add_argument("--aug_hshift_max", type=int, default=8)
    parser.add_argument("--aug_mixup", action="store_true", default=False,
                        help="Stereo Mixup augmentation (Beta(0.2,0.2)).")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base LR before linear scaling by effective batch.")
    parser.add_argument("--max_lr", type=float, default=1e-3,
                        help="Hard ceiling on the batch-scaled LR (divergence guard).")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.9)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--terminal_mse_iters", type=int, default=15000,
                        help="Final N iters: switch pixel loss to MSE, drop aux terms.")
    parser.add_argument("--swa_topk", type=int, default=5,
                        help="Average the top-K best-val checkpoints at the end (0=off).")

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_warmup", type=int, default=2000)

    # Losses
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--stereo_weight", type=float, default=0.02)
    parser.add_argument("--stereo_max_shift", type=int, default=48)
    parser.add_argument("--ffl_weight", type=float, default=0.1,
                        help="Per-view focal-frequency loss weight (0 disables).")

    # Logging
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v9")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--keep_ckpts", type=int, default=5)

    # Evaluation
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False)
    parser.add_argument("--tlsc_base_h", type=int, default=None)
    parser.add_argument("--tlsc_base_w", type=int, default=None)
    parser.add_argument("--ensemble", action="store_true", default=False)
    parser.add_argument("--reparam_eval", action="store_true", default=False,
                        help="Merge RepLK branches to a single DW conv before eval.")
    parser.add_argument("--allow_partial_load", action="store_true", default=False,
                        help="Permit a state-dict load with missing/unexpected keys "
                             "(architecture differs from checkpoint). Off by default so "
                             "an arch mismatch fails loudly instead of scoring a wrong net.")
    parser.add_argument("--use_ema_eval", action="store_true", default=True)
    parser.add_argument("--no_ema_eval", action="store_false", dest="use_ema_eval")


def get_config_v9(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo SR V9 (DECA-SR)")
    _add_common_args(parser)
    args = parser.parse_args(argv)

    preset = SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.decvf_every is not None and args.decvf_every < 1:
        parser.error("--decvf_every must be >= 1; use --no_decvf to disable DECVF")

    args.output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    return args

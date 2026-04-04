from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from .stereo_fusion import resolve_default_stereo_ranges


SIZE_PRESETS = {
    "tiny": {
        "stage_channels": (48, 72, 96),
        "enc_blocks": (3, 4, 6),
        "bottleneck_blocks": 3,
        "decoder_blocks": (3, 3),
        "hr_blocks": (2, 2),
        "deform_heads": 4,
        "deform_points": 9,
    },
    "base": {
        "stage_channels": (64, 96, 128),
        "enc_blocks": (4, 6, 8),
        "bottleneck_blocks": 4,
        "decoder_blocks": (4, 4),
        "hr_blocks": (3, 3),
        "deform_heads": 4,
        "deform_points": 9,
    },
    "large": {
        "stage_channels": (80, 120, 160),
        "enc_blocks": (5, 8, 10),
        "bottleneck_blocks": 5,
        "decoder_blocks": (5, 5),
        "hr_blocks": (4, 4),
        "deform_heads": 8,
        "deform_points": 9,
    },
}


@dataclass
class V5Config:
    scale: int = 4
    in_channels: int = 3
    stage_channels: tuple[int, int, int] = (64, 96, 128)
    enc_blocks: tuple[int, int, int] = (4, 6, 8)
    bottleneck_blocks: int = 4
    decoder_blocks: tuple[int, int] = (4, 4)
    hr_blocks: tuple[int, int] = (3, 3)
    deform_heads: int = 4
    deform_points: int = 9
    stereo_heads: int = 4
    stereo_points: int = 5
    depa_max_disparity: int | None = None
    corr_max_disp: int | None = None
    sample_span: float = 2.0
    residual_range: float = 1.5
    num_refine_iters: int = 2

    data_root: str = "datasets/StereoSR"
    test_source: str = "Flickr1024"
    patch_h: int = 30
    patch_w: int = 90
    augment: bool = True
    num_workers: int = 4

    batch_size: int = 8
    total_iters: int = 250000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_iters: int = 2000
    grad_clip: float = 1.0
    fp16: bool = True
    seed: int = 42
    resume: str | None = None
    pretrain: str | None = None

    pixel_loss: str = "charbonnier"
    charbonnier_eps: float = 1e-3
    mse_weight: float = 0.05
    stereo_weight: float = 0.05
    smooth_weight: float = 0.01

    log_dir: str = "./experiments"
    exp_name: str = "stereo_sr_v5"
    log_interval: int = 100
    val_interval: int = 2500
    save_interval: int = 5000

    ckpt: str | None = None
    save_images: bool = False
    tlsc: bool = False

    def resolve_stereo_ranges(self) -> tuple[int, int]:
        depa_default, corr_default = resolve_default_stereo_ranges(self.scale)
        depa = depa_default if self.depa_max_disparity is None else self.depa_max_disparity
        corr = corr_default if self.corr_max_disp is None else self.corr_max_disp
        return depa, corr


def _preset_defaults(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_size", type=str, default="base", choices=tuple(SIZE_PRESETS))
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--stage_channels", nargs=3, type=int, default=None)
    parser.add_argument("--enc_blocks", nargs=3, type=int, default=None)
    parser.add_argument("--bottleneck_blocks", type=int, default=None)
    parser.add_argument("--decoder_blocks", nargs=2, type=int, default=None)
    parser.add_argument("--hr_blocks", nargs=2, type=int, default=None)
    parser.add_argument("--deform_heads", type=int, default=None)
    parser.add_argument("--deform_points", type=int, default=None)
    parser.add_argument("--stereo_heads", type=int, default=4)
    parser.add_argument("--stereo_points", type=int, default=5)
    parser.add_argument("--depa_max_disparity", type=int, default=None)
    parser.add_argument("--corr_max_disp", type=int, default=None)
    parser.add_argument("--sample_span", type=float, default=2.0)
    parser.add_argument("--residual_range", type=float, default=1.5)
    parser.add_argument("--stochastic_depth", type=float, default=0.1)
    parser.add_argument("--num_refine_iters", type=int, default=2)
    return parser


def get_config_v5(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo SR v5 (DWT + iterative refinement)")
    _preset_defaults(parser)

    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=30)
    parser.add_argument("--patch_w", type=int, default=90)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=250000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)

    parser.add_argument("--pixel_loss", type=str, default="charbonnier", choices=["charbonnier", "l1"])
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--stereo_weight", type=float, default=0.05)
    parser.add_argument("--smooth_weight", type=float, default=0.01)

    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v5")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False)

    args = parser.parse_args(argv)

    preset = SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    args.stage_channels = tuple(args.stage_channels)
    args.enc_blocks = tuple(args.enc_blocks)
    args.decoder_blocks = tuple(args.decoder_blocks)
    args.hr_blocks = tuple(args.hr_blocks)

    depa_default, corr_default = resolve_default_stereo_ranges(args.scale)
    if args.depa_max_disparity is None:
        args.depa_max_disparity = depa_default
    if args.corr_max_disp is None:
        args.corr_max_disp = corr_default

    args.output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    return args

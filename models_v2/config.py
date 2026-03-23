import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description="Stereo SR V2 Training")

    # ── Model ───────────────────────────────────────────────────────────
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--enc_blocks", nargs="+", type=int, default=[4, 6, 6])
    parser.add_argument("--dec_blocks", nargs="+", type=int, default=[4, 4])
    parser.add_argument("--depa_k", type=int, default=8)
    parser.add_argument("--depa_max_disparity", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--corr_max_disp", type=int, default=12)
    parser.add_argument("--stochastic_depth", type=float, default=0.1)
    parser.add_argument("--model_size", type=str, default="base", choices=["tiny", "base", "large"])

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=30, help="LR patch height (x2 baseline)")
    parser.add_argument("--patch_w", type=int, default=90, help="LR patch width (x2 baseline)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    # ── Training ────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=350000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_iters", type=int, default=5000)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)

    # ── Curriculum phases ───────────────────────────────────────────────
    # Patch sizes scale from the x2 baseline (30x90).
    # Phase 1: baseline, Phase 2-4: progressively larger crops.
    parser.add_argument("--phase1_iters", type=int, default=100000)
    parser.add_argument("--phase2_iters", type=int, default=100000)
    parser.add_argument("--phase3_iters", type=int, default=100000)
    parser.add_argument("--phase4_iters", type=int, default=50000)

    # ── Loss weights ────────────────────────────────────────────────────
    parser.add_argument("--w_pixel", type=float, default=1.0)
    parser.add_argument("--w_percep", type=float, default=0.1)
    parser.add_argument("--w_adv", type=float, default=0.01)
    parser.add_argument("--w_freq", type=float, default=0.3)
    parser.add_argument("--w_stereo", type=float, default=0.2)

    # ── Logging ─────────────────────────────────────────────────────────
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr_v2")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=10000)

    # ── Testing ─────────────────────────────────────────────────────────
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False)

    args = parser.parse_args()

    # ── Model size presets ──────────────────────────────────────────────
    size_map = {
        "tiny": {
            "embed_dim": 32, "enc_blocks": [2, 4, 4], "dec_blocks": [2, 2],
            "depa_k": 4, "num_heads": 2, "corr_max_disp": 8,
        },
        "base": {
            "embed_dim": 48, "enc_blocks": [4, 6, 6], "dec_blocks": [4, 4],
            "depa_k": 8, "num_heads": 4, "corr_max_disp": 12,
        },
        "large": {
            "embed_dim": 64, "enc_blocks": [6, 8, 8, 8], "dec_blocks": [4, 4, 4],
            "depa_k": 8, "num_heads": 4, "corr_max_disp": 16,
            "stochastic_depth": 0.2,
        },
    }
    if args.model_size in size_map:
        for k, v in size_map[args.model_size].items():
            setattr(args, k, v)

    args.output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    return args

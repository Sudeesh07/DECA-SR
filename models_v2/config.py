import argparse
import os


SIZE_PRESETS = {
    "tiny": dict(
        embed_dim=32, enc_blocks=[2, 4, 4], dec_blocks=[2, 2],
        depa_k=4, num_heads=2, corr_max_disp=8,
        stochastic_depth=0.05,
    ),
    "base": dict(
        embed_dim=64, enc_blocks=[4, 6, 6], dec_blocks=[4, 4],
        depa_k=8, num_heads=4, corr_max_disp=12,
        stochastic_depth=0.1,
    ),
    "large": dict(
        embed_dim=96, enc_blocks=[6, 8, 8, 8], dec_blocks=[4, 4, 4],
        depa_k=8, num_heads=4, corr_max_disp=16,
        stochastic_depth=0.2,
    ),
}

_PRESET_KEYS = set(SIZE_PRESETS["base"].keys())


def get_config(argv=None):
    parser = argparse.ArgumentParser(description="Stereo SR V2 Training")

    # ── Model ───────────────────────────────────────────────────────────
    parser.add_argument("--model_size", type=str, default="base",
                        choices=list(SIZE_PRESETS.keys()))
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--enc_blocks", nargs="+", type=int, default=None)
    parser.add_argument("--dec_blocks", nargs="+", type=int, default=None)
    parser.add_argument("--depa_k", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--corr_max_disp", type=int, default=None)
    parser.add_argument("--stochastic_depth", type=float, default=None)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--depa_max_disparity", type=int, default=32)
    parser.add_argument("--scam_interval", type=int, default=2,
                        help="Lightweight SCAM every N NAFBlocks (0=disable)")

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
    parser.add_argument("--test_source", type=str, default="Flickr1024")
    parser.add_argument("--patch_h", type=int, default=30)
    parser.add_argument("--patch_w", type=int, default=90)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    # ── Training ────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=350000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay (0=disable)")

    # ── Curriculum phases ───────────────────────────────────────────────
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

    # ── Sanity check modes ──────────────────────────────────────────────
    parser.add_argument("--sanity", action="store_true", default=False,
                        help="Quick convergence test: 2K iters, pixel loss only")
    parser.add_argument("--sanity_full", action="store_true", default=False,
                        help="Multi-phase convergence test: 4K iters, all phases")

    args = parser.parse_args(argv)

    # ── Apply model size preset (only for keys user didn't override) ───
    preset = SIZE_PRESETS[args.model_size]
    for k, v in preset.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    # ── Sanity check overrides ──────────────────────────────────────────
    #
    # --sanity: Pure convergence test (pixel loss only, no phases)
    #   2K iters. Verifies the model can learn basic reconstruction.
    #   Expected: PSNR 8→25 dB. If stalls below 20 at iter 500, broken.
    #
    # --sanity_full: Multi-phase curriculum test (all 4 phases)
    #   4K iters (1K per phase). Verifies:
    #   - Phase transitions don't crash or spike the loss
    #   - Each new loss term activates and produces finite values
    #   - PSNR doesn't regress when new losses are added
    #   - GAN training (phase 4) doesn't destabilize
    #   - Confidence maps are reasonable (not all 0 or all 1)
    #
    #   Expected per-phase behavior:
    #     Phase 1 (pixel):       PSNR climbs to ~24-25 dB
    #     Phase 2 (+freq+stereo): Small dip (~0.2 dB) then recovers
    #     Phase 3 (+percep+conf): Another small dip, then recovers
    #     Phase 4 (+GAN):         PSNR may drop ~0.5 dB (perceptual vs PSNR tradeoff)
    #
    #   Red flags:
    #     - Loss becomes NaN at any phase transition
    #     - PSNR drops >2 dB at a transition and doesn't recover within 200 iters
    #     - adv_d stays at exactly 0.5 (discriminator not learning)
    #     - adv_g explodes (mode collapse)
    #     - conf_reg is exactly 0.693 (ln(2)) — confidence stuck at 0.5

    _SHARED_SANITY = dict(
        log_interval=50,
        val_interval=500,
        save_interval=2000,
    )

    if args.sanity:
        args.total_iters = 2000
        args.warmup_iters = 200
        args.exp_name = f"sanity_{args.model_size}_x{args.scale}"
        args.phase1_iters = 2000
        args.phase2_iters = 0
        args.phase3_iters = 0
        args.phase4_iters = 0
        for k, v in _SHARED_SANITY.items():
            setattr(args, k, v)

    elif args.sanity_full:
        args.total_iters = 4000
        args.warmup_iters = 100
        args.exp_name = f"sanity_full_{args.model_size}_x{args.scale}"
        # 1K iters per phase — enough to see if each phase is stable
        args.phase1_iters = 1000
        args.phase2_iters = 1000
        args.phase3_iters = 1000
        args.phase4_iters = 1000
        for k, v in _SHARED_SANITY.items():
            setattr(args, k, v)
        # Validate at every phase boundary
        args.val_interval = 250

    # ── Directories ─────────────────────────────────────────────────────
    args.output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    return args
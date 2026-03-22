import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description="Stereo Image Super-Resolution (fixed config)")

    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--enc_blocks", nargs="+", type=int, default=[4, 6, 6], help="NAFBlocks per encoder stage")
    parser.add_argument("--enc_channels", nargs="+", type=int, default=None, help="Optional channel list len(enc_blocks)+1")
    parser.add_argument("--deca_k", type=int, default=8)
    parser.add_argument("--deca_max_disparity", type=int, default=32, help="Maximum disparity used by DECA offset sampling")
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--coarse_max_disp", type=int, default=4, help="Coarse disparity search range used by standalone HCPF")
    parser.add_argument("--freq_blocks", type=int, default=2)
    parser.add_argument("--spatial_blocks", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="base", choices=["tiny", "base", "large"])

    parser.add_argument("--data_root", type=str, default="datasets/StereoSR", help="Root directory of the dataset")
    parser.add_argument("--test_source", type=str, default="Flickr1024", help="Test dataset name under data_root/test/")
    parser.add_argument("--patch_h", type=int, default=30, help="Training patch height in LR space")
    parser.add_argument("--patch_w", type=int, default=90, help="Training patch width in LR space")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_iters", type=int, default=350000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_iters", type=int, default=5000)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--stochastic_depth", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)

    parser.add_argument("--phase1_iters", type=int, default=100000, help="Phase 1: pixel loss only, patch 64")
    parser.add_argument("--phase2_iters", type=int, default=100000, help="Phase 2: + frequency + stereo losses, patch 96")
    parser.add_argument("--phase3_iters", type=int, default=100000, help="Phase 3: + perceptual loss, patch 128")
    parser.add_argument("--phase4_iters", type=int, default=50000, help="Phase 4: + adversarial loss, patch 160")

    parser.add_argument("--w_pixel", type=float, default=1.0)
    parser.add_argument("--w_percep", type=float, default=0.1)
    parser.add_argument("--w_adv", type=float, default=0.01)
    parser.add_argument("--w_freq", type=float, default=0.3)
    parser.add_argument("--w_stereo", type=float, default=0.2)

    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="stereo_sr")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=10000)

    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--tlsc", action="store_true", default=False)

    args = parser.parse_args()

    size_map = {
        "tiny": {"embed_dim": 32, "enc_blocks": [2, 4, 4]},
        "base": {"embed_dim": 48, "enc_blocks": [4, 6, 6]},
        "large": {"embed_dim": 64, "enc_blocks": [6, 8, 8, 8], "enc_channels": [64, 64, 128, 128, 128]},
    }
    if args.model_size in size_map:
        for k, v in size_map[args.model_size].items():
            setattr(args, k, v)

    args.output_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    return args

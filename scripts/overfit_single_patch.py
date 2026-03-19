import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running as: python scripts/overfit_single_patch.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.losses_fixed import CombinedLoss, Discriminator, RelativisticAverageGANLoss
from models.network_standalone import build_model_standalone
from utils_fixed import compute_psnr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten.to(device)


def save_tensor_image(img: torch.Tensor, path: str) -> None:
    img = img.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def infer_patch_dir(data_root: str, scale: int, scene: str) -> str:
    return os.path.join(data_root, f"patches_x{scale}", scene)


def build_lr_from_hr(hr: torch.Tensor, scale: int) -> torch.Tensor:
    h, w = hr.shape[-2:]
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    hr = hr[:, :, :h2, :w2]
    lr = F.interpolate(hr, size=(h2 // scale, w2 // scale), mode="bicubic", align_corners=False)
    return lr


def get_phase(iteration: int, args: argparse.Namespace) -> int:
    if iteration < args.phase1_iters:
        return 1
    if iteration < args.phase1_iters + args.phase2_iters:
        return 2
    if iteration < args.phase1_iters + args.phase2_iters + args.phase3_iters:
        return 3
    return 4


def get_patch_size(phase: int, args: argparse.Namespace) -> int:
    sizes = {
        1: args.phase1_patch,
        2: args.phase2_patch,
        3: args.phase3_patch,
        4: args.phase4_patch,
    }
    return sizes.get(phase, args.phase4_patch)


def _safe_pad(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = img.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        # reflect padding requires each pad value to be < corresponding input size.
        can_reflect = (h > 1 and w > 1 and pad_h < h and pad_w < w)
        pad_mode = "reflect" if can_reflect else "replicate"
        img = F.pad(img, (0, pad_w, 0, pad_h), mode=pad_mode)
    return img


def _stereo_augment(
    hr_left: torch.Tensor,
    hr_right: torch.Tensor,
    lr_left: torch.Tensor,
    lr_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if random.random() > 0.5:
        hr_left = torch.flip(hr_left, dims=[3])
        hr_right = torch.flip(hr_right, dims=[3])
        lr_left = torch.flip(lr_left, dims=[3])
        lr_right = torch.flip(lr_right, dims=[3])
        hr_left, hr_right = hr_right, hr_left
        lr_left, lr_right = lr_right, lr_left

    if random.random() > 0.5:
        hr_left = torch.flip(hr_left, dims=[2])
        hr_right = torch.flip(hr_right, dims=[2])
        lr_left = torch.flip(lr_left, dims=[2])
        lr_right = torch.flip(lr_right, dims=[2])

    if random.random() > 0.5:
        perm = torch.randperm(3, device=hr_left.device)
        hr_left = hr_left[:, perm, :, :]
        hr_right = hr_right[:, perm, :, :]
        lr_left = lr_left[:, perm, :, :]
        lr_right = lr_right[:, perm, :, :]

    return hr_left, hr_right, lr_left, lr_right


def build_single_batch(
    hr_left_full: torch.Tensor,
    hr_right_full: torch.Tensor,
    lr_left_full: torch.Tensor,
    lr_right_full: torch.Tensor,
    scale: int,
    patch_size: int,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hr_patch = patch_size * scale

    lr_left = _safe_pad(lr_left_full, patch_size, patch_size)
    lr_right = _safe_pad(lr_right_full, patch_size, patch_size)

    _, _, lr_h, lr_w = lr_left.shape
    top_lr = random.randint(0, lr_h - patch_size)
    left_lr = random.randint(0, lr_w - patch_size)

    top_hr = top_lr * scale
    left_hr = left_lr * scale

    hr_left = _safe_pad(hr_left_full, top_hr + hr_patch, left_hr + hr_patch)
    hr_right = _safe_pad(hr_right_full, top_hr + hr_patch, left_hr + hr_patch)

    lr_left = lr_left[:, :, top_lr:top_lr + patch_size, left_lr:left_lr + patch_size]
    lr_right = lr_right[:, :, top_lr:top_lr + patch_size, left_lr:left_lr + patch_size]
    hr_left = hr_left[:, :, top_hr:top_hr + hr_patch, left_hr:left_hr + hr_patch]
    hr_right = hr_right[:, :, top_hr:top_hr + hr_patch, left_hr:left_hr + hr_patch]

    if augment:
        hr_left, hr_right, lr_left, lr_right = _stereo_augment(
            hr_left, hr_right, lr_left, lr_right
        )

    return lr_left, lr_right, hr_left, hr_right


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    def lr_lambda(step: int) -> float:
        if step < args.warmup_iters:
            return step / max(args.warmup_iters, 1)
        progress = (step - args.warmup_iters) / max(args.total_iters - args.warmup_iters, 1)
        return max(0.5 * (1 + np.cos(np.pi * progress)), 1e-6)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def active_losses(criterion: CombinedLoss) -> list[str]:
    names = ["pixel"]
    if criterion.use_freq:
        names.append("freq")
    if criterion.use_stereo:
        names.append("stereo")
    if criterion.use_percep:
        names.append("percep")
    if criterion.use_adv:
        names.append("adv")
    return names


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image multistage overfit trainer")
    parser.add_argument("--patch_dir", type=str, default=None, help="Folder containing hr0/hr1/lr0/lr1 images")
    parser.add_argument("--data_root", type=str, default="datasets/StereoSR", help="StereoSR root")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--scene", type=str, default="000001", help="Scene folder under patches_x{scale}")

    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "base", "large"])
    parser.add_argument("--deca_k", type=int, default=8)
    parser.add_argument("--deca_max_disparity", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--coarse_max_disp", type=int, default=4)
    parser.add_argument("--spatial_blocks", type=int, default=2)
    parser.add_argument("--freq_blocks", type=int, default=1)
    parser.add_argument("--stochastic_depth", type=float, default=0.0)

    parser.add_argument("--total_iters", type=int, default=1000)
    parser.add_argument("--phase1_iters", type=int, default=250)
    parser.add_argument("--phase2_iters", type=int, default=250)
    parser.add_argument("--phase3_iters", type=int, default=250)
    parser.add_argument("--phase4_iters", type=int, default=250)

    parser.add_argument("--phase1_patch", type=int, default=64)
    parser.add_argument("--phase2_patch", type=int, default=96)
    parser.add_argument("--phase3_patch", type=int, default=128)
    parser.add_argument("--phase4_patch", type=int, default=160)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--w_pixel", type=float, default=1.0)
    parser.add_argument("--w_percep", type=float, default=0.1)
    parser.add_argument("--w_adv", type=float, default=0.01)
    parser.add_argument("--w_freq", type=float, default=0.3)
    parser.add_argument("--w_stereo", type=float, default=0.2)

    parser.add_argument("--out_dir", type=str, default="experiments/overfit_single_patch")
    args = parser.parse_args()

    if args.total_iters <= 0:
        args.total_iters = args.phase1_iters + args.phase2_iters + args.phase3_iters + args.phase4_iters

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.patch_dir is None:
        patch_dir = infer_patch_dir(args.data_root, args.scale, args.scene)
    else:
        patch_dir = args.patch_dir

    hr0_path = os.path.join(patch_dir, "hr0.png")
    hr1_path = os.path.join(patch_dir, "hr1.png")
    lr0_path = os.path.join(patch_dir, "lr0.png")
    lr1_path = os.path.join(patch_dir, "lr1.png")

    if not os.path.exists(hr0_path) or not os.path.exists(hr1_path):
        raise FileNotFoundError(
            f"Missing HR files in {patch_dir}. Expected hr0.png and hr1.png"
        )

    hr_left = load_rgb_tensor(hr0_path, device)
    hr_right = load_rgb_tensor(hr1_path, device)

    if os.path.exists(lr0_path) and os.path.exists(lr1_path):
        lr_left = load_rgb_tensor(lr0_path, device)
        lr_right = load_rgb_tensor(lr1_path, device)
    else:
        lr_left = build_lr_from_hr(hr_left, args.scale)
        lr_right = build_lr_from_hr(hr_right, args.scale)

    model_args = SimpleNamespace(
        scale=args.scale,
        in_channels=3,
        embed_dim=32 if args.model_size == "tiny" else (48 if args.model_size == "base" else 64),
        enc_blocks=[2, 4, 4] if args.model_size == "tiny" else ([4, 6, 6] if args.model_size == "base" else [6, 8, 8, 8]),
        enc_channels=None if args.model_size != "large" else [64, 64, 128, 128, 128],
        deca_k=args.deca_k,
        deca_max_disparity=args.deca_max_disparity,
        num_heads=args.num_heads if args.model_size != "large" else max(args.num_heads, 4),
        coarse_max_disp=args.coarse_max_disp,
        spatial_blocks=args.spatial_blocks,
        freq_blocks=args.freq_blocks,
        stochastic_depth=args.stochastic_depth,
    )

    model = build_model_standalone(model_args).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    criterion = CombinedLoss(
        w_pixel=args.w_pixel,
        w_percep=args.w_percep,
        w_adv=args.w_adv,
        w_freq=args.w_freq,
        w_stereo=args.w_stereo,
        device=str(device),
    ).to(device)
    gan_loss = RelativisticAverageGANLoss()
    discriminator = Discriminator().to(device)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.9, 0.999))
    scheduler_d = build_scheduler(optimizer_d, args)

    best_psnr = -1.0
    current_phase = 0

    print(f"Patch dir: {patch_dir}")
    print(f"LR shape: {tuple(lr_left.shape)} | HR shape: {tuple(hr_left.shape)}")
    print(f"Device: {device}")

    for step in range(1, args.total_iters + 1):
        phase = get_phase(step - 1, args)
        if phase != current_phase:
            current_phase = phase
            criterion.set_phase(phase)
            patch_size = get_patch_size(phase, args)
            print(f"=== Phase {phase} @ step {step}: patch={patch_size}, losses={active_losses(criterion)} ===")

        patch_size = get_patch_size(current_phase, args)
        lr_l, lr_r, hr_l, hr_r = build_single_batch(
            hr_left, hr_right, lr_left, lr_right,
            args.scale, patch_size, args.augment,
        )

        optimizer.zero_grad(set_to_none=True)

        # Frequency stream uses FFT; cuFFT half precision requires power-of-two sizes.
        # Keep AMP behavior close to real training, but auto-fallback for unsupported sizes.
        amp_enabled = args.fp16 and device.type == "cuda" and is_power_of_two(patch_size)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            sr_left, sr_right, conf_left, conf_right = model(lr_l, lr_r)
            loss, loss_dict = criterion(sr_left, sr_right, hr_l, hr_r, conf_left, conf_right)

            if criterion.use_adv and current_phase >= 4:
                with torch.no_grad():
                    d_real = discriminator(hr_l)
                d_fake = discriminator(sr_left)
                loss_adv_g = gan_loss.forward_g(d_real, d_fake)
                loss = loss + args.w_adv * loss_adv_g
                loss_dict["adv_g"] = loss_adv_g

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if criterion.use_adv and current_phase >= 4:
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                d_real = discriminator(hr_l.detach())
                d_fake = discriminator(sr_left.detach())
                loss_d = gan_loss.forward_d(d_real, d_fake)
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()
            scheduler_d.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                sr_l_eval = sr_left.detach().clamp(0, 1)
                sr_r_eval = sr_right.detach().clamp(0, 1)
                psnr_l = compute_psnr(sr_l_eval, hr_l)
                psnr_r = compute_psnr(sr_r_eval, hr_r)
                psnr_avg = 0.5 * (psnr_l + psnr_r)

            lr_now = optimizer.param_groups[0]["lr"]
            loss_items = " | ".join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items() if torch.is_tensor(v)
            )
            print(
                f"step={step:5d} phase={current_phase} patch={patch_size} lr={lr_now:.2e} "
                f"loss={loss.item():.6f} psnr_l={psnr_l:.3f} psnr_r={psnr_r:.3f} avg={psnr_avg:.3f}"
            )
            if args.fp16 and device.type == "cuda" and not amp_enabled:
                print("  AMP auto-disabled for this patch size (non power-of-two FFT).")
            if loss_items:
                print(f"  {loss_items}")

            if psnr_avg > best_psnr:
                best_psnr = psnr_avg
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "phase": current_phase,
                        "best_psnr": best_psnr,
                    },
                    os.path.join(args.out_dir, "best_overfit.pth"),
                )
                save_tensor_image(sr_l_eval, os.path.join(args.out_dir, "best_sr_left.png"))
                save_tensor_image(sr_r_eval, os.path.join(args.out_dir, "best_sr_right.png"))

        if step % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "phase": current_phase,
                    "best_psnr": best_psnr,
                },
                os.path.join(args.out_dir, f"ckpt_step_{step}.pth"),
            )

    print(f"Done. Best PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    main()

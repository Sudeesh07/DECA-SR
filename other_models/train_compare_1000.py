# import argparse
# import json
# import os
# import random
# import time
# import sys

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader


# _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)

# from data.dataset import build_dataset
# from other_models.nafssr import NAFSSR
# from other_models.nafrssr_b import NAFSSR as NAFRSSR_B
# from other_models.mssfnet import MSSFNet
# from models_v2.network_stable import stereo_sr_stable_base
# from utils_fixed import compute_psnr, compute_ssim, setup_logger


# class OurModelAdapter(nn.Module):
#     """Adapter to match output format [B,6,H,W] used by baseline comparison models."""

#     def __init__(self, scale=4):
#         super().__init__()
#         self.net = stereo_sr_stable_base(scale=scale)

#     def forward(self, x):
#         left, right = x.chunk(2, dim=1)
#         sr_l, sr_r, _, _ = self.net(left, right)
#         return torch.cat([sr_l, sr_r], dim=1)


# def seed_everything(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def build_model(name: str, scale: int):
#     if name == "nafssr":
#         return NAFSSR(
#             up_scale=scale,
#             train_size=(1, 6, 30, 90),
#             fast_imp=False,
#             width=48,
#             num_blks=32,
#             drop_path_rate=0.1,
#         )
#     if name == "nafrssr_b":
#         return NAFRSSR_B(
#             up_scale=scale,
#             train_size=(1, 6, 30, 90),
#             fast_imp=False,
#             width=48,
#             num_blks=32,
#             drop_path_rate=0.1,
#         )
#     if name == "mssfnet":
#         return MSSFNet(
#             up_scale=scale,
#             train_size=(1, 6, 30, 90),
#             fast_imp=False,
#             width=64,
#             num_blks=32,
#             drop_path_rate=0.1,
#         )
#     if name == "ours":
#         return OurModelAdapter(scale=scale)
#     raise ValueError(f"Unknown model: {name}")


# def evaluate(model, loader, device, max_samples=50):
#     model.eval()
#     psnr_l, psnr_r, ssim_l, ssim_r = [], [], [], []

#     with torch.no_grad():
#         n = 0
#         for batch in loader:
#             lr_l = batch["lr_left"].to(device)
#             lr_r = batch["lr_right"].to(device)
#             hr_l = batch["hr_left"].to(device)
#             hr_r = batch["hr_right"].to(device)

#             inp = torch.cat([lr_l, lr_r], dim=1)
#             out = model(inp)
#             sr_l, sr_r = out[:, :3], out[:, 3:]

#             min_h = min(sr_l.shape[2], hr_l.shape[2])
#             min_w = min(sr_l.shape[3], hr_l.shape[3])
#             sr_l = sr_l[:, :, :min_h, :min_w].clamp(0, 1)
#             sr_r = sr_r[:, :, :min_h, :min_w].clamp(0, 1)
#             hr_l = hr_l[:, :, :min_h, :min_w]
#             hr_r = hr_r[:, :, :min_h, :min_w]

#             psnr_l.append(compute_psnr(sr_l, hr_l))
#             psnr_r.append(compute_psnr(sr_r, hr_r))
#             ssim_l.append(compute_ssim(sr_l, hr_l))
#             ssim_r.append(compute_ssim(sr_r, hr_r))

#             n += lr_l.size(0)
#             if n >= max_samples:
#                 break

#     psnr_l_m = float(np.mean(psnr_l)) if psnr_l else 0.0
#     psnr_r_m = float(np.mean(psnr_r)) if psnr_r else 0.0
#     ssim_l_m = float(np.mean(ssim_l)) if ssim_l else 0.0
#     ssim_r_m = float(np.mean(ssim_r)) if ssim_r else 0.0

#     return {
#         "psnr_left": psnr_l_m,
#         "psnr_right": psnr_r_m,
#         "psnr_avg": 0.5 * (psnr_l_m + psnr_r_m),
#         "ssim_left": ssim_l_m,
#         "ssim_right": ssim_r_m,
#         "ssim_avg": 0.5 * (ssim_l_m + ssim_r_m),
#     }


# def train_one(model_name, args, train_loader, test_loader, device, logger):
#     seed_everything(args.seed)

#     model = build_model(model_name, args.scale).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     criterion = nn.L1Loss()

#     best = {
#         "psnr_avg": -1e9,
#         "epoch": -1,
#     }

#     history = []
#     t0 = time.time()

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         losses = []

#         for batch in train_loader:
#             lr_l = batch["lr_left"].to(device)
#             lr_r = batch["lr_right"].to(device)
#             hr_l = batch["hr_left"].to(device)
#             hr_r = batch["hr_right"].to(device)

#             inp = torch.cat([lr_l, lr_r], dim=1)
#             out = model(inp)
#             sr_l, sr_r = out[:, :3], out[:, 3:]

#             min_h = min(sr_l.shape[2], hr_l.shape[2])
#             min_w = min(sr_l.shape[3], hr_l.shape[3])
#             sr_l = sr_l[:, :, :min_h, :min_w]
#             sr_r = sr_r[:, :, :min_h, :min_w]
#             hr_l = hr_l[:, :, :min_h, :min_w]
#             hr_r = hr_r[:, :, :min_h, :min_w]

#             loss = criterion(sr_l, hr_l) + criterion(sr_r, hr_r)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optimizer.step()

#             losses.append(loss.item())

#         train_loss = float(np.mean(losses)) if losses else 0.0

#         if (epoch % args.eval_every == 0) or (epoch == args.epochs):
#             metrics = evaluate(model, test_loader, device, max_samples=args.eval_samples)
#             if metrics["psnr_avg"] > best["psnr_avg"]:
#                 best = {
#                     "psnr_avg": metrics["psnr_avg"],
#                     "epoch": epoch,
#                     **metrics,
#                 }
#             logger.info(
#                 f"[{model_name}] Epoch {epoch}/{args.epochs} | "
#                 f"Loss: {train_loss:.6f} | "
#                 f"PSNR(L/R/Avg): {metrics['psnr_left']:.3f}/{metrics['psnr_right']:.3f}/{metrics['psnr_avg']:.3f} | "
#                 f"SSIM Avg: {metrics['ssim_avg']:.4f}"
#             )
#             history.append({"epoch": epoch, "loss": train_loss, **metrics})
#         else:
#             if epoch % args.log_every == 0:
#                 logger.info(f"[{model_name}] Epoch {epoch}/{args.epochs} | Loss: {train_loss:.6f}")

#     total_time = time.time() - t0
#     result = {
#         "model": model_name,
#         "best": best,
#         "final_epoch": args.epochs,
#         "total_time_sec": total_time,
#         "history": history,
#     }
#     return result


# def main():
#     parser = argparse.ArgumentParser("Train 3 comparison models + our model for 1000 epochs")
#     parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--epochs", type=int, default=1000)
#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--weight_decay", type=float, default=0.0)
#     parser.add_argument("--grad_clip", type=float, default=1.0)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--eval_every", type=int, default=50)
#     parser.add_argument("--eval_samples", type=int, default=112)
#     parser.add_argument("--log_every", type=int, default=10)
#     parser.add_argument("--data_root", type=str, default="datasets/StereoSR")
#     parser.add_argument("--test_source", type=str, default="Flickr1024")
#     parser.add_argument("--output_dir", type=str, default="experiments/compare_other_models")
#     parser.add_argument(
#         "--models",
#         nargs="+",
#         default=["nafssr", "nafrssr_b", "mssfnet", "ours"],
#         choices=["nafssr", "nafrssr_b", "mssfnet", "ours"],
#     )
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     logger = setup_logger(args.output_dir, name="compare_other_models")

#     seed_everything(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Build args namespace expected by data.dataset.build_dataset
#     args.patch_h = 30
#     args.patch_w = 90
#     args.augment = True

#     train_dataset = build_dataset(args, split="train")
#     test_dataset = build_dataset(args, split="test")

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=0,
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#     )

#     logger.info(f"Device: {device}")
#     logger.info(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
#     logger.info(f"Models: {args.models}")

#     all_results = []
#     for model_name in args.models:
#         logger.info("=" * 80)
#         logger.info(f"Training {model_name} for {args.epochs} epochs")
#         result = train_one(model_name, args, train_loader, test_loader, device, logger)
#         all_results.append(result)

#     out_json = os.path.join(args.output_dir, "compare_results_1000_epochs.json")
#     with open(out_json, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, indent=2)

#     logger.info("=" * 80)
#     logger.info("Final comparison (best by PSNR avg):")
#     for r in all_results:
#         b = r["best"]
#         logger.info(
#             f"{r['model']:<12} | epoch {b.get('epoch', -1):>4} | "
#             f"PSNR avg {b.get('psnr_avg', 0.0):.3f} | SSIM avg {b.get('ssim_avg', 0.0):.4f}"
#         )
#     logger.info(f"Saved: {out_json}")


# if __name__ == "__main__":
#     main()



import argparse
import json
import os
import random
import time
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataset import build_dataset
from other_models.nafssr import NAFSSR
from other_models.nafrssr_b import NAFSSR as NAFRSSR_B
from other_models.mssfnet import MSSFNet
from models_v2.network_stable import stereo_sr_stable_base
from utils_fixed import compute_psnr, compute_ssim, setup_logger


# =============================================================================
# Model adapters
# =============================================================================

class OurModelAdapter(nn.Module):
    """
    Adapter so our model matches:
        input : [B, 6, H, W]
        output: [B, 6, sH, sW]
    """
    def __init__(self, scale=4):
        super().__init__()
        self.net = stereo_sr_stable_base(scale=scale)

    def forward(self, x):
        left, right = x.chunk(2, dim=1)
        sr_l, sr_r, _, _ = self.net(left, right)
        return torch.cat([sr_l, sr_r], dim=1)


def build_model(name: str, scale: int):
    if name == "nafssr":
        return NAFSSR(
            up_scale=scale,
            train_size=(1, 6, 30, 90),
            fast_imp=False,
            width=48,
            num_blks=32,
            drop_path_rate=0.1,
        )
    elif name == "nafrssr_b":
        return NAFRSSR_B(
            up_scale=scale,
            train_size=(1, 6, 30, 90),
            fast_imp=False,
            width=48,
            num_blks=32,
            drop_path_rate=0.1,
        )
    elif name == "mssfnet":
        return MSSFNet(
            up_scale=scale,
            width=64,
            num_blks=32,
            img_channel=3,
            drop_path_rate=0.1,
            dual=True,
        )
    elif name == "ours":
        return OurModelAdapter(scale=scale)
    else:
        raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, img_size=(30, 90)):
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 6, img_size[0], img_size[1])
        flops = FlopCountAnalysis(model, dummy)
        return int(flops.total())
    except Exception:
        return -1


def evaluate(model, loader, device, max_samples=None):
    model.eval()
    psnr_l, psnr_r, ssim_l, ssim_r = [], [], [], []

    with torch.no_grad():
        n = 0
        for batch in loader:
            lr_l = batch["lr_left"].to(device)
            lr_r = batch["lr_right"].to(device)
            hr_l = batch["hr_left"].to(device)
            hr_r = batch["hr_right"].to(device)

            inp = torch.cat([lr_l, lr_r], dim=1)
            out = model(inp)
            sr_l, sr_r = out[:, :3], out[:, 3:]

            min_h = min(sr_l.shape[2], hr_l.shape[2])
            min_w = min(sr_l.shape[3], hr_l.shape[3])

            sr_l = sr_l[:, :, :min_h, :min_w].clamp(0, 1)
            sr_r = sr_r[:, :, :min_h, :min_w].clamp(0, 1)
            hr_l = hr_l[:, :, :min_h, :min_w]
            hr_r = hr_r[:, :, :min_h, :min_w]

            psnr_l.append(compute_psnr(sr_l, hr_l))
            psnr_r.append(compute_psnr(sr_r, hr_r))
            ssim_l.append(compute_ssim(sr_l, hr_l))
            ssim_r.append(compute_ssim(sr_r, hr_r))

            n += lr_l.size(0)
            if max_samples is not None and n >= max_samples:
                break

    psnr_l_m = float(np.mean(psnr_l)) if psnr_l else 0.0
    psnr_r_m = float(np.mean(psnr_r)) if psnr_r else 0.0
    ssim_l_m = float(np.mean(ssim_l)) if ssim_l else 0.0
    ssim_r_m = float(np.mean(ssim_r)) if ssim_r else 0.0

    return {
        "psnr_left": psnr_l_m,
        "psnr_right": psnr_r_m,
        "psnr_avg": 0.5 * (psnr_l_m + psnr_r_m),
        "ssim_left": ssim_l_m,
        "ssim_right": ssim_r_m,
        "ssim_avg": 0.5 * (ssim_l_m + ssim_r_m),
    }


# =============================================================================
# Training
# =============================================================================

def train_model(model_name, args, train_loader, test_loader, device, logger):
    seed_everything(args.seed)

    model = build_model(model_name, args.scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_iters)
    criterion = nn.L1Loss()

    params = count_parameters(model)
    flops = estimate_flops(model, img_size=(args.patch_h, args.patch_w))

    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Params: {params:,}")
    logger.info(f"FLOPs:  {flops:,}" if flops > 0 else "FLOPs: unavailable")

    best = {
        "iter": -1,
        "psnr_avg": -1e9,
    }

    history = []
    t0 = time.time()
    loss_meter = []

    train_iter = iter(train_loader)

    for step in range(1, args.train_iters + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()

        lr_l = batch["lr_left"].to(device)
        lr_r = batch["lr_right"].to(device)
        hr_l = batch["hr_left"].to(device)
        hr_r = batch["hr_right"].to(device)

        inp = torch.cat([lr_l, lr_r], dim=1)
        out = model(inp)
        sr_l, sr_r = out[:, :3], out[:, 3:]

        min_h = min(sr_l.shape[2], hr_l.shape[2])
        min_w = min(sr_l.shape[3], hr_l.shape[3])

        sr_l = sr_l[:, :, :min_h, :min_w]
        sr_r = sr_r[:, :, :min_h, :min_w]
        hr_l = hr_l[:, :, :min_h, :min_w]
        hr_r = hr_r[:, :, :min_h, :min_w]

        loss = criterion(sr_l, hr_l) + criterion(sr_r, hr_r)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_meter.append(loss.item())

        if step % args.log_every == 0:
            logger.info(f"[{model_name}] Step {step}/{args.train_iters} | L1: {np.mean(loss_meter):.6f}")

        if (step % args.eval_every == 0) or (step == args.train_iters):
            metrics = evaluate(model, test_loader, device, max_samples=args.eval_samples)
            avg_loss = float(np.mean(loss_meter)) if loss_meter else 0.0

            if metrics["psnr_avg"] > best["psnr_avg"]:
                best = {
                    "iter": step,
                    "loss": avg_loss,
                    **metrics,
                }

            logger.info(
                f"[{model_name}] Eval @ {step} | "
                f"L1: {avg_loss:.6f} | "
                f"PSNR(L/R/Avg): {metrics['psnr_left']:.4f}/{metrics['psnr_right']:.4f}/{metrics['psnr_avg']:.4f} | "
                f"SSIM Avg: {metrics['ssim_avg']:.4f}"
            )

            history.append({
                "iter": step,
                "loss": avg_loss,
                **metrics
            })

    total_time = time.time() - t0

    result = {
        "model": model_name,
        "params": params,
        "flops": flops,
        "best": best,
        "history": history,
        "total_time_sec": total_time,
    }

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser("Pairwise Stereo SR Comparison")

    parser.add_argument("--baseline", type=str, default="nafssr",
                        choices=["nafssr", "nafrssr_b", "mssfnet", "ours"])
    parser.add_argument("--candidate", type=str, default="ours",
                        choices=["nafssr", "nafrssr_b", "mssfnet", "ours"])

    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=112)
    parser.add_argument("--log_every", type=int, default=200)

    parser.add_argument("--output_dir", type=str, default="experiments/pair_compare")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, name="pair_compare")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset args expected by build_dataset
    args.patch_h = 30
    args.patch_w = 90
    args.augment = True
    args.data_root = "datasets/StereoSR"
    args.test_source = "Flickr1024"

    train_dataset = build_dataset(args, split="train")
    test_dataset = build_dataset(args, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Device: {device}")
    logger.info(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    logger.info(f"Comparing: baseline={args.baseline} vs candidate={args.candidate}")
    logger.info(f"Train iterations: {args.train_iters}")

    results = []

    for model_name in [args.baseline, args.candidate]:
        result = train_model(model_name, args, train_loader, test_loader, device, logger)
        results.append(result)
        torch.cuda.empty_cache()

    baseline_result = results[0]
    candidate_result = results[1]

    delta_psnr = candidate_result["best"]["psnr_avg"] - baseline_result["best"]["psnr_avg"]
    delta_ssim = candidate_result["best"]["ssim_avg"] - baseline_result["best"]["ssim_avg"]

    summary = {
        "baseline": baseline_result,
        "candidate": candidate_result,
        "delta": {
            "psnr_avg": delta_psnr,
            "ssim_avg": delta_ssim,
        }
    }

    out_json = os.path.join(
        args.output_dir,
        f"pair_{args.baseline}_vs_{args.candidate}.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info("PAIR COMPARISON SUMMARY")
    logger.info("=" * 80)

    logger.info(
        f"{'Model':<15} {'Params':>12} {'FLOPs':>15} {'BestIter':>10} "
        f"{'PSNR Avg':>10} {'SSIM Avg':>10} {'Time(s)':>10}"
    )
    logger.info("-" * 80)

    for r in results:
        b = r["best"]
        logger.info(
            f"{r['model']:<15} "
            f"{r['params']:>12,} "
            f"{r['flops']:>15,} "
            f"{b.get('iter', -1):>10} "
            f"{b.get('psnr_avg', 0.0):>10.4f} "
            f"{b.get('ssim_avg', 0.0):>10.4f} "
            f"{r['total_time_sec']:>10.1f}"
        )

    logger.info("-" * 80)
    logger.info(f"Delta PSNR Avg: {delta_psnr:+.4f} dB")
    logger.info(f"Delta SSIM Avg: {delta_ssim:+.4f}")
    logger.info(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
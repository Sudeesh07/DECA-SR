import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def create_lr_images(hr_dir: str, lr_dir: str, scale: int):
    os.makedirs(lr_dir, exist_ok=True)

    hr_images = sorted(glob.glob(os.path.join(hr_dir, "*")))
    print(f"Generating {scale}× LR images: {len(hr_images)} images")

    for path in tqdm(hr_images):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        lr_w, lr_h = w // scale, h // scale
        img = img.crop((0, 0, lr_w * scale, lr_h * scale))
        lr_img = img.resize((lr_w, lr_h), Image.BICUBIC)

        out_path = os.path.join(lr_dir, os.path.basename(path))
        lr_img.save(out_path, quality=95)


def setup_flickr1024(output_dir: str):
    print("=" * 60)
    print("Flickr1024 Setup Instructions")
    print("=" * 60)
    print()
    print("1. Download Flickr1024 from:")
    print("   https://yingqianwang.github.io/Flickr1024/")
    print()
    print("2. Extract and organize as:")
    print(f"   {output_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── hr/")
    print(f"   │   │   ├── left/   (800 images)")
    print(f"   │   │   └── right/  (800 images)")
    print(f"   │   ├── lr_x2/")
    print(f"   │   │   ├── left/")
    print(f"   │   │   └── right/")
    print(f"   │   └── lr_x4/")
    print(f"   │       ├── left/")
    print(f"   │       └── right/")
    print(f"   ├── val/")
    print(f"   │   └── (same structure, 112 images)")
    print(f"   └── test/")
    print(f"       └── (same structure, 112 images)")
    print()
    for split in ["train", "val", "test"]:
        for res in ["hr", "lr_x2", "lr_x4"]:
            for view in ["left", "right"]:
                os.makedirs(os.path.join(output_dir, split, res, view), exist_ok=True)

    print(f"Directory structure created at {output_dir}")
    print("Place your images in the appropriate directories.")
    print()
    print("3. Generate LR images (if not provided):")
    print(f"   python scripts/prepare_data.py --generate_lr --output {output_dir}/train")


def setup_custom_dataset(input_dir: str, output_dir: str, val_ratio: float = 0.1,
                          test_ratio: float = 0.1):
    left_imgs = sorted(glob.glob(os.path.join(input_dir, "left", "*")))
    right_imgs = sorted(glob.glob(os.path.join(input_dir, "right", "*")))

    assert len(left_imgs) == len(right_imgs), "Mismatched left/right counts"
    n = len(left_imgs)
    print(f"Found {n} stereo pairs")
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)

    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val
    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split_name, split_indices in splits.items():
        for view, imgs in [("left", left_imgs), ("right", right_imgs)]:
            dest_dir = os.path.join(output_dir, split_name, "hr", view)
            os.makedirs(dest_dir, exist_ok=True)
            for idx in split_indices:
                shutil.copy2(imgs[idx], dest_dir)
        print(f"  {split_name}: {len(split_indices)} pairs")


def generate_lr_for_dataset(data_dir: str, scales=(2, 4)):
    for scale in scales:
        for view in ["left", "right"]:
            hr_dir = os.path.join(data_dir, "hr", view)
            lr_dir = os.path.join(data_dir, f"lr_x{scale}", view)
            if os.path.exists(hr_dir):
                create_lr_images(hr_dir, lr_dir, scale)


def verify_dataset(data_dir: str):
    print(f"\nVerifying: {data_dir}")
    issues = 0

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"  ⚠ Missing split: {split}")
            continue
        hr_left = sorted(glob.glob(os.path.join(split_dir, "hr", "left", "*")))
        hr_right = sorted(glob.glob(os.path.join(split_dir, "hr", "right", "*")))

        print(f"  {split}: {len(hr_left)} left, {len(hr_right)} right HR images")

        if len(hr_left) != len(hr_right):
            print(f"    ✗ Mismatched counts!")
            issues += 1
        if hr_left and hr_right:
            img_l = Image.open(hr_left[0])
            img_r = Image.open(hr_right[0])
            if img_l.size != img_r.size:
                print(f"    ✗ Size mismatch: {img_l.size} vs {img_r.size}")
                issues += 1
            else:
                print(f"    ✓ Image size: {img_l.size}")
        for scale in [2, 4]:
            lr_dir = os.path.join(split_dir, f"lr_x{scale}", "left")
            if os.path.exists(lr_dir):
                n_lr = len(glob.glob(os.path.join(lr_dir, "*")))
                print(f"    LR x{scale}: {n_lr} images")
            else:
                print(f"    LR x{scale}: not generated (will be created on-the-fly)")

    if issues == 0:
        print("  ✓ Dataset OK!")
    else:
        print(f"  ✗ {issues} issue(s) found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare stereo SR dataset")
    parser.add_argument("--dataset", type=str, default="flickr1024",
                        choices=["flickr1024", "custom"])
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory for custom dataset")
    parser.add_argument("--output", type=str, default="./data/Flickr1024")
    parser.add_argument("--generate_lr", action="store_true",
                        help="Generate LR images from HR")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset integrity")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    elif args.generate_lr:
        generate_lr_for_dataset(args.output)
    elif args.dataset == "flickr1024":
        setup_flickr1024(args.output)
    elif args.dataset == "custom":
        assert args.input, "Must specify --input for custom dataset"
        setup_custom_dataset(args.input, args.output)

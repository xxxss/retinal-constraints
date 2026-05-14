"""
Download ImageNet validation set from Hugging Face and organize into
the standard ImageFolder structure that PyTorch/timm expects.

Prerequisites:
    1. Create account at https://huggingface.co
    2. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k
       and click "Agree" to accept the usage terms
    3. Run: uv run huggingface-cli login
       and paste your access token from https://huggingface.co/settings/tokens

Usage:
    uv run python download_imagenet.py                    # full val set (50k images, ~6GB)
    uv run python download_imagenet.py --max_samples 5000 # subset (faster, ~600MB)
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ImageNet class index to WordNet ID mapping (first 20 shown, full 1000 loaded dynamically)
def get_wnid_map():
    """Get mapping from class index to WordNet ID (e.g., 0 -> 'n01440764')."""
    import json
    import urllib.request

    cache_path = Path("./data/imagenet_wnids.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Download class index mapping
    url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/raw/main/classes.py"
    print("Downloading class mapping...")

    # Alternative: use the dataset's features directly
    return None  # Will use numeric folder names as fallback


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet validation set")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Download only first N samples (default: all 50,000)")
    parser.add_argument("--output_dir", default=os.path.expanduser("~/datasets/imagenet/val"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_dirs = [d for d in output_dir.iterdir() if d.is_dir()] if output_dir.exists() else []
    if len(existing_dirs) >= 100:
        existing_images = sum(1 for d in existing_dirs for _ in d.glob("*.JPEG"))
        print(f"ImageNet val already exists at {output_dir}")
        print(f"  {len(existing_dirs)} class folders, {existing_images} images")
        print(f"  To re-download, delete {output_dir} first")
        return

    print("Loading ImageNet validation set from Hugging Face...")
    print("(This requires you to have accepted the terms at")
    print(" https://huggingface.co/datasets/ILSVRC/imagenet-1k)")
    print()

    try:
        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            trust_remote_code=True,
        )
    except Exception as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("ERROR: Access denied. Please:")
            print("  1. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            print("  2. Click 'Agree' to accept the terms")
            print("  3. Run: uv run huggingface-cli login")
            print("  4. Try again")
        else:
            print(f"ERROR: {e}")
        return

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Saving {len(ds)} images to {output_dir}...")

    for i, example in enumerate(tqdm(ds, desc="Saving images")):
        label = example["label"]
        image = example["image"]

        # Create class folder (use numeric label, timm handles mapping)
        class_dir = output_dir / f"class_{label:04d}"
        class_dir.mkdir(exist_ok=True)

        # Save image
        img_path = class_dir / f"val_{i:08d}.JPEG"
        if not img_path.exists():
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path, "JPEG", quality=95)

    # Verify
    total_dirs = sum(1 for d in output_dir.iterdir() if d.is_dir())
    total_images = sum(1 for d in output_dir.iterdir() if d.is_dir()
                       for _ in d.glob("*.JPEG"))
    print(f"\nDone! {total_dirs} class folders, {total_images} images")
    print(f"Location: {output_dir}")


if __name__ == "__main__":
    main()

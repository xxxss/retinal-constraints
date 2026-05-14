"""
Download ImageNette — a 10-class subset of ImageNet.
Fast, free, no login required. ~1.5GB.

The 10 classes (all from ImageNet-1k):
  tench, English springer, cassette player, chain saw, church,
  French horn, garbage truck, gas pump, golf ball, parachute
"""

import os
import tarfile
import urllib.request
from pathlib import Path


def main():
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    output_dir = Path(os.path.expanduser("~/datasets"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = output_dir / "imagenette2-320.tgz"
    final_dir = output_dir / "imagenette2-320"

    if final_dir.exists():
        n_val = sum(1 for _ in (final_dir / "val").rglob("*.JPEG"))
        print(f"ImageNette already exists: {final_dir}")
        print(f"  Validation images: {n_val}")
        return

    print(f"Downloading ImageNette (320px version, ~1.5GB)...")
    print(f"  From: {url}")
    print(f"  To:   {tgz_path}")

    def progress_hook(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        mb = count * block_size / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, tgz_path, reporthook=progress_hook)
    print("\n  Download complete.")

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    # Clean up tar
    tgz_path.unlink()

    # Verify
    val_dir = final_dir / "val"
    n_classes = sum(1 for d in val_dir.iterdir() if d.is_dir())
    n_images = sum(1 for _ in val_dir.rglob("*.JPEG"))
    print(f"\nDone!")
    print(f"  Location: {final_dir}")
    print(f"  Validation: {n_classes} classes, {n_images} images")
    print(f"  Training:   {sum(1 for _ in (final_dir / 'train').rglob('*.JPEG'))} images")


if __name__ == "__main__":
    main()

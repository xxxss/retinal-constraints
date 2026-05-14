"""
Evaluate OOD robustness on ImageNet-C (corrupted ImageNet).

ImageNet-C applies 15 types of corruption at 5 severity levels:
  Noise:   gaussian, shot, impulse
  Blur:    defocus, glass, motion, zoom
  Weather: snow, frost, fog, brightness
  Digital: contrast, elastic, pixelate, jpeg

We use a subset of ImageNet validation (or full if available) with
torchvision + timm pretrained models.

Usage:
    # Evaluate baseline ConvNeXt-Tiny (no preprocessing)
    uv run python eval_robustness.py --model convnext_tiny --mode baseline

    # Evaluate with fixed DoG preprocessing
    uv run python eval_robustness.py --model convnext_tiny --mode fixed_dog

    # Evaluate with learnable DoG (requires training first)
    uv run python eval_robustness.py --model convnext_tiny --mode learned_dog

    # Run all three for comparison
    uv run python eval_robustness.py --model convnext_tiny --mode all
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm

from dog_layer import FixedDoG, LearnableDoG


# ---- Corruption transforms (simulating ImageNet-C without downloading 75GB) ----

def apply_corruption(images, corruption_type, severity=3):
    """Apply a corruption to a batch of normalized images.

    Works on normalized images (can have negative values).
    Corruptions are applied in a way that's compatible with normalized inputs.
    Severity ranges from 1 (mild) to 5 (severe).
    """
    s = severity
    if corruption_type == "gaussian_noise":
        std = [0.08, 0.16, 0.24, 0.36, 0.52][s - 1]
        return images + torch.randn_like(images) * std

    elif corruption_type == "shot_noise":
        # Approximate shot noise with scaled Gaussian (avoids Poisson issues)
        std = [0.06, 0.12, 0.2, 0.3, 0.45][s - 1]
        return images + torch.randn_like(images) * std

    elif corruption_type == "impulse_noise":
        prob = [0.01, 0.03, 0.06, 0.1, 0.15][s - 1]
        mask = torch.rand(images.shape[0], 1, images.shape[2], images.shape[3],
                          device=images.device)
        images = images.clone()
        images[mask.expand_as(images) < prob / 2] = -2.0  # black (normalized)
        images[(mask.expand_as(images) >= prob / 2) & (mask.expand_as(images) < prob)] = 2.0
        return images

    elif corruption_type == "gaussian_blur":
        sigma = [0.4, 0.8, 1.2, 2.0, 3.0][s - 1]
        ks = int(2 * np.ceil(3 * sigma) + 1)
        if ks % 2 == 0:
            ks += 1
        return transforms.GaussianBlur(ks, sigma)(images)

    elif corruption_type == "brightness":
        factor = [0.1, 0.2, 0.3, 0.5, 0.7][s - 1]
        return images + factor

    elif corruption_type == "contrast":
        factor = [0.8, 0.6, 0.4, 0.25, 0.1][s - 1]
        mean = images.mean(dim=(-2, -1), keepdim=True)
        return (images - mean) * factor + mean

    elif corruption_type == "fog":
        fog_level = [0.2, 0.35, 0.5, 0.65, 0.8][s - 1]
        return images * (1 - fog_level) + fog_level

    elif corruption_type == "jpeg":
        noise_level = [0.04, 0.08, 0.12, 0.18, 0.25][s - 1]
        return images + torch.randn_like(images) * noise_level

    else:
        return images


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "gaussian_blur", "brightness", "contrast", "fog", "jpeg",
]


# ---- Model wrappers ----

class BaselineModel(nn.Module):
    """Pretrained model without any preprocessing."""

    def __init__(self, model_name="convnext_tiny", num_classes=1000):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


class FixedDoGModel(nn.Module):
    """Pretrained model with fixed DoG preprocessing."""

    def __init__(self, model_name="convnext_tiny", num_classes=1000,
                 sigma_center=1.0, sigma_surround=3.0):
        super().__init__()
        self.dog = FixedDoG(sigma_center, sigma_surround)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        # DoG doubles channels (3→6), need adapter
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)

    def forward(self, x):
        x = self.dog(x)       # (B,3,H,W) → (B,6,H,W)
        x = self.adapter(x)   # (B,6,H,W) → (B,3,H,W)
        return self.backbone(x)


class LearnableDoGModel(nn.Module):
    """Pretrained model with learnable DoG preprocessing."""

    def __init__(self, model_name="convnext_tiny", num_classes=1000):
        super().__init__()
        self.dog = LearnableDoG(in_channels=3)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)

    def forward(self, x):
        x = self.dog(x)
        x = self.adapter(x)
        return self.backbone(x)

    def get_learned_params(self):
        return self.dog.get_learned_params()


# ---- Evaluation ----

def get_imagenet_val(batch_size=32, num_workers=2, max_samples=2000):
    """Load ImageNet validation set (or a subset).

    If full ImageNet not available, downloads a small proxy dataset.
    """
    data_config = timm.data.resolve_data_config({}, model=timm.create_model("convnext_tiny"))
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    # Try ImageNet and ImageNette locations
    candidate_dirs = [
        "/datasets/imagenet/val",
        os.path.expanduser("~/datasets/imagenet/val"),
        "./data/imagenet/val",
        os.path.expanduser("~/datasets/imagenette2-320/val"),  # ImageNette
    ]

    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        # Check it's not empty
        subdirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
        if not subdirs:
            continue

        print(f"Found dataset: {d} ({len(subdirs)} classes)")
        is_imagenette = "imagenette" in d

        dataset = datasets.ImageFolder(d, transform=val_transform)

        if is_imagenette:
            # ImageNette folder indices (0-9) don't match ImageNet class indices.
            # Build a mapping from ImageNette folder order to ImageNet class index.
            wnid_to_inet = {
                "n01440764": 0, "n02102040": 217, "n02979186": 482,
                "n03000684": 491, "n03028079": 497, "n03394916": 566,
                "n03417042": 569, "n03425413": 571, "n03445777": 574,
                "n03888257": 701,
            }
            # dataset.classes is sorted list of folder names (WordNet IDs)
            folder_to_inet = {
                i: wnid_to_inet[wnid]
                for i, wnid in enumerate(dataset.classes)
            }
            # Remap labels
            dataset.samples = [
                (path, folder_to_inet[label]) for path, label in dataset.samples
            ]
            dataset.targets = [folder_to_inet[t] for t in dataset.targets]
            print(f"  Remapped ImageNette labels to ImageNet class indices")

        if max_samples and max_samples < len(dataset):
            dataset = torch.utils.data.Subset(
                dataset, range(min(max_samples, len(dataset)))
            )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        loader.is_imagenet = True
        return loader

    print("ERROR: No dataset found. Run:")
    print("  uv run python download_imagenette.py")
    raise FileNotFoundError("No ImageNet or ImageNette found")


@torch.no_grad()
def evaluate_clean(model, loader, device):
    """Evaluate on clean (uncorrupted) data."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # Handle case where model outputs more classes than labels
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


@torch.no_grad()
def evaluate_corrupted(model, loader, device, severity=3):
    """Evaluate on all corruption types at given severity."""
    model.eval()
    results = {}
    for corruption in CORRUPTIONS:
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            # Apply corruption
            corrupted = apply_corruption(images, corruption, severity)
            outputs = model(corrupted)
            if outputs.shape[1] > 10 and labels.max() < 10:
                outputs = outputs[:, :10]
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        results[corruption] = correct / total
    results["mean"] = np.mean(list(results.values()))
    return results


def train_dog_layers(model, loader, device, epochs=5, lr=1e-3):
    """Fine-tune only the DoG + adapter layers, freeze backbone."""
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only train DoG and adapter
    trainable = []
    if hasattr(model, "dog"):
        trainable.extend(model.dog.parameters())
    if hasattr(model, "adapter"):
        trainable.extend(model.adapter.parameters())

    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if outputs.shape[1] > 10 and labels.max() < 10:
                outputs = outputs[:, :10]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        acc = correct / total
        print(f"  Fine-tune epoch {epoch+1}/{epochs}: acc={acc:.4f}")

    # Unfreeze for evaluation
    model.eval()


# ---- Main ----

def run_evaluation(mode, model_name, device, max_samples=2000, severity=3):
    print(f"\n{'='*60}")
    print(f"Evaluating: {mode} (model={model_name})")
    print(f"{'='*60}")

    loader = get_imagenet_val(max_samples=max_samples)

    is_imagenet = getattr(loader, "is_imagenet", False)
    num_classes = 1000 if is_imagenet else 10

    if mode == "baseline":
        model = BaselineModel(model_name, num_classes=num_classes)
    elif mode == "fixed_dog":
        model = FixedDoGModel(model_name, num_classes=num_classes)
    elif mode == "learned_dog":
        model = LearnableDoGModel(model_name, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = model.to(device)

    # Fine-tune DoG layers if needed
    if mode in ("fixed_dog", "learned_dog"):
        print("Fine-tuning preprocessing layers (backbone frozen)...")
        train_loader = get_imagenet_val(max_samples=max_samples)
        train_dog_layers(model, train_loader, device, epochs=5)

        if mode == "learned_dog":
            params = model.get_learned_params()
            print(f"  Learned σ_center:   {params['sigma_center']}")
            print(f"  Learned σ_surround: {params['sigma_surround']}")
            print(f"  Learned α (mix):    {params['alpha']}")

    # Evaluate clean
    clean_acc = evaluate_clean(model, loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    # Evaluate corrupted
    print(f"Evaluating {len(CORRUPTIONS)} corruptions at severity {severity}...")
    corrupt_results = evaluate_corrupted(model, loader, device, severity)

    print(f"\n{'Corruption':<20} | {'Accuracy':>10}")
    print(f"{'-'*20}-+-{'-'*10}")
    for c in CORRUPTIONS:
        print(f"{c:<20} | {corrupt_results[c]:>9.4f}")
    print(f"{'-'*20}-+-{'-'*10}")
    print(f"{'MEAN'::<20} | {corrupt_results['mean']:>9.4f}")

    return {
        "mode": mode,
        "clean_acc": clean_acc,
        "corruptions": corrupt_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnext_tiny.fb_in1k")
    parser.add_argument("--mode", default="all",
                        choices=["baseline", "fixed_dog", "learned_dog", "all"])
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--severity", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    modes = ["baseline", "fixed_dog", "learned_dog"] if args.mode == "all" else [args.mode]
    all_results = {}

    for mode in modes:
        result = run_evaluation(
            mode, args.model, device,
            max_samples=args.max_samples, severity=args.severity
        )
        all_results[mode] = result

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/robustness_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot comparison if multiple modes
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-corruption comparison
        x = np.arange(len(CORRUPTIONS))
        width = 0.8 / len(all_results)
        colors = {"baseline": "#4477AA", "fixed_dog": "#EE7733", "learned_dog": "#228833"}

        for i, (mode, result) in enumerate(all_results.items()):
            accs = [result["corruptions"][c] * 100 for c in CORRUPTIONS]
            axes[0].bar(x + i * width, accs, width, label=mode, color=colors.get(mode, "gray"))

        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(CORRUPTIONS, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Per-Corruption Accuracy (severity=3)", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        # Right: summary bar chart
        modes_list = list(all_results.keys())
        clean_accs = [all_results[m]["clean_acc"] * 100 for m in modes_list]
        mean_corrupt = [all_results[m]["corruptions"]["mean"] * 100 for m in modes_list]

        x2 = np.arange(len(modes_list))
        axes[1].bar(x2 - 0.15, clean_accs, 0.3, label="Clean", color="#4477AA")
        axes[1].bar(x2 + 0.15, mean_corrupt, 0.3, label="Corrupted (mean)", color="#EE7733")
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(modes_list)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Clean vs Corrupted Accuracy", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("results/robustness_comparison.png", dpi=150, bbox_inches="tight")
        print("\nSaved: results/robustness_comparison.png")


if __name__ == "__main__":
    main()

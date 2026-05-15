"""
OOD robustness evaluation v2: adds AdaptiveDoG mode.

Key difference from v1: the adaptive model trains with noise augmentation
so the noise estimator learns to detect degraded inputs and widen the
receptive field accordingly (mimicking retinal light adaptation).

Compares 4 modes:
  1. baseline:      ResNet18
  2. fixed_dog:     Fixed DoG → ResNet18
  3. learned_dog:   Learnable DoG → ResNet18  (overfits to clean — see v1)
  4. adaptive_dog:  Noise-adaptive DoG → ResNet18  (NEW)

Usage:
    python3 eval_cifar_v2.py                    # run all 4 modes
    python3 eval_cifar_v2.py --mode adaptive_dog  # run only adaptive
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
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

from dog_layer import FixedDoG, LearnableDoG
from dog_layer_adaptive import AdaptiveDoG


# ---- Corruptions ----

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "gaussian_blur", "brightness", "contrast", "fog", "jpeg",
]


def apply_corruption(images, corruption_type, severity=3):
    s = severity
    if corruption_type == "gaussian_noise":
        std = [0.02, 0.04, 0.06, 0.08, 0.12][s - 1]
        return images + torch.randn_like(images) * std
    elif corruption_type == "shot_noise":
        std = [0.015, 0.03, 0.05, 0.07, 0.1][s - 1]
        return images + torch.randn_like(images) * std
    elif corruption_type == "impulse_noise":
        prob = [0.01, 0.03, 0.06, 0.1, 0.15][s - 1]
        mask = torch.rand_like(images[:, :1, :, :])
        out = images.clone()
        out[mask.expand_as(out) < prob / 2] = 0.0
        out[(mask.expand_as(out) >= prob / 2) & (mask.expand_as(out) < prob)] = 1.0
        return out
    elif corruption_type == "gaussian_blur":
        sigma = [0.3, 0.5, 0.8, 1.2, 1.8][s - 1]
        ks = max(3, int(2 * np.ceil(2 * sigma) + 1))
        if ks % 2 == 0:
            ks += 1
        return transforms.GaussianBlur(ks, sigma)(images)
    elif corruption_type == "brightness":
        factor = [0.05, 0.1, 0.15, 0.2, 0.3][s - 1]
        return (images + factor).clamp(0, 1)
    elif corruption_type == "contrast":
        factor = [0.8, 0.6, 0.4, 0.25, 0.1][s - 1]
        mean = images.mean(dim=(-2, -1), keepdim=True)
        return ((images - mean) * factor + mean).clamp(0, 1)
    elif corruption_type == "fog":
        fog_level = [0.15, 0.25, 0.4, 0.55, 0.7][s - 1]
        return (images * (1 - fog_level) + fog_level).clamp(0, 1)
    elif corruption_type == "jpeg":
        noise = [0.02, 0.04, 0.06, 0.1, 0.15][s - 1]
        return (images + torch.randn_like(images) * noise).clamp(0, 1)
    return images


# ---- Models ----

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None, num_classes=10)

    def forward(self, x):
        return self.net(x)


class FixedDoGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dog = FixedDoG(sigma_center=1.0, sigma_surround=2.0, kernel_size=7)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)
        self.net = models.resnet18(weights=None, num_classes=10)

    def forward(self, x):
        return self.net(self.adapter(self.dog(x)))


class LearnableDoGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dog = LearnableDoG(in_channels=3, init_sigma_center=1.0,
                                init_sigma_surround=2.0, kernel_size=7)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)
        self.net = models.resnet18(weights=None, num_classes=10)

    def forward(self, x):
        return self.net(self.adapter(self.dog(x)))

    def get_learned_params(self):
        return self.dog.get_learned_params()


class AdaptiveDoGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dog = AdaptiveDoG(in_channels=3, kernel_size=9)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)
        self.net = models.resnet18(weights=None, num_classes=10)

    def forward(self, x):
        return self.net(self.adapter(self.dog(x)))

    def get_noise_level(self, x):
        return self.dog.get_noise_level(x)


# ---- Data ----

def get_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ---- Training ----

def train_model(model, epochs, device, lr=0.01, noise_augment=False):
    """
    Train model. If noise_augment=True, randomly corrupt 50% of batches
    during training so the adaptive layer learns to detect noise.
    """
    model = model.to(device)
    train_loader, test_loader = get_cifar10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    aug_corruptions = ["gaussian_noise", "shot_noise", "gaussian_blur", "contrast", "fog"]

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Noise augmentation: randomly corrupt 50% of batches
            if noise_augment and torch.rand(1).item() > 0.5:
                c = aug_corruptions[torch.randint(len(aug_corruptions), (1,)).item()]
                s = torch.randint(1, 4, (1,)).item()  # severity 1-3
                images = apply_corruption(images, c, s)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = evaluate_clean(model, test_loader, device)
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train={correct/total:.4f} test={test_acc:.4f}")

    return model, test_loader


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


@torch.no_grad()
def evaluate_corrupted(model, loader, device, severity=3):
    model.eval()
    results = {}
    for corruption in CORRUPTIONS:
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            corrupted = apply_corruption(images, corruption, severity)
            correct += (model(corrupted).argmax(1) == labels).sum().item()
            total += images.size(0)
        results[corruption] = correct / total
    results["mean"] = np.mean([results[c] for c in CORRUPTIONS])
    return results


# ---- Main ----

def run_one(mode, epochs, device):
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    noise_augment = False
    if mode == "baseline":
        model = BaselineModel()
    elif mode == "fixed_dog":
        model = FixedDoGModel()
    elif mode == "learned_dog":
        model = LearnableDoGModel()
    elif mode == "adaptive_dog":
        model = AdaptiveDoGModel()
        noise_augment = True  # key: train with noise so estimator learns
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t0 = time.time()
    model, test_loader = train_model(model, epochs, device, noise_augment=noise_augment)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.0f}s")

    clean = evaluate_clean(model, test_loader, device)
    print(f"  Clean accuracy: {clean:.4f}")

    if mode == "learned_dog":
        params = model.get_learned_params()
        print(f"  Learned σ_center:   {params['sigma_center']}")
        print(f"  Learned σ_surround: {params['sigma_surround']}")
        print(f"  Learned α (mix):    {params['alpha']}")

    if mode == "adaptive_dog":
        # Show noise estimation on clean vs corrupted
        sample_batch = next(iter(test_loader))[0][:8].to(device)
        clean_noise = model.get_noise_level(sample_batch)
        noisy_batch = apply_corruption(sample_batch, "gaussian_noise", severity=4)
        noisy_noise = model.get_noise_level(noisy_batch)
        print(f"  Noise estimator — clean: {clean_noise.mean():.3f}, "
              f"noisy: {noisy_noise.mean():.3f}")

    print(f"  Evaluating corruptions...")
    corrupt = evaluate_corrupted(model, test_loader, device)

    print(f"\n  {'Corruption':<20} | {'Accuracy':>10}")
    print(f"  {'-'*20}-+-{'-'*10}")
    for c in CORRUPTIONS:
        print(f"  {c:<20} | {corrupt[c]:>9.4f}")
    print(f"  {'-'*20}-+-{'-'*10}")
    print(f"  {'MEAN':<20} | {corrupt['mean']:>9.4f}")
    print(f"  {'CLEAN':<20} | {clean:>9.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/cifar_{mode}_model.pt")

    return {"clean": clean, "corruptions": corrupt, "train_time": train_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mode", default="all",
                        choices=["baseline", "fixed_dog", "learned_dog", "adaptive_dog", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    modes = (["baseline", "fixed_dog", "learned_dog", "adaptive_dog"]
             if args.mode == "all" else [args.mode])
    all_results = {}

    for mode in modes:
        all_results[mode] = run_one(mode, args.epochs, device)

    # Save results
    serializable = {}
    for mode, result in all_results.items():
        serializable[mode] = {
            "clean": float(result["clean"]),
            "train_time": float(result["train_time"]),
            "corruptions": {k: float(v) for k, v in result["corruptions"].items()},
        }
    with open("results/cifar_robustness_v2.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Plot
    if len(all_results) > 1:
        colors = {"baseline": "#4477AA", "fixed_dog": "#EE7733",
                  "learned_dog": "#228833", "adaptive_dog": "#CC3311"}
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        x = np.arange(len(CORRUPTIONS))
        width = 0.8 / len(all_results)
        for i, (mode, result) in enumerate(all_results.items()):
            accs = [result["corruptions"][c] * 100 for c in CORRUPTIONS]
            axes[0].bar(x + i * width, accs, width, label=mode, color=colors.get(mode))
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(CORRUPTIONS, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Per-Corruption Accuracy (severity=3)", fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3, axis="y")

        modes_list = list(all_results.keys())
        clean_accs = [all_results[m]["clean"] * 100 for m in modes_list]
        mean_corrupt = [all_results[m]["corruptions"]["mean"] * 100 for m in modes_list]
        x2 = np.arange(len(modes_list))
        bars1 = axes[1].bar(x2 - 0.15, clean_accs, 0.3, label="Clean", color="#4477AA")
        bars2 = axes[1].bar(x2 + 0.15, mean_corrupt, 0.3, label="Corrupted (mean)", color="#EE7733")
        for bar, val in zip(bars1, clean_accs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.1f}', ha='center', fontsize=8)
        for bar, val in zip(bars2, mean_corrupt):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.1f}', ha='center', fontsize=8)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(modes_list, fontsize=8)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Clean vs Corrupted", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("results/cifar_robustness_v2.png", dpi=150, bbox_inches="tight")
        print("\nSaved: results/cifar_robustness_v2.png")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<15} | {'Clean':>8} | {'Corrupted':>10} | {'Drop':>8}")
    print(f"{'-'*15}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
    for mode in modes:
        c = all_results[mode]["clean"] * 100
        m = all_results[mode]["corruptions"]["mean"] * 100
        print(f"{mode:<15} | {c:>7.1f}% | {m:>9.1f}% | {c-m:>7.1f}%")


if __name__ == "__main__":
    main()

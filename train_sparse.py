"""
Step 2: Train retinal bottleneck models WITH sparsity penalty.

Biological motivation: every spike costs metabolic energy (ATP).
The brain can only afford 2-4% of neurons active at any time.
We model this by adding L1 penalty on activations to the loss.

    total_loss = classification_loss + λ * mean(|activations|)

Sweep λ to find the trade-off curve: accuracy vs sparsity.

Usage:
    # Train one model: bottleneck=2, sparsity_lambda=0.001
    uv run python train_sparse.py --bottleneck 2 --sparsity 0.001

    # Sweep all sparsity levels (the core experiment)
    uv run python train_sparse.py --sweep

    # Quick test
    uv run python train_sparse.py --bottleneck 2 --sparsity 0.001 --epochs 2
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import RetinalBottleneckModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_cifar10_grayscale(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


class RetinalBottleneckWithHooks(nn.Module):
    """Wraps RetinalBottleneckModel to capture intermediate activations."""

    def __init__(self, retina_out_channels=2, **kwargs):
        super().__init__()
        self.model = RetinalBottleneckModel(
            retina_out_channels=retina_out_channels, **kwargs
        )
        self.activations = []

    def forward(self, x):
        self.activations = []

        # Manually forward through retina, capturing activations after each ReLU
        h = x
        for layer in self.model.retina:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                self.activations.append(h)
        retina_out = h

        # Forward through VVS, capturing activations
        h = retina_out
        for layer in self.model.vvs:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                self.activations.append(h)
        vvs_out = h

        # Classifier
        out = self.model.classifier(vvs_out)
        return out

    def sparsity_loss(self):
        """L1 norm of all activations, averaged.

        This is the "metabolic cost" — penalizes every non-zero activation.
        Biological interpretation: each spike costs energy, so the network
        learns to use as few spikes (non-zero activations) as possible.
        """
        if not self.activations:
            return torch.tensor(0.0)
        total = sum(act.abs().mean() for act in self.activations)
        return total / len(self.activations)

    def compute_sparsity_stats(self):
        """Compute what fraction of activations are near-zero."""
        if not self.activations:
            return 0.0, 0.0
        total_elements = 0
        near_zero = 0
        for act in self.activations:
            total_elements += act.numel()
            near_zero += (act.abs() < 1e-5).sum().item()
        frac_silent = near_zero / total_elements
        mean_activation = sum(
            act.abs().mean().item() for act in self.activations
        ) / len(self.activations)
        return frac_silent, mean_activation

    def get_retina_output(self, x):
        return self.model.get_retina_output(x)


def train_one_epoch(model, loader, optimizer, criterion, device, sparsity_lambda):
    model.train()
    total_loss, total_cls_loss, total_sp_loss = 0, 0, 0
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        cls_loss = criterion(outputs, labels)
        sp_loss = model.sparsity_loss()
        loss = cls_loss + sparsity_lambda * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_sp_loss += sp_loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return (
        total_loss / total,
        total_cls_loss / total,
        total_sp_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_sparsity = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

        frac_silent, mean_act = model.compute_sparsity_stats()
        all_sparsity.append(frac_silent)

    avg_sparsity = sum(all_sparsity) / len(all_sparsity)
    return total_loss / total, correct / total, avg_sparsity


def train_model(
    bottleneck_width, sparsity_lambda, epochs=20, lr=1e-4, device=None
):
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Training: bottleneck={bottleneck_width}, λ_sparse={sparsity_lambda}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    model = RetinalBottleneckWithHooks(retina_out_channels=bottleneck_width)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    train_loader, test_loader = get_cifar10_grayscale()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_acc": [], "test_acc": [],
        "train_loss": [], "test_loss": [],
        "sparsity": [],  # fraction of silent neurons
        "sparsity_lambda": sparsity_lambda,
        "bottleneck": bottleneck_width,
    }

    for epoch in range(epochs):
        t0 = time.time()
        loss, cls_loss, sp_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, sparsity_lambda
        )
        test_loss, test_acc, sparsity = evaluate(
            model, test_loader, criterion, device
        )
        elapsed = time.time() - t0

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(loss)
        history["test_loss"].append(test_loss)
        history["sparsity"].append(sparsity)

        print(
            f"  Epoch {epoch+1:2d}/{epochs} "
            f"| acc={test_acc:.4f} silent={sparsity:.1%} "
            f"| cls={cls_loss:.4f} sp={sp_loss:.4f} "
            f"| {elapsed:.1f}s"
        )

    # Save
    os.makedirs("results", exist_ok=True)
    tag = f"bn{bottleneck_width}_sp{sparsity_lambda}"
    torch.save(model.state_dict(), f"results/{tag}_model.pt")
    with open(f"results/{tag}_history.json", "w") as f:
        json.dump(history, f)

    print(f"  Final: acc={test_acc:.4f}, silent={sparsity:.1%}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Retinal Bottleneck + Sparsity")
    parser.add_argument("--bottleneck", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=0.001,
                        help="Sparsity penalty λ. 0=no penalty. Default: 0.001")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep sparsity λ: 0, 0.0001, 0.001, 0.01, 0.1")
    args = parser.parse_args()

    device = get_device()

    if args.sweep:
        lambdas = [0, 0.0001, 0.001, 0.01, 0.1]
        results = {}
        for lam in lambdas:
            _, history = train_model(
                args.bottleneck, lam,
                epochs=args.epochs, lr=args.lr, device=device
            )
            results[lam] = {
                "acc": history["test_acc"][-1],
                "sparsity": history["sparsity"][-1],
            }

        print(f"\n{'='*60}")
        print(f"SPARSITY SWEEP RESULTS (bottleneck={args.bottleneck})")
        print(f"{'='*60}")
        print(f"{'λ':>10} | {'Test Acc':>10} | {'Silent %':>10}")
        print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for lam in lambdas:
            r = results[lam]
            print(f"{lam:>10.4f} | {r['acc']:>9.4f} | {r['sparsity']:>9.1%}")
    else:
        train_model(
            args.bottleneck, args.sparsity,
            epochs=args.epochs, lr=args.lr, device=device
        )


if __name__ == "__main__":
    main()

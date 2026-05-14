"""
Train retinal bottleneck models with different bottleneck widths.

Usage:
    # Train a single model with bottleneck width = 2
    uv run python train.py --bottleneck 2

    # Train all bottleneck widths for comparison (the core experiment)
    uv run python train.py --all

    # Quick test (2 epochs)
    uv run python train.py --bottleneck 2 --epochs 2
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
    """Load CIFAR-10, convert to grayscale (matching Lindsey 2019)."""
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train_model(bottleneck_width, epochs=20, lr=1e-4, device=None):
    """Train one model and return results."""
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Training: bottleneck_width = {bottleneck_width}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    model = RetinalBottleneckModel(retina_out_channels=bottleneck_width)
    model = model.to(device)

    # Print param count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    train_loader, test_loader = get_cifar10_grayscale()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        print(
            f"  Epoch {epoch+1:2d}/{epochs} "
            f"| train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

    # Save model and history
    os.makedirs("results", exist_ok=True)
    tag = f"bottleneck_{bottleneck_width}"
    torch.save(model.state_dict(), f"results/{tag}_model.pt")
    with open(f"results/{tag}_history.json", "w") as f:
        json.dump(history, f)

    print(f"  Final test accuracy: {test_acc:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Retinal Bottleneck Experiment")
    parser.add_argument("--bottleneck", type=int, default=2,
                        help="Bottleneck width (key variable). Default: 2")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs. Default: 20")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate. Default: 1e-4")
    parser.add_argument("--all", action="store_true",
                        help="Train all bottleneck widths: 1, 2, 4, 8, 16, 32")
    args = parser.parse_args()

    device = get_device()

    if args.all:
        # Core experiment: sweep bottleneck widths
        widths = [1, 2, 4, 8, 16, 32]
        results = {}
        for w in widths:
            _, history = train_model(w, epochs=args.epochs, lr=args.lr, device=device)
            results[w] = history["test_acc"][-1]

        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Bottleneck':>12} | {'Test Accuracy':>14}")
        print(f"{'-'*12}-+-{'-'*14}")
        for w in widths:
            print(f"{w:>12} | {results[w]:>13.4f}")
    else:
        train_model(args.bottleneck, epochs=args.epochs, lr=args.lr, device=device)


if __name__ == "__main__":
    main()

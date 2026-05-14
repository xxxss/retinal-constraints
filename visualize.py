"""
Visualize learned receptive fields (convolutional filters) of trained models.

This is the key result: when bottleneck is narrow, the first layer should
learn center-surround (Mexican hat / DoG) receptive fields — the same
structure found in biological retinal ganglion cells.

Usage:
    # Visualize filters from a trained model
    uv run python visualize.py --bottleneck 2

    # Compare all bottleneck widths side by side
    uv run python visualize.py --compare
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no GUI popups
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RetinalBottleneckModel


def load_model(bottleneck_width):
    """Load a trained model."""
    path = f"results/bottleneck_{bottleneck_width}_model.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run train.py first."
        )
    model = RetinalBottleneckModel(retina_out_channels=bottleneck_width)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_first_layer_filters(model):
    """Extract first conv layer filters (retina_1)."""
    # First layer is retina[0] (Conv2d)
    first_conv = model.retina[0]
    filters = first_conv.weight.data.cpu().numpy()
    # Shape: (out_channels, in_channels=1, H, W)
    return filters[:, 0, :, :]  # squeeze input channel dim


def get_last_retina_filters(model):
    """Extract last retina conv layer filters (the bottleneck layer)."""
    # Find the last Conv2d in retina
    last_conv = None
    for module in model.retina:
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    filters = last_conv.weight.data.cpu().numpy()
    return filters


def plot_filters(filters, title, save_path=None):
    """Plot convolutional filters as a grid."""
    n = len(filters)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    vmax = max(abs(filters.min()), abs(filters.max()))

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            im = ax.imshow(
                filters[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                interpolation="nearest"
            )
            ax.set_title(f"#{i}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


def plot_filter_cross_section(filters, title, save_path=None):
    """Plot 1D cross-sections through filter centers.

    If filters are center-surround (Mexican hat), the cross-section
    should show a positive peak in the center with negative flanks
    (or vice versa).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    center = filters.shape[1] // 2

    for i, f in enumerate(filters):
        cross = f[center, :]  # horizontal cross-section through center
        ax.plot(cross, label=f"#{i}", alpha=0.7)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Weight")
    ax.set_title(f"{title} — Cross-sections (center row)")
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


def compare_all():
    """Compare first-layer filters across different bottleneck widths."""
    model_files = sorted(glob.glob("results/bottleneck_*_model.pt"))
    if not model_files:
        print("No trained models found. Run: uv run python train.py --all")
        return

    widths = []
    for f in model_files:
        w = int(f.split("bottleneck_")[1].split("_model")[0])
        widths.append(w)
    widths.sort()

    n = len(widths)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, w in enumerate(widths):
        model = load_model(w)
        filters = get_first_layer_filters(model)

        # Top row: first filter of each model
        vmax = max(abs(filters[0].min()), abs(filters[0].max()))
        axes[0, col].imshow(
            filters[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="nearest"
        )
        axes[0, col].set_title(f"BN={w}\nFilter #0", fontsize=11)
        axes[0, col].axis("off")

        # Bottom row: cross-section of first filter
        center = filters.shape[1] // 2
        cross = filters[0, center, :]
        axes[1, col].plot(cross, "b-", linewidth=2)
        axes[1, col].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, col].set_title("Cross-section", fontsize=10)
        axes[1, col].set_xlabel("Position")
        if col == 0:
            axes[1, col].set_ylabel("Weight")

    fig.suptitle(
        "Bottleneck Width vs Learned Receptive Fields\n"
        "(Narrow bottleneck → center-surround / Mexican hat)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path = "results/comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Retinal Filters")
    parser.add_argument("--bottleneck", type=int, default=2,
                        help="Bottleneck width to visualize")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all trained bottleneck widths")
    args = parser.parse_args()

    if args.compare:
        compare_all()
    else:
        model = load_model(args.bottleneck)
        filters = get_first_layer_filters(model)

        print(f"First layer filters shape: {filters.shape}")
        print(f"  (retina_hidden_channels filters, each {filters.shape[1]}x{filters.shape[2]})")

        plot_filters(
            filters,
            f"First Layer Filters (bottleneck={args.bottleneck})",
            save_path=f"results/filters_bn{args.bottleneck}.png",
        )
        plot_filter_cross_section(
            filters[:8],  # show first 8
            f"Bottleneck={args.bottleneck}",
            save_path=f"results/cross_section_bn{args.bottleneck}.png",
        )


if __name__ == "__main__":
    main()

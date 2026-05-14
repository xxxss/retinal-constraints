"""
Visualize the BOTTLENECK layer filters — this is where center-surround
receptive fields should emerge.

The bottleneck layer (retina_2) has only K output channels.
Each filter is (retina_hidden_channels=32, 9, 9), so we compute
the effective receptive field by combining retina_1 and retina_2.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RetinalBottleneckModel


def compute_effective_rf(model, image_size=32):
    """
    Compute effective receptive fields using gradient-based method.
    For each bottleneck channel, find which input pixels it responds to most.
    """
    model.eval()
    # Create a blank input that requires grad
    x = torch.zeros(1, 1, image_size, image_size, requires_grad=True)

    retina_out = model.get_retina_output(x)
    # retina_out shape: (1, bottleneck_channels, H, W)

    n_channels = retina_out.shape[1]
    rfs = []

    for ch in range(n_channels):
        if x.grad is not None:
            x.grad.zero_()
        # Sum activations at center pixel of this channel
        center_h, center_w = retina_out.shape[2] // 2, retina_out.shape[3] // 2
        target = retina_out[0, ch, center_h, center_w]
        target.backward(retain_graph=True)

        # Gradient w.r.t. input = effective receptive field
        rf = x.grad[0, 0].detach().numpy().copy()
        rfs.append(rf)

    return np.array(rfs)


def main():
    model = RetinalBottleneckModel(retina_out_channels=2)
    model.load_state_dict(
        torch.load("results/bottleneck_2_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    # === 1. Direct bottleneck layer weights ===
    # retina[2] is the second Conv2d (retina_2), the bottleneck layer
    # Its weight shape: (2, 32, 9, 9) — 2 output channels, 32 input channels
    bottleneck_conv = model.retina[2]  # Conv2d after first ReLU
    weights = bottleneck_conv.weight.data.numpy()
    print(f"Bottleneck layer weight shape: {weights.shape}")
    print(f"  = {weights.shape[0]} output channels × {weights.shape[1]} input channels × {weights.shape[2]}×{weights.shape[3]}")

    # Average across input channels to get 2D summary
    avg_filters = weights.mean(axis=1)  # (2, 9, 9)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        vmax = max(abs(avg_filters[i].min()), abs(avg_filters[i].max()))
        axes[0, i].imshow(avg_filters[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
        axes[0, i].set_title(f"Bottleneck Channel #{i}\n(avg across 32 input channels)", fontsize=11)
        axes[0, i].axis("off")

        # Cross-section
        center = avg_filters[i].shape[0] // 2
        cross = avg_filters[i][center, :]
        axes[1, i].plot(cross, "b-", linewidth=2, marker="o")
        axes[1, i].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, i].set_title(f"Cross-section (center row)", fontsize=10)
        axes[1, i].set_xlabel("Position")
        axes[1, i].set_ylabel("Weight")

    fig.suptitle("Bottleneck Layer Filters (retina_out_channels=2)\n"
                 "Should show center-surround structure if bottleneck works",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/bottleneck_filters.png", dpi=150, bbox_inches="tight")
    print("Saved: results/bottleneck_filters.png")

    # === 2. Effective receptive fields via gradient ===
    print("\nComputing effective receptive fields (gradient method)...")
    rfs = compute_effective_rf(model)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        # Crop to center 15x15 for visibility
        c = rfs[i].shape[0] // 2
        r = 7
        rf_crop = rfs[i][c-r:c+r+1, c-r:c+r+1]
        vmax = max(abs(rf_crop.min()), abs(rf_crop.max()))

        axes[0, i].imshow(rf_crop, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
        axes[0, i].set_title(f"Effective RF — Channel #{i}\n(gradient w.r.t. input)", fontsize=11)
        axes[0, i].axis("off")

        # Cross-section
        center = rf_crop.shape[0] // 2
        cross = rf_crop[center, :]
        axes[1, i].plot(cross, "b-", linewidth=2, marker="o")
        axes[1, i].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, i].set_title("Cross-section", fontsize=10)
        axes[1, i].set_xlabel("Position")
        axes[1, i].set_ylabel("Gradient magnitude")

    fig.suptitle("Effective Receptive Fields (bottleneck=2)\n"
                 "Center-surround = Mexican hat = positive center, negative surround (or vice versa)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/effective_rf.png", dpi=150, bbox_inches="tight")
    print("Saved: results/effective_rf.png")


if __name__ == "__main__":
    main()

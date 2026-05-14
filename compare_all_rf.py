"""
Compare effective receptive fields AND accuracy across all bottleneck widths.
This is the core figure of the experiment.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import torch

from model import RetinalBottleneckModel


def compute_effective_rf(model, image_size=32):
    model.eval()
    x = torch.zeros(1, 1, image_size, image_size, requires_grad=True)
    retina_out = model.get_retina_output(x)
    n_channels = retina_out.shape[1]
    rfs = []
    for ch in range(min(n_channels, 4)):  # show up to 4 channels
        if x.grad is not None:
            x.grad.zero_()
        center_h, center_w = retina_out.shape[2] // 2, retina_out.shape[3] // 2
        target = retina_out[0, ch, center_h, center_w]
        target.backward(retain_graph=True)
        rf = x.grad[0, 0].detach().numpy().copy()
        rfs.append(rf)
    return np.array(rfs)


def main():
    widths = [1, 2, 4, 8, 16, 32]
    accuracies = {}

    # Load accuracies
    for w in widths:
        try:
            with open(f"results/bottleneck_{w}_history.json") as f:
                hist = json.load(f)
                accuracies[w] = hist["test_acc"][-1]
        except FileNotFoundError:
            pass

    # === Figure 1: Effective RFs across bottleneck widths ===
    fig, axes = plt.subplots(3, len(widths), figsize=(3.5 * len(widths), 10))

    for col, w in enumerate(widths):
        model = RetinalBottleneckModel(retina_out_channels=w)
        model.load_state_dict(
            torch.load(f"results/bottleneck_{w}_model.pt", map_location="cpu", weights_only=True)
        )
        rfs = compute_effective_rf(model)

        # Row 0: First channel effective RF (2D)
        c = rfs[0].shape[0] // 2
        r = 7
        rf_crop = rfs[0][c-r:c+r+1, c-r:c+r+1]
        vmax = max(abs(rf_crop.min()), abs(rf_crop.max())) or 1e-6
        axes[0, col].imshow(rf_crop, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            interpolation="nearest")
        acc_str = f"{accuracies.get(w, 0)*100:.1f}%" if w in accuracies else "?"
        axes[0, col].set_title(f"BN={w}\nacc={acc_str}", fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: Cross-section of first channel
        center = rf_crop.shape[0] // 2
        cross = rf_crop[center, :]
        axes[1, col].plot(cross, "b-", linewidth=2, marker="o", markersize=3)
        axes[1, col].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, col].set_title("Channel #0 cross-section", fontsize=9)
        if col == 0:
            axes[1, col].set_ylabel("Weight")

        # Row 2: Second channel RF (if exists), else average of all
        if len(rfs) > 1:
            rf2 = rfs[1][c-r:c+r+1, c-r:c+r+1]
            vmax2 = max(abs(rf2.min()), abs(rf2.max())) or 1e-6
            axes[2, col].imshow(rf2, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2,
                                interpolation="nearest")
            axes[2, col].set_title("Channel #1", fontsize=9)
        else:
            axes[2, col].text(0.5, 0.5, "Only 1\nchannel", ha="center", va="center",
                              transform=axes[2, col].transAxes, fontsize=10)
        axes[2, col].axis("off")

    fig.suptitle(
        "Effective Receptive Fields vs Bottleneck Width\n"
        "Narrow bottleneck (left) → center-surround structure | Wide (right) → unstructured",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("results/all_effective_rf.png", dpi=150, bbox_inches="tight")
    print("Saved: results/all_effective_rf.png")

    # === Figure 2: Accuracy vs Bottleneck Width ===
    if accuracies:
        fig2, ax = plt.subplots(figsize=(8, 5))
        ws = sorted(accuracies.keys())
        accs = [accuracies[w] * 100 for w in ws]
        ax.plot(ws, accs, "bo-", linewidth=2, markersize=8)
        for w, a in zip(ws, accs):
            ax.annotate(f"{a:.1f}%", (w, a), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10)
        ax.set_xlabel("Bottleneck Width (channels)", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title("Accuracy vs Bottleneck Width\n"
                      "(Narrower bottleneck = stronger compression = lower accuracy)",
                      fontsize=13, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ws)
        ax.set_xticklabels([str(w) for w in ws])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/accuracy_vs_bottleneck.png", dpi=150, bbox_inches="tight")
        print("Saved: results/accuracy_vs_bottleneck.png")


if __name__ == "__main__":
    main()

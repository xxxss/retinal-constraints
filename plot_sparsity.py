"""Plot sparsity sweep results: accuracy vs sparsity trade-off."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import glob
import os
import numpy as np


def main():
    # Load all sparse experiment results
    files = sorted(glob.glob("results/bn2_sp*_history.json"))
    # Also include the baseline (no sparsity) from step 1
    baseline_file = "results/bottleneck_2_history.json"

    results = []
    for f in files:
        with open(f) as fh:
            h = json.load(fh)
        results.append({
            "lambda": h["sparsity_lambda"],
            "acc": h["test_acc"][-1] * 100,
            "sparsity": h["sparsity"][-1] * 100,
            "acc_history": [a * 100 for a in h["test_acc"]],
            "sparsity_history": [s * 100 for s in h["sparsity"]],
        })

    results.sort(key=lambda x: x["lambda"])

    # === Figure 1: Accuracy vs Sparsity trade-off ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lambdas = [r["lambda"] for r in results]
    accs = [r["acc"] for r in results]
    sparsities = [r["sparsity"] for r in results]

    # Left: Accuracy and Sparsity vs Lambda
    ax1 = axes[0]
    color_acc = "tab:blue"
    color_sp = "tab:red"

    ax1.set_xlabel("Sparsity Penalty λ", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=12)
    l1 = ax1.plot(range(len(lambdas)), accs, "o-", color=color_acc, linewidth=2,
                  markersize=8, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silent Neurons (%)", color=color_sp, fontsize=12)
    l2 = ax2.plot(range(len(lambdas)), sparsities, "s-", color=color_sp, linewidth=2,
                  markersize=8, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor=color_sp)

    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels([str(l) for l in lambdas])
    ax1.set_title("Accuracy vs Sparsity Trade-off", fontsize=13, fontweight="bold")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Scatter plot — accuracy vs sparsity
    ax3 = axes[1]
    ax3.scatter(sparsities, accs, s=100, c=range(len(lambdas)), cmap="viridis",
                zorder=5, edgecolors="black")
    for i, r in enumerate(results):
        ax3.annotate(f"λ={r['lambda']}", (r["sparsity"], r["acc"]),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax3.set_xlabel("Silent Neurons (%)", fontsize=12)
    ax3.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax3.set_title("Accuracy vs Sparsity\n(ideal = top-right corner)", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Mark the "biological zone" — brain is 96-98% silent
    ax3.axvspan(90, 100, alpha=0.1, color="green", label="Biological zone (96-98% silent)")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("results/sparsity_tradeoff.png", dpi=150, bbox_inches="tight")
    print("Saved: results/sparsity_tradeoff.png")

    # === Figure 2: Training curves ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        epochs = range(1, len(r["acc_history"]) + 1)
        axes2[0].plot(epochs, r["acc_history"], label=f"λ={r['lambda']}", linewidth=1.5)
        axes2[1].plot(epochs, r["sparsity_history"], label=f"λ={r['lambda']}", linewidth=1.5)

    axes2[0].set_xlabel("Epoch")
    axes2[0].set_ylabel("Test Accuracy (%)")
    axes2[0].set_title("Accuracy During Training", fontsize=13, fontweight="bold")
    axes2[0].legend(fontsize=9)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_xlabel("Epoch")
    axes2[1].set_ylabel("Silent Neurons (%)")
    axes2[1].set_title("Sparsity During Training", fontsize=13, fontweight="bold")
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/sparsity_training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: results/sparsity_training_curves.png")

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'λ':>8} | {'Accuracy':>10} | {'Silent':>10} | {'Acc Drop':>10}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    baseline_acc = results[0]["acc"]
    for r in results:
        drop = r["acc"] - baseline_acc
        print(f"{r['lambda']:>8} | {r['acc']:>9.1f}% | {r['sparsity']:>9.1f}% | {drop:>+9.1f}%")


if __name__ == "__main__":
    main()

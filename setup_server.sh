#!/bin/bash
# One-click setup for GPU cloud server
# Usage: bash setup_server.sh
set -e

echo "=== Retinal Constraints Experiment Setup ==="

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Install project dependencies
echo "Installing Python dependencies..."
uv sync

# 3. Download ImageNette (small, fast, no login)
echo "Downloading ImageNette dataset..."
uv run python download_imagenette.py

# 4. Quick sanity check
echo "Running sanity check..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run experiments:"
echo "  # Step 1: Bottleneck experiment (Lindsey 2019 replication)"
echo "  uv run python train.py --all --epochs 20"
echo "  uv run python compare_all_rf.py"
echo ""
echo "  # Step 2: Sparsity penalty experiment"
echo "  uv run python train_sparse.py --sweep --epochs 20"
echo "  uv run python plot_sparsity.py"
echo ""
echo "  # Step 3: OOD robustness (the key experiment)"
echo "  uv run python eval_robustness.py --mode all"
echo ""
echo "  # If you have ImageNet val set, put it at ~/datasets/imagenet/val/"
echo "  # and the scripts will auto-detect it."

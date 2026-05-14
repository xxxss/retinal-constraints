"""
Difference-of-Gaussians (DoG) preprocessing layer.

Two versions:
  1. FixedDoG — fixed σ values, no learnable params (reproduces PNAS 2025 approach)
  2. LearnableDoG — σ_center, σ_surround, mixing weight are learnable

Both compute:  output = α * Gauss(σ_center) + (1-α) * [input - Gauss(σ_surround)]

When α=0: pure DoG (center-surround, like retinal ganglion cells)
When α=1: pure smoothing (like low-light adaptation)
The network learns where on this spectrum to operate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian_kernel_2d(sigma, kernel_size=None):
    """Create a 2D Gaussian kernel."""
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d / gauss_2d.sum()
    return gauss_2d


class FixedDoG(nn.Module):
    """Fixed Difference-of-Gaussians preprocessing.

    Applies DoG filter with fixed σ values to each input channel independently.
    No learnable parameters — this is the PNAS 2025 baseline.
    """

    def __init__(self, sigma_center=1.0, sigma_surround=3.0, kernel_size=13):
        super().__init__()
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.kernel_size = kernel_size

        # Precompute DoG kernel
        k_center = gaussian_kernel_2d(sigma_center, kernel_size)
        k_surround = gaussian_kernel_2d(sigma_surround, kernel_size)
        dog_kernel = k_center - k_surround  # center - surround

        # Shape: (1, 1, H, W) — will be applied per-channel via groups
        self.register_buffer("dog_kernel", dog_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape
        # Apply DoG to each channel independently using groups=C
        kernel = self.dog_kernel.expand(C, 1, -1, -1)
        padding = self.kernel_size // 2
        dog_out = F.conv2d(x, kernel, padding=padding, groups=C)

        # Concatenate original + DoG (gives the network both raw and filtered)
        return torch.cat([x, dog_out], dim=1)  # output: (B, 2C, H, W)

    def extra_repr(self):
        return f"σ_center={self.sigma_center}, σ_surround={self.sigma_surround}, fixed"


class LearnableDoG(nn.Module):
    """Learnable Difference-of-Gaussians preprocessing.

    σ_center, σ_surround, and mixing weight α are learnable parameters
    that get optimized end-to-end with the classification loss.

    For each input channel, produces 2 output channels:
      - Original (passthrough)
      - Learned DoG filtered version

    The network learns the optimal center-surround balance.
    """

    def __init__(
        self,
        in_channels=3,
        init_sigma_center=1.0,
        init_sigma_surround=3.0,
        kernel_size=13,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Learnable parameters (one set per input channel)
        # Use log-space to ensure σ > 0
        self.log_sigma_center = nn.Parameter(
            torch.full((in_channels,), math.log(init_sigma_center))
        )
        self.log_sigma_surround = nn.Parameter(
            torch.full((in_channels,), math.log(init_sigma_surround))
        )
        # Mixing weight: how much of original vs DoG to keep
        # α=0 → pure DoG, α=1 → pure original
        self.alpha_logit = nn.Parameter(torch.zeros(in_channels))

    def _build_dog_kernels(self):
        """Build DoG kernels from current learnable parameters."""
        sigma_c = self.log_sigma_center.exp()   # (C,)
        sigma_s = self.log_sigma_surround.exp()  # (C,)
        ks = self.kernel_size
        half = ks // 2

        # Grid for Gaussian computation
        coords = torch.arange(ks, device=sigma_c.device, dtype=torch.float32) - half
        grid = coords[:, None] ** 2 + coords[None, :] ** 2  # (ks, ks)

        # Compute per-channel Gaussians: (C, ks, ks)
        g_center = torch.exp(-grid[None] / (2 * sigma_c[:, None, None] ** 2))
        g_center = g_center / g_center.sum(dim=(-2, -1), keepdim=True)

        g_surround = torch.exp(-grid[None] / (2 * sigma_s[:, None, None] ** 2))
        g_surround = g_surround / g_surround.sum(dim=(-2, -1), keepdim=True)

        dog = g_center - g_surround  # (C, ks, ks)
        return dog.unsqueeze(1)  # (C, 1, ks, ks) for grouped conv

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels

        # Build kernels from current parameters
        dog_kernels = self._build_dog_kernels()  # (C, 1, ks, ks)
        padding = self.kernel_size // 2

        # Apply per-channel DoG
        dog_out = F.conv2d(x, dog_kernels, padding=padding, groups=C)

        # Mix original and DoG
        alpha = torch.sigmoid(self.alpha_logit)  # (C,), range [0, 1]
        alpha = alpha[None, :, None, None]  # (1, C, 1, 1)
        mixed = alpha * x + (1 - alpha) * dog_out

        # Concatenate original + mixed DoG
        return torch.cat([x, mixed], dim=1)  # (B, 2C, H, W)

    def get_learned_params(self):
        """Return current learned values for visualization."""
        with torch.no_grad():
            return {
                "sigma_center": self.log_sigma_center.exp().cpu().numpy(),
                "sigma_surround": self.log_sigma_surround.exp().cpu().numpy(),
                "alpha": torch.sigmoid(self.alpha_logit).cpu().numpy(),
            }

    def extra_repr(self):
        with torch.no_grad():
            sc = self.log_sigma_center.exp().mean().item()
            ss = self.log_sigma_surround.exp().mean().item()
            a = torch.sigmoid(self.alpha_logit).mean().item()
        return f"σ_center≈{sc:.2f}, σ_surround≈{ss:.2f}, α≈{a:.2f}, learnable"

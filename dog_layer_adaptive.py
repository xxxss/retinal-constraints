"""
Adaptive DoG layer — σ values adjust based on input noise level.

Key insight from the previous experiment:
  - Fixed DoG (σ=1.0/2.0) → robust but not optimal
  - Learned DoG → overfit to clean data, σ too small → fragile

Biological solution: retina adapts receptive field size based on conditions:
  - Bright/clean → small σ (Mexican hat, high resolution)
  - Dark/noisy  → large σ (bell shape, noise reduction)

This layer estimates input noise level and interpolates between
a sharp filter (small σ) and a smooth filter (large σ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveDoG(nn.Module):
    """
    Noise-adaptive Difference-of-Gaussians.

    Estimates input signal quality, then interpolates between:
      - Sharp DoG (small σ) for clean inputs → high resolution
      - Smooth DoG (large σ) for noisy inputs → noise reduction

    This mimics retinal light adaptation:
      bright → Mexican hat (decorrelation)
      dark   → bell shape (spatial averaging / denoising)
    """

    def __init__(
        self,
        in_channels=3,
        sigma_sharp_center=0.8,
        sigma_sharp_surround=1.5,
        sigma_smooth_center=2.0,
        sigma_smooth_surround=4.0,
        kernel_size=13,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Precompute two fixed DoG kernels
        k_sharp = self._make_dog(sigma_sharp_center, sigma_sharp_surround, kernel_size)
        k_smooth = self._make_dog(sigma_smooth_center, sigma_smooth_surround, kernel_size)

        # Shape: (1, 1, H, W) for depthwise conv
        self.register_buffer("kernel_sharp", k_sharp.unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_smooth", k_smooth.unsqueeze(0).unsqueeze(0))

        # Noise estimator: small conv network that estimates input "quality"
        # Outputs a single scalar per image: 0 = clean, 1 = noisy
        self.noise_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),     # downsample to 8x8
            nn.Flatten(),
            nn.Linear(in_channels * 64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),                 # output in [0, 1]
        )

    @staticmethod
    def _make_dog(sigma_center, sigma_surround, kernel_size):
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g_center = torch.exp(-x ** 2 / (2 * sigma_center ** 2))
        g_center = g_center[:, None] * g_center[None, :]
        g_center = g_center / g_center.sum()

        g_surround = torch.exp(-x ** 2 / (2 * sigma_surround ** 2))
        g_surround = g_surround[:, None] * g_surround[None, :]
        g_surround = g_surround / g_surround.sum()

        return g_center - g_surround

    def forward(self, x):
        B, C, H, W = x.shape
        padding = self.kernel_size // 2

        # Estimate noise level per image: (B, 1)
        noise_level = self.noise_estimator(x)  # (B, 1)

        # Apply both DoG filters
        k_sharp = self.kernel_sharp.expand(C, 1, -1, -1)
        k_smooth = self.kernel_smooth.expand(C, 1, -1, -1)

        dog_sharp = F.conv2d(x, k_sharp, padding=padding, groups=C)
        dog_smooth = F.conv2d(x, k_smooth, padding=padding, groups=C)

        # Interpolate: noisy → use smooth, clean → use sharp
        alpha = noise_level[:, :, None, None]  # (B, 1, 1, 1)
        dog_adaptive = (1 - alpha) * dog_sharp + alpha * dog_smooth

        # Concatenate original + adaptive DoG
        return torch.cat([x, dog_adaptive], dim=1)  # (B, 2C, H, W)

    def get_noise_level(self, x):
        """Return estimated noise level for debugging."""
        with torch.no_grad():
            return self.noise_estimator(x).squeeze(-1).cpu().numpy()

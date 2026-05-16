"""
Adaptive DoG layer v2 — σ values adjust based on input noise level.

v1 problem: noise estimator used AdaptiveAvgPool which smooths out
the very noise signal we're trying to detect.

v2 fix: use LOCAL VARIANCE as noise feature. Noisy images have high
pixel-level variance within small patches; clean images have low variance.
This is how real noise estimation works (and analogous to how retinal
horizontal cells detect local contrast).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDoG(nn.Module):
    """
    Noise-adaptive Difference-of-Gaussians.

    Estimates input signal quality via local variance, then interpolates:
      - Sharp DoG (small σ) for clean inputs → high resolution
      - Smooth DoG (large σ) for noisy inputs → noise reduction
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

        self.register_buffer("kernel_sharp", k_sharp.unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_smooth", k_smooth.unsqueeze(0).unsqueeze(0))

        # v2 noise estimator: based on local variance
        # Step 1: compute local variance via depthwise conv (not learnable)
        # Step 2: small MLP maps variance stats to noise level
        self.noise_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 16),  # mean + std of local variance per channel
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
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

    def _estimate_noise(self, x):
        """Estimate noise level from local variance.

        Clean image: smooth regions have low local variance
        Noisy image: even smooth regions have high local variance

        This is analogous to how horizontal cells in the retina
        detect local contrast by comparing nearby photoreceptor outputs.
        """
        B, C, H, W = x.shape

        # Compute local mean with 5x5 average pooling
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)

        # Local variance = E[x^2] - E[x]^2
        local_var = F.avg_pool2d(x ** 2, kernel_size=5, stride=1, padding=2) - local_mean ** 2
        local_var = local_var.clamp(min=0)  # numerical stability

        # Per-channel statistics: mean and std of local variance
        # Shape: (B, C) each
        var_mean = local_var.mean(dim=(-2, -1))  # avg local variance per channel
        var_std = local_var.std(dim=(-2, -1))     # how variable the variance is

        # Concatenate: (B, 2*C)
        features = torch.cat([var_mean, var_std], dim=1)

        # MLP → noise level in [0, 1]
        return self.noise_mlp(features)  # (B, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        padding = self.kernel_size // 2

        # Estimate noise level per image
        noise_level = self._estimate_noise(x)  # (B, 1)

        # Apply both DoG filters
        k_sharp = self.kernel_sharp.expand(C, 1, -1, -1)
        k_smooth = self.kernel_smooth.expand(C, 1, -1, -1)

        dog_sharp = F.conv2d(x, k_sharp, padding=padding, groups=C)
        dog_smooth = F.conv2d(x, k_smooth, padding=padding, groups=C)

        # Interpolate: clean(α→0) → sharp, noisy(α→1) → smooth
        alpha = noise_level[:, :, None, None]  # (B, 1, 1, 1)
        dog_adaptive = (1 - alpha) * dog_sharp + alpha * dog_smooth

        return torch.cat([x, dog_adaptive], dim=1)

    def get_noise_level(self, x):
        """Return estimated noise level for debugging."""
        with torch.no_grad():
            return self._estimate_noise(x).squeeze(-1).cpu().numpy()

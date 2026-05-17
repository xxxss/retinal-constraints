"""
Retina Preprocessor — GPU version using PyTorch MPS/CUDA.

Uses the full AdaptiveDoG with all components validated in experiments:
  - Photoreceptor log compression
  - Adaptive DoG (sharp/smooth interpolation)
  - Noise estimation via local variance

Keeps a persistent GPU tensor pipeline to avoid CPU↔GPU transfers per frame.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class RetinaPreprocessorGPU:
    def __init__(
        self,
        log_scale=10.0,
        dog_sigma_sharp_center=0.8,
        dog_sigma_sharp_surround=1.5,
        dog_sigma_smooth_center=2.0,
        dog_sigma_smooth_surround=4.0,
        dog_strength=0.5,
        noise_threshold=0.005,
        kernel_size=13,
        device=None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.log_scale = log_scale
        self.log_norm = float(np.log1p(log_scale)) if log_scale > 0 else 1.0
        self.dog_strength = dog_strength
        self.noise_threshold = noise_threshold
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Build DoG kernels on GPU
        k_sharp = self._make_dog(dog_sigma_sharp_center, dog_sigma_sharp_surround, kernel_size)
        k_smooth = self._make_dog(dog_sigma_smooth_center, dog_sigma_smooth_surround, kernel_size)

        # Shape: (3, 1, K, K) for depthwise conv on 3 channels
        self.kernel_sharp = k_sharp.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).to(self.device)
        self.kernel_smooth = k_smooth.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).to(self.device)

        # Temporal state
        self.prev_gray = None
        self.frame_count = 0

        print(f"RetinaPreprocessorGPU on {self.device}")

    @staticmethod
    def _make_dog(sigma_center, sigma_surround, size):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        gc = torch.exp(-x ** 2 / (2 * sigma_center ** 2))
        gc = (gc[:, None] * gc[None, :])
        gc = gc / gc.sum()
        gs = torch.exp(-x ** 2 / (2 * sigma_surround ** 2))
        gs = (gs[:, None] * gs[None, :])
        gs = gs / gs.sum()
        return gc - gs

    def _estimate_local_variance(self, x):
        """Estimate noise via local variance. Returns per-image scalar."""
        # x: (1, 3, H, W)
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_var = F.avg_pool2d(x ** 2, kernel_size=5, stride=1, padding=2) - local_mean ** 2
        local_var = local_var.clamp(min=0)
        # Mean variance across all pixels and channels
        return local_var.mean().item()

    @torch.no_grad()
    def process_tensor(self, x):
        """Process (1, 3, H, W) float tensor on GPU. Returns same shape."""
        # 1. Photoreceptor log compression
        if self.log_scale > 0:
            x = torch.log1p(x * self.log_scale) / self.log_norm

        # 2. Adaptive DoG
        if self.dog_strength > 0:
            dog_sharp = F.conv2d(x, self.kernel_sharp, padding=self.padding, groups=3)
            dog_smooth = F.conv2d(x, self.kernel_smooth, padding=self.padding, groups=3)

            # Noise-adaptive: use local variance to choose filter
            noise_level = self._estimate_local_variance(x)
            alpha = min(1.0, noise_level / max(self.noise_threshold, 1e-8))
            alpha = min(alpha, 1.0)

            dog = (1 - alpha) * dog_sharp + alpha * dog_smooth
            x = x + self.dog_strength * dog

        return x.clamp(0, 1)

    def process_image(self, bgr_frame):
        """Process OpenCV BGR frame. Returns BGR uint8."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        out = self.process_tensor(tensor)

        out_np = out[0].cpu().permute(1, 2, 0).numpy()
        out_np = (out_np * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def process_video_frame(self, bgr_frame):
        """Process video frame. Returns (processed_bgr, stats)."""
        self.frame_count += 1

        # Temporal change stats
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        changed_ratio = 1.0
        if self.prev_gray is not None:
            delta = cv2.absdiff(gray, self.prev_gray)
            changed_ratio = float((delta > 15).mean())
        self.prev_gray = gray

        # Process
        processed = self.process_image(bgr_frame)

        # Get noise estimate for display
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        noise_var = self._estimate_local_variance(tensor)

        stats = {
            "frame": self.frame_count,
            "changed": changed_ratio,
            "noise_var": noise_var,
        }
        return processed, stats

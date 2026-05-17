"""
Retina Preprocessor v2 — fast, using OpenCV native operations.

v1 problems:
  - Used PyTorch conv2d on CPU → 3fps on 1080p
  - Temporal blending caused ghosting artifacts

v2 fixes:
  - All operations use OpenCV (C++ backend), no PyTorch at inference
  - Temporal: only track change ratio for stats, no frame blending
  - DoG via cv2.GaussianBlur (hardware-optimized)
"""

import cv2
import numpy as np


class RetinaPreprocessor:
    """
    Fast retinal preprocessing using OpenCV.

    Usage:
        retina = RetinaPreprocessor()

        # Single image
        processed = retina.process_image(frame)

        # Video frame (tracks temporal stats)
        processed, stats = retina.process_video_frame(frame)
    """

    def __init__(
        self,
        log_scale=10.0,
        dog_sigma_center=1.5,
        dog_sigma_surround=3.0,
        dog_strength=0.3,
        delta_threshold=15,
    ):
        self.log_scale = log_scale
        self.log_norm = np.log1p(log_scale).astype(np.float32) if log_scale > 0 else 1.0
        self.dog_sigma_center = dog_sigma_center
        self.dog_sigma_surround = dog_sigma_surround
        self.dog_strength = dog_strength
        self.delta_threshold = delta_threshold

        # Temporal state
        self.prev_gray = None
        self.frame_count = 0

        # Precompute kernel sizes (must be odd)
        self.ksize_center = self._sigma_to_ksize(dog_sigma_center)
        self.ksize_surround = self._sigma_to_ksize(dog_sigma_surround)

    @staticmethod
    def _sigma_to_ksize(sigma):
        k = int(2 * np.ceil(3 * sigma) + 1)
        if k % 2 == 0:
            k += 1
        return k

    def process_image(self, bgr_frame):
        """Process single frame. Input/output: BGR uint8."""
        img = bgr_frame.astype(np.float32) / 255.0

        # 1. Photoreceptor: log compression
        if self.log_scale > 0:
            img = np.log1p(img * self.log_scale) / self.log_norm

        # 2. Ganglion cell: DoG = GaussianBlur(small σ) - GaussianBlur(large σ)
        if self.dog_strength > 0:
            g_center = cv2.GaussianBlur(img, (self.ksize_center, self.ksize_center),
                                         self.dog_sigma_center)
            g_surround = cv2.GaussianBlur(img, (self.ksize_surround, self.ksize_surround),
                                           self.dog_sigma_surround)
            dog = g_center - g_surround
            img = img + self.dog_strength * dog

        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def process_video_frame(self, bgr_frame):
        """Process video frame. Returns (processed_frame, stats_dict)."""
        self.frame_count += 1

        # Compute temporal change stats (but don't blend frames)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        changed_ratio = 1.0

        if self.prev_gray is not None and self.delta_threshold > 0:
            delta = cv2.absdiff(gray, self.prev_gray)
            change_mask = (delta > self.delta_threshold).astype(np.float32)
            changed_ratio = float(change_mask.mean())

        self.prev_gray = gray

        # Always process the full frame (no blending = no ghosting)
        processed = self.process_image(bgr_frame)

        stats = {
            "frame": self.frame_count,
            "changed": changed_ratio,
            "skipped": 1.0 - changed_ratio,
        }
        return processed, stats

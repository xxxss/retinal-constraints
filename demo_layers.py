"""
Retinal processing pipeline — biologically accurate layer structure.

Signal flow matches the actual retina:

  ① Photoreceptors: log(light) → cone outputs (R, G, B)
  ② Horizontal cells: lateral connections compute local average
                      → feedback to photoreceptors (gain control)
                      → provides "surround" signal to bipolar cells
  ③ Bipolar cells: center(from photoreceptor) - surround(from horizontal)
                   → ON-bipolar: relu(center - surround)
                   → OFF-bipolar: relu(surround - center)
                   This is WHERE center-surround (DoG) actually happens
  ④ Amacrine cells: temporal processing (frame differencing)
  ⑤ Ganglion cells: color opponent recoding + spike output
                    → Parasol/M: luminance ON/OFF (R+G)
                    → Midget/P:  R-G opponent
                    → Bistratified/K: B-Y opponent

Usage:
    uv run python demo_layers.py
    uv run python demo_layers.py --input video.mp4
"""

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class RetinaLayerVis:
    """Biologically accurate retinal pipeline with visualization."""

    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.log_scale = 10.0
        self.log_norm = float(np.log1p(10.0))

        # Horizontal cell neighborhood (for computing local average / surround)
        self.horizontal_size = 15

        # M/P/K pathway parameters — different receptive field sizes
        # Parasol (M): large RF, low resolution → motion/depth
        # Midget (P):  small RF, high resolution → detail/color
        # Bistratified (K): medium RF → blue-yellow
        ks = 13
        self.padding = ks // 2

        # M pathway: large receptive fields
        self.k_center_M = self._make_gaussian(1.5, ks).unsqueeze(0).unsqueeze(0).to(self.device)
        self.k_surround_M = self._make_gaussian(4.0, ks).unsqueeze(0).unsqueeze(0).to(self.device)

        # P pathway: small receptive fields (high acuity)
        self.k_center_P = self._make_gaussian(0.5, ks).unsqueeze(0).unsqueeze(0).to(self.device)
        self.k_surround_P = self._make_gaussian(1.5, ks).unsqueeze(0).unsqueeze(0).to(self.device)

        # K pathway: medium receptive fields
        self.k_center_K = self._make_gaussian(1.0, ks).unsqueeze(0).unsqueeze(0).to(self.device)
        self.k_surround_K = self._make_gaussian(3.0, ks).unsqueeze(0).unsqueeze(0).to(self.device)

        # Compression stride per pathway
        # M: aggressive compression (low res anyway)
        # P: minimal compression (need to preserve detail)
        # K: moderate
        self.stride_M = 6
        self.stride_P = 2
        self.stride_K = 4

        # Temporal state (amacrine cells)
        self.prev_gray = None
        self.bg_model = None       # slow background model
        self.fast_buffer = None    # fast temporal buffer

        # Spike state (ganglion cells)
        self.spike_accumulator = None  # accumulate spikes over time for rate estimate
        self.spike_frame_count = 0

        print(f"RetinaLayerVis on {self.device}")

    @staticmethod
    def _make_gaussian(sigma, size):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-x**2 / (2 * sigma**2))
        g2d = g[:, None] * g[None, :]
        return g2d / g2d.sum()

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

    def _tensor_to_bgr(self, t):
        out = t[0].cpu().permute(1, 2, 0).numpy()
        out = (out * 255).clip(0, 255).astype(np.uint8)
        if out.shape[2] == 1:
            return cv2.cvtColor(out[:, :, 0], cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def _single_channel_vis(self, t, colormap=None):
        """Signed tensor → visualization. Gray=0, bright=positive, dark=negative."""
        d = t[0, 0].cpu().numpy()
        vmax = max(abs(d.min()), abs(d.max()), 1e-6)
        normalized = ((d / vmax) * 0.5 + 0.5)
        normalized = (normalized * 255).clip(0, 255).astype(np.uint8)
        if colormap is not None:
            return cv2.applyColorMap(normalized, colormap)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    @torch.no_grad()
    def process(self, bgr_frame):
        x = self._to_tensor(bgr_frame)  # (1, 3, H, W) RGB
        layers = {}

        # ================================================================
        # Layer 0: Raw input
        # ================================================================
        layers["0_input"] = bgr_frame.copy()

        # ================================================================
        # Layer 1: PHOTORECEPTORS
        # Log compression: output ≈ log(light intensity)
        # Compresses dynamic range so moonlight and sunlight both usable
        # Three cone types (R, G, B) each do this independently
        # ================================================================
        x_photo = torch.log1p(x.clamp(min=0) * self.log_scale) / self.log_norm
        layers["1_photoreceptor"] = self._tensor_to_bgr(x_photo)

        R = x_photo[:, 0:1]
        G = x_photo[:, 1:2]
        B = x_photo[:, 2:3]

        # Cone visualization
        zeros = torch.zeros_like(R)
        r_img = self._tensor_to_bgr(torch.cat([R, zeros, zeros], dim=1))
        g_img = self._tensor_to_bgr(torch.cat([zeros, G, zeros], dim=1))
        b_img = self._tensor_to_bgr(torch.cat([zeros, zeros, B], dim=1))
        h, w = r_img.shape[:2]
        w3 = w // 3
        cones = np.zeros_like(r_img)
        cones[:, :w3] = cv2.resize(r_img, (w3, h))
        cones[:, w3:w3*2] = cv2.resize(g_img, (w3, h))
        cones[:, w3*2:] = cv2.resize(b_img, (w - w3*2, h))
        layers["1b_cones"] = cones

        # ================================================================
        # Layer 2: HORIZONTAL CELLS
        # Lateral connections across photoreceptors via gap junctions.
        # Computes local weighted average (≈ Gaussian due to signal decay
        # through gap junctions).
        #
        # Two functions:
        #   a) Gain control: adapt to local brightness (light adaptation)
        #      → divides photoreceptor output by local mean
        #   b) Provides "surround" signal to bipolar cells
        #      → this surround signal is what creates center-surround
        # ================================================================
        hs = self.horizontal_size
        hp = hs // 2
        # This IS the horizontal cell output: local weighted average
        # In biology: signal spreads through gap junctions, decays with distance
        horizontal_output = F.avg_pool2d(x_photo, kernel_size=hs, stride=1, padding=hp)

        # 2a) Gain control: photoreceptor / local_mean
        x_adapted = x_photo / (horizontal_output + 0.05)
        x_adapted = x_adapted / (x_adapted.max() + 1e-6)
        layers["2a_gain_ctrl"] = self._tensor_to_bgr(x_adapted.clamp(0, 1))

        # Gain map visualization
        gain = 1.0 / (horizontal_output + 0.05)
        gain_vis = gain[0].mean(dim=0).cpu().numpy()
        gain_vis = gain_vis / max(gain_vis.max(), 1e-6)
        layers["2b_gain_map"] = cv2.applyColorMap(
            (gain_vis * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )

        # 2b) Surround signals — DIFFERENT per pathway
        # M pathway gets wide surround, P pathway gets narrow surround
        surround_M = F.conv2d(x_adapted, self.k_surround_M.expand(3, 1, -1, -1),
                              padding=self.padding, groups=3)
        surround_P = F.conv2d(x_adapted, self.k_surround_P.expand(3, 1, -1, -1),
                              padding=self.padding, groups=3)

        layers["2c_surround"] = self._tensor_to_bgr(surround_M.clamp(0, 1))

        # ================================================================
        # Layer 3: BIPOLAR CELLS — separate processing per M/P/K pathway
        #
        # Each pathway has its own center-surround with different σ:
        #   M (parasol): large RF (σ_c=1.5, σ_s=4.0) → coarse, fast
        #   P (midget):  small RF (σ_c=0.5, σ_s=1.5) → fine, slow
        #   K (bistrat): medium RF (σ_c=1.0, σ_s=3.0)
        # ================================================================
        R_a = x_adapted[:, 0:1]
        G_a = x_adapted[:, 1:2]
        B_a = x_adapted[:, 2:3]

        # --- M pathway bipolar cells (luminance = R+G) ---
        lum = (R_a + G_a) / 2.0
        center_M = F.conv2d(lum, self.k_center_M, padding=self.padding)
        surr_M = F.conv2d(lum, self.k_surround_M[:, :, :, :1].expand(1, 1, -1, -1)
                          if False else self.k_surround_M, padding=self.padding)
        dog_M = center_M - surr_M

        # --- P pathway bipolar cells (R and G separate, small RF) ---
        center_P_r = F.conv2d(R_a, self.k_center_P, padding=self.padding)
        surr_P_r = F.conv2d(R_a, self.k_surround_P, padding=self.padding)
        center_P_g = F.conv2d(G_a, self.k_center_P, padding=self.padding)
        surr_P_g = F.conv2d(G_a, self.k_surround_P, padding=self.padding)
        # R-G opponent: center red, surround green (and vice versa)
        dog_P = (center_P_r - surr_P_g)  # red center, green surround

        # --- K pathway bipolar cells (B vs R+G) ---
        center_K = F.conv2d(B_a, self.k_center_K, padding=self.padding)
        surr_K = F.conv2d(lum, self.k_surround_K, padding=self.padding)  # surround is yellow (R+G)
        dog_K = center_K - surr_K

        # General DoG visualization (use M pathway)
        layers["3a_dog"] = self._single_channel_vis(dog_M)

        # ON/OFF split (shown for M pathway)
        on_M = torch.relu(dog_M)
        off_M = torch.relu(-dog_M)
        on_vis = on_M[0, 0].cpu().numpy()
        on_vis = (on_vis / max(on_vis.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
        off_vis = off_M[0, 0].cpu().numpy()
        off_vis = (off_vis / max(off_vis.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
        on_off = np.zeros((*on_vis.shape, 3), dtype=np.uint8)
        on_off[:, :, 1] = on_vis
        on_off[:, :, 2] = off_vis
        layers["3b_on_off"] = on_off

        # ================================================================
        # Layer 4: AMACRINE CELLS
        # >30 types in mammalian retina. We model three key functions:
        #
        # 4a. Starburst amacrine cells → DIRECTION SELECTIVITY
        #     Detect motion direction at each pixel via optical flow
        #     Output: velocity vectors (dx, dy) per pixel
        #
        # 4b. Wide-field amacrine cells → OBJECT/BACKGROUND SEPARATION
        #     Maintain slow background model, extract foreground
        #     Global flow = camera motion, local deviations = object motion
        #
        # 4c. Transient amacrine cells → MULTI-TIMESCALE
        #     Fast buffer (flicker detection) vs slow buffer (gradual change)
        # ================================================================
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.prev_gray is not None:
            # --- 4a: Optical flow (starburst amacrine → direction selectivity) ---
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            # flow shape: (H, W, 2) → dx, dy per pixel
            dx, dy = flow[:, :, 0], flow[:, :, 1]
            speed = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)  # -pi to pi

            # Visualize as HSV: hue=direction, saturation=1, value=speed
            hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
            hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # hue
            hsv[:, :, 1] = 255  # saturation
            hsv[:, :, 2] = np.clip(speed * 20, 0, 255).astype(np.uint8)  # value=speed
            layers["4a_flow"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # --- 4b: Object/background separation (wide-field amacrine) ---
            if self.bg_model is None:
                self.bg_model = gray.copy()
            else:
                self.bg_model = 0.95 * self.bg_model + 0.05 * gray  # slow update

            # Foreground = differs significantly from background
            fg_mask = np.abs(gray - self.bg_model)
            fg_vis = np.clip(fg_mask * 3, 0, 255).astype(np.uint8)
            layers["4b_foreground"] = cv2.applyColorMap(fg_vis, cv2.COLORMAP_HOT)

            # --- 4c: Multi-timescale (transient amacrine) ---
            if self.fast_buffer is None:
                self.fast_buffer = gray.copy()

            fast_delta = np.abs(gray - self.fast_buffer)     # fast change (~2 frames)
            slow_delta = np.abs(gray - self.bg_model)        # slow change (~20 frames)
            self.fast_buffer = 0.5 * self.fast_buffer + 0.5 * gray  # fast update

            # Visualize: red=fast transient, green=slow change
            multi_vis = np.zeros((*gray.shape, 3), dtype=np.uint8)
            multi_vis[:, :, 2] = np.clip(fast_delta * 5, 0, 255).astype(np.uint8)  # red=fast
            multi_vis[:, :, 1] = np.clip(slow_delta * 3, 0, 255).astype(np.uint8)  # green=slow
            layers["4c_multiscale"] = multi_vis

        else:
            h_frame, w_frame = bgr_frame.shape[:2]
            layers["4a_flow"] = np.zeros_like(bgr_frame)
            layers["4b_foreground"] = np.zeros_like(bgr_frame)
            layers["4c_multiscale"] = np.zeros_like(bgr_frame)

        self.prev_gray = gray

        # ================================================================
        # Layer 5: GANGLION CELLS — M/P/K pathways with different properties
        #
        # Each pathway uses the DoG output from its own bipolar cells:
        #   M (parasol): dog_M, large RF → ON/OFF luminance
        #   P (midget):  dog_P, small RF → R-G opponent ON/OFF
        #   K (bistrat): dog_K, medium RF → B-Y opponent ON/OFF
        #
        # Each pathway will be compressed at a DIFFERENT ratio:
        #   M: stride=6 (low res, fast — for motion)
        #   P: stride=2 (high res, slow — for detail)
        #   K: stride=4 (medium)
        # ================================================================

        # 5a. Parasol / M: luminance ON/OFF
        lum_on = torch.relu(dog_M)
        lum_off = torch.relu(-dog_M)
        lon = lum_on[0, 0].cpu().numpy()
        loff = lum_off[0, 0].cpu().numpy()
        lon = (lon / max(lon.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
        loff = (loff / max(loff.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
        m_vis = np.zeros((*lon.shape, 3), dtype=np.uint8)
        m_vis[:, :, 1] = lon
        m_vis[:, :, 2] = loff
        layers["5a_parasol_M"] = m_vis

        # 5b. Midget / P: R-G opponent ON/OFF
        rg_on = torch.relu(dog_P)    # red > green
        rg_off = torch.relu(-dog_P)  # green > red
        layers["5b_midget_P"] = self._single_channel_vis(dog_P, cv2.COLORMAP_COOL)

        # 5c. Bistratified / K: B-Y opponent ON/OFF
        by_on = torch.relu(dog_K)    # blue > yellow
        by_off = torch.relu(-dog_K)  # yellow > blue
        layers["5c_bistrat_K"] = self._single_channel_vis(dog_K, cv2.COLORMAP_COOL)

        # ================================================================
        # Layer 5d: SPIKE ENCODING (ganglion cell digitization)
        # Ganglion cells convert continuous signals into discrete spikes.
        # The firing RATE encodes signal strength:
        #   strong signal → high spike probability per timestep
        #   weak signal   → low spike probability
        #   zero signal   → silent (no energy cost)
        #
        # We model each frame as one "timestep":
        #   spike = 1 if random() < signal_strength, else 0
        # Over many frames, the average spike count ≈ firing rate
        # ================================================================

        # Pack all ganglion outputs (use absolute values for spike probability)
        # M channel: use ON signal strength
        spike_input = lum_on[0, 0].cpu().numpy()
        # Zero out border to avoid padding artifacts
        b = self.padding
        spike_input[:b, :] = 0; spike_input[-b:, :] = 0
        spike_input[:, :b] = 0; spike_input[:, -b:] = 0
        # Normalize to [0, 1] probability range
        spike_prob = spike_input / max(spike_input.max(), 1e-6)

        # Generate spikes: Poisson-like process (Bernoulli per pixel per frame)
        spikes_this_frame = (np.random.random(spike_prob.shape) < spike_prob).astype(np.float32)

        # Accumulate for rate estimation
        self.spike_frame_count += 1
        if self.spike_accumulator is None or self.spike_accumulator.shape != spikes_this_frame.shape:
            self.spike_accumulator = spikes_this_frame.copy()
        else:
            # Exponential moving average of spike rate
            self.spike_accumulator = 0.9 * self.spike_accumulator + 0.1 * spikes_this_frame

        # Visualize instantaneous spikes (white dots on black = individual spikes)
        spike_vis = (spikes_this_frame * 255).astype(np.uint8)
        layers["5d_spikes"] = cv2.cvtColor(spike_vis, cv2.COLOR_GRAY2BGR)

        # Visualize accumulated rate (brighter = higher firing rate)
        rate_vis = (self.spike_accumulator / max(self.spike_accumulator.max(), 1e-6) * 255)
        rate_vis = rate_vis.clip(0, 255).astype(np.uint8)
        layers["5e_spike_rate"] = cv2.applyColorMap(rate_vis, cv2.COLORMAP_INFERNO)

        # Sparsity: what fraction of neurons are silent this frame
        spike_sparsity = 1.0 - spikes_this_frame.mean()
        layers["_spike_sparsity"] = spike_sparsity

        # ================================================================
        # Layer 6: OPTIC NERVE — M/P/K compressed SEPARATELY
        #
        # Each pathway is compressed at a different ratio,
        # matching biological receptive field sizes:
        #   M (parasol): stride=6, large RF → 36:1 per pathway
        #   P (midget):  stride=2, small RF →  4:1 per pathway
        #   K (bistrat): stride=4, medium   → 16:1 per pathway
        #
        # In the LGN, these arrive at different layers:
        #   M → layers 1-2 (magnocellular)
        #   P → layers 3-6 (parvocellular)
        #   K → interleaved (koniocellular)
        # ================================================================
        border = self.padding

        # M pathway: ON + OFF, compress aggressively
        m_nerve = torch.cat([lum_on, lum_off], dim=1)
        m_nerve = m_nerve[:, :, border:-border, border:-border]
        m_compressed = F.avg_pool2d(m_nerve, kernel_size=self.stride_M, stride=self.stride_M)

        # P pathway: R-G ON + OFF, compress minimally (preserve detail)
        p_nerve = torch.cat([rg_on, rg_off], dim=1)
        p_nerve = p_nerve[:, :, border:-border, border:-border]
        p_compressed = F.avg_pool2d(p_nerve, kernel_size=self.stride_P, stride=self.stride_P)

        # K pathway: B-Y ON + OFF, moderate compression
        k_nerve = torch.cat([by_on, by_off], dim=1)
        k_nerve = k_nerve[:, :, border:-border, border:-border]
        k_compressed = F.avg_pool2d(k_nerve, kernel_size=self.stride_K, stride=self.stride_K)

        oh, ow = m_nerve.shape[2], m_nerve.shape[3]

        # Visualize: show each pathway at its own resolution, upsampled for display
        m_up = F.interpolate(m_compressed, size=(oh, ow), mode="nearest")
        p_up = F.interpolate(p_compressed, size=(oh, ow), mode="nearest")
        k_up = F.interpolate(k_compressed, size=(oh, ow), mode="nearest")

        # 6a: M pathway compressed (luminance ON-OFF, coarse/blocky)
        comp_lum = m_up[:, 0:1] - m_up[:, 1:2]
        layers["6a_compressed"] = self._single_channel_vis(comp_lum)

        # Reconstruct from all three pathways
        rec_lum = m_up[:, 0:1] - m_up[:, 1:2]
        rec_rg = p_up[:, 0:1] - p_up[:, 1:2]
        rec_by = k_up[:, 0:1] - k_up[:, 1:2]

        base_lum = F.avg_pool2d(x_adapted.mean(dim=1, keepdim=True),
                                kernel_size=self.stride_M, stride=self.stride_M)
        base_lum = F.interpolate(base_lum, size=(oh, ow), mode="nearest")

        rec_r = base_lum + rec_lum + rec_rg
        rec_g = base_lum + rec_lum - rec_rg
        rec_b = base_lum + rec_lum + rec_by
        reconstructed = torch.cat([rec_r, rec_g, rec_b], dim=1)

        rmin, rmax = reconstructed.min(), reconstructed.max()
        if rmax - rmin > 1e-6:
            reconstructed = (reconstructed - rmin) / (rmax - rmin)
        layers["6b_reconstructed"] = self._tensor_to_bgr(reconstructed.clamp(0, 1))

        # Compression stats
        total_input = oh * ow * 3  # original pixels
        total_compressed = (m_compressed.numel() + p_compressed.numel() + k_compressed.numel())
        ratio = total_input / max(total_compressed, 1)
        layers["_compression_ratio"] = ratio
        layers["_m_size"] = tuple(m_compressed.shape[2:])
        layers["_p_size"] = tuple(p_compressed.shape[2:])
        layers["_k_size"] = tuple(k_compressed.shape[2:])

        # Also keep the pre-compression output for comparison
        # Final: the gain-controlled color image (what photoreceptors + horizontal cells produce)
        # This is the "analog" signal before ganglion cells encode it into M/P/K spikes
        layers["final"] = self._tensor_to_bgr(x_adapted.clamp(0, 1))

        return layers


def run_demo(args):
    retina = RetinaLayerVis()

    cap = cv2.VideoCapture(0 if args.input == "0" else args.input)
    if not cap.isOpened():
        print("Error:", args.input)
        return

    print("Press 'q' to quit, 's' to screenshot")
    print()
    print("Signal flow:")
    print("  Photoreceptor → Horizontal → Bipolar(ON/OFF) → Amacrine → Ganglion(M/P/K)")

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.input != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        h, w = frame.shape[:2]
        if max(h, w) > 480:
            scale = 480 / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            h, w = frame.shape[:2]

        t0 = time.time()
        L = retina.process(frame)
        fps = 1.0 / max(time.time() - t0, 1e-6)

        # 4-column layout
        pw = w // 4 * 4  # panel width divisible by 4
        ph = h
        def r(img):
            return cv2.resize(img, (pw, ph))
        def lbl(img, text, color=(255, 255, 255)):
            out = img.copy()
            cv2.putText(out, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
            cv2.putText(out, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            return out

        row1 = np.hstack([
            lbl(r(L["0_input"]),           "0 Input"),
            lbl(r(L["1_photoreceptor"]),    "1 Photoreceptor"),
            lbl(r(L["1b_cones"]),           "1b Cones R|G|B"),
            lbl(r(L["2a_gain_ctrl"]),       "2a Horizontal"),
        ])
        row2 = np.hstack([
            lbl(r(L["2b_gain_map"]),        "2b Gain map"),
            lbl(r(L["2c_surround"]),        "2c Surround"),
            lbl(r(L["3a_dog"]),             "3a Bipolar DoG"),
            lbl(r(L["3b_on_off"]),          "3b ON(g)+OFF(r)"),
        ])
        row3 = np.hstack([
            lbl(r(L["4a_flow"]),            "4a Flow (hue=dir)"),
            lbl(r(L["4b_foreground"]),      "4b Foreground"),
            lbl(r(L["4c_multiscale"]),      "4c Fast(R)/Slow(G)"),
            lbl(r(L["5a_parasol_M"]),       "5a Ganglion M"),
        ])
        row4 = np.hstack([
            lbl(r(L["5b_midget_P"]),        "5b Ganglion P", (200,200,255)),
            lbl(r(L["5c_bistrat_K"]),       "5c Ganglion K", (255,200,100)),
            lbl(r(L["5d_spikes"]),          "5d Spikes"),
            lbl(r(L["5e_spike_rate"]),      "5e Spike rate"),
        ])
        row5 = np.hstack([
            lbl(r(L["6a_compressed"]),      "6a Compressed"),
            lbl(r(L["6b_reconstructed"]),   "6b Reconstructed"),
            lbl(r(L["final"]),              "Retina output"),
            lbl(r(L["0_input"]),            "Original"),
        ])

        cr = L["_compression_ratio"]
        sp = L["_spike_sparsity"]
        ms = L["_m_size"]
        ps = L["_p_size"]
        ks_ = L["_k_size"]
        info = np.zeros((22, pw * 4, 3), dtype=np.uint8)
        cv2.putText(info,
            "FPS:{:.0f} | Total:{:.0f}:1 | M:{}x{}(s{}) P:{}x{}(s{}) K:{}x{}(s{}) | Silent:{:.0%}".format(
                fps, cr, ms[1], ms[0], retina.stride_M,
                ps[1], ps[0], retina.stride_P,
                ks_[1], ks_[0], retina.stride_K, sp),
            (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)

        combined = np.vstack([row1, row2, row3, row4, row5, info])
        cv2.imshow("Retina Pipeline", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"retina_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="0")
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()

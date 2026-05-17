# Retinal Constraints — Project Documentation

## What is this project

A biologically-inspired vision processing system that replicates the human retina's signal processing pipeline. Started from reading James V. Stone's "Vision and Brain: How We Perceive the World" (MIT Press, 2012) and evolved through experiments into a working real-time demo.

**Long-term goal**: build a practical image/video processing system inspired by retinal architecture, not just academic experiments.

**GitHub**: https://github.com/xxxss/retinal-constraints

## Project origin and knowledge base

This project was born from an LLM Wiki knowledge base at `~/wiki/` (following Karpathy's LLM Wiki pattern). Key wiki pages:

- `~/wiki/sources/vision-and-brain.md` — detailed chapter-by-chapter notes from Stone's book
- `~/wiki/wiki/analyses/retinal-dog-robustness.md` — full experimental results (6 rounds)
- `~/wiki/wiki/concepts/receptive-fields.md` — Mexican hat, center-surround theory
- `~/wiki/wiki/concepts/efficient-coding.md` — why compression works
- `~/wiki/wiki/concepts/bayesian-perception.md` — why priors matter

## Two parts of the project

### Part 1: Validation experiments (CIFAR-10 + ResNet18)

Proved that retinal preprocessing improves OOD robustness. Run on GPU servers.

### Part 2: Real-time demo (local Mac, webcam)

Visualizes the complete retinal pipeline in real-time. Run locally.

---

## Part 1: Experiments

### Files

| File | Purpose |
|------|---------|
| `model.py` | Retinal bottleneck CNN (Lindsey 2019 replication) |
| `train.py` | Train bottleneck models, sweep widths 1-32 |
| `train_sparse.py` | Train with L1 sparsity penalty |
| `visualize.py` | Visualize learned conv filters |
| `visualize_bottleneck.py` | Visualize effective receptive fields via gradient |
| `compare_all_rf.py` | Compare RFs across all bottleneck widths |
| `plot_sparsity.py` | Plot sparsity vs accuracy trade-off |
| `dog_layer.py` | FixedDoG and LearnableDoG modules (PyTorch) |
| `dog_layer_adaptive.py` | AdaptiveDoG with local-variance noise estimation |
| `eval_cifar.py` | OOD robustness eval: baseline / fixed_dog / learned_dog |
| `eval_cifar_v2.py` | Adds adaptive_dog mode |
| `eval_ablation.py` | Ablation: baseline vs noise_aug vs adaptive_dog |
| `eval_ablation_v3.py` | Adds auxiliary supervision for noise estimator |
| `eval_ablation_v4.py` | Adds L1 sparsity on top of best model |
| `eval_retina_layers.py` | Tests log compression layer contribution |
| `eval_robustness.py` | ImageNet-scale eval (needs ImageNet data) |

### Key experimental results

**Experiment 1 — Bottleneck (Lindsey 2019 replication)**:
```
Bottleneck width 2 → network spontaneously learns center-surround (Mexican hat)
receptive fields, identical to biological retinal ganglion cells.
No bottleneck (width 32) → no such structure emerges.
```

**Experiment 2 — Sparsity penalty**: L1 penalty on ReLU activations ineffective (47% → 47.5% silent). ReLU inherently provides ~50% sparsity; pushing further requires architectural changes (SNN).

**Experiment 3-5 — OOD Robustness (final result)**:
```
Mode                      |  Clean  | Corrupted |  Drop
baseline                  |  83.7%  |   67.5%   |  16.2%
baseline + noise_aug      |  83.2%  |   68.7%   |  14.6%
+ adaptive DoG            |  83.2%  |   70.5%   |  12.7%
+ aux supervision         |  83.4%  |   71.4%   |  12.0%  ← best

Three components independently contribute:
  Noise augmentation: +1.2%
  DoG structure:      +1.8%
  Aux supervision:    +0.9%
  Total:              +3.9% (26% relative improvement in robustness)
```

**Experiment 6 — Log compression**: Improves blur/contrast robustness (+4.3%/+4.7%) but worsens impulse noise (-8.3%). Net effect neutral. Needs horizontal cell gain control to compensate.

**Key finding**: Noise estimator never learned to discriminate clean vs noisy at CIFAR-10 resolution (32x32). May work at higher resolution (ImageNet 224x224) — untested.

### How to run experiments

```bash
# On local Mac (MPS GPU)
cd ~/projects/retinal-constraints
uv run python train.py --all --epochs 20          # bottleneck sweep
uv run python compare_all_rf.py                    # visualize RFs

# On GPU server (needs CUDA 13+, see setup_server.sh)
git clone https://github.com/xxxss/retinal-constraints.git
cd retinal-constraints
pip install torch torchvision timm matplotlib       # use pip, not uv (faster)
python eval_cifar_v2.py --epochs 50                  # robustness eval
python eval_ablation_v3.py                           # ablation study
```

### Saved results

All in `results/` directory:
- `bottleneck_*_model.pt` — trained bottleneck models (widths 1-32)
- `bn2_sp*_model.pt` — sparsity experiment models
- `cifar_*_model.pt` — robustness experiment models
- `*.json` — training histories
- `*.png` — visualization plots
- `results_ablation*.txt` — GPU server experiment logs

---

## Part 2: Real-time demo

### Files

| File | Purpose |
|------|---------|
| `demo_layers.py` | **Main demo** — full retinal pipeline visualization (webcam) |
| `demo_video.py` | Side-by-side: original vs retina + YOLO detection (OpenCV backend) |
| `demo_video_gpu.py` | Same but with GPU AdaptiveDoG |
| `retina_preprocessor.py` | OpenCV-based retina module (fast, CPU) |
| `retina_preprocessor_gpu.py` | PyTorch-based retina module (GPU, full pipeline) |

### demo_layers.py — the main demo

```bash
uv run python demo_layers.py              # webcam
uv run python demo_layers.py --input v.mp4  # video file
```

Displays a 4-column × 5-row grid showing every layer of retinal processing:

```
┌────────────┬────────────┬────────────┬────────────┐
│ 0 Input    │ 1 Photo-   │ 1b Cones   │ 2a Horiz   │
│            │ receptor   │  R|G|B     │ gain ctrl  │
├────────────┼────────────┼────────────┼────────────┤
│ 2b Gain    │ 2c Surround│ 3a Bipolar │ 3b ON+OFF  │
│ map        │            │ DoG        │ (grn/red)  │
├────────────┼────────────┼────────────┼────────────┤
│ 4a Flow    │ 4b Fore-   │ 4c Fast/   │ 5a Gang-   │
│            │ ground     │ Slow       │ lion M     │
├────────────┼────────────┼────────────┼────────────┤
│ 5b Gang P  │ 5c Gang K  │ 5d Spikes  │ 5e Spike   │
│ R-G        │ B-Y        │            │ rate       │
├────────────┼────────────┼────────────┼────────────┤
│ 6a Comp-   │ 6b Recon-  │ Retina     │ Original   │
│ ressed     │ structed   │ output     │            │
├────────────┴────────────┴────────────┴────────────┤
│ FPS | M:WxH(s6) P:WxH(s2) K:WxH(s4) | Silent:%   │
└───────────────────────────────────────────────────┘
```

Keys: `q` = quit, `s` = screenshot

### Retinal pipeline architecture (biologically accurate)

```
Signal flow in demo_layers.py:

① PHOTORECEPTORS (line ~130)
   Input RGB → log(1 + 10x) / log(11)
   - Compresses dynamic range (moonlight to sunlight)
   - Each cone type (R, G, B) processes independently

② HORIZONTAL CELLS (line ~165)
   Two functions:
   a) Gain control: x / local_mean(x)
      - Per-pixel brightness normalization
      - Shadows boosted, highlights suppressed
      - Implemented via F.avg_pool2d (15x15 window)
   b) Surround signal → passed to bipolar cells
      - Gaussian blur of adapted signal
      - This provides the "surround" in center-surround

③ BIPOLAR CELLS (line ~210)
   center - surround = DoG (Difference of Gaussians)
   **This is WHERE center-surround actually happens**
   
   Three pathways with DIFFERENT receptive field sizes:
     M (parasol):  σ_center=1.5, σ_surround=4.0 (large, coarse)
     P (midget):   σ_center=0.5, σ_surround=1.5 (small, fine)
     K (bistrat):  σ_center=1.0, σ_surround=3.0 (medium)
   
   Each splits into ON/OFF:
     ON  = relu(center - surround)  → brightening
     OFF = relu(surround - center)  → darkening

④ AMACRINE CELLS (line ~270)
   Three types:
   a) Starburst → optical flow (cv2.calcOpticalFlowFarneback)
      Visualized as HSV: hue=direction, brightness=speed
   b) Wide-field → foreground/background separation
      Background model: 0.95 * bg + 0.05 * current (slow update)
      Foreground = |current - background|
   c) Transient → multi-timescale
      Fast buffer (0.5 decay): detects flicker
      Slow buffer (background): detects gradual change
      Visualized: red=fast, green=slow

⑤ GANGLION CELLS (line ~330)
   Three types outputting to optic nerve:
     Parasol (M): luminance ON/OFF from dog_M
     Midget (P):  R-G opponent ON/OFF from dog_P
     Bistratified (K): B-Y opponent ON/OFF from dog_K
   
   + Spike encoding: continuous → probabilistic binary spikes
     spike = 1 if random() < signal_strength else 0
     ~85-95% of neurons silent per frame (energy efficiency)

⑥ OPTIC NERVE (line ~400)
   M/P/K pathways compressed SEPARATELY at different ratios:
     M: stride=6 (36:1, coarse — for motion)
     P: stride=2 (4:1, fine — for detail)
     K: stride=4 (16:1, medium)
   
   Reconstruction from compressed signals requires adding back
   base luminance (DC component lost by DoG).
```

### Key biological insights embedded in the code

1. **DoG is NOT a separate step** — it emerges from horizontal cells providing surround to bipolar cells
2. **Different pathways have different resolutions** — M is coarse/fast, P is fine/slow
3. **Horizontal cell gain control uses avg_pool** — because gap junctions spread signals laterally with distance decay ≈ Gaussian
4. **Border artifacts** from zero-padding are cropped (line ~400) — DoG + zero padding = false edges
5. **Spike encoding is sparse** — matches biological data (2-4% neurons active at any time)
6. **Log compression before DoG** — photoreceptors compress first, then ganglion cells encode edges

---

## What's been explored but NOT yet done

### Tried and found ineffective
- L1 sparsity penalty on ReLU networks (needs SNN for real sparsity)
- Learnable DoG without constraints (overfits, σ shrinks too small)
- Noise estimator at 32x32 resolution (can't discriminate clean/noisy)

### Not yet implemented
- **Local gain control (horizontal cells) as trainable preprocessing layer** for ConvNeXt/RepViT
- **ON/OFF pathway split fed to backbone network** (currently visualization only)
- **M/P/K channels as actual model input** instead of RGB
- **Attention-driven foveation** (coarse-to-fine: YOLO detects → crop high-res)
- **ImageNet-scale validation** of robustness improvements
- **Triton/CUDA kernel fusion** for speed optimization
- **Video temporal sparsity** — only process changed regions (DeltaCNN-style)

### Open research questions
1. Does the noise estimator work at ImageNet resolution (224x224)?
2. Does M/P/K separation improve downstream task performance vs RGB?
3. Can horizontal cell gain control replace traditional data augmentation for robustness?
4. What happens if you train a backbone network directly on M/P/K output instead of RGB?

---

## Dependencies

```bash
# Local Mac development
uv sync  # installs from pyproject.toml

# Key packages: torch, torchvision, timm, matplotlib, opencv-python, ultralytics (YOLOv8)

# GPU server (use pip, faster than uv)
pip install torch torchvision timm matplotlib
```

## Related materials

- **Knowledge base wiki**: `~/wiki/` (Obsidian-compatible markdown)
- **Source book**: `~/wiki/raw/Vision and Brain How We Perceive the World (James V. Stone).pdf`
- **Original Lindsey 2019 code**: https://github.com/ganguli-lab/RetinalResources
- **Paper references**: PNAS 2025 (DoG+ViT), NatComm 2024 (photoreceptor adaptation), DeltaCNN (CVPR 2022), Eventful Transformers (ICCV 2023)

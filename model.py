"""
Retinal Bottleneck Model — PyTorch reimplementation of Lindsey et al. 2019
"A Unified Theory of Early Visual Representations from Retina to Cortex"

Core idea: a CNN with a narrow bottleneck between "retina" and "cortex" layers.
When bottleneck is narrow (few channels), the retina layers spontaneously learn
center-surround (Mexican hat) receptive fields — just like biological retinas.

Architecture:
    Input(32x32 grayscale)
    → Retina: Conv layers compressing to K output channels (the bottleneck)
    → Cortex (VVS): Conv layers with wider channels
    → Dense classifier → 10 classes (CIFAR-10)
"""

import torch
import torch.nn as nn


class RetinalBottleneckModel(nn.Module):
    """
    Args:
        retina_out_channels: bottleneck width (key variable!)
            - 32 = no bottleneck (control)
            - 2  = strong bottleneck → should learn center-surround RFs
            - 1  = extreme bottleneck
        retina_hidden_channels: width of retina hidden layers
        retina_layers: number of conv layers in retina
        vvs_channels: width of cortex (VVS) layers
        vvs_layers: number of conv layers in cortex
        filter_size: conv kernel size
        num_classes: output classes
    """

    def __init__(
        self,
        retina_out_channels: int = 2,
        retina_hidden_channels: int = 32,
        retina_layers: int = 2,
        vvs_channels: int = 32,
        vvs_layers: int = 2,
        filter_size: int = 9,
        num_classes: int = 10,
    ):
        super().__init__()
        self.retina_out_channels = retina_out_channels
        padding = filter_size // 2  # 'same' padding

        # === Retina layers ===
        retina = []
        in_ch = 1  # grayscale input
        for i in range(retina_layers):
            is_last = (i == retina_layers - 1)
            out_ch = retina_out_channels if is_last else retina_hidden_channels
            retina.append(
                nn.Conv2d(in_ch, out_ch, filter_size, padding=padding)
            )
            retina.append(nn.ReLU())
            in_ch = out_ch
        self.retina = nn.Sequential(*retina)

        # === Cortex (VVS) layers ===
        vvs = []
        in_ch = retina_out_channels
        for i in range(vvs_layers):
            vvs.append(
                nn.Conv2d(in_ch, vvs_channels, filter_size, padding=padding)
            )
            vvs.append(nn.ReLU())
            in_ch = vvs_channels
        self.vvs = nn.Sequential(*vvs)

        # === Classifier head ===
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vvs_channels * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        retina_out = self.retina(x)
        vvs_out = self.vvs(retina_out)
        return self.classifier(vvs_out)

    def get_retina_output(self, x):
        """Get bottleneck activations for visualization."""
        return self.retina(x)

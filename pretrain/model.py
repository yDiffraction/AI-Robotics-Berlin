"""FPGA-exportable spiking conv net for GenX320 frames on the KV260 (riverops).

Constraints from documentation/DEPLOY.md and the riverops exporter:
  * input  [B, T, 2, 80, 80]  (despot: downsampling=4, 100 Hz -> 10 ms frames)
  * only spiking conv layers (no Linear, no ReduceMean) -> last layer is a
    conv whose output is [B, T, num_classes, 1, 1], read out as spike counts
  * square kernels/strides, no dilation/groups
Strided SpikingConvBN2d (instead of the AvgPool variants) keeps the
fusion-aware QAT path (prepare_qat(..., fused_bn=True)) available.
"""
from __future__ import annotations

import torch
from torch import nn
from riverops.nn import SpikingConv2d, SpikingConvBN2d

INPUT_HW = 80


class ThrowSNN(nn.Module):
    def __init__(self, num_classes: int = 11, widths=(16, 32, 64, 64), tau: float = 2.0):
        super().__init__()
        c1, c2, c3, c4 = widths
        kw = dict(initial_tau=tau, bias=False)  # riverops 0.2.0: ATan surrogate is built in
        self.backbone = nn.Sequential(
            SpikingConvBN2d(2, c1, kernel_size=5, stride=2, padding=2, first_layer=True, **kw),  # 80 -> 40
            SpikingConvBN2d(c1, c2, kernel_size=3, stride=2, padding=1, **kw),                   # 40 -> 20
            SpikingConvBN2d(c2, c3, kernel_size=3, stride=2, padding=1, **kw),                   # 20 -> 10
            SpikingConvBN2d(c3, c4, kernel_size=3, stride=2, padding=1, **kw),                   # 10 -> 5
        )
        # Classification head: one 5x5 conv -> [B, T, K, 1, 1]. Swap this for fine-tuning.
        self.head = SpikingConv2d(c4, num_classes, kernel_size=5, initial_tau=tau, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 2, 80, 80] -> output spikes [B, T, K]."""
        return self.head(self.backbone(x)).flatten(2)

    def export_module(self) -> nn.Sequential:
        """Pure conv stack (what the FPGA runs); readout/flatten stays on the host."""
        return nn.Sequential(*self.backbone, self.head)

    def reset_head(self, num_classes: int) -> None:
        old = self.head
        self.head = SpikingConv2d(old.conv.in_channels, num_classes, kernel_size=5,
                                  initial_tau=2.0, bias=True).to(next(self.parameters()).device)


def load_pretrained(path: str, num_classes: int = 2, device: str = "cpu",
                    freeze_first: int = 0) -> ThrowSNN:
    """Load the DVS-Gesture-pretrained backbone and attach a fresh `num_classes` head.

    freeze_first: number of backbone layers to freeze (0 = fine-tune everything;
    1-2 is a good choice with only a few dozen own clips).
    """
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = ThrowSNN(cfg["num_classes"], tuple(cfg["widths"]))
    model.load_state_dict(ck["model"])
    model.to(device).reset_head(num_classes)
    for layer in list(model.backbone)[:freeze_first]:
        for p in layer.parameters():
            p.requires_grad_(False)
    return model


def rate_readout(spikes: torch.Tensor) -> torch.Tensor:
    """Output spikes [B, T, K] -> firing rate per class [B, K] (argmax = prediction)."""
    return spikes.mean(1)

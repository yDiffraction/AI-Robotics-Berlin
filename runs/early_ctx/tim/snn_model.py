"""(copy of pretrain/model.py with an input-size argument)
FPGA-exportable spiking conv net for GenX320 frames on the KV260 (riverops).

Constraints from documentation/DEPLOY.md and the riverops exporter:
  * input  [B, T, 2, 80, 80]  (despot: downsampling=4, 100 Hz -> 10 ms frames)
  * only spiking conv layers (no Linear, no ReduceMean) -> last layer is a
    conv whose output is [B, T, num_classes, 1, 1], read out as spike counts
  * square kernels/strides, no dilation/groups
Strided SpikingConvBN2d (instead of the AvgPool variants) keeps the
fusion-aware QAT path (prepare_qat(..., fused_bn=True)) available.

The backbone is a list of (out_channels, kernel, stride) layers ("arch").
"""
from __future__ import annotations

import torch
from torch import nn
from riverops.nn import SpikingConv2d, SpikingConvBN2d

INPUT_HW = 80

# v1 (first pretraining run): ~5.9 M MAC / frame
ARCH_V1 = [(16, 5, 2), (32, 3, 2), (64, 3, 2), (64, 3, 2)]

# Shape-sweep candidates, all <= ~15x v1 compute (see arch_cost)
ARCHS = {
    "v1":    ARCH_V1,
    # same topology, ~4x wider
    "wide":  [(64, 5, 2), (128, 3, 2), (256, 3, 2), (256, 3, 2)],
    # keep 80x80 resolution in layer 1 (small ball), deeper
    "hires": [(16, 3, 1), (32, 3, 2), (48, 3, 1), (96, 3, 2), (128, 3, 2), (128, 3, 1), (128, 3, 2)],
    # cheap stride-2 stem, depth at 40x40 and 10x10
    "mid":   [(32, 5, 2), (64, 3, 1), (96, 3, 2), (128, 3, 2), (128, 3, 1), (128, 3, 2)],
    # few-data option: ~6k weights, trained from scratch
    "tiny":  [(8, 5, 2), (16, 3, 2), (16, 3, 2), (16, 3, 2)],
}


def arch_from_config(cfg: dict) -> list[tuple[int, int, int]]:
    if "arch" in cfg:
        return [tuple(a) for a in cfg["arch"]]
    c1, c2, c3, c4 = cfg["widths"]  # v1 checkpoints
    return [(c1, 5, 2), (c2, 3, 2), (c3, 3, 2), (c4, 3, 2)]


def arch_cost(arch, num_classes: int = 11, in_ch: int = 2, hw: int = INPUT_HW) -> dict:
    """Dense multiply-accumulates per 10 ms frame, neuron count and parameter count."""
    macs = neurons = params = 0
    c = in_ch
    for out, k, s in arch:
        hw = (hw + 2 * (k // 2) - k) // s + 1
        macs += hw * hw * out * c * k * k
        neurons += hw * hw * out
        params += out * c * k * k
        c = out
    macs += num_classes * c * hw * hw
    params += num_classes * c * hw * hw + num_classes
    return {"MMAC_per_frame": macs / 1e6, "neurons": neurons + num_classes, "params": params,
            "final_hw": hw}


class ThrowSNN(nn.Module):
    def __init__(self, num_classes: int = 11, widths=None, tau: float = 2.0, arch=None, hw: int = INPUT_HW):
        super().__init__()
        if arch is None:
            arch = ARCH_V1 if widths is None else arch_from_config({"widths": widths})
        self.arch = [tuple(a) for a in arch]
        kw = dict(initial_tau=tau, bias=False)  # riverops 0.2.0: ATan surrogate is built in
        layers, c = [], 2
        for i, (out, k, s) in enumerate(self.arch):
            layers.append(SpikingConvBN2d(c, out, kernel_size=k, stride=s, padding=k // 2,
                                          first_layer=(i == 0), **kw))
            c, hw = out, (hw + 2 * (k // 2) - k) // s + 1
        self.backbone = nn.Sequential(*layers)
        self.final_hw, self.final_ch = hw, c
        # Classification head: one conv covering the whole final map -> [B, T, K, 1, 1].
        self.head = SpikingConv2d(c, num_classes, kernel_size=hw, initial_tau=tau, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 2, 80, 80] -> output spikes [B, T, K]."""
        return self.head(self.backbone(x)).flatten(2)

    def export_module(self) -> nn.Sequential:
        """Pure conv stack (what the FPGA runs); readout/flatten stays on the host."""
        return nn.Sequential(*self.backbone, self.head)

    def reset_head(self, num_classes: int) -> None:
        self.head = SpikingConv2d(self.final_ch, num_classes, kernel_size=self.final_hw,
                                  initial_tau=2.0, bias=True).to(next(self.parameters()).device)


def load_pretrained(path: str, num_classes: int = 2, device: str = "cpu",
                    freeze_first: int = 0) -> ThrowSNN:
    """Load a DVS-Gesture-pretrained backbone and attach a fresh `num_classes` head.

    freeze_first: number of backbone layers to freeze (0 = fine-tune everything;
    1-2 is a good choice with only a few dozen own clips).
    """
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = ThrowSNN(cfg["num_classes"], arch=arch_from_config(cfg))
    model.load_state_dict(ck["model"])
    model.to(device).reset_head(num_classes)
    for layer in list(model.backbone)[:freeze_first]:
        for p in layer.parameters():
            p.requires_grad_(False)
    return model


def rate_readout(spikes: torch.Tensor) -> torch.Tensor:
    """Output spikes [B, T, K] -> firing rate per class [B, K] (argmax = prediction)."""
    return spikes.mean(1)


if __name__ == "__main__":
    base = arch_cost(ARCH_V1)["MMAC_per_frame"]
    for name, a in ARCHS.items():
        c = arch_cost(a)
        print(f"{name:6s} {c['MMAC_per_frame']:6.1f} MMAC/frame ({c['MMAC_per_frame'] / base:4.1f}x)  "
              f"neurons {c['neurons']:7d}  params {c['params']:8d}  final {c['final_hw']}x{c['final_hw']}")

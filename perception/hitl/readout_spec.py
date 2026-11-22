"""Host readout rules of the board models, read from a model folder's readout.json.

Two rules exist:
  ema   (v3, perception/model_v3): y = 0.85 * y + 0.15 * spike[0]; alarm when y reaches theta (>=)
  diff  (throwdet, e.g. perception/model_d_neg11): y = (1 - alpha) * y + alpha * (spike[1] - spike[0]);
        alarm when y > threshold for hold_frames consecutive frames
Both: then refractory_ms (1 s) without alarms. Written as one linear filter over a signed sum of output channels,
so the same spec drives the live alarm (Live), the differentiable training readout (filter_torch) and scoring.
"""
import json
from pathlib import Path


class Spec:
    def __init__(self, model_dir):
        r = json.loads((Path(model_dir) / "readout.json").read_text())
        self.raw = r
        if "spike[1]" in str(r.get("score", "")):
            a = float(r["alpha"])
            self.kind, self.signs, self.decay, self.gain = "diff", {0: -1.0, 1: 1.0}, 1.0 - a, a
            self.theta, self.hold, self.strict = float(r["threshold"]), int(r.get("hold_frames", 1)), True
            self.theta_range, self.y_pos, self.y_neg = (-0.6, 0.6), 1.0, -1.0
            self.pos_channel = 1
        else:
            ch = int(r.get("output_channel", 0))
            self.kind, self.signs, self.decay, self.gain = "ema", {ch: 1.0}, 0.85, 0.15
            self.theta, self.hold, self.strict = float(r["theta"]), 1, False
            self.theta_range, self.y_pos, self.y_neg = (0.3, 0.99), 1.0, 0.0
            self.pos_channel = ch
        self.refr = int(r.get("refractory_ms", 1000)) // 10
        self.n_used = max(self.signs) + 1          # output channels that matter

    def with_theta(self, theta):
        """readout.json for an export with a new theta."""
        r = dict(self.raw)
        r["threshold" if self.kind == "diff" else "theta"] = float(theta)
        return r

    def signal(self, out):
        """One frame: sum of sign * spike over the used channels (out: sequence of channel outputs)."""
        return sum(s * float(out[c] > 0) for c, s in self.signs.items())

    def above(self, y, theta):
        return y > theta if self.strict else y >= theta

    def alarms(self, spikes, theta=None):
        """spikes [T, K] (or [T] for single-channel models) -> alarm frame indices."""
        th = self.theta if theta is None else theta
        live = Live(self, th)
        return [t for t in range(len(spikes)) if live.step(t, spikes[t])[1]]

    def filter_torch(self, sp, L=60):
        """Differentiable readout: sp [B, T, K] spikes -> y [B, T]."""
        import torch
        import torch.nn.functional as F
        x = sum(s * sp[..., c] for c, s in self.signs.items())
        k = self.gain * self.decay ** torch.arange(L, device=sp.device, dtype=sp.dtype)
        return F.conv1d(F.pad(x[:, None], (L - 1, 0)), k.flip(0).view(1, 1, L))[:, 0]


class Live:
    """Per-frame readout state (the board's host side)."""

    def __init__(self, spec, theta=None):
        self.spec = spec
        self.theta = spec.theta if theta is None else theta
        self.y, self.run, self.since = 0.0, 0, 10 ** 9

    def step(self, k, out):
        sp = self.spec
        o = out if hasattr(out, "__len__") else [out]
        self.y = sp.decay * self.y + sp.gain * sp.signal(o)
        self.run = self.run + 1 if sp.above(self.y, self.theta) else 0
        self.since += 1
        if self.run == sp.hold and self.since >= sp.refr:
            self.since = 0
            return self.y, True
        return self.y, False

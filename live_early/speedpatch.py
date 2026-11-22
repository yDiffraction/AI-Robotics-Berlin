"""Optional speed patch: riverops 0.2.0 lif_step does `if tau <= 0.0` on a tensor every
timestep, forcing a GPU sync. Same math, no check (tau = 1/sigmoid(.) > 1 always)."""
import riverops.nn.neuron as _neuron
from riverops.nn.functional import _as_tensor_like
from riverops.nn.quantization import fake_quantize_potentials, fake_quantize_threshold
from riverops.nn.surrogate import ATan

_SPIKE = ATan()


def _fast_single_step(self, x, potentials):
    if self.quant is not None:  # keep the reference path for QAT / fixed point
        return _orig(self, x, potentials)
    tau = self.tau
    v = potentials * (1.0 - 1.0 / tau) + x
    spikes = _SPIKE(v - self.v_threshold)
    if self.reset_mode == "SOFT":
        nv = v - spikes * self.v_threshold
    else:
        nv = (1.0 - spikes) * v
    return spikes, nv


_orig = _neuron.PLIFNode._single_step
_neuron.PLIFNode._single_step = _fast_single_step

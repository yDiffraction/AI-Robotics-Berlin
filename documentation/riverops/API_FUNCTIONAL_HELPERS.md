Functional helpers

Import these functions from riverops.nn.functional.
Manage state one timestep at a time

if_step and lif_step return (spikes, next_potentials). Initialize potentials to zero and pass each returned potential tensor into the next step:

import torch
from riverops.nn import functional

currents = torch.rand(2, 4, 3, 8, 8)  # [B, T, C, H, W]
potentials = torch.zeros_like(currents[:, 0])
spikes_over_time = []
for current_t in currents.unbind(dim=1):
    spikes_t, potentials = functional.lif_step(current_t, potentials, tau=2.0)
    spikes_over_time.append(spikes_t)

spikes = torch.stack(spikes_over_time, dim=1)

riverops.nn.functional.if_step(input_current, potentials, *, threshold=1.0, reset_mode='HARD', surrogate_function=None, detach_reset=False, quant=None)

riverops.nn.functional.lif_step(input_current, potentials, *, tau=2.0, threshold=1.0, reset_mode='HARD', surrogate_function=None, detach_reset=False, quant=None)

Apply a stateless PyTorch module to a sequence

seq_to_ann_forward combines the batch and time dimensions, applies the module, then restores those two dimensions. Use it for operations without state that depends on timestep order.

riverops.nn.functional.seq_to_ann_forward(x_seq, stateless_module)

    Apply stateless modules on [B, T, …] by flattening batch and time.


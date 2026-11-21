Getting started

riverops works with PyTorch models, tensors, and optimizers. This guide assumes you have used a PyTorch layer before; no previous spiking-network experience is needed to run the example.
Install

riverops requires Python 3.10 or later. Download the riverops wheel supplied with your release, then install it into your project:

uv add ./riverops-0.1.0-py3-none-any.whl

Or install it into an existing Python environment:

python -m pip install ./riverops-0.1.0-py3-none-any.whl

The required Python dependencies are installed automatically.

If you are working from the repository, run uv sync from its root instead. The repository examples and API reference describe that checkout; use the matching library version when following them.
Understand the input shape

riverops convolutional layers take a five-dimensional tensor:

[batch, time, channels, height, width]

For example, [4, 8, 3, 32, 32] holds four sequences, each with eight timesteps and three channels of 32x32 values. Keep the time dimension even for a single timestep: [4, 1, 3, 32, 32].

Your data loader should provide sequences in this order. riverops does not automatically add a time dimension or encode ordinary images as spike trains.
Run your first layer

A SpikingConvBN2d combines a convolution, batch normalization, and a spiking neuron. The neuron accumulates input over time and emits a spike when its membrane potential reaches its threshold.

import torch
from riverops.nn import SpikingConvBN2d, surrogate

layer = SpikingConvBN2d(
    3,
    16,
    kernel_size=3,
    padding=1,
    bias=False,
    first_layer=True,
    initial_tau=2.0,
    v_threshold=1.0,
    surrogate_function=surrogate.ATan(alpha=2.0),
)

x = torch.randn(4, 8, 3, 32, 32)
spikes = layer(x)
print(spikes.shape)  # torch.Size([4, 8, 16, 32, 32])

The output keeps the batch and time dimensions. Each output value is a spike, represented as 0 or 1 in a floating-point tensor.

first_layer=True marks the input-facing convolution, which may receive non-spike values. Later convolutions normally receive the preceding layer’s spikes.

Pass a complete sequence in one call. The neuron starts with zero membrane potentials each time you call the layer; it carries them between timesteps within that call.

Continue with Build a model to turn these spikes into predictions.

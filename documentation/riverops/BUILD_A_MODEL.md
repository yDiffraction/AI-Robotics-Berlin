Build a model

Combine riverops layers with torch.nn.Module and torch.nn.Sequential just as you would in a PyTorch model.
A small classifier

This model uses two spiking convolutional layers. It averages their output over the image dimensions, sums over time, and passes the result to a regular PyTorch classifier.

import torch
from torch import nn
from riverops.nn import SpikingConvBN2d, surrogate


class TinySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            SpikingConvBN2d(
                3, 16, kernel_size=3, padding=1, bias=False,
                first_layer=True,
                surrogate_function=surrogate.ATan(alpha=2.0),
            ),
            SpikingConvBN2d(
                16, 32, kernel_size=3, padding=1, bias=False,
                surrogate_function=surrogate.ATan(alpha=2.0),
            ),
        )
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        spikes = self.features(x)         # [B, T, 32, H, W]
        counts = spikes.mean(dim=(3, 4))   # [B, T, 32]
        return self.head(counts.sum(1))   # [B, 10]


model = TinySNN()
x = torch.randn(4, 8, 3, 32, 32)
logits = model(x)
print(logits.shape)  # torch.Size([4, 10])

The ten outputs per sequence are class scores. Use these scores with torch.nn.CrossEntropyLoss; a softmax is not needed before that loss.
Choose your layers

Need
	

Use

Convolution followed by a neuron
	

SpikingConv2d

Add batch normalization during training
	

SpikingConvBN2d

Add dropout during training
	

SpikingConvBNDropout2d

Pool before the neuron emits spikes
	

A SpikingConvAvgPool2d variant

Downsample an existing sequence
	

AvgPool2d

Upsample an existing sequence
	

NearestUpsample

Reuse earlier output through feedback
	

RecurrentSpikingConv2d

Use the riverops versions of convolution, normalization, pooling, and upsampling when operating directly on [B, T, C, H, W] tensors. The classifier above uses torch.nn.Linear after reducing the sequence to an ordinary [B, C] tensor.

The API reference lists the available combinations and their arguments.
Choose neuron behavior

The convolutional modules use PLIFNode, a leaky integrate-and-fire neuron. Its leak is learnable by default. Start with the defaults, or set the following arguments on your convolutional module:

Argument
	

Effect

initial_tau=2.0
	

Sets the initial membrane time constant.

learnable_leak=False
	

Keeps the leak fixed during training.

v_threshold=1.0
	

Sets the membrane potential needed to emit a spike.

reset_mode="HARD"
	

Resets the membrane potential to zero after a spike.

reset_mode="SOFT"
	

Subtracts the threshold after a spike.

In learnable mode, initial_tau must be greater than one. With a fixed leak, it may be any positive value. For models trained with a fixed leak, keep learnable_leak=False when recreating the model.

Use IFNode if you want a standalone integrate-and-fire neuron without leak.
Add recurrence

A recurrent convolution adds spikes from an input path and a feedback path. The feedback path receives the previous timestep’s combined spikes.

from riverops.nn import RecurrentSpikingConv2d

recurrent = RecurrentSpikingConv2d(16, 16, kernel_size=3, padding=1)
input_spikes = torch.randint(0, 2, (4, 8, 16, 16, 16)).float()
output_spikes = recurrent(input_spikes)
print(output_spikes.shape)  # torch.Size([4, 8, 16, 16, 16])

Because the two paths are added, a combined output may be 2 as well as 0 or 1.
Sequence state

IFNode, PLIFNode, and RecurrentSpikingConv2d start every forward call with zero membrane potentials. Recurrent feedback also starts at zero. State is carried only between timesteps within that call, so independent sequences do not need a manual reset.

Calling a model once per timestep therefore does not preserve its sequence state. If you need to manage state yourself, use the single-step functions in the functional API.

Continue with Training and evaluation.

Fusion

Fusion combines supported layers into a simpler form for evaluation. For example, it folds batch normalization into the convolution that precedes it.

Train the original model first. Use its evaluation mode when preparing a fused copy so the fusion uses BatchNorm’s learned running statistics.
Fuse a model

Continue with the model from Build a model:

import torch
from riverops.nn import fuse_model

model.eval()
fused_model = fuse_model(model).eval()

with torch.no_grad():
    original_output = model(x)
    fused_output = fused_model(x)

print((original_output - fused_output).abs().max())

fuse_model returns a copy and recursively fuses supported riverops child modules. Your original model is preserved.

Compare outputs and your task metrics before using the fused version. Small floating-point differences can affect spikes near a neuron’s threshold.
Fuse one module

For a standalone BatchNorm or pooling variant, call its .fuse() method directly. Plain SpikingConv2d needs no fusion:

from riverops.nn import SpikingConvBN2d

layer = SpikingConvBN2d(3, 16, kernel_size=3, padding=1, first_layer=True)
layer.eval()
fused_layer = layer.fuse()

Use fuse_model for a model containing such modules; it traverses the model’s children rather than replacing the root module itself.
Supported combinations

    Convolution followed by batch normalization.

    Convolution followed by supported local average pooling.

    Spiking convolution modules with a .fuse() method, including their supported BatchNorm, dropout, and pooling combinations.

Dropout is omitted from fused modules because it is disabled during evaluation.

For average-pooling fusion, use a convolution with stride and dilation (1, 1) and a riverops AvgPool2d with zero padding. This pool uses count_include_pad=True. The helper rejects unsupported dilation or pool padding; compare outputs to verify equivalence for your convolution geometry.

fuse_model handles supported riverops modules; it does not rewrite arbitrary PyTorch layer sequences.

For individual convolution/pooling pairs, see the fusion helpers.

Continue with Quantization and fixed point if you want to evaluate reduced-precision behavior.

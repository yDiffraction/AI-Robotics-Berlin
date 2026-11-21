Quantization and fixed point

Quantization represents values with a limited number of bits. riverops lets you train with these precision limits or evaluate their effect on a trained model.

Workflow
	

Use it when…

Quantization-aware training (QAT)
	

You want to fine-tune a model while simulating reduced precision.

Post-training evaluation (PTQ-style)
	

You want to evaluate fixed-point effects without further training.

These tools prepare supported riverops modules. Standard PyTorch layers, such as the classifier’s torch.nn.Linear, remain floating-point. Fixed-point evaluation is approximate emulation, not a guarantee of bit-for-bit hardware agreement or faster execution.
Choose a format

FixedPointFormat describes the total width, the fractional width, and whether negative values are allowed:

from riverops.nn import FixedPointFormat

fmt = FixedPointFormat(total_bits=8, frac_bits=4, signed=True)
print(fmt.scale)      # 16
print(fmt.min_value)  # -8.0
print(fmt.max_value)  # 7.9375

This format has a step size of 1 / 16. Conversion rounds to that grid and clips values outside its range.
QuantConfig

Choose formats for weights, biases, membrane potentials, and thresholds. Enable each kind of quantization explicitly:

from riverops.nn import FixedPointFormat, QuantConfig

weight_fmt = FixedPointFormat(total_bits=8, frac_bits=4)
potential_fmt = FixedPointFormat(total_bits=16, frac_bits=8)
bias_fmt = FixedPointFormat(total_bits=24, frac_bits=12)

quant_config = QuantConfig(
    weight_format=weight_fmt,
    bias_format=bias_fmt,
    potential_format=potential_fmt,
    threshold_format=potential_fmt,
    enable_weight_quant=True,
    enable_bias_quant=True,
    enable_potential_quant=True,
    enable_threshold_quant=True,
    qat_mode=True,
)

These are example formats, not recommended settings for every model. Compare accuracy with your floating-point model when choosing your own.

potential_format also controls intermediate neuron calculations during fixed-point emulation; there is no separate compute format.
Quantization-aware training

Start with the trained model from Training and evaluation. prepare_qat returns a copy with fake quantization enabled by the configuration. Fake quantization keeps floating-point tensors while snapping forward values to the chosen grid; QAT keeps gradients usable for training.

import torch
from riverops.nn import prepare_qat

qat_model = prepare_qat(model, quant_config, fused_bn=True).train()
qat_optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-4)

Continue your training loop with qat_model and qat_optimizer. Create the optimizer after preparing the copy so it updates the correct parameters.

fused_bn=True enables fusion-aware QAT for SpikingConvBN2d and SpikingConvBNDropout2d while retaining normal BatchNorm training. Pooling variants do not enable that path. After fine-tuning, fuse and evaluate the trained copy:

from riverops.nn import fuse_model

qat_model.eval()
fused_qat_model = fuse_model(qat_model).eval()
with torch.no_grad():
    qat_output = fused_qat_model(x)

Post-training evaluation

prepare_ptq returns an evaluation copy, fuses supported modules, and enables approximate fixed-point emulation:

from riverops.nn import prepare_ptq

ptq_model = prepare_ptq(model, quant_config, fuse=True)
with torch.no_grad():
    ptq_output = ptq_model(x)

Here model is a container such as TinySNN. For a standalone module such as SpikingConvBN2d, fuse it directly before preparing it:

from riverops.nn import prepare_fixed_point

layer.eval()
fixed_layer = prepare_fixed_point(layer.fuse(), quant_config).eval()

Here layer is the standalone module from Fusion. Unfused BatchNorm or pooling modules are not accepted by prepare_fixed_point.
Compare results

Run the same validation data through the floating model, its fused copy, and the QAT or PTQ version. Compare task accuracy as well as output spikes: a small change near the firing threshold can change later timesteps.

For direct conversion of tensors and integer fixed-point arithmetic, see the quantization API.

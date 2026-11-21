Quantization

For choosing formats and preparing a model, see Quantization and fixed point.
Formats and settings

total_bits includes the sign bit for signed formats. frac_bits must be non-negative and smaller than total_bits.

class riverops.nn.FixedPointFormat(total_bits, frac_bits, signed=True)

    Signed or unsigned fixed-point format description.

Set both a format and its corresponding enable_* flag to enable quantization. potential_format is also used for intermediate fixed-point neuron calculations.

class riverops.nn.QuantConfig(weight_format=None, potential_format=None, threshold_format=None, enable_weight_quant=False, enable_potential_quant=False, enable_threshold_quant=False, qat_mode=False, bias_format=None, enable_bias_quant=False)

    Quantization settings shared across riverops modules.

Floating-point quantization helpers

These return floating-point tensors on a fixed-point grid.

riverops.nn.quantize_uniform(x, fmt)

riverops.nn.fixed_resize(value, fmt)

    Approximate fixed_pkg resize by quantizing/dequantizing to a format.

The following helpers apply the matching format and enable flag from a QuantConfig:

riverops.nn.fake_quantize_weights(weight, quant)

riverops.nn.fake_quantize_bias(bias, quant)

riverops.nn.fake_quantize_potentials(value, quant)

riverops.nn.fake_quantize_threshold(value, quant)

Integer conversion

to_fixed produces an integer representation. Use from_fixed to decode it back to a floating-point tensor. resize_fixed converts an integer tensor between two formats.

riverops.nn.to_fixed(x, fmt)

riverops.nn.from_fixed(x_fixed, fmt)

riverops.nn.resize_fixed(x_fixed, src_fmt, dst_fmt)

clip_to_format clips an existing integer representation to the format’s integer range without changing its fractional scale.

riverops.nn.clip_to_format(x_fixed, fmt)

Integer arithmetic

Both helpers return integer tensors in dst_fmt. Decode their results with from_fixed(result, dst_fmt).

riverops.nn.add_then_resize(lhs_fixed, rhs_fixed, src_fmt, dst_fmt)

riverops.nn.mul_then_resize(lhs_fixed, rhs_fixed, lhs_fmt, rhs_fmt, dst_fmt)
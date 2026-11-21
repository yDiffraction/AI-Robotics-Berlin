Model preparation

These functions return copies. Set the input model to evaluation mode before fusion, and create a new optimizer for a QAT copy before training it.
Whole-model fusion

Fuses supported riverops child modules recursively. For a standalone BatchNorm or pooling variant, call its .fuse() method directly. Plain SpikingConv2d needs no fusion.

riverops.nn.fuse_model(module)

    Recursively fuse riverops modules that expose .fuse().

    Returns a fused copy and leaves module untouched. This is intended for eval, inference, or export preparation. It does not perform graph validation, torch.fx traversal, or arbitrary PyTorch pattern matching.

Quantization-aware training

Attaches the supplied quantization configuration to supported modules. fused_bn=True enables fusion-aware QAT for SpikingConvBN2d and SpikingConvBNDropout2d. Pooling variants do not enable that path.

riverops.nn.prepare_qat(module, quant_config, *, fused_bn=False)

    Enable existing fake quantization paths for QAT fine-tuning.

    Returns a prepared copy and leaves module untouched.

Post-training evaluation

Returns an evaluation copy and, by default, fuses supported child modules before enabling approximate fixed-point emulation.

riverops.nn.prepare_ptq(module, quant_config, *, fuse=True)

    Prepare a trained floating-point model for post-training quantized evaluation.

    Returns a copy in eval mode, fuses riverops modules, and enables approximate fixed-point emulation through prepare_fixed_point. This is not QAT, integer-only execution, or bit-accurate hardware export.

Fixed-point preparation

Requires SpikingConv2d modules. Fuse standalone BatchNorm or pooling variants with .fuse(); for a container model, use fuse_model first. A plain SpikingConv2d can be prepared directly.

riverops.nn.prepare_fixed_point(module, quant_config)

    Enable approximate fixed-point emulation on fused SpikingConv2d modules.

    Returns a prepared copy and leaves module untouched. The emulation uses FixedPointFormat quantize/dequantize steps after key arithmetic operations. It is intended for fused SpikingConv2d modules and is not a bit-accurate VHDL fixed_pkg implementation or optimized execution path. Treat it as evaluation-oriented emulation, not the main training path.

Fusion helpers

For direct use on individual layer pairs. Call .eval() on BatchNorm before fusing it.

riverops.nn.compute_fused_conv_bn_params(conv, bn)

    Fold BatchNorm running statistics into floating-point Conv2d parameters.

riverops.nn.fuse_conv_bn(conv, bn)

riverops.nn.fuse_conv_avgpool(conv, pool)

Sequential layer fusion for export

fuse_for_export returns an evaluation copy and scans torch.nn.Sequential containers for adjacent convolution/BatchNorm and convolution/riverops average-pooling pairs. It is separate from fuse_model, which calls the .fuse() methods of spiking convolution child modules.

riverops.nn.fuse_for_export(module)
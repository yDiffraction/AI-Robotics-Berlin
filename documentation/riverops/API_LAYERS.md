Layers

riverops spatial layers use [B, T, C, H, W] tensors. They apply their operation at every timestep, preserving the batch and time dimensions.
Convolution

This wrapper uses zero padding, dilation 1, and groups 1.

first_layer=True marks a convolution receiving non-spike input. Pass an optional QuantConfig as quant to use its weight and bias quantization flags.

class riverops.nn.Conv2d(*args, first_layer=False, quant=None, **kwargs)

    Time-aware conv layer with optional weight quantization.

Batch normalization

Batch normalization operates over the combined batch and time dimensions. Use the usual torch.nn.BatchNorm2d constructor arguments.

class riverops.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, *, bias=True)

Average pooling

Local average pooling uses ceil_mode=False and count_include_pad=True. When stride is omitted, it defaults to kernel_size.

class riverops.nn.AvgPool2d(kernel_size, stride=None, padding=0)

    Fusion-friendly average pooling with explicit [B, T, C, H, W] support.

Dropout

Use the usual torch.nn.Dropout2d constructor arguments, including p and inplace. Call .eval() to disable dropout for evaluation.

class riverops.nn.Dropout2d(p=0.5, inplace=False)

Upsampling

Increase spatial dimensions using nearest-neighbor interpolation.

class riverops.nn.NearestUpsample(scale_factor)

    Nearest-neighbor upsampling on [B, T, C, H, W] tensors.


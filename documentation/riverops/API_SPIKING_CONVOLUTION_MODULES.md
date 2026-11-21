Spiking convolution modules

These modules take [B, T, C, H, W] inputs and return output spikes in the same layout. They use PLIFNode with learnable leak by default.

Import these classes from riverops.nn, or directly from riverops.nn.spiking_conv. Import RecurrentSpikingConv2d from riverops.nn or riverops.nn.recurrent.
Choose a module

Module
	

Operations, in order

SpikingConv2d
	

Convolution, neuron

SpikingConvBN2d
	

Convolution, BatchNorm, neuron

SpikingConvBNDropout2d
	

Convolution, BatchNorm, dropout, neuron

SpikingConvAvgPool2d
	

Convolution, average pooling, neuron

SpikingConvBNAvgPool2d
	

Convolution, BatchNorm, average pooling, neuron

SpikingConvBNDropoutAvgPool2d
	

Convolution, BatchNorm, dropout, average pooling, neuron

Convolution arguments follow PyTorch naming. Set first_layer=True on the input-facing module. All variants share the same neuron settings.

The BatchNorm, dropout, and pooling variants expose .fuse(), which returns a new SpikingConv2d for evaluation. See Fusion for supported conditions and whole-model preparation.
Convolution and neuron

class riverops.nn.SpikingConv2d(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, first_layer=False, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    Conv2d -> PLIFNode.

BatchNorm variants

SpikingConvBN2d applies convolution, batch normalization, and then the neuron. SpikingConvBNDropout2d adds dropout between batch normalization and the neuron.

For evaluation, call .eval() before .fuse() to fold the running statistics into a new SpikingConv2d. Fusion requires running statistics; keep bn_track_running_stats=True when using that workflow.

class riverops.nn.SpikingConvBN2d(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, first_layer=False, bn_eps=1e-05, bn_momentum=0.1, bn_affine=True, bn_track_running_stats=True, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    Conv2d -> BatchNorm2d -> PLIFNode.

class riverops.nn.SpikingConvBNDropout2d(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, first_layer=False, bn_eps=1e-05, bn_momentum=0.1, bn_affine=True, bn_track_running_stats=True, dropout_p=0.5, dropout_inplace=False, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    Conv2d -> BatchNorm2d -> Dropout2d -> PLIFNode.

Pooling variants

Pooling occurs before spike generation. pool_stride defaults to pool_kernel_size when omitted.

class riverops.nn.SpikingConvAvgPool2d(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, first_layer=False, pool_kernel_size=2, pool_stride=None, pool_padding=0, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    Conv2d -> AvgPool2d -> PLIFNode.

class riverops.nn.SpikingConvBNAvgPool2d(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, first_layer=False, bn_eps=1e-05, bn_momentum=0.1, bn_affine=True, bn_track_running_stats=True, pool_kernel_size=2, pool_stride=None, pool_padding=0, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    Conv2d -> BatchNorm2d -> AvgPool2d -> PLIFNode.

class riverops.nn.SpikingConvBNDropoutAvgPool2d(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, first_layer=False, bn_eps=1e-05, bn_momentum=0.1, bn_affine=True, bn_track_running_stats=True, dropout_p=0.5, dropout_inplace=False, pool_kernel_size=2, pool_stride=None, pool_padding=0, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    Conv2d -> BatchNorm2d -> Dropout2d -> AvgPool2d -> PLIFNode.

Recurrent convolution

The recurrent module adds its input and feedback paths in the spike domain. Both paths’ membrane potentials and the previous output start at zero for each call. Feedback uses the previous timestep’s combined output.

class riverops.nn.RecurrentSpikingConv2d(in_channels, hidden_channels, kernel_size, *, stride=1, padding=0, bias=True, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None)

    SpikingConv2d input path + SpikingConv2d feedback path.

    Input layout is always [B, T, C, H, W]. At each timestep, the module applies the input and feedback paths, then sums their spikes. Both paths’ membrane potentials and previous spikes start at zero on each forward call and are carried between timesteps within the sequence.

The feedback convolution uses stride 1; choose a kernel and padding that preserve its spatial shape, such as kernel_size=3, padding=1.

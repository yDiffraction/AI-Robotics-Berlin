Neurons

Neurons process a complete sequence and return a tensor of spikes with the same shape. Each call begins with zero membrane potentials; timesteps are processed in order along dimension 1. Both IFNode and PLIFNode accept [B, T, F] and [B, T, C, H, W] inputs.
Integrate and fire

IFNode accumulates input without leak.

class riverops.nn.IFNode(*, v_threshold=1.0, reset_mode='HARD', surrogate_function=None, quant=None)

Leaky integrate and fire

PLIFNode learns its leak by default. initial_tau must be greater than one in learnable mode. For a fixed leak, use learnable_leak=False and any positive initial_tau.

class riverops.nn.PLIFNode(*, initial_tau=2.0, learnable_leak=True, v_threshold=1.0, reset_mode='HARD', surrogate_function=None, quant=None)

Common arguments

Argument
	

Meaning

v_threshold
	

Spike threshold; defaults to 1.0.

reset_mode
	

"HARD" resets to zero; "SOFT" subtracts the threshold. The default is "HARD".

surrogate_function
	

Callable used for spike generation and its backward gradient. Defaults to ATan.

quant
	

Optional quantization configuration.

For explicit timestep-by-timestep state, use functional helpers. Their if_step and lif_step functions also accept detach_reset=True to detach spikes from the reset calculation’s gradient. This argument is not accepted by the IFNode or PLIFNode constructors.
Surrogate gradient

ATan uses a threshold in its forward pass and an arctangent surrogate gradient in its backward pass.

class riverops.nn.surrogate.ATan(alpha=2.0)

    Arctangent surrogate gradient.


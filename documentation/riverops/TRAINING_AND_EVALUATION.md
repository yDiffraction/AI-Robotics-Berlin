Training and evaluation

Use normal PyTorch losses and optimizers. The examples below continue with model and x from Build a model.
Train a batch

riverops uses a surrogate gradient to train through spike generation. The ATan surrogate used in the example model supplies the backward gradient; the forward output still consists of spikes.

import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.tensor([0, 1, 2, 3])  # One class label per sequence in x.

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = criterion(logits, target)
loss.backward()
optimizer.step()

For real data, put the same steps inside your data-loader loop:

for x_batch, target_batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x_batch), target_batch)
    loss.backward()
    optimizer.step()

Each x_batch should have shape [B, T, C, H, W]; each target_batch should contain one class label per sequence.
Evaluate predictions

Switch to evaluation mode so batch normalization uses its running statistics and dropout is disabled.

model.eval()
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=1)

This floating-point evaluation is the starting point for comparing prepared versions of your model.
Prepare a trained model

Goal
	

Next step

Combine supported layers for evaluation
	

Fuse the model

Continue training with reduced-precision effects
	

Quantization-aware training

Evaluate a trained model with fixed-point effects
	

Post-training evaluation

Produce a supported model graph
	

Export

The preparation helpers return copies. Keep evaluating your original model alongside the prepared one with the same inputs and task metrics.

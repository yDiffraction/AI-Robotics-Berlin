# SNN pretraining on DVS Gesture (for throw vs. fake fine-tuning)

The goal is a backbone that is already in the **board's format**, so fine-tuning on our own GenX320 clips starts from useful motion filters.

## Board format the model matches

| What | Value | Source |
|---|---|---|
| Input | `[B, T, 2, 80, 80]`: binary polarity frames, **10 ms** each | despot `downsampling: 4`, `frequency: 100` (documentation/DEPLOY.md) |
| Layers | 4 × `SpikingConvBN2d` (stride 2: 80→40→20→10→5), widths 16-32-64-64 | riverops export supports conv-LIF only |
| Head | `SpikingConv2d(64, K, 5)` → `[B, T, K, 1, 1]`; class = most output spikes | no Linear/ReduceMean in the exporter |
| Params | ~79k (11-class head), ~74k with a 2-class head | |

`ThrowSNN.export_module()` gives the pure conv stack. `export_for_river(fuse_model(...))` was verified to produce a River Model Graph (`checkpoints/dvsg_best_river_graph.json`).

## Data

DVS Gesture comes from the Zenodo mirror of Tonic's npy release (figshare blocks scripted downloads):

- train: users 1–23, 1077 recordings
- test: users 24–29, 264 recordings
- 11 arm/hand gestures

DVS128 coordinates are rescaled 128→80 and binned to 10 ms. Each recording is stored as its set of active cells, so the input is binary. Binary input is used on purpose: it is the most sensor-agnostic choice between DVS128 and GenX320.

Augmentation:

- ±6 px random shift
- 10 % event dropout
- 0.05 % random noise cells

## Result (21 Nov 2026, Intel Arc 140T, 35 min)

| | |
|---|---|
| Best checkpoint | `checkpoints/dvsg_best.pt`, epoch 27 of 30 |
| Held-out users (24–29), per 0.64 s window | **87.9 %** (chance 9 %) |
| Held-out users, per recording (vote over 4 windows) | **91.3 %** |
| Train accuracy | 93 % (small gap, no strong overfitting) |
| Fused (BN-folded) vs. float model | identical predictions; 0.002 % of output spikes differ |

This is float accuracy. Quantized (QAT / fixed-point) accuracy has not been measured yet.

## Run

```bash
python -I pretrain/prep_dvsgesture.py data/dvsgesture_raw/extracted/ibmGestureTrain data/dvsg_train.npz
python -I pretrain/prep_dvsgesture.py data/dvsgesture_raw/extracted/ibmGestureTest  data/dvsg_test.npz
python pretrain/train_dvsgesture.py --epochs 30 --max-minutes 35
```

The runs used system Python 3.14 with `torch 2.13+xpu` (Intel Arc 140T) and the riverops 0.2.0 wheel installed with `--no-deps`. The project `.venv` has CPU-only torch.

`speedpatch.py` removes a per-timestep GPU sync in riverops' PLIF neuron. It produces identical outputs and is ~25 % faster, and it only applies when no quantization config is set.

## Fine-tune on throw / fake

```python
from model import load_pretrained, rate_readout
model = load_pretrained("pretrain/checkpoints/dvsg_best.pt", num_classes=2, device="xpu", freeze_first=1)
# x: [B, T, 2, 80, 80] binary 10 ms frames from GenX320 (despot format), e.g. T=90 for -600..+300 ms
# loss = F.mse_loss(rate_readout(model(x)), F.one_hot(y, 2).float())
```

Note that a frozen layer's BatchNorm still updates its running stats in `train()` mode. Put frozen layers in `eval()` if that matters. After fine-tuning, run `prepare_qat(model, quant_config, fused_bn=True)` with 8-bit weights and Q4.12 potentials to match `config.yaml` `scaling: 4096`, then fuse and export.

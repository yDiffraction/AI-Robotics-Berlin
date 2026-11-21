Export

riverops.export captures supported PyTorch models and produces a River Model Graph as JSON. The graph describes model structure for downstream hardware tools; exporting it does not generate firmware.
Export a small model

Start with an evaluation model and an example input with the shape you want to export:

import torch
from torch import nn
from riverops.nn import SpikingConv2d
from riverops.export import export_for_river

export_model = nn.Sequential(
    SpikingConv2d(3, 8, kernel_size=3, padding=1, first_layer=True),
    SpikingConv2d(8, 3, kernel_size=3, padding=1, bias=False),
).eval()

example_input = torch.rand(2, 4, 3, 16, 16)
model_json = export_for_river(export_model, example_input)

with open("model.json", "w") as output:
    output.write(model_json)

The output contains input dimensions, nodes, connections, and model outputs. The export IR and JSON schema retain names such as ConvLIFSpec and ConvLIF2d; these are distinct from the Python model-building class SpikingConv2d.
Check model support

Export is still a work in progress. Start with a small supported convolutional sequence before exporting a larger network.

    Only a subset of operations and convolution properties can be represented.

    Graph cycles and multiple model inputs or outputs are not fully supported.

    String padding is not supported by the export path.

    The graph representation does not capture the neuron’s reset mode.

The examples/models directory contains supported models and examples of patterns that are not supported. In particular, the TinySNN training example has a Linear classification head; do not assume that the whole classifier can be exported.
Export function

riverops.export.export_for_river(model, input_tensor)

    Simple helper to go all the way from a model to the JSON string representation of it.


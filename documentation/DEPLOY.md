
Deploying a new model on the Kria

Once your new SNN is ready, you will want to deploy it on the Kria dev board.

There might be scripts to help you do this, but just in case here is a list of what needs to be copied on the board. You can easily do that using SCP.
Moving parts

In order to update a model on the Kria, there are many moving parts:

    spleat.bit.bin and spleat.dtbo: those are the bitstream and a device-tree overlay for Linux.
    config.yaml: this is a configuration file for the river server.
    metavision_camera.json: configuration file for the Despot application.
    weights.txt: model parameters for the SNN.
    ROS node: if your SNN is new, you probably need to update the ROS node on the board. This is not covered here.

Bitstream and Device Tree Overlay

Those files go to /usr/lib/firmware/xilinx/spleat/spleat.bit.bin and /usr/lib/firmware/xilinx/spleat/spleat.dtbo.

You can generate them using riversdk.dev.

    Export the JSON configuration of your model.
    Log into the River SDK.
    Click "Open firmware jobs".
    Click "New firmware".
    Fill the form, submit and wait for the output.

Note: make sure to select the "KV260" board and the "GenX320" Prophesee IP.
Server configuration file

The server configuration is used by the software talking to the FPGA. With the right configuration file, it knows the right details about the SNN. It is located on the Kria under /root/.config/spleat-server/config.yaml.

It looks like this:

application:
  host: "0.0.0.0"
  port: 3333
  weights_file: /root/weights.txt  # Change it if you name your file differently (not recommended).
  max_concurrency: 40  # You can keep it like that.

inputs:
  channels: 2
  height: 80
  width: 80
  scaling: 4096.0

outputs:
  channels: 5
  height: 1
  width: 1

Important details:

    Use 0.0.0.0 for the host and 3333 for the port, it will protecty you from headaches.
    Scaling depends on the quantization of the potentials. It must be 2**fraction_bits. Example: 4096 for Q4.12.
    Other values: channels, width and height of the input and outputs.

Despot configuration file

metavision_camera.json contains configuration for the camera and the despot application. It should normally be placed in /root/metavision_camera.json on the Kria.

By default it has this content :

{
  "frequency": 100.0,
  "downsampling": 4,
  "noise_filtering_factor": 0.5,

  "biases": {
    "bias_diff": 51,
    "bias_diff_off": 40,
    "bias_diff_on": 40,
    "bias_fo": 29,
    "bias_hpf": 0,
    "bias_refr": 82
  }
}

Some details:

    "frequency" represents how often frames are constructed out of events from camera.
    "downsampling" expects a value in [1, 2, 4, 8, 16, 32, 64] to downsample both dmensions (height, width) at once. The native camera definition is 320x320; this is what you will get with downsampling of 1. With a downsampling of 2, for instance, it becomes 160x160 and with 4 it becomes 80x80.
    "noise_filtering_factor" allows to reduce noise. With a very high value like 100.0 there is almost no noise filtering, with a very low value like 0.0001 almost everything is filtered out, including useful information. Values between 0.1 and 1.0 make the most sense. If you don't want any noise filtering at all, just remove the line.
    "biases" are configuaration values for the Prophesee camera sensorpart. You can find more information at docs.prophesee.ai.

Model parameters file

By default, the weights file is in /root/weights.txt.

It is a long list of integer values which contains all the parameters of your SNN:

127
127
-127
108
128
-127
100
127
81
...

CAUTION: the file must not have an empty lines (including at the end)!

The layout is as follow:

[weights layer 1]
[weights layer 2]
...
[weights layer N]
[biases layer 1]
[biases layer 2]
...
[biases layer N]
threshold layer 1
leak layer 1
threshold layer 2
leak layer 2
...
threshold layer N
leak layer N


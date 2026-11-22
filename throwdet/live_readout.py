#!/usr/bin/env python3
"""Host-side live readout of the throw-motion detector: board output spikes -> alarm.

The board runs the network every 10 ms frame and outputs one spike count per output neuron
(neuron 0 = no throw motion, neuron 1 = throw motion). Feed those two numbers into
ThrowReadout.update() every frame; it returns True on the frame an alarm fires.

    from live_readout import ThrowReadout
    ro = ThrowReadout("throwdet/models/D_e10_fold3/readout.json")
    for s0, s1 in board_outputs():            # every 10 ms
        if ro.update(s0, s1):
            trigger_dodge()                   # e.g. send the dodge key to dodge_controller

Pure Python, no dependencies, so it can go into a ROS node or the dodge supervisor as is.
CLI for testing: reads lines "s0 s1" (one per frame) from stdin and prints the alarms.

    printf "0 0\\n0 1\\n..." | python throwdet/live_readout.py throwdet/models/D_e10_fold3/readout.json
"""
import json
import sys


class ThrowReadout:
    def __init__(self, readout_json, refractory_frames=None):
        r = json.load(open(readout_json)) if isinstance(readout_json, str) else readout_json
        self.alpha = float(r["alpha"])
        self.threshold = float(r["threshold"])
        self.hold = int(r.get("hold_frames", 1))
        self.refractory = refractory_frames if refractory_frames is not None else int(r.get("refractory_ms", 1000) // 10)
        self.score = 0.0
        self.run = 0
        self.since_alarm = 10**9
        self.frame = 0

    def update(self, s0, s1):
        """One 10 ms frame. Returns True if an alarm fires on this frame."""
        self.score = (1.0 - self.alpha) * self.score + self.alpha * (float(s1) - float(s0))
        self.run = self.run + 1 if self.score > self.threshold else 0
        self.since_alarm += 1
        self.frame += 1
        if self.run == self.hold and self.since_alarm >= self.refractory:
            self.since_alarm = 0
            return True
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ro = ThrowReadout(sys.argv[1])
    for line in sys.stdin:
        parts = line.split()
        if len(parts) < 2:
            continue
        if ro.update(parts[0], parts[1]):
            print(f"ALARM at frame {ro.frame} ({ro.frame / 100:.2f} s), score {ro.score:+.3f}", flush=True)


if __name__ == "__main__":
    main()

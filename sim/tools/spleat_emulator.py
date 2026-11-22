#!/usr/bin/env python3
"""Stand-in for the SPLEAT server (camera + SNN on the FPGA).

Speaks the same wire protocol as the board: each frame is a 4-byte big-endian
length followed by a CBOR map {"id", "payload", "latency", "error"}, where
"payload" is a byte string of little-endian float32 class scores. gesture_node
connects to it exactly as it would to the real server.

Modes:
  manual    type a class index (Enter) to "show" that gesture; empty line = idle
  sequence  play a fixed script, e.g. --sequence "1:3,10:2,7:4" (class:seconds)
  random    random scores, same as the Durance qualia_sim

Standard library only, so it runs in the sim container and on the host.
"""

import argparse
import os
import random
import socket
import struct
import sys
import threading
import time


def cbor_head(major, value):
    if value < 24:
        return bytes([(major << 5) | value])
    if value < 0x100:
        return bytes([(major << 5) | 24, value])
    if value < 0x10000:
        return bytes([(major << 5) | 25]) + struct.pack(">H", value)
    if value < 0x100000000:
        return bytes([(major << 5) | 26]) + struct.pack(">I", value)
    return bytes([(major << 5) | 27]) + struct.pack(">Q", value)


def cbor_text(text):
    data = text.encode()
    return cbor_head(3, len(data)) + data


def encode_frame(frame_id, scores, latency_us=0):
    payload = struct.pack(f"<{len(scores)}f", *scores)
    body = (
        cbor_head(5, 4)
        + cbor_text("id") + cbor_head(0, frame_id)
        + cbor_text("payload") + cbor_head(2, len(payload)) + payload
        + cbor_text("latency") + cbor_head(0, latency_us)
        + cbor_text("error") + cbor_text("")
    )
    return struct.pack(">I", len(body)) + body


class Source:
    """Holds the class currently being 'shown' to the camera."""

    def __init__(self, classes, idle_class, noise):
        self.classes = classes
        self.idle_class = idle_class
        self.noise = noise
        self.active = idle_class
        self.lock = threading.Lock()

    def set(self, index):
        with self.lock:
            self.active = index

    def scores(self):
        with self.lock:
            active = self.active
        scores = [random.uniform(0.0, self.noise) for _ in range(self.classes)]
        scores[active] += 1.0
        total = sum(scores)
        return [s / total for s in scores]


def manual_input(source):
    print(f"Enter a class index 0..{source.classes - 1}, empty line = idle "
          f"(class {source.idle_class}), q = quit", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "q":
            os._exit(0)
        if line == "":
            source.set(source.idle_class)
        elif line.isdigit() and int(line) < source.classes:
            source.set(int(line))
        else:
            print(f"ignored: {line!r}", flush=True)
            continue
        print(f"showing class {source.active}", flush=True)


def play_sequence(source, script, loop):
    steps = []
    for item in script.split(","):
        index, seconds = item.split(":")
        steps.append((int(index), float(seconds)))
    while True:
        for index, seconds in steps:
            print(f"showing class {index} for {seconds:.1f}s", flush=True)
            source.set(index)
            time.sleep(seconds)
        if not loop:
            source.set(source.idle_class)
            print("sequence done, idle", flush=True)
            return


def serve(args, source):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"SPLEAT emulator listening on {args.host}:{args.port} "
          f"({args.classes} classes, {args.rate:.0f} Hz, mode={args.mode})", flush=True)
    period = 1.0 / args.rate
    frame_id = 0
    while True:
        conn, peer = server.accept()
        print(f"client connected from {peer[0]}", flush=True)
        next_tick = time.monotonic()
        try:
            while True:
                if args.mode == "random":
                    scores = [random.random() for _ in range(args.classes)]
                else:
                    scores = source.scores()
                conn.sendall(encode_frame(frame_id, scores))
                frame_id += 1
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except (BrokenPipeError, ConnectionResetError):
            print("client disconnected", flush=True)
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("manual", "sequence", "random"), default="manual")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--classes", type=int, default=11,
                        help="number of scores per frame; must match msgs/Gestures.msg")
    parser.add_argument("--idle-class", type=int, default=None,
                        help="class sent when nothing is shown (default: last class)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="frames per second (camera frame frequency)")
    parser.add_argument("--noise", type=float, default=0.3,
                        help="random score mass added to every class")
    parser.add_argument("--sequence", default="1:3,10:2,7:4,10:2",
                        help="class:seconds list for --mode sequence")
    parser.add_argument("--loop", action="store_true", help="repeat the sequence forever")
    args = parser.parse_args()

    idle = args.classes - 1 if args.idle_class is None else args.idle_class
    source = Source(args.classes, idle, args.noise)
    if args.mode == "manual":
        threading.Thread(target=manual_input, args=(source,), daemon=True).start()
    elif args.mode == "sequence":
        threading.Thread(target=play_sequence, args=(source, args.sequence, args.loop),
                         daemon=True).start()
    try:
        serve(args, source)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

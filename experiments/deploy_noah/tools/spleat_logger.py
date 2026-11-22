#!/usr/bin/env python3
"""Log every inference result from a SPLEAT server (the board or the emulator) to CSV.

Connects as an extra TCP client, decodes each CBOR frame and writes one row per frame:
  host_time_ns   laptop wall clock when the frame arrived (time.time_ns)
  id             server frame id
  latency        server-reported latency field (unit as reported by the server)
  error          server error string
  argmax, s0..sN class scores

Press Enter while it runs to write a MARK row (e.g. at the moment a throw is released);
marks let you align score traces with the phone video and the take list.

  uv run tools/spleat_logger.py --host durance-kria.local --out logs/slot1.csv
"""

import argparse
import csv
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cbor2


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf += chunk
    return bytes(buf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="durance-kria.local")
    p.add_argument("--port", type=int, default=3333)
    p.add_argument("--out", default=f"logs/spleat-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    addr = socket.gethostbyname(args.host)
    sock = socket.create_connection((addr, args.port), timeout=5)
    sock.settimeout(None)
    print(f"connected to {args.host} ({addr}):{args.port}, logging to {out}. Enter = MARK, Ctrl-C = stop")

    lock = threading.Lock()
    f = out.open("w", newline="")
    writer = csv.writer(f)
    header_written = False
    marks = 0

    def mark_loop():
        nonlocal marks
        for line in sys.stdin:
            with lock:
                marks += 1
                writer.writerow(["MARK", time.time_ns(), marks, line.strip()])
                f.flush()
            print(f"MARK {marks} {line.strip()}")

    threading.Thread(target=mark_loop, daemon=True).start()

    frames, t_start = 0, time.monotonic()
    try:
        while True:
            size = struct.unpack(">I", recv_exact(sock, 4))[0]
            now = time.time_ns()
            msg = cbor2.loads(recv_exact(sock, size))
            payload = msg.get("payload", b"")
            scores = struct.unpack(f"<{len(payload) // 4}f", payload) if payload else ()
            with lock:
                if not header_written:
                    writer.writerow(["kind", "host_time_ns", "id", "latency", "error", "argmax"]
                                    + [f"s{i}" for i in range(len(scores))])
                    header_written = True
                best = max(range(len(scores)), key=scores.__getitem__) if scores else -1
                writer.writerow(["FRAME", now, msg.get("id"), msg.get("latency"), msg.get("error", ""), best]
                                + [f"{s:.5g}" for s in scores])
            frames += 1
            if not args.quiet and frames % 100 == 0:
                fps = frames / (time.monotonic() - t_start)
                print(f"{frames} frames, {fps:.1f} fps, last latency={msg.get('latency')}, argmax={best}")
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        print(f"{frames} frames written to {out}")


if __name__ == "__main__":
    main()

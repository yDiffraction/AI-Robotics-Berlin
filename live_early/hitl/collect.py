#!/usr/bin/env python3
"""Run the detector live on the board and record a human's "it should spike NOW" button presses.

camera (despot_debug_camera over SSH) or a replay -> binary 10 ms frames -> SPLEAT on the Kria (and/or the float model)
-> host readout (EMA 0.85, alarm when y crosses theta, 1 s refractory) -> web page at http://127.0.0.1:8771

On the page: press SPACE (or the big button; hold it if you like) the moment the network should fire, i.e. when a throw
wind-up starts. BACKSPACE undoes the last press (within 5 s). The page shows the board's alarms next to your presses and
a running score (hit / miss / false alarm), so you see where the model is wrong while you record.

Everything is recorded to live_early/hitl/sessions/<stamp>/:
  chunk_NNNN.npz  frames  uint8 [n, 1600]   bit-packed 2x80x80 binary frames (np.unpackbits -> (2, 80, 80))
                  spike   int8  [n]          board output channel 0 (-1 when no board)
                  spike_f int8  [n]          float model channel 0 (-1 when not run)
                  y, alarm, held, t          readout, alarm flag, button held, wall time (s)
  events.jsonl    {"k": frame, "ev": "down" | "up" | "undo", "t": wall}
  meta.json       model folder, weights md5, theta, args

  python live_early/hitl/collect.py                                  # camera -> board (v3 must be loaded, see board_ctl.py)
  python live_early/hitl/collect.py --backend both                   # + float model alongside
  python live_early/hitl/collect.py --replay rec.bin --speed 0.5     # label a recording at half speed
  python live_early/hitl/collect.py --replay sessions/<stamp>        # re-label a recorded session
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_spec = importlib.util.spec_from_file_location("live_viewer", HERE.parent / "viewer.py")
viewer = importlib.util.module_from_spec(_spec)   # live_early/viewer.py: camera_frames, replay_frames, Board, Float, Readout
_spec.loader.exec_module(viewer)

GRID, CHUNK = 80, 6000          # 60 s per chunk file


def start_camera_tunnel(host, port):
    """despot_debug_camera on the board connects to ITS 127.0.0.1:port, and an SSH reverse tunnel carries that to
    this laptop's port. Works over the IPv6 link-local address (no IPv4 on the board subnet needed)."""
    command = f"""pkill -f "despot_debug_camera --host 127.0.0.1 --port {port}" 2>/dev/null; sleep 0.3
despot_debug_camera --host 127.0.0.1 --port {port} >/tmp/hitl_camera.log 2>&1 &
camera_pid=$!
trap 'kill "$camera_pid" 2>/dev/null; wait "$camera_pid" 2>/dev/null' EXIT
trap 'exit' HUP INT TERM
read -r stop
"""
    return subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes",
                             "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
                             "-R", f"{port}:127.0.0.1:{port}", "--", host, command], stdin=subprocess.PIPE)


def session_frames(path: Path):
    for f in sorted(path.glob("chunk_*.npz")):
        for row in np.load(f)["frames"]:
            yield np.unpackbits(row).reshape(2, GRID, GRID).astype(np.float32)


def paced(gen, speed, stop):
    """Replay at `speed` x real time (frames are 10 ms)."""
    t0 = time.perf_counter()
    for k, f in enumerate(gen):
        if stop.is_set():
            return
        yield f
        if speed > 0:
            time.sleep(max(0.0, t0 + (k + 1) * 0.01 / speed - time.perf_counter()))


class Recorder:
    """Buffers per-frame rows and writes 60 s chunks from a background thread (compression must not stall the loop)."""

    def __init__(self, out: Path, chunk=CHUNK):
        self.out, self.chunk = out, chunk
        out.mkdir(parents=True, exist_ok=True)
        self.cols = {k: [] for k in ("frames", "spike", "spike_f", "y", "alarm", "held", "t", "watch")}
        self.n_chunk = 0
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._writer, daemon=True)
        self.th.start()
        self.events = open(out / "events.jsonl", "a", buffering=1)

    def add(self, frame, spike, spike_f, y, alarm, held, t, watch=0):
        c = self.cols
        c["watch"].append(watch)
        c["frames"].append(np.packbits(frame.astype(bool).ravel()))
        c["spike"].append(spike); c["spike_f"].append(spike_f); c["y"].append(y)
        c["alarm"].append(alarm); c["held"].append(held); c["t"].append(t)
        if len(c["frames"]) >= self.chunk:
            self.flush()

    def event(self, k, ev):
        self.events.write(json.dumps({"k": k, "ev": ev, "t": time.time()}) + "\n")

    def flush(self):
        c = self.cols
        if not c["frames"]:
            return
        data = dict(frames=np.stack(c["frames"]), spike=np.array(c["spike"], np.int8), spike_f=np.array(c["spike_f"], np.int8),
                    y=np.array(c["y"], np.float32), alarm=np.array(c["alarm"], np.uint8), held=np.array(c["held"], np.uint8),
                    t=np.array(c["t"], np.float64), watch=np.array(c["watch"], np.uint8))
        self.q.put((self.out / f"chunk_{self.n_chunk:04d}.npz", data))
        self.n_chunk += 1
        self.cols = {k: [] for k in c}

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, data = item
            tmp = path.with_name(path.name + ".part")      # complete files only ever appear as chunk_*.npz
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **data)
            os.replace(tmp, path)

    def close(self):
        self.flush()
        self.q.put(None)
        self.th.join()
        self.events.close()


class Score:
    """Live hit / miss / false-alarm count. A press at p is hit by an alarm in [p - pre, p + post]."""

    def __init__(self, pre, post):
        self.pre, self.post = pre, post
        self.open, self.all = [], []          # presses not yet scored / all valid presses
        self.alarms_open, self.alarms_all = [], []
        self.hits = self.misses = self.fas = 0
        self.lat = []

    def alarm(self, k):
        self.alarms_open.append(k)
        self.alarms_all.append(k)

    def update(self, k):
        while self.open and self.open[0] + self.post < k:              # press window over
            p = self.open.pop(0)
            hit = [a for a in self.alarms_all if p - self.pre <= a <= p + self.post]
            if hit:
                self.hits += 1
                self.lat.append((hit[0] - p) * 10)
            else:
                self.misses += 1
        while self.alarms_open and self.alarms_open[0] + self.pre < k:  # no later press can claim it
            a = self.alarms_open.pop(0)
            if not any(p - self.pre <= a <= p + self.post for p in self.all):
                self.fas += 1

    def press(self, k):
        self.open.append(k)
        self.all.append(k)

    def undo(self, k):
        if self.open and k - self.open[-1] <= 500:
            self.all.remove(self.open.pop())
            return True
        return False

    def as_dict(self):
        return {"hits": self.hits, "misses": self.misses, "fas": self.fas,
                "lat_ms": round(float(np.median(self.lat)), 0) if self.lat else None}


PAGE = (HERE / "collect.html").read_text(encoding="utf-8") if (HERE / "collect.html").exists() else ""


class State:
    def __init__(self, cfg):
        self.lock = threading.Lock()
        self.cfg = cfg
        self.rows = deque(maxlen=600)
        self.seq, self.k = 0, 0
        self.img = np.zeros((GRID, GRID), np.uint8)
        self.held = False
        self.pending = []           # (event) from the browser, applied by the frame loop
        self.score = {}
        self.fps, self.rtt, self.minutes = 0.0, -1.0, 0.0
        self.n_press = 0
        self.done = False
        self.extra = {}             # live_loop.py: control panel state, sent with every stream message
        self.on_post = None         # live_loop.py: handler for other POST paths -> (code, json)


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ("/down", "/up", "/undo"):
                if state.on_post is None:
                    return self.send_error(404)
                code, body = state.on_post(self.path)
                return self._send(code, json.dumps(body).encode(), "application/json")
            with state.lock:
                state.pending.append(self.path[1:])
            self._send(204, b"", "text/plain")

        def do_GET(self):
            if self.path == "/":
                page = (HERE / "collect.html").read_text(encoding="utf-8")   # fresh: page fixes need only a reload
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if self.path != "/stream":
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen, first = None, True
            try:
                while not state.done:
                    with state.lock:
                        n_new = len(state.rows) if seen is None else min(len(state.rows), state.seq - seen)
                        rows = list(state.rows)[len(state.rows) - n_new:] if n_new else []
                        msg = {"rows": rows, "img": base64.b64encode(state.img.tobytes()).decode(), "k": state.k,
                               "fps": round(state.fps, 1), "rtt_ms": state.rtt, "score": state.score,
                               "presses": state.n_press, "minutes": round(state.minutes, 1), **state.extra}
                        seen = state.seq
                    if first:
                        msg["cfg"], first = state.cfg, False
                    self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE.parent / "model_v3"), help="folder with readout.json and model.pt; must match the board")
    ap.add_argument("--backend", choices=["board", "float", "both"], default="board")
    ap.add_argument("--host", default="192.168.77.1",
                    help="SPLEAT server: 192.168.77.1, durance-kria.local, an IPv6 link-local like fe80::...%%50, "
                         "or 192.168.123.99 on the dog")
    ap.add_argument("--tunnel", type=int, default=-1, help="camera over an SSH reverse tunnel (default: on for IPv6 hosts)")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--ssh", default=None, help="SSH target for despot_debug_camera (default root@<host>)")
    ap.add_argument("--listen-port", type=int, default=3334)
    ap.add_argument("--no-start", action="store_true", help="despot_debug_camera is started by hand")
    ap.add_argument("--replay", help=".bin / .npy recording or a session folder instead of the camera")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (0 = as fast as possible, for tests)")
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--pre", type=int, default=60, help="score window: alarm up to this many frames before the press counts")
    ap.add_argument("--post", type=int, default=30, help="... and up to this many after")
    ap.add_argument("--out", default=str(HERE / "sessions"))
    ap.add_argument("--tag", default="", help="appended to the session folder name")
    ap.add_argument("--http-port", type=int, default=8771)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (tests)")
    ap.add_argument("--auto-press", help="tests: comma-separated frame indices to press at (no browser)")
    a = ap.parse_args()
    a.ssh = a.ssh or f"root@{a.host}"

    model_dir = Path(a.model)
    ro = json.loads((model_dir / "readout.json").read_text())
    theta = a.theta if a.theta is not None else float(ro["theta"])
    board = viewer.Board(a.host, a.port) if a.backend in ("board", "both") else None
    flt = viewer.Float(model_dir / "model.pt") if a.backend in ("float", "both") else None
    readout = viewer.Readout(0.85, theta)
    stamp = time.strftime("%Y%m%d_%H%M%S") + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(Path(a.out) / stamp)
    md5 = hashlib.md5((model_dir / "weights.txt").read_bytes()).hexdigest() if (model_dir / "weights.txt").exists() else None
    (rec.out / "meta.json").write_text(json.dumps({
        "model": str(model_dir.resolve()), "weights_md5": md5, "theta": theta, "alpha": 0.85,
        "backend": a.backend, "source": a.replay or "camera", "speed": a.speed, "started": time.time(),
        "score_window": [a.pre, a.post], "args": vars(a)}, indent=1))
    print(f"recording to {rec.out}", flush=True)

    cfg = {"theta": theta, "primary": "board" if board else "float", "model": model_dir.resolve().name,
           "source": a.replay or "camera", "out": str(rec.out)}
    state = State(cfg)
    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(state))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"label page: http://127.0.0.1:{a.http_port}/", flush=True)

    stop = threading.Event()
    if a.replay and Path(a.replay).is_dir():
        frames = paced(session_frames(Path(a.replay)), a.speed, stop)
    elif a.replay:
        a.fast, a.loop = True, False
        frames = paced(viewer.replay_frames(a, stop), a.speed, stop)
    else:
        if a.tunnel == 1 or (a.tunnel < 0 and ":" in a.host):
            viewer.start_camera = start_camera_tunnel
        frames = viewer.camera_frames(a, stop)
    auto = set(int(x) for x in a.auto_press.split(",")) if a.auto_press else set()

    score = Score(a.pre, a.post)
    recent, rtts = deque(maxlen=3), []
    t_fps, n_fps, t_start = time.perf_counter(), 0, time.time()
    try:
        for k, f in enumerate(frames):
            if a.max_frames and k >= a.max_frames:
                break
            with state.lock:
                evs, state.pending = state.pending, []
            if k in auto:
                evs += ["down", "up"]
            for ev in evs:
                if ev == "down" and not state.held:
                    state.held = True
                    score.press(k)
                    rec.event(k, "down")
                elif ev == "up":
                    state.held = False
                    rec.event(k, "up")
                elif ev == "undo" and score.undo(k):
                    rec.event(k, "undo")
            s_b = s_f = -1
            if board:
                t0 = time.perf_counter()
                ob, _ = board.infer(f)
                rtts.append((time.perf_counter() - t0) * 1e3)
                s_b = int(ob[0] > 0)
            if flt:
                of, _ = flt.infer(f)
                s_f = int(of[0] > 0)
            s = s_b if board else s_f
            y, al = readout.step(k, float(s))
            if al:
                score.alarm(k)
            score.update(k)
            rec.add(f, s_b, s_f, y, int(al), int(state.held), time.time())
            n_fps += 1
            now = time.perf_counter()
            with state.lock:
                state.rows.append([round(y, 3), s, int(al), int(state.held), int("down" in evs)])
                state.seq += 1
                state.k = k
                state.score = score.as_dict()
                state.n_press = len(score.all)
                recent.append(f)
                if k % 4 == 0:
                    acc = np.zeros((2, GRID, GRID), bool)
                    for g in recent:
                        acc |= g > 0
                    state.img = acc[1].astype(np.uint8) * 2 | acc[0].astype(np.uint8)
                if now - t_fps >= 1.0:
                    state.fps = n_fps / (now - t_fps)
                    state.rtt = round(float(np.median(rtts[-100:])), 2) if rtts else -1.0
                    state.minutes = (time.time() - t_start) / 60
                    t_fps, n_fps = now, 0
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        state.done = True
        score.update(10 ** 9)
        rec.close()
        summary = {"session": str(rec.out), "frames": state.k + 1, "presses": state.n_press, **score.as_dict()}
        if rtts:
            summary["rtt_ms_median"] = round(float(np.median(rtts)), 2)
        (rec.out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

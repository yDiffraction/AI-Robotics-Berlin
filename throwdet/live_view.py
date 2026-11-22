#!/usr/bin/env python3
"""Live view of the GenX320 camera on the Kria with the throw-motion detector overlaid.

The board streams raw events to this laptop (despot_debug_camera, the same sender that
`river-tools capture` and tools/record_tool.py use). This script bins them exactly like the
training data (10 ms frames, 4x4 blocks -> binary 2x80x80), runs the detector on every frame and
shows the event image with the decision on top: THROW MOTION (red) or "no throw motion" (green),
the score trace with the alarm threshold, and the alarms.

Two inference backends:
  --backend mac     the float network (model.pt) in PyTorch on this laptop. Works with any
                    firmware on the board; good to judge a model before deploying it.
  --backend board   every frame is sent to the SPLEAT server on the Kria (TCP 3333, CBOR
                    {"id", "payload": float32[2,80,80]}) and the FPGA returns the output spikes.
                    Needs the 2-class firmware and the model's weights.txt on the board
                    (throwdet/deploy_board.sh).

The decision rule is the one of the robot: throwdet/live_readout.py with the model's
readout.json (score = EMA of spike[1] - spike[0], alarm when it stays above the threshold).

  .venv/bin/python throwdet/live_view.py                                  # model D, mac backend
  .venv/bin/python throwdet/live_view.py --model throwdet/models/A_neg --backend board
  .venv/bin/python throwdet/live_view.py --replay dataset/01/samples/sample_x.bin   # no camera
  .venv/bin/python throwdet/live_view.py --save takes/live.bin            # also record the events

Stop despot_send_and_forget on the board first: both apps need the camera.
Close the window (or press q) to stop; the camera sender on the board is stopped with it.
"""
import argparse
import faulthandler
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from live_readout import ThrowReadout  # noqa: E402

EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
SENSOR, HW, BIN_US = 320, 80, 10_000
BLOCK = SENSOR // HW
HEADER = struct.Struct("!I")
ROOT = Path(__file__).resolve().parent
LOG_PATH = Path("/tmp/live_view.log")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ event sources

def start_camera(host, port):
    """Start despot_debug_camera on the board; it connects back to this laptop on `port`."""
    command = f"""set -- $SSH_CONNECTION
despot_debug_camera --host "$1" --port {port} >/tmp/live_view_camera.log 2>&1 &
camera_pid=$!
trap 'kill "$camera_pid" 2>/dev/null; wait "$camera_pid" 2>/dev/null' EXIT
trap 'exit' HUP INT TERM
read -r stop
"""
    return subprocess.Popen(["ssh", "-4", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                             "-o", "ConnectTimeout=10", "--", host, command], stdin=subprocess.PIPE)


def stop_camera(proc):
    try:
        proc.communicate(input=b"\n", timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def recv_exact(sock, n, stop):
    buf = bytearray()
    while len(buf) < n:
        if stop.is_set():
            return None
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def camera_source(args, sink, state, stop):
    """Receive event batches from the board. If the camera sender stops or the connection drops
    (e.g. the camera is shaken hard, the cable is touched), the sender is restarted."""
    with socket.create_server(("", args.listen_port)) as server:
        server.settimeout(0.5)
        restarts = 0
        while not stop.is_set():
            proc = start_camera(args.host, args.listen_port)
            state["status"] = f"waiting for the camera on {args.host}" + (f" (restart {restarts})" if restarts else "")
            try:
                while not stop.is_set() and proc.poll() is None:
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    state["status"] = "camera connected" + (f" (restarts: {restarts})" if restarts else "")
                    with conn:
                        conn.settimeout(0.5)
                        while not stop.is_set():
                            head = recv_exact(conn, HEADER.size, stop)
                            data = head and recv_exact(conn, HEADER.unpack(head)[0], stop)
                            if data is None:
                                break
                            data = data[:len(data) // EVENT_DTYPE.itemsize * EVENT_DTYPE.itemsize]
                            if data:
                                sink(np.frombuffer(data, EVENT_DTYPE))
                    log(f"camera connection closed after {state.get('frames', 0)} frames")
                    break
            except Exception:
                log("camera receive error:\n" + traceback.format_exc())
            finally:
                stop_camera(proc)
            if not stop.is_set():
                restarts += 1
                state["status"] = f"camera stopped, restarting the sender (restart {restarts})"
                log(f"camera sender ended (exit {proc.returncode}), restarting")
                time.sleep(1.0)


def replay_source(args, sink, state, stop):
    raw = Path(args.replay).read_bytes()
    n = HEADER.unpack(raw[:4])[0]
    ev = np.frombuffer(raw, EVENT_DTYPE, offset=4, count=n // EVENT_DTYPE.itemsize)
    t = ev["t"].astype(np.int64)
    state["status"] = f"replay {Path(args.replay).name} ({(t[-1] - t[0]) / 1e6:.1f} s)"
    while not stop.is_set():
        start_wall, t0 = time.monotonic(), int(t[0])
        edges = np.searchsorted(t, np.arange(t0, int(t[-1]) + 20_000, 20_000))
        for a, b in zip(edges[:-1], edges[1:]):
            if stop.is_set():
                return
            due = start_wall + (int(t[a]) - t0) / 1e6 if a < len(t) else 0
            time.sleep(max(0.0, due - time.monotonic()))
            if b > a:
                sink(ev[a:b])
        if not args.loop:
            state["status"] += " - finished"
            return
        state["reset"] = True


# ------------------------------------------------------------------ binning

class Binner:
    """Raw events -> binary [2, 80, 80] frames of 10 ms, as throwdet/prep.py encodes the training clips."""

    def __init__(self, out, display, save=None):
        self.out, self.display, self.save = out, display, save
        self.frame_idx = None
        self.cur = np.zeros((2, HW, HW), np.uint8)
        self.lock = threading.Lock()

    def __call__(self, ev):
        ev = ev[(ev["x"] < SENSOR) & (ev["y"] < SENSOR) & (ev["p"] < 2)]
        if not len(ev):
            return
        if self.save is not None:
            self.save.write(ev)
        self.display(ev)
        f = ev["t"].astype(np.int64) // BIN_US
        if self.frame_idx is None or f[0] < self.frame_idx - 100 or f[0] > self.frame_idx + 200:
            # first batch, the camera clock restarted (sender restart) or jumped: start over
            self.frame_idx = int(f[0])
            self.cur[:] = 0
        p, y, x = ev["p"].astype(np.int64), ev["y"].astype(np.int64) // BLOCK, ev["x"].astype(np.int64) // BLOCK
        for fi in np.unique(f):
            fi = int(fi)
            while fi > self.frame_idx:  # close the current frame (and empty frames in a gap)
                self.out.put(self.cur.copy())
                self.cur[:] = 0
                self.frame_idx += 1
            if fi == self.frame_idx:
                m = f == fi
                self.cur[p[m], y[m], x[m]] = 1


class Saver:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "wb")
        self.fh.write(HEADER.pack(0))
        self.size = 0

    def write(self, ev):
        b = ev.astype(EVENT_DTYPE).tobytes()
        self.fh.write(b)
        self.size += len(b)

    def close(self):
        self.fh.seek(0)
        self.fh.write(HEADER.pack(self.size))
        self.fh.close()
        print(f"saved {self.size // EVENT_DTYPE.itemsize} events to {self.path}")


# ------------------------------------------------------------------ inference backends

def make_stateful(net, torch):
    """riverops neurons start every forward() call from zero potentials. For a live stream the
    potentials are carried over from call to call instead, as on the FPGA, so the network can be
    fed a few frames at a time and gives the same spikes as one call on the whole stream."""
    from riverops.nn.neuron import IFNode, PLIFNode
    nodes = [m for m in net.modules() if isinstance(m, (IFNode, PLIFNode))]

    def stateful_forward(self, x):
        pot = getattr(self, "_live_potentials", None)
        if pot is None or pot.shape != x[:, 0].shape:
            pot = torch.zeros_like(x[:, 0])
        spikes = []
        for x_t in x.unbind(dim=1):
            s, pot = self._step_forward(x_t, pot)
            spikes.append(s)
        self._live_potentials = pot
        return torch.stack(spikes, dim=1)

    for m in nodes:
        m.forward = stateful_forward.__get__(m)
    return nodes


class MacBackend:
    """Float network (model.pt) in PyTorch, neuron potentials kept across frames like on the FPGA."""

    def __init__(self, model_dir, device="cpu"):
        import torch
        from model import ThrowSNN
        self.torch = torch
        torch.set_num_threads(2)
        ck = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=False)
        self.net = ThrowSNN(2, arch=ck["arch"], tau=ck["args"]["tau"])
        self.net.load_state_dict(ck["model"])
        self.net.eval().to(device)
        self.nodes = make_stateful(self.net, torch)
        self.dev = device

    def reset(self):
        for m in self.nodes:
            m._live_potentials = None

    def run(self, frames):
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack(frames))[None].float().to(self.dev)
            return self.net(x)[0].cpu().numpy()


class BoardBackend:
    """Send every frame to the SPLEAT server on the Kria and read back the output spikes."""

    def __init__(self, host, port):
        self.addr = (host.split("@")[-1], port)
        self.sock = None
        self.reconnect()
        self.msg_id = 0
        import cbor2
        self.cbor2 = cbor2
        self.warned = False

    def reconnect(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = socket.create_connection(self.addr, timeout=5)
        self.sock.settimeout(5)

    def _rx(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("SPLEAT server closed the connection")
            buf += chunk
        return bytes(buf)

    def run(self, frames):
        out = []
        for fr in frames:
            self.msg_id += 1
            msg = self.cbor2.dumps({"id": self.msg_id, "payload": fr.astype("<f4").tobytes()})
            self.sock.sendall(HEADER.pack(len(msg)) + msg)
            while True:  # the server broadcasts to all clients; take the reply to our message
                reply = self.cbor2.loads(self._rx(HEADER.unpack(self._rx(4))[0]))
                if reply.get("id") == self.msg_id:
                    break
            if reply.get("error"):
                raise RuntimeError(f"SPLEAT server error: {reply['error']}")
            s = np.frombuffer(reply.get("payload") or b"", "<f4")
            if len(s) < 2:
                if not self.warned:
                    print(f"WARNING: the board returns {len(s)} outputs, the throw detector needs 2.", flush=True)
                    self.warned = True
                s = np.zeros(2, np.float32)
            s = s[:2]  # 11-output firmware: outputs 0 and 1 are the detector (export.py --pad-outputs 11)
            out.append(s)
        return np.array(out)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models/D_e10_fold3"), help="model folder (model.pt, readout.json)")
    ap.add_argument("--readout", default=None, help="readout json (default: <model>/readout.json)")
    ap.add_argument("--backend", choices=["mac", "board"], default="mac")
    ap.add_argument("--host", default="root@192.168.77.1", help="ssh target of the Kria")
    ap.add_argument("--listen-port", type=int, default=3333, help="port on this laptop the camera sender connects to")
    ap.add_argument("--spleat-port", type=int, default=3333, help="SPLEAT server port on the board")
    ap.add_argument("--replay", default=None, help="play a recorded .bin in real time instead of the camera")
    ap.add_argument("--loop", action="store_true", help="repeat the replay")
    ap.add_argument("--save", default=None, help="also write the received events to this .bin")
    ap.add_argument("--window-s", type=float, default=6.0, help="seconds of score history shown")
    ap.add_argument("--duration", type=float, default=None, help="close the window after this many seconds")
    ap.add_argument("--screenshot", default=None, help="save the window as PNG when it closes")
    args = ap.parse_args()
    fault_log = open(LOG_PATH, "a")
    faulthandler.enable(fault_log)  # a hard crash leaves its stack in /tmp/live_view.log
    log(f"start: {' '.join(sys.argv[1:])}")

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    readout_path = args.readout or str(Path(args.model) / "readout.json")
    ro = ThrowReadout(readout_path)
    backend = MacBackend(args.model) if args.backend == "mac" else BoardBackend(args.host, args.spleat_port)
    print(f"model {Path(args.model).name}, backend {args.backend}, readout alpha {ro.alpha} "
          f"threshold {ro.threshold} hold {ro.hold}", flush=True)

    stop = threading.Event()
    frames_q = queue.Queue()
    state = {"status": "starting", "reset": False}
    img_lock = threading.Lock()
    img_acc = np.zeros((2, SENSOR, SENSOR), np.uint16)

    def display(ev):
        with img_lock:
            np.add.at(img_acc, (ev["p"].astype(np.int64), ev["y"].astype(np.int64), ev["x"].astype(np.int64)), 1)

    saver = Saver(args.save) if args.save else None
    binner = Binner(frames_q, display, saver)
    n_hist = int(args.window_s * 100)
    hist = {"score": deque([0.0] * n_hist, maxlen=n_hist), "alarm": deque([False] * n_hist, maxlen=n_hist)}
    live = {"score": 0.0, "last_alarm": -1e9, "alarms": 0, "frames": 0, "lag_ms": 0.0, "infer_ms": 0.0, "fps": 0.0}
    lock = threading.Lock()

    def infer():
        t_fps, n_fps = time.monotonic(), 0
        while not stop.is_set():
            try:
                frames = [frames_q.get(timeout=0.2)]
            except queue.Empty:
                continue
            while len(frames) < 50:
                try:
                    frames.append(frames_q.get_nowait())
                except queue.Empty:
                    break
            if state.get("reset"):
                ro.score, ro.run = 0.0, 0
                if hasattr(backend, "reset"):
                    backend.reset()
                state["reset"] = False
            t0 = time.monotonic()
            try:
                sp = backend.run(frames)
            except Exception as e:  # keep the window alive, show the error, reconnect
                state["status"] = f"inference error: {e}"
                log("inference error:\n" + traceback.format_exc())
                time.sleep(0.5)
                if hasattr(backend, "reconnect"):
                    try:
                        backend.reconnect()
                    except Exception as e2:
                        log(f"reconnect failed: {e2}")
                continue
            dt = (time.monotonic() - t0) * 1000
            with lock:
                for s0, s1 in sp:
                    fired = ro.update(s0, s1)
                    hist["score"].append(ro.score)
                    hist["alarm"].append(fired)
                    if fired:
                        live["alarms"] += 1
                        live["last_alarm"] = time.monotonic()
                        print(f"ALARM {live['alarms']} at {time.strftime('%H:%M:%S')}, score {ro.score:+.3f}", flush=True)
                live["score"] = ro.score
                live["frames"] += len(frames)
                live["infer_ms"] = dt / len(frames)
                live["lag_ms"] = frames_q.qsize() * 10.0
                n_fps += len(frames)
                if time.monotonic() - t_fps > 1:
                    live["fps"] = n_fps / (time.monotonic() - t_fps)
                    t_fps, n_fps = time.monotonic(), 0

    source = replay_source if args.replay else camera_source
    threads = [threading.Thread(target=source, args=(args, binner, state, stop), daemon=True),
               threading.Thread(target=infer, daemon=True)]
    for th in threads:
        th.start()

    # ---------------- window
    matplotlib.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(11, 6.6), facecolor="black")
    fig.canvas.manager.set_window_title("Throw-motion detector - live")
    ax_img = fig.add_axes([0.02, 0.05, 0.5, 0.83])
    ax_sc = fig.add_axes([0.58, 0.12, 0.39, 0.5], facecolor="#111111")
    ax_img.set_axis_off()
    im = ax_img.imshow(np.zeros((SENSOR, SENSOR, 3), np.uint8), interpolation="nearest")
    banner = fig.text(0.27, 0.93, "", ha="center", va="center", fontsize=26, weight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.4", fc="#1f6f3f", ec="none"))
    info = fig.text(0.58, 0.88, "", ha="left", va="top", fontsize=10, color="#dddddd", family="monospace")
    status = fig.text(0.02, 0.01, "", ha="left", va="bottom", fontsize=9, color="#aaaaaa")
    tx = np.arange(-n_hist, 0) / 100.0
    line, = ax_sc.plot(tx, np.zeros(n_hist), color="#4fc3f7", lw=1.5)
    ax_sc.axhline(ro.threshold, color="#ff5252", ls="--", lw=1)
    ax_sc.text(tx[0], ro.threshold, " alarm threshold", color="#ff5252", va="bottom", fontsize=8)
    alarm_marks = ax_sc.scatter([], [], color="#ff5252", s=40, zorder=3)
    ax_sc.set_xlim(tx[0], 0)
    ax_sc.set_ylim(-1.05, 1.05)
    ax_sc.set_xlabel("seconds", color="#aaaaaa")
    ax_sc.set_ylabel("score  EMA(spike[throw] - spike[none])", color="#aaaaaa", fontsize=9)
    ax_sc.tick_params(colors="#aaaaaa")
    for s in ax_sc.spines.values():
        s.set_color("#444444")
    on_c, off_c = np.array([255, 255, 255], np.float32), np.array([60, 140, 255], np.float32)
    decay = np.zeros((SENSOR, SENSOR, 3), np.float32)

    def draw(_):
        try:
            return _draw()
        except Exception:
            log("draw error:\n" + traceback.format_exc())
            return ()

    def _draw():
        nonlocal decay
        with img_lock:
            acc = img_acc.copy()
            img_acc[:] = 0
        decay *= 0.55
        decay[decay < 1] = 0
        decay[acc[0] > 0] = off_c
        decay[acc[1] > 0] = on_c
        im.set_data(decay.astype(np.uint8))
        with lock:
            sc = np.array(hist["score"])
            al = np.array(hist["alarm"])
            lv = dict(live)
        line.set_ydata(sc)
        k = np.flatnonzero(al)
        alarm_marks.set_offsets(np.c_[tx[k], sc[k]] if len(k) else np.zeros((0, 2)))
        recent_alarm = time.monotonic() - lv["last_alarm"] < 1.0
        above = lv["score"] > ro.threshold
        if recent_alarm:
            banner.set_text("THROW MOTION  - ALARM")
            banner.get_bbox_patch().set_facecolor("#d32f2f")
        elif above:
            banner.set_text("throw motion?")
            banner.get_bbox_patch().set_facecolor("#ef6c00")
        else:
            banner.set_text("no throw motion")
            banner.get_bbox_patch().set_facecolor("#1f6f3f")
        info.set_text(f"model    {Path(args.model).name}  ({args.backend})\n"
                      f"score    {lv['score']:+.3f}   threshold {ro.threshold:+.3f}  hold {ro.hold}\n"
                      f"alarms   {lv['alarms']}\n"
                      f"frames   {lv['frames']}   {lv['fps']:.0f} frames/s (camera: 100)\n"
                      f"compute  {lv['infer_ms']:.2f} ms/frame   backlog {lv['lag_ms']:.0f} ms")
        status.set_text(state["status"])
        return im, line, alarm_marks, banner, info, status

    done = threading.Event()

    def shutdown(*_):
        """Stop the camera sender on the board, save the recording, then quit (also when the
        GUI event loop does not return after the window is closed, as with the macOS backend)."""
        if done.is_set():
            return
        done.set()
        if args.screenshot:
            fig.savefig(args.screenshot, facecolor="black")
        stop.set()
        for th in threads:
            th.join(timeout=6)
        if saver is not None:
            saver.close()
        print(f"stopped: {live['alarms']} alarms in {live['frames'] / 100:.1f} s", flush=True)
        os._exit(0)

    def on_key(e):
        if e.key in ("q", "escape"):
            shutdown()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", shutdown)
    signal.signal(signal.SIGINT, lambda *_: shutdown())
    if args.duration:
        timer = fig.canvas.new_timer(interval=int(args.duration * 1000))
        timer.single_shot = True
        timer.add_callback(shutdown)
        timer.start()
    _anim = FuncAnimation(fig, draw, interval=40, cache_frame_data=False)  # noqa: F841 (keep a reference)
    try:
        plt.show()
    finally:
        shutdown()


if __name__ == "__main__":
    main()

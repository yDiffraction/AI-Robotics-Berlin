#!/usr/bin/env python3
"""Ball-dodge host loop: events -> binary 10 ms frames -> SNN on the Kria (SPLEAT server) -> BallTrigger -> events.

The laptop builds the frames itself (same binning as the recordings: 80x80 cells, 2 polarities, cell = 1 if any
event, 10 ms), so the board sees exactly the input format the model was trained on; this replaces
despot_send_and_forget (which sends event counts and only reached ~47 frames/s live). Rows >= --roi-rows are
zeroed for ROI models.

Sources
  camera (default)  listens on --listen-port for despot_debug_camera (started over SSH with river_tools, or by
                    hand with --no-start); stop record_tool.py / live_vis.py first, they use the same port.
  --file X.bin      a whole recording (record_tool .bin), replayed through the board. Virtual clock (frame
                    index x 10 ms) unless --realtime. With --labels (the take's labels.csv: file,label,
                    decision_s) the triggers are scored against the labelled throws / feints.

Outputs: every arm / dash / disarm is printed, sent as JSON over UDP (--udp host:port, for the ROS node
go2_sim_tools/ball_trigger_udp.py -> /reflex/arm|dash|disarm) and logged; --save writes the per-frame spikes
(npz) so thresholds can be swept offline (--sweep, no board needed).

  python tools/ball_live.py --roi-rows 16 --theta 3 --udp 192.168.77.1:5555
  python tools/ball_live.py --file dataset/03/samples/sample_20261121_200828_124951.bin \
         --labels dataset/03/samples/labels.csv --roi-rows 16 --save runs/ball/stream/x.npz
  python tools/ball_live.py --sweep runs/ball/stream/x.npz --labels dataset/03/samples/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_trigger_logic import ARM, DASH, DISARM, BallTrigger, TriggerParams  # noqa: E402

SENSOR, GRID, FRAME_US = 320, 80, 10_000
REC = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])     # record_tool .bin (4-byte header)


# ----------------------------------------------------------------------------- CBOR / SPLEAT client
def cbor_head(major, value):
    if value < 24:
        return bytes([(major << 5) | value])
    for info, fmt in ((24, ">B"), (25, ">H"), (26, ">I"), (27, ">Q")):
        if value < 1 << (8 * struct.calcsize(fmt)):
            return bytes([(major << 5) | info]) + struct.pack(fmt, value)
    raise ValueError(value)


def cbor_decode(b, i=0):
    major, info = b[i] >> 5, b[i] & 31
    i += 1
    if info < 24:
        value = info
    else:
        size = 1 << (info - 24)
        value = int.from_bytes(b[i:i + size], "big")
        i += size
    if major == 0:
        return value, i
    if major == 1:
        return -1 - value, i
    if major in (2, 3):
        data = bytes(b[i:i + value])
        return (data if major == 2 else data.decode()), i + value
    if major == 5:
        out = {}
        for _ in range(value):
            key, i = cbor_decode(b, i)
            out[key], i = cbor_decode(b, i)
        return out, i
    if major == 7:
        if info == 26:
            return struct.unpack(">f", value.to_bytes(4, "big"))[0], i
        if info == 27:
            return struct.unpack(">d", value.to_bytes(8, "big"))[0], i
        return {20: False, 21: True, 22: None}.get(info, value), i
    return None, i


class Spleat:
    """One request per frame, wait for its result (the server broadcasts results; ours carries our id)."""

    def __init__(self, host, port=3333):
        self.addr = (host, port)
        self.sock = None
        self.next_id = 0
        self.connect()

    def connect(self):
        self.sock = socket.create_connection(self.addr, timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(1.0)

    def _rx(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise ConnectionError("SPLEAT server closed the connection")
            buf += c
        return bytes(buf)

    def infer(self, frame):
        fid, self.next_id = self.next_id, self.next_id + 1
        payload = frame.astype("<f4").tobytes()
        body = (cbor_head(5, 2) + cbor_head(3, 2) + b"id" + cbor_head(0, fid)
                + cbor_head(3, 7) + b"payload" + cbor_head(2, len(payload)) + payload)
        self.sock.sendall(struct.pack(">I", len(body)) + body)
        while True:
            msg, _ = cbor_decode(self._rx(struct.unpack(">I", self._rx(4))[0]))
            if msg.get("id") == fid:
                if msg.get("error"):
                    raise ConnectionError(f"SPLEAT server error: {msg['error']}")
                return np.frombuffer(msg["payload"], "<f4"), fid, msg.get("latency", -1)


# ----------------------------------------------------------------------------- frames
def frames_from_file(path):
    """Whole recording -> (frame index array [N], binary frames generator). Same binning as the dataset clips."""
    raw = Path(path).read_bytes()
    ev = np.frombuffer(raw[4:4 + (len(raw) - 4) // REC.itemsize * REC.itemsize], dtype=REC)
    t = ev["t"].astype(np.int64)
    t0 = int(t.min())
    n = (t - t0) // FRAME_US
    order = np.argsort(n, kind="stable")
    n, ev = n[order], ev[order]
    cell = (np.minimum(ev["p"], 1).astype(np.int64) * GRID + np.minimum(ev["y"], SENSOR - 1) * GRID // SENSOR) * GRID \
        + np.minimum(ev["x"], SENSOR - 1) * GRID // SENSOR
    bounds = np.searchsorted(n, np.arange(int(n[-1]) + 2))
    n_frames = int(n[-1]) + 1

    def gen():
        for k in range(n_frames):
            f = np.zeros(2 * GRID * GRID, np.float32)
            f[cell[bounds[k]:bounds[k + 1]]] = 1.0
            yield f.reshape(2, GRID, GRID)
    return n_frames, gen()


def frames_from_camera(args):
    """despot_debug_camera -> binary frames, as tools/live_vis.py (empty frames for pauses, at most 20)."""
    from river_tools.capture.events import EVENT_DTYPE
    camera = None
    with socket.create_server(("", args.listen_port)) as server:
        if not args.no_start:
            from river_tools.capture.remote import start_camera
            camera = start_camera(args.ssh, args.listen_port)
        print(f"waiting for despot_debug_camera on port {args.listen_port}", flush=True)
        conn, addr = server.accept()
        print(f"camera connected from {addr[0]}", flush=True)
        frame = np.zeros(2 * GRID * GRID, np.float32)
        cur = None
        try:
            with conn:
                while True:
                    h = conn.recv(4, socket.MSG_WAITALL)
                    if len(h) < 4:
                        return
                    size = struct.unpack(">I", h)[0]
                    data = conn.recv(size, socket.MSG_WAITALL) if size else b""
                    if not data:
                        continue
                    ev = np.frombuffer(data, dtype=EVENT_DTYPE, count=len(data) // EVENT_DTYPE.itemsize)
                    x = np.minimum(ev["x"], SENSOR - 1).astype(np.intp)
                    y = np.minimum(ev["y"], SENSOR - 1).astype(np.intp)
                    p = np.minimum(ev["polarity"], 1).astype(np.intp)
                    cell = (p * GRID + y * GRID // SENSOR) * GRID + x * GRID // SENSOR
                    num = ev["timestamp"] // FRAME_US
                    for k in np.unique(num):
                        if cur is None:
                            cur = k
                        if k > cur:
                            yield frame.reshape(2, GRID, GRID)
                            for _ in range(min(int(k - cur - 1), 20)):
                                yield np.zeros((2, GRID, GRID), np.float32)
                            frame = np.zeros(2 * GRID * GRID, np.float32)
                            cur = k
                        frame[cell[num == k]] = 1.0
        finally:
            if camera is not None:
                from river_tools.capture.remote import stop_camera
                stop_camera(camera)


# ----------------------------------------------------------------------------- labels / scoring
def load_labels(path, rec_name):
    """labels.csv of a take (file,label,decision_s,...) -> list of (label, decision_s) for this recording."""
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("file") == rec_name and r.get("label") in ("throw", "feint"):
            out.append((r["label"], float(r["decision_s"])))
    return sorted(out, key=lambda v: v[1])


def score_events(dash_t, labels, duration, pre=0.4, post=0.8):
    """dash times vs labelled throws / feints (decision_s = swing peak, ~ the release).
    throw hit: a dash in [decision - pre, decision + post]; feint false: a dash in the same window;
    every other dash is an idle false trigger (per minute of the time outside all labelled windows)."""
    dash_t = np.asarray(dash_t, float)
    used = np.zeros(len(dash_t), bool)
    hits, lat, fe_fired = [], [], []
    for lab, d in labels:
        m = (dash_t >= d - pre) & (dash_t <= d + post)
        used |= m
        if lab == "throw":
            hits.append(bool(m.any()))
            if m.any():
                lat.append(float(dash_t[m][0] - d))
        else:
            fe_fired.append(bool(m.any()))
    covered = len(labels) * (pre + post)
    idle_min = max(duration - covered, 1e-9) / 60
    n_idle = int((~used).sum())
    return {"throws": len(hits), "throw_hit": int(sum(hits)), "feints": len(fe_fired), "feint_fired": int(sum(fe_fired)),
            "idle_dashes": n_idle, "idle_per_min": round(n_idle / idle_min, 2), "idle_minutes": round(idle_min, 2),
            "dash_minus_decision_s": {"median": float(np.median(lat)) if lat else None,
                                      "min": float(min(lat)) if lat else None, "max": float(max(lat)) if lat else None}}


def run_trigger(spikes, params, dt=FRAME_US / 1e6):
    """Offline: per-frame spikes [N, C] -> list of (event, t, frame)."""
    trig = BallTrigger(params)
    out = []
    for k, s in enumerate(spikes):
        t = k * dt
        for ev in trig.step([float(v) for v in s], t, k):
            out.append((ev if isinstance(ev, str) else ev[0], t, k))
    return out


class Combiner(object):
    """Dash policy over two per-frame candidates: snn (BallTrigger evidence >= theta, after min_fill) and tracker
    (causal_ball_tracker fired). mode snn | tracker | or | and (both within the last `and_window` frames).
    Refractory as in BallTrigger: after a dash nothing fires for `refractory` frames."""

    def __init__(self, mode, refractory_frames, and_window=15):
        self.mode, self.refr, self.and_window = mode, refractory_frames, and_window
        self.last_snn = self.last_tr = -10 ** 9
        self.until = -1

    def step(self, k, snn, tr):
        if snn:
            self.last_snn = k
        if tr:
            self.last_tr = k
        if k < self.until:
            return False
        fire = {"snn": snn, "tracker": tr, "or": snn or tr,
                "and": (snn or tr) and k - self.last_snn <= self.and_window and k - self.last_tr <= self.and_window}[self.mode]
        if fire:
            self.until = k + self.refr
        return fire


def combined_dashes(spikes, tr_fires, a):
    """Offline: dash frames of the combined policy (snn candidate = BallTrigger evidence >= theta)."""
    trig = BallTrigger(trigger_params(a, refractory=0.0))
    comb = Combiner(a.fire, int(a.refractory * 100), a.and_window)
    out = []
    for k, s in enumerate(spikes):
        trig.step([float(v) for v in s], k * FRAME_US / 1e6, k)
        snn = trig.fill >= trig.min_fill and trig.evidence >= a.theta - 1e-9
        if comb.step(k, snn, bool(tr_fires[k]) if tr_fires is not None else False):
            out.append(k)
    return out


def trigger_params(a, refractory=None):
    return TriggerParams(window=a.window, theta=a.theta, refractory=a.refractory if refractory is None else refractory,
                         n_channels=None,
                         swing_channel=a.swing_channel, watchdog=1e9 if a.file or a.sweep else a.watchdog)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.77.1", help="SPLEAT server (board)")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--ssh", default="root@192.168.77.1", help="SSH target for starting despot_debug_camera")
    ap.add_argument("--listen-port", type=int, default=3334)
    ap.add_argument("--no-start", action="store_true", help="despot_debug_camera is started by hand")
    ap.add_argument("--file", help="replay a whole recording instead of the camera")
    ap.add_argument("--realtime", action="store_true", help="--file: pace at 100 Hz")
    ap.add_argument("--labels", help="labels.csv of the take(s): file,label,decision_s")
    ap.add_argument("--roi-rows", type=int, default=80)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--theta", type=float, default=3.0)
    ap.add_argument("--refractory", type=float, default=2.0)
    ap.add_argument("--watchdog", type=float, default=0.2)
    ap.add_argument("--swing-channel", type=int, default=None)
    ap.add_argument("--udp", help="host:port to send trigger events to (JSON)")
    ap.add_argument("--save", help="npz with the per-frame output spikes, latency and round trip")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds of frames")
    ap.add_argument("--sweep", help="npz from --save: score thresholds offline (no board)")
    ap.add_argument("--thetas", default="1,2,3,4,5,6")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fire", default="snn", choices=["snn", "tracker", "or", "and"],
                    help="dash policy: SNN only (BallTrigger), the host's causal ball tracker only, or both combined")
    ap.add_argument("--and-window", type=int, default=15, help="frames within which both must fire (--fire and)")
    ap.add_argument("--no-board", action="store_true", help="no SNN (tracker only, --fire tracker)")
    a = ap.parse_args()

    if a.sweep:
        z = np.load(a.sweep, allow_pickle=True)
        rec = str(z["rec"])
        labels = load_labels(a.labels, rec) if a.labels else []
        dur = len(z["out"]) * FRAME_US / 1e6
        res = {}
        trf = z["tracker"] if "tracker" in z.files else None
        for th in [float(v) for v in a.thetas.split(",")]:
            a.theta = th
            if a.fire == "snn":
                ev = run_trigger(z["out"], trigger_params(a))
                dash = [t for e, t, _ in ev if e == DASH]
            else:
                dash = [k * FRAME_US / 1e6 for k in combined_dashes(z["out"], trf, a)]
            res[th] = score_events(dash, labels, dur)
            print(f"{a.fire} theta {th:4.1f}: {json.dumps(res[th])}")
            if a.fire == "tracker":
                break
        return

    udp = None
    if a.udp:
        uh, up = a.udp.rsplit(":", 1)
        udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (uh, int(up)))
    if a.no_board and a.fire != "tracker":
        raise SystemExit("--no-board needs --fire tracker")
    board = None if a.no_board else Spleat(a.host, a.port)
    trig = BallTrigger(trigger_params(a, refractory=None if a.fire == "snn" else 0.0))
    tracker = None
    if a.fire != "snn" or a.save:
        from causal_ball_tracker import CausalBallTracker
        tracker = CausalBallTracker()
    comb = Combiner(a.fire, int(a.refractory * 100), a.and_window)
    tr_fires = []
    if a.file:
        n_frames, frames = frames_from_file(a.file)
        print(f"{a.file}: {n_frames} frames ({n_frames / 100 / 60:.1f} min)", flush=True)
    else:
        frames = frames_from_camera(a)
    outs, lats, rtts, events = [], [], [], []
    t_start = time.perf_counter()
    for k, f in enumerate(frames):
        if a.seconds and k >= a.seconds * 100:
            break
        trf = bool(tracker.step(f.any(0))) if tracker is not None else False
        tr_fires.append(trf)
        if a.roi_rows < GRID:
            f = f.copy()
            f[:, a.roi_rows:, :] = 0
        t0 = time.perf_counter()
        if board is not None:
            out, fid, lat = board.infer(f)
        else:
            out, fid, lat = np.zeros(11, np.float32), k, 0
        t1 = time.perf_counter()
        rtts.append(t1 - t0)
        lats.append(lat)
        outs.append(out.copy())
        t = k * FRAME_US / 1e6 if a.file and not a.realtime else t1 - t_start
        evs = trig.step([float(v) for v in out], t, fid) if board is not None else []
        if a.fire != "snn":
            snn = board is not None and trig.fill >= trig.min_fill and trig.evidence >= a.theta - 1e-9
            evs = [ARM, DASH] if comb.step(k, snn, trf) else []
        for ev in evs:
            name = ev if isinstance(ev, str) else ev[0]
            msg = {"event": name, "t": round(t, 3), "frame": k, "host_time": time.time()}
            events.append(msg)
            if not a.quiet:
                print(json.dumps(msg), flush=True)
            if udp:
                udp[0].sendto(json.dumps(msg).encode(), udp[1])
        if a.file and a.realtime:
            time.sleep(max(0.0, t_start + (k + 1) * FRAME_US / 1e6 - time.perf_counter()))
    wall = time.perf_counter() - t_start
    outs = np.array(outs)
    rtts = np.array(rtts) * 1e3
    lats = np.array(lats)
    summary = {"frames": len(outs), "fps": round(len(outs) / wall, 1),
               "rtt_ms": {"median": round(float(np.median(rtts)), 2), "p99": round(float(np.percentile(rtts, 99)), 2)},
               "board_latency_us": {"median": float(np.median(lats)), "p99": float(np.percentile(lats, 99))},
               "spikes_per_channel": outs.sum(0).astype(int).tolist(), "events": len(events)}
    if a.file and a.labels:
        labels = load_labels(a.labels, Path(a.file).name)
        summary["score"] = score_events([e["t"] for e in events if e["event"] == DASH], labels, len(outs) / 100)
    print(json.dumps(summary, indent=1), flush=True)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save, out=outs, latency_us=lats, rtt_ms=rtts, rec=Path(a.file).name if a.file else "camera",
                            tracker=np.array(tr_fires, bool),
                            roi_rows=a.roi_rows, events=json.dumps(events), summary=json.dumps(summary))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Throw Reflex bridge: event camera -> SNN on the FPGA -> throw detection -> /reflex/dash, plus the web UI.

One self-contained file (Python 3.10+ stdlib, numpy, rclpy; copy it to the Kria next to go2_dodge.py and
model/readout.json; robot.launch.py starts both nodes):
  camera   despot_debug_camera streams the GenX320 events; they are binned into the frames the model was trained
           on: 10 ms, 2 polarities x 80x80 cells, cell = 1 if any event of that polarity in its 4x4 pixel block
  SNN      every frame goes to the SPLEAT server (the spiking CNN on the FPGA, TCP 3333) -> output spikes
  readout  ema   y = alpha * y + (1 - alpha) * spike[channel] per frame, alarm on the upward crossing of theta
                 (Tim's v3 early throw-motion detector: channel 0, alpha 0.85, theta 0.85 PROVISIONAL)
           diff  d = sum over the last `window` frames of (spike[channel] - spike[neg_channel]), alarm when
                 d >= theta (2-output ball models)
           then `refractory` s without alarms. An alarm is a detection: /snn/detection and the web page get it.
  reflex   a detection publishes /reflex/dash (go2_dodge.py dodges) ONLY while auto-dodge is on. It starts off;
           Y on the Go2 remote toggles it, so do the web page and /snn/auto_set. X stays the manual dodge
           (handled by go2_dodge.py).
  web      http://<host>:8080 (dodge/web/): camera feed, SNN spikes, readout vs theta, robot state, the switch.

Run (ROS 2 Humble + the go2-kria overlay; normally through robot.launch.py):
  python3 snn_bridge.py                                              # on the Kria: camera and FPGA local
  python3 snn_bridge.py --ros-args -p camera:=ssh -p spleat_host:=192.168.77.1     # on the laptop, board on LAN
  python3 snn_bridge.py --no-ros --ros-args -p camera:=replay:rec.bin -p snn:=mock  # desk test, no ROS at all
--no-ros runs everything except ROS (auto-dodge then only switches on the page and nothing is published).

Parameters (--ros-args -p name:=value; '' = automatic):
  camera        local    start `despot_debug_camera --host 127.0.0.1 --port <listen_port>` here, stop it on exit
                ssh      start it on ssh_host over SSH (it connects back to this machine), as river_tools does
                listen   only listen on listen_port (camera started by hand)
                replay:<file.bin>  play a record_tool recording in real time, looped
  listen_port 3334, ssh_host root@192.168.77.1, camera_bin despot_debug_camera
  snn           fpga (SPLEAT server spleat_host:spleat_port, 127.0.0.1:3333) | mock (no board: ch0 spikes while
                >= 6 cells are active in the rows < 16 of the 80x80 frame within the last 2 frames, held for
                mock_hold frames like an SNN burst, plus a little noise; reported as mock everywhere)
  readout_file  model/readout.json next to this script; readout_mode ema|diff and theta default to it
  theta, alpha, channel 0, neg_channel 1 (diff), window 10 (diff), refractory 1.0 s
  stale 0.3     s: no camera frame for longer (or the FPGA lost) resets the readout to 0; a frame older than this
                never publishes /reflex/dash
  auto false    auto-dodge at start
  remote both   where Y is read: lowstate (/lowstate wireless_remote bytes 2-3) | wirelesscontroller | both | none
  toggle_button y, controller_topic /wirelesscontroller
  web_host 0.0.0.0, web_port 8080, web_dir web/ next to this script
  color false   poll the Go2 front camera (videohub, api_id 1001) at color_hz 5 for the page

ROS: publishes /snn/detection (std_msgs/String JSON {t, k, y, theta, acted, reason}), /reflex/dash
(std_msgs/Empty, only auto on + fresh data), /snn/auto (std_msgs/Bool, transient local, at start and on change),
/api/videohub/request (color on); subscribes /snn/auto_set (std_msgs/Bool), /lowstate and /wirelesscontroller (Y),
/dodge/event (go2_dodge.py state, shown on the page), /api/videohub/response.

Web API (the page in web/ depends on it):
  GET  /api/state   snapshot {model:{name, readout, theta, channel, mock, ...}, camera:{state: streaming|waiting|
                    replay|offline, rate (events/s)}, fpga:{state: connected|offline|mock, latency_ms, fps},
                    ros:{available, auto}, robot:{state: idle|dash|hold|return|unknown, last: last /dodge/event},
                    color:{enabled, available, seq}, counters:{detections, dodges}, recent:[...], server_time}
                    (numbers are 0 while unknown, never null)
  GET  /stream      server-sent events, data = JSON:
                    frame      <= 25 Hz {t, w:160, h:160, on, off, rate}: on / off = base64 of the 160x160 bitmap
                               (sensor / 2, row-major, 1 bit per pixel, MSB first, 3200 bytes) of the events since
                               the previous frame message (~40 ms)
                    snn        <= 25 Hz {frames:[{k, s:[ch0, ch1, ch2], y}], theta, latency_ms, fps}: all SNN frames
                               since the previous message (a new page first gets the last 6 s)
                    detection  {t, k, y, theta, acted, reason}  (acted = /reflex/dash was published)
                    status     1 Hz and on any change: the /api/state snapshot
                    robot      a /dodge/event, forwarded
                    color      {seq} when a new color JPEG is there
  POST /api/auto {enabled}, /api/theta {theta}, /api/color {enabled}  -> the snapshot
  GET  /color.jpg   latest color JPEG (404 if none);  GET / and any file in web_dir: the page
"""
from __future__ import annotations

import atexit
import base64
import json
import math
import os
import queue
import random
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
SENSOR, GRID, DISP = 320, 80, 160      # sensor pixels, SNN cells, display pixels per side
FRAME_US = 10_000                       # one SNN frame = 10 ms of camera time
EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])  # despot stream and .bin files
MAX_GAP = 20                            # empty frames sent for a pause in the events (as tools/ball_live.py)
HISTORY = 600                           # SNN frames kept for a page that connects (6 s)
TICK = 0.04                             # web update period (25 Hz)
REPLAY_CHUNK_US = 5_000                 # replay: event batch size
CAMERA_LOG = "/tmp/snn_bridge_camera.log"
VIDEOHUB_GET_IMAGE = 1001               # Go2 videohub api_id: one JPEG of the front camera
# bits of WirelessController.keys / LowState.wireless_remote[2:4] (as go2_dodge.py)
BUTTONS = {"r1": 0, "l1": 1, "start": 2, "select": 3, "r2": 4, "l2": 5, "f1": 6, "f2": 7,
           "a": 8, "b": 9, "x": 10, "y": 11, "up": 12, "right": 13, "down": 14, "left": 15}

DEFAULTS = {
    "camera": "local", "listen_port": 3334, "ssh_host": "root@192.168.77.1", "camera_bin": "despot_debug_camera",
    "snn": "fpga", "spleat_host": "127.0.0.1", "spleat_port": 3333, "mock_hold": 15,
    "readout_file": "", "readout_mode": "", "theta": "", "alpha": "", "channel": 0, "neg_channel": 1,
    "window": 10, "refractory": 1.0, "stale": 0.3, "model_name": "",
    "auto": False, "remote": "both", "toggle_button": "y", "controller_topic": "/wirelesscontroller",
    "web_host": "0.0.0.0", "web_port": 8080, "web_dir": "",
    "color": False, "color_hz": 5.0,
}


# ----------------------------------------------------------------------------- parameters
def coerce(name, value, default):
    """A parameter value (any type from the command line) -> the type of its default."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes", "on"):
                    return True
                if low in ("false", "0", "no", "off", ""):
                    return False
                raise ValueError(value)
            return bool(value)
        if isinstance(default, int):
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(value)
            return int(value)
        if isinstance(default, float):
            return float(value)
        return "" if value is None else str(value)
    except (TypeError, ValueError):
        raise SystemExit("parameter %s: %r is not a %s" % (name, value, type(default).__name__))


def scalar(text):
    """YAML-like scalar of a `name:=value` override: true/false, int, float, quoted or plain string."""
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def parse_cli_params(argv):
    """`-p name:=value` pairs (as after --ros-args) -> {name: value}; used with --no-ros."""
    out, i = {}, 0
    while i < len(argv):
        if argv[i] in ("-p", "--param") and i + 1 < len(argv):
            name, sep, value = argv[i + 1].partition(":=")
            if not sep:
                raise SystemExit("bad parameter %r, use -p name:=value" % argv[i + 1])
            out[name.strip()] = scalar(value)
            i += 2
        else:
            i += 1
    return out


class Config(object):
    """All parameters, typed like DEFAULTS. Raises SystemExit with a readable message on a bad value."""

    def __init__(self, values=None):
        values = dict(values or {})
        unknown = sorted(set(values) - set(DEFAULTS))
        if unknown:
            raise SystemExit("unknown parameter(s) %s; known: %s" % (", ".join(unknown), ", ".join(DEFAULTS)))
        for name, default in DEFAULTS.items():
            setattr(self, name, coerce(name, values.get(name, default), default))
        self.camera_mode, _, self.replay_path = self.camera.partition(":")
        need = []
        if self.camera_mode not in ("local", "ssh", "listen", "replay") or (self.camera_mode == "replay"
                                                                              and not self.replay_path):
            need.append("camera must be local, ssh, listen or replay:<file.bin>")
        if self.snn not in ("fpga", "mock"):
            need.append("snn must be fpga or mock")
        if self.readout_mode not in ("", "ema", "diff"):
            need.append("readout_mode must be ema or diff")
        if self.remote not in ("lowstate", "wirelesscontroller", "both", "none"):
            need.append("remote must be lowstate, wirelesscontroller, both or none")
        if self.toggle_button != "none" and self.toggle_button.lower() not in BUTTONS:
            need.append("toggle_button must be none or one of %s" % ", ".join(BUTTONS))
        if self.window < 1 or self.refractory < 0 or not 0.05 <= self.stale <= 5 or not 0 < self.color_hz <= 30:
            need.append("window >= 1, refractory >= 0, stale 0.05..5 s, color_hz 0..30")
        if not 0 <= self.web_port <= 65535 or not 0 <= self.listen_port <= 65535:
            need.append("ports must be 0..65535")
        if need:
            raise SystemExit("snn_bridge: " + "; ".join(need))
        self.toggle_button = self.toggle_button.lower()


def finite(v, nd=3):
    """Rounded number for JSON (browsers cannot parse NaN / Infinity): 0 when unknown."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0
    return round(v, nd) if math.isfinite(v) else 0


# ----------------------------------------------------------------------------- CBOR / SPLEAT server
def cbor_head(major, value):
    if value < 24:
        return bytes([(major << 5) | value])
    for info, fmt in ((24, ">B"), (25, ">H"), (26, ">I"), (27, ">Q")):
        if value < 1 << (8 * struct.calcsize(fmt)):
            return bytes([(major << 5) | info]) + struct.pack(fmt, value)
    raise ValueError(value)


def cbor_decode(buf, i=0, depth=0):
    """Minimal CBOR decoder (definite lengths) -> (value, end). Enough for the SPLEAT maps."""
    if depth > 16:
        raise ValueError("CBOR nested too deep")
    if i >= len(buf):
        raise ValueError("truncated CBOR")
    major, info = buf[i] >> 5, buf[i] & 31
    i += 1
    if info < 24:
        arg = info
    elif info <= 27:
        n = 1 << (info - 24)
        if i + n > len(buf):
            raise ValueError("truncated CBOR")
        arg = int.from_bytes(bytes(buf[i:i + n]), "big")
        i += n
    else:
        raise ValueError("indefinite or reserved CBOR length (info %d)" % info)
    if major == 0:
        return arg, i
    if major == 1:
        return -1 - arg, i
    if major in (2, 3):
        if i + arg > len(buf):
            raise ValueError("truncated CBOR string")
        data = bytes(buf[i:i + arg])
        return (data if major == 2 else data.decode("utf-8")), i + arg
    if major in (4, 5):
        items = []
        for _ in range(arg * (2 if major == 5 else 1)):
            v, i = cbor_decode(buf, i, depth + 1)
            items.append(v)
        if major == 4:
            return items, i
        try:
            return dict(zip(items[0::2], items[1::2])), i
        except TypeError:
            raise ValueError("unhashable CBOR map key")
    if major == 6:
        return cbor_decode(buf, i, depth + 1)                       # tag: keep the tagged value
    if info == 25:
        return struct.unpack(">e", bytes(buf[i - 2:i]))[0], i
    if info == 26:
        return struct.unpack(">f", bytes(buf[i - 4:i]))[0], i
    if info == 27:
        return struct.unpack(">d", bytes(buf[i - 8:i]))[0], i
    return {20: False, 21: True, 22: None, 23: None}.get(arg, arg), i


def spleat_request(fid, payload):
    """One SPLEAT request: 4-byte big-endian length + CBOR {"id": fid, "payload": float32 LE bytes}."""
    body = (cbor_head(5, 2) + cbor_head(3, 2) + b"id" + cbor_head(0, fid)
            + cbor_head(3, 7) + b"payload" + cbor_head(2, len(payload)) + payload)
    return struct.pack(">I", len(body)) + body


def decode_result(body):
    """One SPLEAT result (CBOR {id, payload: float32 LE per output, latency (us), error}) ->
    (id or None, float32 array, latency_us or None, error text). ValueError for anything else."""
    msg, end = cbor_decode(body)
    if end != len(body) or not isinstance(msg, dict):
        raise ValueError("not a SPLEAT result")
    payload = msg.get("payload")
    err = msg.get("error")
    err = "" if err in (None, False, "") else str(err)
    if not isinstance(payload, bytes) or len(payload) % 4:
        if err:
            payload = b""
        else:
            raise ValueError("SPLEAT result without a float32 payload")
    fid = msg.get("id")
    if not isinstance(fid, int) or isinstance(fid, bool):
        fid = None
    lat = msg.get("latency")
    lat = float(lat) if isinstance(lat, (int, float)) and not isinstance(lat, bool) else None
    return fid, np.frombuffer(payload, "<f4"), lat, err


class SpleatClient(object):
    """Client of the SPLEAT server: one request per frame, then wait for the result with our id (the server
    broadcasts every result to every client, so foreign results are skipped). Ids start at a random offset:
    other clients count from 0."""

    MAX_MSG = 1 << 20

    def __init__(self, host, port, timeout=1.0):
        self.addr = (host, int(port))
        self.timeout = timeout
        self.sock = None
        self.buf = bytearray()
        self.next_id = random.randrange(1 << 20, 1 << 30)

    def connect(self):
        self.close()
        s = socket.create_connection(self.addr, timeout=2.0)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock, self.buf = s, bytearray()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def infer(self, frame):
        """frame [2, 80, 80] (0/1) -> (outputs float32 array, server latency in us or None)."""
        if self.sock is None:
            raise ConnectionError("not connected")
        fid = self.next_id
        self.next_id += 1
        self.sock.settimeout(self.timeout)
        self.sock.sendall(spleat_request(fid, np.ascontiguousarray(frame, "<f4").tobytes()))
        deadline = time.monotonic() + self.timeout
        while True:
            rid, out, lat, err = decode_result(self._message(deadline))
            if rid == fid:
                if err:
                    raise ConnectionError("SPLEAT server error: %s" % err)
                return out, lat

    def _message(self, deadline):
        while True:
            if len(self.buf) >= 4:
                n = struct.unpack_from(">I", self.buf)[0]
                if n > self.MAX_MSG:
                    raise ConnectionError("SPLEAT stream out of sync (length %d)" % n)
                if len(self.buf) >= 4 + n:
                    body = bytes(self.buf[4:4 + n])
                    del self.buf[:4 + n]
                    return body
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("no SPLEAT result within %.1f s" % self.timeout)
            self.sock.settimeout(left)
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("SPLEAT server closed the connection")
            self.buf += chunk
            quickack = getattr(socket, "TCP_QUICKACK", None)       # Linux: ACK at once, no delayed-ACK stalls
            if quickack is not None:
                try:
                    self.sock.setsockopt(socket.IPPROTO_TCP, quickack, 1)
                except OSError:
                    pass


class MockSnn(object):
    """Stand-in for the FPGA (snn:=mock): an 'airspace tripwire'. Channel 0 spikes when >= `cells` cells (either
    polarity) are active in the rows < `rows` of the 80x80 frame within the last 2 frames; the spike is held for
    `hold` frames (like a burst of the real SNN, so the EMA readout can reach theta), plus rare noise spikes on
    channels 0-2. 11 outputs like the board. On the 03 recordings this fires shortly after every visible throw."""

    def __init__(self, rows=16, cells=6, hold=15, noise=0.002, outputs=11, seed=None):
        self.rows, self.cells, self.hold, self.noise, self.outputs = rows, cells, max(int(hold), 1), noise, outputs
        self.prev, self.left = 0, 0
        self.rng = random.Random(seed)

    def infer(self, frame):
        t0 = time.perf_counter()
        n = int(np.count_nonzero(frame[:, :self.rows, :].any(axis=0)))
        if n + self.prev >= self.cells:
            self.left = self.hold
        self.prev = n
        out = np.zeros(self.outputs, np.float32)
        if self.left > 0:
            out[0] = 1.0
            self.left -= 1
        for c in range(min(3, self.outputs)):
            if self.rng.random() < self.noise:
                out[c] = 1.0
        return out, (time.perf_counter() - t0) * 1e6


# ----------------------------------------------------------------------------- readout
def load_readout(path):
    """readout.json (Tim's format: output_channel, readout, rule 'y_t = 0.85 * y_{t-1} + 0.15 * spike_t', theta)
    -> {mode, alpha, theta, channel, window, name, error}; missing values stay None."""
    info = {"mode": None, "alpha": None, "theta": None, "channel": None, "window": None, "name": None,
            "file": str(path), "error": ""}
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        info["error"] = str(e)
        return info
    if not isinstance(d, dict):
        info["error"] = "not a JSON object"
        return info
    info["mode"] = "diff" if str(d.get("readout", "")).lower() == "diff" else "ema"
    m = re.search(r"=\s*([0-9]*\.?[0-9]+)\s*\*\s*y", str(d.get("rule", "")))
    for key, cast in (("alpha", float), ("theta", float), ("window", int), ("name", str)):
        if d.get(key) is not None:
            info[key] = cast(d[key])
    if info["alpha"] is None and m:
        info["alpha"] = float(m.group(1))
    if d.get("output_channel") is not None:
        info["channel"] = int(d["output_channel"])
    return info


class EmaReadout(object):
    """y = alpha * y + (1 - alpha) * spike per frame; alarm when y rises past theta (prev < theta <= y) and the
    last alarm is >= refractory frames ago (Tim's perception readout, frame index k = 10 ms of camera time)."""

    mode = "ema"

    def __init__(self, theta, alpha=0.85, refractory=100, channel=0):
        self.theta, self.alpha, self.refractory, self.channel = float(theta), float(alpha), int(refractory), channel
        self.y, self.last = 0.0, -10 ** 9

    @property
    def value(self):
        return self.y

    def reset(self):
        self.y = 0.0

    def step(self, k, out):
        prev = self.y
        self.y = self.alpha * prev + (1.0 - self.alpha) * (1.0 if out[self.channel] > 0 else 0.0)
        alarm = prev < self.theta <= self.y and k - self.last >= self.refractory
        if alarm:
            self.last = k
        return self.y, alarm


class DiffReadout(object):
    """d = sum over the last `window` frames of (spike[channel] - spike[neg_channel]); alarm when d >= theta and
    the last alarm is >= refractory frames ago (the ball models' evidence, as tools/ball_trigger_logic.py)."""

    mode = "diff"

    def __init__(self, theta, window=10, refractory=100, channel=0, neg_channel=1):
        self.theta, self.window, self.refractory = float(theta), int(window), int(refractory)
        self.channel, self.neg_channel = channel, neg_channel
        self.buf = deque(maxlen=self.window)
        self.d, self.last = 0, -10 ** 9

    @property
    def value(self):
        return float(self.d)

    def reset(self):
        self.buf.clear()
        self.d = 0

    def step(self, k, out):
        v = int(out[self.channel] > 0) - int(out[self.neg_channel] > 0)
        if len(self.buf) == self.window:
            self.d -= self.buf[0]
        self.buf.append(v)
        self.d += v
        alarm = self.d >= self.theta and k - self.last >= self.refractory
        if alarm:
            self.last = k
        return float(self.d), alarm


# ----------------------------------------------------------------------------- events -> frames, display
ZERO_FRAME = np.zeros((2, GRID, GRID), np.uint8)
ZERO_FRAME.flags.writeable = False


class Binner(object):
    """Events -> binary 10 ms frames [2, 80, 80] on the camera clock (cell = 1 if any event of that polarity in its
    4x4 block, polarity 0 = OFF first). feed() returns the frames this batch completed, in order: a frame is done
    when an event of a later frame arrives; a pause yields at most MAX_GAP empty frames; late events of a frame
    already done are dropped; the camera clock jumping back by > 1 s (camera restarted) starts over."""

    def __init__(self):
        self.cur = None                       # frame number (camera time // 10 ms) being filled
        self.frame = np.zeros(2 * GRID * GRID, np.uint8)
        self.late = 0

    def reset(self):
        self.cur = None
        self.frame = np.zeros(2 * GRID * GRID, np.uint8)

    def feed(self, x, y, p, t):
        """x, y in 0..319, p in 0..1 (integer arrays), t in us."""
        if not len(t):
            return []
        num = np.asarray(t, np.int64) // FRAME_US
        if len(num) > 1 and np.any(num[1:] < num[:-1]):
            order = np.argsort(num, kind="stable")
            num, x, y, p = num[order], x[order], y[order], p[order]
        cell = (p * GRID + y // (SENSOR // GRID)) * GRID + x // (SENSOR // GRID)
        if self.cur is not None and num[-1] < self.cur - 100:
            self.reset()
        if self.cur is None:
            self.cur = int(num[0])
        out = []
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(num)) + 1, [len(num)]))
        for i in range(len(bounds) - 1):
            s, e = int(bounds[i]), int(bounds[i + 1])
            b = int(num[s])
            if b < self.cur:
                self.late += e - s
                continue
            if b > self.cur:
                out.append(self.frame.reshape(2, GRID, GRID))
                out.extend([ZERO_FRAME] * min(b - self.cur - 1, MAX_GAP))
                self.frame = np.zeros(2 * GRID * GRID, np.uint8)
                self.cur = b
            self.frame[cell[s:e]] = 1
        return out


class Display(object):
    """Events since the last take() at half sensor resolution (160x160), ON (polarity 1) and OFF separately."""

    def __init__(self):
        self.lock = threading.Lock()
        self.acc = np.zeros(2 * DISP * DISP, bool)
        self.n = 0

    def add(self, x, y, p):
        idx = (p * DISP + y // 2) * DISP + x // 2
        with self.lock:
            self.acc[idx] = True
            self.n += len(idx)

    def take(self):
        """-> (on, off, events): packed bitmaps (row-major, MSB first, 3200 bytes each) and the event count."""
        with self.lock:
            acc, n = self.acc, self.n
            self.acc, self.n = np.zeros(2 * DISP * DISP, bool), 0
        return np.packbits(acc[DISP * DISP:]), np.packbits(acc[:DISP * DISP]), n


def b64(a):
    return base64.b64encode(np.asarray(a, np.uint8).tobytes()).decode("ascii")


def read_bin(path):
    """record_tool .bin (4-byte header, then events x >u2, y >u2, p u1, t >i8 us) -> x, y, p, t (native arrays,
    sorted by t). The header is skipped: a take cut off by a crash may have a header behind the file."""
    size = os.path.getsize(path)
    if size < 4:
        raise ValueError("not a recording (shorter than its header)")
    n = (size - 4) // EVENT_DTYPE.itemsize
    ev = np.fromfile(path, dtype=EVENT_DTYPE, count=n, offset=4)
    x = np.minimum(ev["x"], SENSOR - 1).astype(np.uint16)
    y = np.minimum(ev["y"], SENSOR - 1).astype(np.uint16)
    p = np.minimum(ev["p"], 1).astype(np.uint8)
    t = ev["t"].astype(np.int64)
    if n > 1 and np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        x, y, p, t = x[order], y[order], p[order], t[order]
    return x, y, p, t


def recv_exact(sock, n, stop):
    """n bytes from a socket with a timeout set, or None when the peer closed or `stop` is set."""
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        if stop.is_set():
            return None
        try:
            r = sock.recv_into(view[got:], n - got)
        except socket.timeout:
            continue
        except OSError:
            return None
        if not r:
            return None
        got += r
    return buf


class Camera(threading.Thread):
    """Event source thread: despot_debug_camera over TCP (local / ssh / listen) or a .bin recording (replay).
    Every batch goes to bridge.feed(); the state goes to bridge.camera_status()."""

    def __init__(self, bridge, cfg):
        super().__init__(name="camera", daemon=True)
        self.bridge, self.cfg, self.log, self.stop = bridge, cfg, bridge.log, bridge.stop
        self.mode = cfg.camera_mode
        self.proc = self.guard = None
        self.proc_lock = threading.Lock()
        self.fails, self.next_start = 0, 0.0
        atexit.register(self.stop_proc)

    def run(self):
        try:
            if self.mode == "replay":
                self._replay(self.cfg.replay_path)
            else:
                self._serve()
        except Exception as e:  # never die silently: the page shows it
            self.log.error("camera thread failed: %r" % (e,))
            self.bridge.camera_status("offline", "camera thread failed: %s" % e)
        finally:
            self.stop_proc()

    # --- live camera ---------------------------------------------------------------
    def _serve(self):
        port = self.cfg.listen_port
        while not self.stop.is_set():
            try:
                server = socket.create_server(("", port))
            except OSError as e:
                self.bridge.camera_status("offline", "cannot listen on port %d: %s" % (port, e))
                self.stop.wait(2.0)
                continue
            with server:
                server.settimeout(0.5)
                self.bridge.camera_status("waiting", "")
                self.log.info("waiting for despot_debug_camera on port %d (camera:=%s)" % (port, self.mode))
                while not self.stop.is_set():
                    self._keep_proc()
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue
                    self.log.info("camera connected from %s" % addr[0])
                    self.bridge.camera_status("connected", "")
                    try:
                        self._read(conn)
                    finally:
                        conn.close()
                        self.bridge.camera_lost()
                    if not self.stop.is_set():
                        self.log.warning("camera connection closed, waiting for it again")
                        self.bridge.camera_status("waiting", "")

    def _read(self, conn):
        """Batches: 4-byte big-endian size, then that many bytes of events (13 bytes each; empty batches occur)."""
        conn.settimeout(0.5)
        while not self.stop.is_set():
            head = recv_exact(conn, 4, self.stop)
            if head is None:
                return
            size = struct.unpack(">I", head)[0]
            if size > 64 << 20:
                self.log.error("camera stream out of sync (batch of %d bytes), reconnecting" % size)
                return
            data = recv_exact(conn, size, self.stop) if size else b""
            if data is None:
                return
            n = len(data) // EVENT_DTYPE.itemsize
            if n:
                self.fails = 0
                ev = np.frombuffer(data, dtype=EVENT_DTYPE, count=n)
                self.bridge.feed(ev["x"], ev["y"], ev["p"], ev["t"])

    def _keep_proc(self):
        """local / ssh: (re)start the camera process, with a growing pause when it keeps failing."""
        if self.mode not in ("local", "ssh"):
            return
        with self.proc_lock:
            proc = self.proc
        if proc is not None:
            code = proc.poll()
            if code is None:
                return
            self.stop_proc()
            self.fails += 1
            self.next_start = time.monotonic() + min(2.0 * self.fails, 10.0)
            what = "despot_debug_camera" if self.mode == "local" else "ssh to %s" % self.cfg.ssh_host
            self.log.warning("%s exited with code %s (log: %s), restarting" % (what, code, CAMERA_LOG))
            self.bridge.camera_status("offline", "%s exited with code %s" % (what, code))
            return
        if time.monotonic() < self.next_start:
            return
        try:
            with self.proc_lock:
                if self.mode == "local":
                    self._start_local()
                else:
                    self._start_ssh()
        except OSError as e:
            self.fails += 1
            self.next_start = time.monotonic() + min(2.0 * self.fails, 10.0)
            if self.fails == 1:
                self.log.error("cannot start the camera: %s" % e)
            self.bridge.camera_status("offline", "cannot start the camera: %s" % e)
            return
        self.bridge.camera_status("waiting", "")

    def _start_local(self):
        cmd = [self.cfg.camera_bin, "--host", "127.0.0.1", "--port", str(self.cfg.listen_port)]
        try:
            logf = open(CAMERA_LOG, "ab")
        except OSError:
            logf = None
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf or subprocess.DEVNULL,
                                         stderr=subprocess.STDOUT)
        finally:
            if logf is not None:
                logf.close()
        # Guard: if this process dies without cleaning up (SIGKILL), the guard's stdin closes and it stops the
        # camera, so the next start finds the camera free. Killed first on a normal stop.
        try:
            self.guard = subprocess.Popen(
                ["sh", "-c", 'read -r line; grep -q despot /proc/%d/cmdline 2>/dev/null && kill %d'
                 % (self.proc.pid, self.proc.pid)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.guard = None
        self.log.info("started %s (pid %d, log %s)" % (" ".join(cmd), self.proc.pid, CAMERA_LOG))

    def _start_ssh(self):
        # As river_tools.capture.remote.start_camera: the board connects back to the SSH client address; the
        # session ends (and the trap stops the camera) when we close its stdin or die.
        script = ("set -- $SSH_CONNECTION\n"
                  "%s --host \"$1\" --port %d >%s 2>&1 &\n"
                  "camera_pid=$!\n"
                  "trap 'kill \"$camera_pid\" 2>/dev/null; wait \"$camera_pid\" 2>/dev/null' EXIT\n"
                  "trap 'exit' HUP INT TERM\n"
                  "read -r stop\n") % (self.cfg.camera_bin, self.cfg.listen_port, CAMERA_LOG)
        self.proc = subprocess.Popen(
            ["ssh", "-4", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "--", self.cfg.ssh_host, script],
            stdin=subprocess.PIPE)
        self.log.info("started despot_debug_camera on %s over SSH" % self.cfg.ssh_host)

    def stop_proc(self):
        """Stop the camera process (idempotent; also called at exit)."""
        with self.proc_lock:
            proc, guard = self.proc, self.guard
            self.proc = self.guard = None
        if guard is not None:
            try:
                guard.kill()
                guard.wait(1)
            except Exception:
                pass
        if proc is None:
            return
        try:
            if self.mode == "ssh":
                try:
                    proc.communicate(input=b"\n", timeout=3)
                except Exception:
                    proc.kill()
                    proc.wait(2)
            elif proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(2)
        except Exception as e:
            self.log.warning("stopping the camera: %r" % (e,))

    # --- replay ---------------------------------------------------------------------
    def _replay(self, path):
        try:
            x, y, p, t = read_bin(path)
        except (OSError, ValueError) as e:
            self.log.error("replay %s: %s" % (path, e))
            self.bridge.camera_status("offline", "replay %s: %s" % (path, e))
            return
        if not len(t):
            self.bridge.camera_status("offline", "replay %s: no events" % path)
            return
        dur = (t[-1] - t[0]) / 1e6
        self.log.info("replaying %s: %d events, %.1f s, looped" % (path, len(t), dur))
        self.bridge.camera_status("replay", "")
        n_chunks = int((t[-1] - t[0]) // REPLAY_CHUNK_US) + 1
        idx = np.searchsorted(t, t[0] + REPLAY_CHUNK_US * np.arange(n_chunks + 1, dtype=np.int64))
        span, offset = int(t[-1] - t[0]) + FRAME_US, 0
        while not self.stop.is_set():
            wall0 = time.monotonic()
            for i in range(n_chunks):
                delay = wall0 + (i + 1) * REPLAY_CHUNK_US / 1e6 - time.monotonic()
                if delay > 0 and self.stop.wait(delay):
                    return
                s, e = idx[i], idx[i + 1]
                if e > s:
                    self.bridge.feed(x[s:e], y[s:e], p[s:e], t[s:e] + offset)
            offset += span                     # the camera clock keeps running over the loop


# ----------------------------------------------------------------------------- Go2 remote
def lowstate_keys(remote):
    """LowState.wireless_remote (40 bytes) -> the 16 key bits, or None (header 0x55 0x51 missing)."""
    if len(remote) >= 4 and int(remote[0]) == 0x55 and int(remote[1]) == 0x51:
        return int(remote[2]) | int(remote[3]) << 8
    return None


class RemoteButton(object):
    """Presses of one remote button from several sources (/lowstate, /wirelesscontroller): a press is a rising
    edge of its bit; the first message of a source only sets the reference (a button held at start is no press);
    presses within `debounce` s of the last one count once (both topics can carry the same press)."""

    def __init__(self, bit, debounce=0.4):
        self.mask, self.debounce = 1 << bit, debounce
        self.last, self.pressed_at = {}, -1e9

    def update(self, source, keys, now=None):
        now = time.monotonic() if now is None else now
        keys = int(keys)
        was = self.last.get(source)
        self.last[source] = keys
        if was is None or not keys & self.mask or was & self.mask:
            return False
        if now - self.pressed_at < self.debounce:
            return False
        self.pressed_at = now
        return True


def jpeg_bytes(binary):
    """unitree_api Response.binary (int8[], rclpy gives array('b')) -> bytes (& 0xFF)."""
    try:
        return memoryview(binary).cast("B").tobytes()
    except (TypeError, ValueError):
        return bytes((int(v) & 0xFF) for v in binary)


# ----------------------------------------------------------------------------- server-sent events
def sse(event, data):
    return ("event: %s\ndata: %s\n\n" % (event, json.dumps(data, separators=(",", ":")))).encode()


class SseClient(object):
    MAX_QUEUE = 256

    def __init__(self):
        self.q = deque()
        self.ev = threading.Event()
        self.dead = False

    def push(self, msg):
        if len(self.q) >= self.MAX_QUEUE:      # the page fell far behind: drop it, its browser reconnects
            self.dead = True
        else:
            self.q.append(msg)
        self.ev.set()

    def pop_all(self, timeout):
        if not self.q:
            self.ev.wait(timeout)
        self.ev.clear()
        out = []
        while self.q:
            out.append(self.q.popleft())
        return out


class Hub(object):
    """All connected pages. publish() never blocks; each page has its own queue and thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()

    def subscribe(self):
        c = SseClient()
        with self.lock:
            self.clients.add(c)
        return c

    def unsubscribe(self, c):
        with self.lock:
            self.clients.discard(c)

    def count(self):
        with self.lock:
            return len(self.clients)

    def publish(self, event, data):
        with self.lock:
            clients = list(self.clients)
        if clients:
            msg = sse(event, data)
            for c in clients:
                c.push(msg)

    def close_all(self):
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            c.dead = True
            c.ev.set()


# ----------------------------------------------------------------------------- the bridge
class Log(object):
    """ROS logger when there is a node, else stdout."""

    def __init__(self, ros_logger=None):
        self.ros = ros_logger

    def _out(self, level, msg):
        if self.ros is not None:
            getattr(self.ros, level)(msg)
        else:
            print("%s [snn_bridge] %s%s" % (time.strftime("%H:%M:%S"), "" if level == "info" else level.upper()
                                           + ": ", msg), flush=True)

    def info(self, msg):
        self._out("info", msg)

    def warning(self, msg):
        self._out("warning", msg)

    def error(self, msg):
        self._out("error", msg)


class Bridge(object):
    """Camera -> frames -> SNN -> readout -> detections, the shared state and the web server. Threads: camera
    (Camera), snn (_snn_loop), web ticker (_ticker, 25 Hz), HTTP (_web_loop + one thread per request); the ROS
    executor (if any) calls set_auto / on_robot_event / on_color_jpeg and RosLink publishes for us."""

    def __init__(self, cfg, log=None):
        self.cfg, self.log = cfg, log or Log()
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.hub = Hub()
        self.ros = None                                     # RosLink, set by main() with ROS
        self.web_dir = Path(cfg.web_dir).expanduser() if cfg.web_dir else HERE / "web"
        # model and readout
        ro = load_readout(cfg.readout_file or HERE / "model" / "readout.json")
        if ro["error"]:
            self.log.warning("readout file %s: %s (using defaults)" % (ro["file"], ro["error"]))
        mode = cfg.readout_mode or ro["mode"] or "ema"
        same = mode == (ro["mode"] or "ema")
        if str(cfg.theta).strip():
            theta = float(cfg.theta)
        elif same and ro["theta"] is not None:
            theta = ro["theta"]
        else:
            theta = 0.85 if mode == "ema" else 3.0
        refr = int(round(cfg.refractory * 1e6 / FRAME_US))
        if mode == "ema":
            alpha = float(cfg.alpha) if str(cfg.alpha).strip() else (ro["alpha"] if ro["alpha"] is not None else 0.85)
            if not 0 < alpha < 1:
                raise SystemExit("alpha must be in (0, 1)")
            self.readout = EmaReadout(theta, alpha, refr, cfg.channel)
        else:
            self.readout = DiffReadout(theta, cfg.window, refr, cfg.channel, cfg.neg_channel)
        self.readout_info = ro
        self.mock = MockSnn(hold=cfg.mock_hold) if cfg.snn == "mock" else None
        self.model_name = cfg.model_name or ro["name"] or (
            "Mock SNN (airspace tripwire)" if self.mock else
            "Early throw-motion SNN (v3)" if mode == "ema" else "Ball SNN (diff readout)")
        self.n_out = max(3, cfg.channel + 1, cfg.neg_channel + 1 if mode == "diff" else 0)
        # camera and frames
        self.binner, self.display = Binner(), Display()
        self.frames = queue.Queue(maxsize=64)
        self.k = 0                                          # frames made (SNN time, 10 ms each)
        self.ev_total, self.last_event, self.last_frame = 0, 0.0, 0.0
        self.camera_state, self.camera_error = "replay" if cfg.camera_mode == "replay" else "waiting", ""
        self.dropped = self.skipped = 0
        # SNN
        self.fpga_state, self.fpga_error = ("mock" if self.mock else "offline"), ""
        self.lat, self.rtt = deque(maxlen=100), deque(maxlen=100)
        self.done = 0                                       # frames through the SNN
        self.hist, self.pending, self.published_k = deque(maxlen=HISTORY), [], 0
        self.rate_win, self.ev_rate, self.fps = deque(), 0.0, 0.0
        # control and robot
        self.auto, self.auto_source, self.auto_time = bool(cfg.auto), "start", time.time()
        self.detections = self.dodges = 0
        self.last_detection, self.last_acted = None, 0.0
        self.robot_state, self.robot_last, self.robot_out_m, self.robot_listeners = None, {}, None, 0
        self.recent = deque(maxlen=12)
        # color camera
        self.color_enabled, self.color_error = bool(cfg.color), ""
        self.jpeg, self.jpeg_seq, self.jpeg_time = None, 0, 0.0
        # web
        self.web_port, self.web_error = None, ""
        self.threads = []
        self.camera = None

    # --- lifecycle ----------------------------------------------------------------------
    def start(self):
        self.log.info("model %s: %s readout, theta %g, channel %d, refractory %.1f s%s" % (
            self.model_name, self.readout.mode, self.readout.theta, self.cfg.channel, self.cfg.refractory,
            ", alpha %g" % self.readout.alpha if self.readout.mode == "ema" else ", window %d" % self.cfg.window))
        self.log.info("auto-dodge %s at start (Y on the remote, the web page or /snn/auto_set toggle it)"
                      % ("ON" if self.auto else "off"))
        self.camera = Camera(self, self.cfg)
        self.threads = [self.camera,
                        threading.Thread(target=self._snn_loop, name="snn", daemon=True),
                        threading.Thread(target=self._ticker, name="ticker", daemon=True),
                        threading.Thread(target=self._web_loop, name="web", daemon=True)]
        for th in self.threads:
            th.start()

    def close(self):
        self.stop.set()
        self.hub.close_all()
        for th in self.threads:
            th.join(timeout=6.0 if th is self.camera else 2.0)
        if self.camera is not None:
            self.camera.stop_proc()

    # --- camera side (camera thread) -----------------------------------------------------
    def camera_status(self, state, error=""):
        with self.lock:
            self.camera_state, self.camera_error = state, error

    def camera_lost(self):
        self.binner.reset()

    def feed(self, x, y, p, t):
        """One batch of events (any integer arrays; t in us on the camera clock)."""
        if not len(t):
            return
        x = np.minimum(x, SENSOR - 1).astype(np.intp)
        y = np.minimum(y, SENSOR - 1).astype(np.intp)
        p = np.minimum(p, 1).astype(np.intp)
        now = time.monotonic()
        self.display.add(x, y, p)
        frames = self.binner.feed(x, y, p, np.asarray(t, np.int64))
        with self.lock:
            self.ev_total += len(t)
            self.last_event = now
            if frames:
                self.last_frame = now
        for f in frames:
            self.k += 1
            try:
                self.frames.put_nowait((self.k, f, now))
            except queue.Full:
                self.dropped += 1

    # --- SNN side (snn thread) -----------------------------------------------------------
    def _drain(self):
        n = 0
        while True:
            try:
                self.frames.get_nowait()
                n += 1
            except queue.Empty:
                return n

    def _snn_down(self, error):
        with self.lock:
            was = self.fpga_state
            self.fpga_state, self.fpga_error = "offline", error
        self.readout.reset()
        self._drain()
        if was != "offline":
            self.log.warning("SNN offline: %s (readout reset, retrying)" % error)

    def _snn_loop(self):
        cfg, client, backoff, idle_reset = self.cfg, None, 0.5, True
        while not self.stop.is_set():
            if self.mock is None and client is None:
                client = SpleatClient(cfg.spleat_host, cfg.spleat_port)
                try:
                    client.connect()
                except OSError as e:
                    client = None
                    self._snn_down("SPLEAT server %s:%d: %s" % (cfg.spleat_host, cfg.spleat_port, e))
                    self.stop.wait(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue
                backoff = 0.5
                self._drain()
                self.readout.reset()
                with self.lock:
                    self.fpga_state, self.fpga_error = "connected", ""
                self.log.info("SPLEAT server %s:%d connected" % (cfg.spleat_host, cfg.spleat_port))
            try:
                k, frame, born = self.frames.get(timeout=0.1)
            except queue.Empty:
                if not idle_reset and time.monotonic() - self.last_frame > cfg.stale:
                    self.readout.reset()                       # camera silent: never act on old evidence
                    idle_reset = True
                continue
            if time.monotonic() - born > cfg.stale:            # behind: drop the backlog, start over
                self.skipped += self._drain() + 1
                self.readout.reset()
                continue
            idle_reset = False
            t0 = time.perf_counter()
            try:
                out, lat_us = self.mock.infer(frame) if self.mock is not None else client.infer(frame)
            except (OSError, ConnectionError, ValueError) as e:
                if client is not None:
                    client.close()
                    client = None
                self._snn_down(str(e))
                continue
            rtt = (time.perf_counter() - t0) * 1e3
            if len(out) < self.n_out:
                out = np.concatenate((out, np.zeros(self.n_out - len(out), np.float32)))
            y, alarm = self.readout.step(k, out)
            row = (k, int(out[0] > 0), int(out[1] > 0), int(out[2] > 0), round(float(y), 4))
            with self.lock:
                self.done += 1
                self.rtt.append(rtt)
                self.lat.append(lat_us / 1e3 if lat_us is not None and lat_us >= 0 else rtt)
                self.hist.append(row)
                self.pending.append(row)
            if alarm:
                self._detection(k, y, born)
        if client is not None:
            client.close()

    def _detection(self, k, y, born):
        acted, reason = False, ""
        with self.lock:
            auto = self.auto
        if not auto:
            reason = "auto-dodge off"
        elif self.ros is None:
            reason = "no ROS"
        elif time.monotonic() - born > self.cfg.stale:
            reason = "stale frame"
        else:
            try:
                self.ros.publish_dash()
                acted = True
            except Exception as e:  # publishing must never kill the SNN loop
                reason = "publish failed: %s" % e
        det = {"t": round(time.time(), 3), "k": k, "y": round(float(y), 4), "theta": self.readout.theta,
               "acted": acted, "reason": reason}
        with self.lock:
            self.detections += 1
            self.last_detection = det
            if acted:
                self.last_acted = time.monotonic()
            self._note("SNN", "dodge" if acted else "detected (%s)" % reason)
        if self.ros is not None:
            try:
                self.ros.publish_detection(det)
            except Exception as e:
                self.log.warning("publishing /snn/detection: %r" % (e,))
        self.hub.publish("detection", det)
        self.log.info("THROW DETECTED k=%d y=%.3f -> %s" % (k, y, "/reflex/dash" if acted else "no dodge: " + reason))

    # --- control (HTTP / ROS threads) ----------------------------------------------------
    def _note(self, source, action):
        self.recent.appendleft({"t": round(time.time(), 3), "source": source, "action": action})

    def set_auto(self, enabled, source):
        enabled = bool(enabled)
        with self.lock:
            changed = enabled != self.auto
            self.auto = enabled
            if changed:
                self.auto_source, self.auto_time = source, time.time()
                self._note(source, "auto-dodge %s" % ("on" if enabled else "off"))
        if changed:
            self.log.info("auto-dodge %s (%s)" % ("ON: detections move the robot" if enabled else "off", source))
            if self.ros is not None:
                self.ros.publish_auto(enabled)
        return changed

    def set_theta(self, theta):
        theta = float(theta)
        if not math.isfinite(theta):
            raise ValueError("theta must be a finite number")
        lo, hi = (0.01, 0.99) if self.readout.mode == "ema" else (1.0, float(self.readout.window))
        self.readout.theta = round(min(max(theta, lo), hi), 4)
        self.log.info("theta set to %g" % self.readout.theta)

    def set_color(self, enabled):
        with self.lock:
            self.color_enabled = bool(enabled)
        self.log.info("color camera %s" % ("on" if enabled else "off"))

    def on_robot_event(self, data):
        """A /dodge/event of go2_dodge.py (dict)."""
        ev = str(data.get("event", ""))
        with self.lock:
            self.robot_last = data
            if ev in ("dash", "hold", "return", "idle"):
                self.robot_state = ev
            elif ev == "stop":
                self.robot_state = "idle"
            if data.get("out_m") is not None:
                self.robot_out_m = finite(data.get("out_m"))
            if ev == "dash":
                self.dodges += 1
                src = str(data.get("source", ""))
                self._note("X button" if src.startswith("button") else
                           "SNN" if src == "/reflex/dash" and time.monotonic() - self.last_acted < 1.0 else src,
                           "dodge")
            elif ev in ("ignored", "stop"):
                self._note("robot", "%s (%s)" % (ev, data.get("reason") or data.get("state") or ""))
        self.hub.publish("robot", data)

    def on_color_jpeg(self, jpeg):
        with self.lock:
            self.jpeg, self.jpeg_time, self.color_error = jpeg, time.monotonic(), ""
            self.jpeg_seq += 1
            seq = self.jpeg_seq
        self.hub.publish("color", {"seq": seq})

    def color_failed(self, error):
        with self.lock:
            self.color_error = error

    # --- state -------------------------------------------------------------------------
    def _camera_view(self, now):
        if self.camera_state == "connected":
            return "streaming" if now - self.last_event < self.cfg.stale else "waiting"
        return self.camera_state

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            ro = self.readout
            model = {"name": self.model_name, "readout": ro.mode, "theta": ro.theta, "channel": self.cfg.channel,
                     "mock": self.mock is not None, "y": finite(ro.value, 4), "refractory": self.cfg.refractory}
            if ro.mode == "ema":
                model["alpha"] = ro.alpha
            else:
                model.update(window=ro.window, neg_channel=ro.neg_channel)
            if self.ros is None:
                robot_state = "unknown"
            elif self.robot_listeners == 0:
                robot_state = "unknown"                      # no go2_dodge (nothing listens to /reflex/dash)
            else:
                robot_state = self.robot_state or "idle"
            color_ok = self.jpeg is not None and now - self.jpeg_time < 3.0
            return {
                "model": model,
                "camera": {"state": self._camera_view(now), "rate": int(self.ev_rate), "source": self.cfg.camera,
                           "error": self.camera_error, "dropped": self.dropped, "late": self.binner.late},
                "fpga": {"state": self.fpga_state, "latency_ms": finite(np.median(self.lat), 2) if self.lat else 0,
                         "rtt_ms": finite(np.median(self.rtt), 2) if self.rtt else 0, "fps": finite(self.fps, 1),
                         "host": "mock" if self.mock else "%s:%d" % (self.cfg.spleat_host, self.cfg.spleat_port),
                         "error": self.fpga_error, "skipped": self.skipped},
                "ros": {"available": self.ros is not None, "auto": self.auto, "auto_source": self.auto_source,
                        "auto_time": round(self.auto_time, 3),
                        "remote": bool(self.ros is not None and self.ros.remote_ok)},
                "robot": {"state": robot_state, "last": dict(self.robot_last), "listeners": self.robot_listeners,
                          "out_m": self.robot_out_m if self.robot_out_m is not None else 0},
                "color": {"enabled": self.color_enabled, "available": color_ok, "seq": self.jpeg_seq,
                          "error": self.color_error},
                "counters": {"detections": self.detections, "dodges": self.dodges},
                "last_detection": self.last_detection or {},
                "recent": list(self.recent),
                "server_time": round(time.time(), 3),
            }

    def _batch(self, rows):
        return {"frames": [{"k": r[0], "s": [r[1], r[2], r[3]], "y": r[4]} for r in rows],
                "theta": self.readout.theta, "latency_ms": finite(np.median(self.lat), 2) if self.lat else 0,
                "fps": finite(self.fps, 1)}

    # --- web ---------------------------------------------------------------------------
    def connect_page(self):
        """New SSE page -> (client, first messages: status, the last 6 s of SNN frames, the color seq)."""
        with self.lock:                                      # atomically with the ticker: no frame twice
            client = self.hub.subscribe()
            rows = [r for r in self.hist if r[0] <= self.published_k]
            first = [b"retry: 1500\n\n", sse("status", self.snapshot())]
            if rows:
                first.append(sse("snn", self._batch(rows)))
            if self.jpeg_seq:
                first.append(sse("color", {"seq": self.jpeg_seq}))
        return client, first

    def _ticker(self):
        """25 Hz: event image, SNN batch, rates; status at 1 Hz and on any change."""
        next_t, last_status, last_sig, idle = time.monotonic(), 0.0, None, 0
        while not self.stop.is_set():
            next_t += TICK
            delay = next_t - time.monotonic()
            if delay < -0.5:
                next_t = time.monotonic()
            elif delay > 0 and self.stop.wait(delay):
                break
            now = time.monotonic()
            with self.lock:
                self.rate_win.append((now, self.ev_total, self.done))
                while len(self.rate_win) > 2 and now - self.rate_win[0][0] > 1.0:
                    self.rate_win.popleft()
                t0, e0, d0 = self.rate_win[0]
                if now - t0 >= 0.2:
                    self.ev_rate, self.fps = (self.ev_total - e0) / (now - t0), (self.done - d0) / (now - t0)
                rows, self.pending = self.pending, []
                if rows:
                    self.published_k = rows[-1][0]
                clients = self.hub.count()
                if clients and rows:
                    self.hub.publish("snn", self._batch(rows))
            on, off, n = self.display.take()
            if not clients:
                last_sig = None
                continue
            if n or idle < 12:                                # after activity: ~0.5 s of empty images, then quiet
                idle = 0 if n else idle + 1
                self.hub.publish("frame", {"t": round(time.time(), 3), "w": DISP, "h": DISP, "on": b64(on),
                                           "off": b64(off), "rate": int(self.ev_rate)})
            snap = self.snapshot()
            sig = (snap["camera"]["state"], snap["fpga"]["state"], snap["ros"]["auto"], snap["robot"]["state"],
                   snap["color"]["enabled"], snap["color"]["available"], snap["model"]["theta"],
                   snap["counters"]["detections"], snap["counters"]["dodges"], snap["robot"]["listeners"],
                   snap["camera"]["error"], snap["fpga"]["error"])
            if sig != last_sig or now - last_status >= 1.0:
                self.hub.publish("status", snap)
                last_sig, last_status = sig, now

    def _web_loop(self):
        handler = make_handler(self)
        failed = False
        while not self.stop.is_set():
            try:
                server = ThreadingHTTPServer((self.cfg.web_host, self.cfg.web_port), handler)
            except OSError as e:
                self.web_error = "web server on port %d: %s" % (self.cfg.web_port, e)
                if not failed:
                    self.log.error(self.web_error + " (retrying every 3 s)")
                failed = True
                self.stop.wait(3.0)
                continue
            server.daemon_threads = True
            server.timeout = 0.5
            self.web_port, self.web_error = server.server_address[1], ""
            self.log.info("web UI: %s" % "  ".join(local_urls(self.cfg.web_host, self.web_port)))
            if not (self.web_dir / "index.html").is_file():
                self.log.warning("no %s: serving a placeholder page" % (self.web_dir / "index.html"))
            try:
                while not self.stop.is_set():
                    server.handle_request()
            finally:
                server.server_close()


def local_urls(host, port):
    """Addresses a browser can use for the web server (ip -4 addr, BusyBox too)."""
    if host not in ("", "0.0.0.0"):
        return ["http://%s:%d/" % (host, port)]
    addrs = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2).stdout
        addrs = [a for a in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", out) if not a.startswith("127.")]
    except (OSError, subprocess.SubprocessError):
        pass
    return ["http://%s:%d/" % (a, port) for a in (addrs or ["127.0.0.1"])]


# ----------------------------------------------------------------------------- HTTP
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                 ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
                 ".jpeg": "image/jpeg", ".ico": "image/x-icon", ".webp": "image/webp", ".woff2": "font/woff2",
                 ".woff": "font/woff", ".txt": "text/plain; charset=utf-8", ".map": "application/json",
                 ".webmanifest": "application/manifest+json"}

PLACEHOLDER = b"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Throw Reflex</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font:15px/1.5 -apple-system,system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;padding:32px">
<h1 style="font-weight:600">Throw Reflex</h1><p>The web UI files are missing (web/index.html next to
snn_bridge.py). The bridge runs: <a href="/api/state">/api/state</a>, <a href="/stream">/stream</a>.</p></body></html>
"""


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0", "on", "off"):
        return v.strip().lower() in ("true", "1", "on")
    raise ValueError("enabled must be true or false")


def make_handler(bridge):
    web = bridge.web_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ThrowReflex/1"
        sys_version = ""

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data, separators=(",", ":")).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/state":
                    return self._json(bridge.snapshot())
                if path == "/stream":
                    return self._stream()
                if path == "/color.jpg":
                    jpeg = bridge.jpeg
                    if jpeg is None:
                        return self._send(404, b"no color image yet\n", "text/plain")
                    return self._send(200, jpeg, "image/jpeg")
                return self._static(path)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 4096:
                    raise ValueError("body too large")
                data = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
                if path == "/api/auto":
                    bridge.set_auto(as_bool(data["enabled"]), "web")
                elif path == "/api/theta":
                    bridge.set_theta(data["theta"])
                elif path == "/api/color":
                    bridge.set_color(as_bool(data["enabled"]))
                else:
                    return self._json({"error": "not found"}, 404)
            except KeyError as e:
                return self._json({"error": "missing field %s" % e}, 400)
            except (TypeError, ValueError) as e:
                return self._json({"error": str(e)}, 400)
            self._json(bridge.snapshot())

        def _static(self, path):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            f = (web / rel).resolve()
            if "\x00" in rel or web not in f.parents or not f.is_file():
                if rel == "index.html":
                    return self._send(200, PLACEHOLDER, "text/html; charset=utf-8")
                return self._send(404, b"not found\n", "text/plain")
            self._send(200, f.read_bytes(), CONTENT_TYPES.get(f.suffix.lower(), "application/octet-stream"))

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            client, first = bridge.connect_page()
            try:
                self.wfile.write(b"".join(first))
                self.wfile.flush()
                while not bridge.stop.is_set() and not client.dead:
                    msgs = client.pop_all(15.0)
                    self.wfile.write(b"".join(msgs) if msgs else b": ping\n\n")
                    self.wfile.flush()
            except OSError:                                   # page closed
                pass
            finally:
                bridge.hub.unsubscribe(client)

    return Handler


# ----------------------------------------------------------------------------- ROS
class RosLink(object):
    """The ROS side: topics in and out. Callbacks run in the executor (main) thread; publish_* may be called from
    any thread (the SNN thread publishes /reflex/dash directly: lowest latency)."""

    def __init__(self, node, bridge, cfg):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from std_msgs.msg import Bool, Empty, String
        self.node, self.bridge, self.log = node, bridge, bridge.log
        self.Bool, self.Empty, self.String = Bool, Empty, String
        self.lock = threading.Lock()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.dash_pub = node.create_publisher(Empty, "/reflex/dash", 10)
        self.det_pub = node.create_publisher(String, "/snn/detection", 10)
        self.auto_pub = node.create_publisher(Bool, "/snn/auto", latched)
        node.create_subscription(Bool, "/snn/auto_set", lambda m: bridge.set_auto(bool(m.data), "/snn/auto_set"), 10)
        node.create_subscription(String, "/dodge/event", self.on_dodge_event, 10)
        # Y on the remote toggles auto-dodge
        self.remote_ok = False
        if cfg.toggle_button != "none" and cfg.remote != "none":
            self.button = RemoteButton(BUTTONS[cfg.toggle_button])
            try:
                from unitree_go.msg import LowState, WirelessController
            except ImportError as e:
                self.log.warning("unitree_go messages not found (%s): no %s toggle from the remote"
                                 % (e, cfg.toggle_button.upper()))
            else:
                if cfg.remote in ("lowstate", "both"):
                    node.create_subscription(LowState, "/lowstate", self.on_lowstate, qos_profile_sensor_data)
                if cfg.remote in ("wirelesscontroller", "both"):
                    node.create_subscription(WirelessController, cfg.controller_topic,
                                             lambda m: self.on_keys("wirelesscontroller", m.keys),
                                             qos_profile_sensor_data)
                self.remote_ok = True
        # Go2 front camera (videohub), polled only while enabled
        self.video_pub, self.pending_id, self.pending_t, self.req_n = None, None, 0.0, 0
        try:
            from unitree_api.msg import Request, Response
        except ImportError as e:
            bridge.color_failed("unitree_api messages not found")
            self.log.warning("unitree_api messages not found (%s): no color camera" % e)
        else:
            self.Request = Request
            self.video_pub = node.create_publisher(Request, "/api/videohub/request", 10)
            node.create_subscription(Response, "/api/videohub/response", self.on_video,
                                     QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
            node.create_timer(1.0 / cfg.color_hz, self.color_tick)
        node.create_timer(0.5, self.graph_tick)
        self.graph_tick()
        self.publish_auto(bridge.auto)

    # --- out ----------------------------------------------------------------------------
    def publish_dash(self):
        with self.lock:
            self.dash_pub.publish(self.Empty())

    def publish_detection(self, det):
        with self.lock:
            self.det_pub.publish(self.String(data=json.dumps(det)))

    def publish_auto(self, enabled):
        with self.lock:
            self.auto_pub.publish(self.Bool(data=bool(enabled)))

    # --- in -----------------------------------------------------------------------------
    def on_lowstate(self, m):
        keys = lowstate_keys(m.wireless_remote)
        if keys is not None:
            self.on_keys("lowstate", keys)

    def on_keys(self, source, keys):
        if self.button.update(source, keys):
            self.bridge.set_auto(not self.bridge.auto, "Y button" if self.bridge.cfg.toggle_button == "y"
                                 else "%s button" % self.bridge.cfg.toggle_button.upper())

    def on_dodge_event(self, m):
        try:
            data = json.loads(m.data)
            if not isinstance(data, dict):
                raise ValueError
        except ValueError:
            data = {"event": "unknown", "raw": m.data[:200]}
        self.bridge.on_robot_event(data)

    def graph_tick(self):
        self.bridge.robot_listeners = self.dash_pub.get_subscription_count()

    def color_tick(self):
        if not self.bridge.color_enabled:
            self.pending_id = None
            return
        now = time.monotonic()
        if self.pending_id is not None and now - self.pending_t < 1.0:
            return                                            # one request at a time (1 s timeout)
        self.req_n += 1
        req = self.Request()
        req.header.identity.id = (time.monotonic_ns() // 1000) * 100 + self.req_n % 100
        req.header.identity.api_id = VIDEOHUB_GET_IMAGE
        with self.lock:
            self.video_pub.publish(req)
        self.pending_id, self.pending_t = req.header.identity.id, now

    def on_video(self, m):
        if m.header.identity.api_id != VIDEOHUB_GET_IMAGE:
            return
        if m.header.identity.id == self.pending_id:
            self.pending_id = None
        if m.header.status.code != 0:
            return self.bridge.color_failed("videohub status %d %s" % (m.header.status.code, m.data[:80]))
        jpeg = jpeg_bytes(m.binary)
        if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8":
            return self.bridge.color_failed("videohub sent no JPEG")
        self.bridge.on_color_jpeg(jpeg)


def declare_params(node):
    """All DEFAULTS as ROS parameters; dynamic typing so `-p theta:=1` or `-p web_port:=8080.0` are fine."""
    try:
        from rcl_interfaces.msg import ParameterDescriptor
        desc = ParameterDescriptor(dynamic_typing=True)
    except Exception:
        desc = None
    out = {}
    for name, default in DEFAULTS.items():
        p = node.declare_parameter(name, default, desc) if desc is not None else node.declare_parameter(name, default)
        out[name] = p.value
    return out


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return
    if "--no-ros" in argv:
        argv.remove("--no-ros")
        bridge = Bridge(Config(parse_cli_params(argv)))
        signal.signal(signal.SIGTERM, _raise_interrupt)
        bridge.start()
        try:
            while not bridge.stop.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            bridge.close()
        return

    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
    except ImportError as e:
        raise SystemExit("rclpy not found (%s): source ROS 2 first, or run with --no-ros" % e)
    try:
        from rclpy.executors import ExternalShutdownException
    except ImportError:
        ExternalShutdownException = KeyboardInterrupt
    signal.signal(signal.SIGTERM, _raise_interrupt)          # before rclpy.init: rclpy chains to it
    rclpy.init(args=[sys.argv[0]] + argv)
    node = rclpy.create_node("snn_bridge")
    bridge = None
    try:
        bridge = Bridge(Config(declare_params(node)), Log(node.get_logger()))
        bridge.ros = RosLink(node, bridge, bridge.cfg)
        bridge.start()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if bridge is not None:
            bridge.close()
        node.destroy_node()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

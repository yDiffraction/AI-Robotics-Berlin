#!/usr/bin/env python3
"""Tests for dodge/snn_bridge.py without ROS (Python 3.10+ and numpy only).

  python3 dodge/tests/test_snn_bridge.py            (or: python3 -m unittest discover dodge/tests)

Pure parts: CBOR / SPLEAT messages, event binning, the display bitmaps, the ema / diff readouts (ema checked
against Tim's perception readout), the remote Y edge, readout.json, the mock SNN, .bin reading, parameters.
In-process: the SPLEAT client and the bridge's FPGA loop against a fake SPLEAT server, and the web API
(/api/state, POST /api/*, /stream) with a synthetic recording replayed through the mock SNN.
"""
import array
import base64
import json
import os
import random
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import snn_bridge as B  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def cbor(v):
    """Tiny CBOR encoder for the test messages (int, str, bytes, float32, None, bool, dict)."""
    if v is None:
        return b"\xf6"
    if v is True or v is False:
        return b"\xf5" if v else b"\xf4"
    if isinstance(v, int):
        return B.cbor_head(0, v) if v >= 0 else B.cbor_head(1, -1 - v)
    if isinstance(v, float):
        return b"\xfa" + struct.pack(">f", v)
    if isinstance(v, bytes):
        return B.cbor_head(2, len(v)) + v
    if isinstance(v, str):
        return B.cbor_head(3, len(v.encode())) + v.encode()
    if isinstance(v, dict):
        return B.cbor_head(5, len(v)) + b"".join(cbor(k) + cbor(x) for k, x in v.items())
    raise TypeError(v)


def result(fid, outs, latency=1234, error=None):
    body = cbor({"id": fid, "payload": np.asarray(outs, "<f4").tobytes(), "latency": latency, "error": error})
    return struct.pack(">I", len(body)) + body


class TimReadout:
    """Reference: perception/viewer.py Readout on origin/Tim (copied unchanged)."""

    def __init__(self, alpha, theta, refractory_frames=100):
        self.alpha, self.theta, self.refr = alpha, theta, refractory_frames
        self.y, self.last = 0.0, -10 ** 9

    def step(self, k, spike):
        prev = self.y
        self.y = self.alpha * self.y + (1 - self.alpha) * spike
        alarm = prev < self.theta <= self.y and k - self.last >= self.refr
        if alarm:
            self.last = k
        return self.y, alarm


class FakeSpleat(threading.Thread):
    """SPLEAT server stand-in: answers every request with 11 outputs (ch0 = 1 if the frame has any event in
    rows < 16), after first sending a result for another client's id (the real server broadcasts)."""

    def __init__(self, error=None):
        super().__init__(daemon=True)
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.error, self.requests, self.shapes_ok = error, 0, True
        self.stop = threading.Event()

    def run(self):
        self.server.settimeout(0.2)
        while not self.stop.is_set():
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:                                 # closed by close()
                return
            threading.Thread(target=self.serve, args=(conn,), daemon=True).start()

    def serve(self, conn):
        with conn:
            self._serve(conn)

    def _serve(self, conn):
        conn.settimeout(0.2)
        buf = b""
        while not self.stop.is_set():
            try:
                chunk = conn.recv(1 << 16)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while len(buf) >= 4 and len(buf) >= 4 + struct.unpack(">I", buf[:4])[0]:
                n = struct.unpack(">I", buf[:4])[0]
                msg, _ = B.cbor_decode(buf[4:4 + n])
                buf = buf[4 + n:]
                frame = np.frombuffer(msg["payload"], "<f4")
                self.shapes_ok &= frame.size == 2 * 80 * 80
                self.requests += 1
                outs = np.zeros(11, np.float32)
                outs[0] = float(frame.reshape(2, 80, 80)[:, :16, :].any())
                conn.sendall(result(msg["id"] + 7, np.ones(11)) + result(msg["id"], outs, error=self.error))

    def close(self):
        self.stop.set()
        self.server.close()


def write_bin(path, x, y, p, t):
    ev = np.zeros(len(t), B.EVENT_DTYPE)
    ev["x"], ev["y"], ev["p"], ev["t"] = x, y, p, t
    with open(path, "wb") as fh:
        fh.write(struct.pack(">I", ev.nbytes))
        fh.write(ev.tobytes())


def synthetic_recording(path, seconds=3.0, throw_at=1.0, seed=1):
    """Noise in the lower half every 10 ms, and a 'ball' crossing the rows < 64 px (cells < 16) for 0.3 s."""
    rng = np.random.default_rng(seed)
    xs, ys, ps, ts = [], [], [], []
    for k in range(int(seconds * 100)):
        n = 40
        xs.append(rng.integers(0, 320, n))
        ys.append(rng.integers(160, 320, n))
        ps.append(rng.integers(0, 2, n))
        ts.append(1_000_000 + k * 10_000 + np.sort(rng.integers(0, 10_000, n)))
        if throw_at <= k / 100 < throw_at + 0.3:
            m = 30
            xs.append(rng.integers(100, 220, m))
            ys.append(rng.integers(0, 60, m))
            ps.append(rng.integers(0, 2, m))
            ts.append(1_000_000 + k * 10_000 + np.sort(rng.integers(0, 10_000, m)))
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    write_bin(path, np.concatenate(xs)[order], np.concatenate(ys)[order], np.concatenate(ps)[order], t[order])


def frames_of(binner, events):
    x, y, p, t = (np.asarray(v, np.int64) for v in zip(*events))
    return binner.feed(x, y, p, t)


# ----------------------------------------------------------------------------- CBOR / SPLEAT
class TestCbor(unittest.TestCase):
    def test_request_roundtrip(self):
        payload = np.arange(2 * 80 * 80, dtype="<f4").tobytes()
        for fid in (0, 23, 24, 255, 256, 65536, 1 << 33):
            msg = B.spleat_request(fid, payload)
            n = struct.unpack(">I", msg[:4])[0]
            self.assertEqual(n, len(msg) - 4)
            d, end = B.cbor_decode(msg[4:])
            self.assertEqual(end, n)
            self.assertEqual(d["id"], fid)
            self.assertEqual(d["payload"], payload)

    def test_result(self):
        body = result(41, [1, 0, 0.5] + [0] * 8, latency=1500)[4:]
        fid, out, lat, err = B.decode_result(body)
        self.assertEqual((fid, lat, err), (41, 1500.0, ""))
        self.assertEqual(out.dtype, np.dtype("<f4"))
        self.assertEqual(list(out[:3]), [1.0, 0.0, 0.5])
        fid, out, lat, err = B.decode_result(cbor({"id": 3, "error": "bad shape", "latency": None}))
        self.assertEqual((fid, err, len(out)), (3, "bad shape", 0))

    def test_values(self):
        self.assertEqual(B.cbor_decode(b"\x39\x01\x00")[0], -257)
        self.assertEqual(B.cbor_decode(b"\xf9\x3c\x00")[0], 1.0)                       # half float
        self.assertEqual(B.cbor_decode(b"\xfb" + struct.pack(">d", 2.5))[0], 2.5)
        self.assertEqual(B.cbor_decode(b"\x83\x01\x02\x03")[0], [1, 2, 3])
        self.assertEqual(B.cbor_decode(b"\xc1\x1a\x00\x00\x00\x05")[0], 5)             # tag skipped
        self.assertEqual([B.cbor_decode(b)[0] for b in (b"\xf4", b"\xf5", b"\xf6")], [False, True, None])

    def test_bad(self):
        for bad in (b"", b"\x5a\x00\x00\x00\x09ab", b"\xa1\x61", b"\x9f", b"\x1c"):
            with self.assertRaises(ValueError):
                B.cbor_decode(bad)
        with self.assertRaises(ValueError):
            B.decode_result(cbor({"id": 1, "payload": b"abc"}))                      # not float32
        with self.assertRaises(ValueError):
            B.decode_result(cbor({"id": 1, "payload": b"abcd"}) + b"\x00")           # trailing bytes
        with self.assertRaises(ValueError):
            B.decode_result(cbor(5))


class TestSpleatClient(unittest.TestCase):
    def test_matches_own_id(self):
        srv = FakeSpleat()
        srv.start()
        try:
            c = B.SpleatClient("127.0.0.1", srv.port)
            c.connect()
            f = np.zeros((2, 80, 80), np.uint8)
            out, lat = c.infer(f)
            self.assertEqual(out[0], 0.0)                     # the foreign result (all ones) was skipped
            self.assertEqual(lat, 1234.0)
            f[1, 3, 40] = 1
            self.assertEqual(c.infer(f)[0][0], 1.0)
            self.assertTrue(srv.shapes_ok)
            c.close()
        finally:
            srv.close()

    def test_error_and_timeout(self):
        srv = FakeSpleat(error="model not loaded")
        srv.start()
        try:
            c = B.SpleatClient("127.0.0.1", srv.port)
            c.connect()
            with self.assertRaises(ConnectionError):
                c.infer(np.zeros((2, 80, 80), np.uint8))
            c.close()
        finally:
            srv.close()
        silent = socket.create_server(("127.0.0.1", 0))      # accepts, never answers
        c = B.SpleatClient("127.0.0.1", silent.getsockname()[1], timeout=0.3)
        try:
            c.connect()
            t0 = time.monotonic()
            with self.assertRaises(OSError):
                c.infer(np.zeros((2, 80, 80), np.uint8))
            self.assertLess(time.monotonic() - t0, 1.0)
        finally:
            c.close()
            silent.close()

    def test_ids_not_from_zero(self):
        self.assertGreaterEqual(B.SpleatClient("127.0.0.1", 1).next_id, 1 << 20)


# ----------------------------------------------------------------------------- frames and display
class TestBinner(unittest.TestCase):
    def test_cells_and_order(self):
        b = B.Binner()
        out = frames_of(b, [(0, 0, 1, 5), (3, 3, 1, 9_999), (319, 319, 0, 10_000), (4, 0, 1, 10_001)])
        self.assertEqual(len(out), 1)                       # frame 0 done when frame 1 starts
        f0 = out[0]
        self.assertEqual(f0.shape, (2, 80, 80))
        self.assertEqual(int(f0.sum()), 1)                  # (0,0) and (3,3) share the 4x4 block
        self.assertEqual(f0[1, 0, 0], 1)
        out = frames_of(b, [(10, 10, 0, 20_000)])
        f1 = out[0]
        self.assertEqual(f1[0, 79, 79], 1)                  # x = y = 319, OFF
        self.assertEqual(f1[1, 0, 1], 1)                    # x = 4 -> cell 1
        self.assertEqual(int(f1.sum()), 2)

    def test_gap_late_unsorted_and_clock_jump(self):
        b = B.Binner()
        frames_of(b, [(0, 0, 1, 0)])
        out = frames_of(b, [(0, 0, 1, 500_000)])            # 0.5 s pause: frame 0 + at most MAX_GAP empty frames
        self.assertEqual(len(out), 1 + B.MAX_GAP)
        self.assertTrue(all(int(f.sum()) == 0 for f in out[1:]))
        out = frames_of(b, [(8, 8, 1, 400_000)])            # late event of a frame already sent: dropped
        self.assertEqual((out, b.late), ([], 1))
        out = frames_of(b, [(40, 40, 1, 520_000), (0, 0, 0, 510_000)])   # unsorted batch
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1][0, 0, 0], 1)                # frame 51 has the OFF event
        frames_of(b, [(0, 0, 1, 3_000_000)])                # frame 300
        out = frames_of(b, [(0, 0, 1, 1_500_000)])          # back by 1.5 s: camera restarted, start over
        self.assertEqual((out, b.cur), ([], 150))
        out = frames_of(b, [(0, 0, 1, 1_000_000)])          # back by 0.5 s only: late events
        self.assertEqual((out, b.cur, b.late), ([], 150, 2))

    def test_display_bitmaps(self):
        d = B.Display()
        d.add(np.array([0, 2, 319]), np.array([0, 0, 319]), np.array([1, 1, 0]))
        on, off, n = d.take()
        self.assertEqual((len(on), len(off), n), (3200, 3200, 3))
        bits_on = np.unpackbits(on).reshape(160, 160)
        self.assertEqual(on[0], 0b11000000)                 # MSB first: pixels (0,0) and (1,0)
        self.assertEqual(int(bits_on.sum()), 2)
        self.assertEqual(np.unpackbits(off).reshape(160, 160)[159, 159], 1)
        self.assertEqual(len(B.b64(on)), 4268)
        self.assertEqual(base64.b64decode(B.b64(on)), on.tobytes())
        on, off, n = d.take()                               # take() starts over
        self.assertEqual((int(on.sum()), int(off.sum()), n), (0, 0, 0))


# ----------------------------------------------------------------------------- readouts
class TestReadout(unittest.TestCase):
    def test_ema_equals_tims_readout(self):
        rng = random.Random(3)
        for theta in (0.5, 0.7, 0.85):
            ours, ref = B.EmaReadout(theta, 0.85, 100), TimReadout(0.85, theta)
            bursty = 0
            for k in range(20000):
                if rng.random() < 0.01:
                    bursty = rng.randint(3, 30)
                s = 1.0 if bursty > 0 or rng.random() < 0.05 else 0.0
                bursty -= 1
                y1, a1 = ours.step(k, np.array([s, 0, 0], np.float32))
                y2, a2 = ref.step(k, s)
                self.assertAlmostEqual(y1, y2, places=12)
                self.assertEqual(a1, a2)

    def test_ema_crossing_refractory_reset(self):
        r = B.EmaReadout(0.85, 0.85, refractory=100)
        alarms = [k for k in range(40) if r.step(k, [1.0])[1]]
        self.assertEqual(alarms, [11])                      # 1 - 0.85^12 >= 0.85 at the 12th spike, once
        r.reset()
        self.assertEqual(r.value, 0.0)
        alarms = [k for k in range(40, 80) if r.step(k, [1.0])[1]]
        self.assertEqual(alarms, [])                        # crossing again within 1 s: refractory
        r.reset()
        alarms = [k for k in range(200, 240) if r.step(k, [1.0])[1]]
        self.assertEqual(alarms, [211])
        r2 = B.EmaReadout(0.85, 0.85, refractory=0, channel=2)
        self.assertFalse(any(r2.step(k, [1.0, 1.0, 0.0])[1] for k in range(30)))   # watches channel 2 only

    def test_diff(self):
        r = B.DiffReadout(3, window=10, refractory=50)
        seq = [(1, 0)] * 2 + [(1, 1)] * 2 + [(1, 0)]
        out = [r.step(k, [a, b]) for k, (a, b) in enumerate(seq)]
        self.assertEqual([v for v, _ in out], [1, 2, 2, 2, 3])
        self.assertEqual([a for _, a in out], [False] * 4 + [True])
        self.assertFalse(r.step(5, [1, 0])[1])              # d = 4 >= 3 but refractory
        for k in range(6, 20):
            r.step(k, [0, 1])
        self.assertEqual(r.value, -10)                      # the window keeps 10 frames
        r.reset()
        self.assertEqual(r.value, 0)

    def test_load_readout(self):
        ro = B.load_readout(os.path.join(os.path.dirname(B.__file__), "model", "readout.json"))
        self.assertEqual((ro["mode"], ro["alpha"], ro["theta"], ro["channel"], ro["error"]),
                         ("ema", 0.85, 0.85, 0, ""))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            with open(p, "w") as fh:
                json.dump({"readout": "diff", "theta": 4, "window": 12, "name": "ball r1"}, fh)
            ro = B.load_readout(p)
            self.assertEqual((ro["mode"], ro["theta"], ro["window"], ro["name"]), ("diff", 4.0, 12, "ball r1"))
            self.assertTrue(B.load_readout(os.path.join(d, "missing.json"))["error"])


# ----------------------------------------------------------------------------- remote, mock, files, params
class TestRemote(unittest.TestCase):
    def test_y_edges(self):
        y = 1 << B.BUTTONS["y"]
        btn = B.RemoteButton(B.BUTTONS["y"], debounce=0.4)
        self.assertFalse(btn.update("lowstate", y, now=0.0))                # first message: reference only
        self.assertFalse(btn.update("lowstate", y, now=0.1))                # still held
        self.assertFalse(btn.update("lowstate", 0, now=0.2))
        self.assertTrue(btn.update("lowstate", y, now=1.0))                 # press
        self.assertFalse(btn.update("lowstate", y | 1 << 10, now=1.1))      # X added while Y held: no Y press
        self.assertFalse(btn.update("wirelesscontroller", 0, now=1.0))
        self.assertFalse(btn.update("wirelesscontroller", y, now=1.05))     # same press on the other topic
        btn.update("lowstate", 0, now=1.2)
        self.assertTrue(btn.update("lowstate", y, now=1.6))                 # next press after the debounce
        self.assertFalse(btn.update("lowstate", 1 << 10, now=3.0))          # other buttons do nothing

    def test_lowstate_keys(self):
        r = np.zeros(40, np.uint8)
        self.assertIsNone(B.lowstate_keys(r))
        r[:4] = [0x55, 0x51, 0x00, 0x08]                                    # bit 11 (Y) in the high byte
        self.assertEqual(B.lowstate_keys(r), 1 << 11)
        self.assertEqual(B.lowstate_keys(bytes([0x55, 0x51, 0x00, 0x04]) + bytes(36)), 1 << 10)
        self.assertIsNone(B.lowstate_keys([0x55]))

    def test_jpeg_bytes(self):
        raw = bytes([0xFF, 0xD8, 0x00, 0x7F, 0x80, 0xFF, 0xD9])
        signed = array.array("b", [v - 256 if v > 127 else v for v in raw])
        self.assertEqual(B.jpeg_bytes(signed), raw)
        self.assertEqual(B.jpeg_bytes(list(signed)), raw)
        self.assertEqual(B.jpeg_bytes(np.frombuffer(raw, np.int8)), raw)


class TestMock(unittest.TestCase):
    def test_tripwire_hold(self):
        m = B.MockSnn(hold=15, noise=0.0)
        empty = np.zeros((2, 80, 80), np.uint8)
        low = empty.copy()
        low[0, 40:60, 10:60] = 1                            # lots of motion, but low in the image
        top = empty.copy()
        top[1, 5, 10:13] = 1                                # 3 cells in the airspace
        self.assertEqual(m.infer(low)[0][0], 0.0)
        self.assertEqual(m.infer(top)[0][0], 0.0)           # 3 cells
        out, lat = m.infer(top)                             # 3 + 3 within 2 frames: fires
        self.assertEqual(out[0], 1.0)
        self.assertEqual(len(out), 11)
        self.assertGreaterEqual(lat, 0)
        spikes = [m.infer(empty)[0][0] for _ in range(20)]
        self.assertEqual(sum(spikes), 14)                   # held for 15 frames in total

    def test_noise_is_rare(self):
        m = B.MockSnn(noise=0.002, seed=5)
        empty = np.zeros((2, 80, 80), np.uint8)
        n = sum(m.infer(empty)[0][:3].sum() for _ in range(5000))
        self.assertLess(n, 80)
        r = B.EmaReadout(0.85, 0.85)
        m = B.MockSnn(noise=0.002, seed=6)
        self.assertFalse(any(r.step(k, m.infer(empty)[0])[1] for k in range(5000)))   # noise never alarms


class TestFilesAndParams(unittest.TestCase):
    def test_read_bin(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.bin")
            write_bin(p, [1, 400, 3], [2, 5, 330], [1, 0, 7], [300, 100, 200])
            x, y, p_, t = B.read_bin(p)
            self.assertEqual(list(t), [100, 200, 300])      # sorted
            self.assertEqual(list(x), [319, 3, 1])          # clipped to the sensor
            self.assertEqual(list(y), [5, 319, 2])
            self.assertEqual(list(p_), [0, 1, 1])
            with open(p, "ab") as fh:
                fh.write(b"\x00" * 5)                       # a cut-off event is ignored
            self.assertEqual(len(B.read_bin(p)[3]), 3)

    def test_params(self):
        self.assertEqual(B.parse_cli_params(["--ros-args", "-p", "camera:=replay:/a b.bin", "-p", "theta:=0.9",
                                             "-p", "auto:=true", "-p", "web_port:=8081", "-p", "ssh_host:='x'"]),
                         {"camera": "replay:/a b.bin", "theta": 0.9, "auto": True, "web_port": 8081, "ssh_host": "x"})
        c = B.Config({"camera": "replay:/x.bin", "theta": 0.9, "auto": "true", "web_port": 8081.0, "color_hz": 2})
        self.assertEqual((c.camera_mode, c.replay_path, c.theta, c.auto, c.web_port, c.color_hz),
                         ("replay", "/x.bin", "0.9", True, 8081, 2.0))
        for bad in ({"camera": "usb"}, {"camera": "replay:"}, {"snn": "gpu"}, {"web_port": 1.5},
                    {"auto": "maybe"}, {"toggle_button": "z"}, {"nonsense": 1}, {"readout_mode": "max"}):
            with self.assertRaises(SystemExit):
                B.Config(bad)

    def test_theta_from_file_and_overrides(self):
        self.assertEqual(B.Bridge(B.Config({"snn": "mock"})).readout.theta, 0.85)
        b = B.Bridge(B.Config({"snn": "mock", "theta": "0.7"}))
        self.assertEqual((b.readout.mode, b.readout.theta, b.readout.alpha), ("ema", 0.7, 0.85))
        b = B.Bridge(B.Config({"snn": "mock", "readout_mode": "diff"}))
        self.assertEqual((b.readout.mode, b.readout.theta, b.readout.window), ("diff", 3.0, 10))
        b.set_theta(50)
        self.assertEqual(b.readout.theta, 10.0)             # clamped to the window


# ----------------------------------------------------------------------------- bridge, in process
class FakeRos:
    remote_ok = True

    def __init__(self):
        self.dashes, self.detections, self.autos = 0, [], []

    def publish_dash(self):
        self.dashes += 1

    def publish_detection(self, det):
        self.detections.append(det)

    def publish_auto(self, v):
        self.autos.append(v)


def get(port, path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=3) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(port, path, body):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def read_sse(port, seconds, until=None):
    """Events of /stream for `seconds` (or until until(event, data) is true) -> list of (event, data)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=1)
    s.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
    buf, events, t_end = b"", [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            lines = dict(line.split(": ", 1) for line in block.decode().split("\n") if ": " in line)
            if "event" in lines:
                events.append((lines["event"], json.loads(lines["data"])))
                if until and until(*events[-1]):
                    s.close()
                    return events
    s.close()
    return events


class TestBridge(unittest.TestCase):
    def wait(self, cond, timeout=5.0):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if cond():
                return True
            time.sleep(0.02)
        return False

    def test_web_api_and_stream(self):
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, "rec.bin")
            synthetic_recording(rec)
            web = os.path.join(d, "web")
            os.mkdir(web)
            with open(os.path.join(web, "index.html"), "w") as fh:
                fh.write("<!doctype html><title>t</title>")
            b = B.Bridge(B.Config({"camera": "replay:" + rec, "snn": "mock", "web_port": 0, "web_host": "127.0.0.1",
                                   "web_dir": web}))
            b.start()
            try:
                self.assertTrue(self.wait(lambda: b.web_port))
                port = b.web_port
                code, ctype, body = get(port, "/api/state")
                s = json.loads(body)
                self.assertEqual((code, ctype), (200, "application/json"))
                self.assertEqual(set(s) >= {"model", "camera", "fpga", "ros", "robot", "color", "counters",
                                            "server_time"}, True)
                self.assertTrue({"name", "readout", "theta", "channel", "mock"} <= set(s["model"]))
                self.assertEqual((s["model"]["readout"], s["model"]["mock"], s["fpga"]["state"],
                                  s["camera"]["state"], s["ros"], s["robot"]["state"]),
                                 ("ema", True, "mock", "replay", dict(s["ros"], available=False, auto=False),
                                  "unknown"))
                self.assertTrue({"latency_ms", "fps"} <= set(s["fpga"]) and "rate" in s["camera"])
                self.assertEqual(s["color"], dict(s["color"], enabled=False, available=False, seq=0))
                self.assertEqual(s["counters"], {"detections": 0, "dodges": 0})
                self.assertEqual(get(port, "/")[2], b"<!doctype html><title>t</title>")
                with self.assertRaises(urllib.error.HTTPError):
                    get(port, "/color.jpg")
                with self.assertRaises(urllib.error.HTTPError):
                    get(port, "/../snn_bridge.py")

                events = read_sse(port, 4.0, until=lambda e, dt: e == "detection")
                kinds = [e for e, _ in events]
                self.assertEqual(kinds[0], "status")
                self.assertIn("frame", kinds)
                self.assertIn("snn", kinds)
                self.assertEqual(kinds[-1], "detection")    # the 'ball' at 1.0 s
                fr = next(dt for e, dt in events if e == "frame")
                self.assertEqual((fr["w"], fr["h"], len(base64.b64decode(fr["on"])),
                                  len(base64.b64decode(fr["off"]))), (160, 160, 3200, 3200))
                ks = [f["k"] for e, dt in events if e == "snn" for f in dt["frames"]]
                self.assertEqual(ks, sorted(set(ks)))           # every frame once, in order
                self.assertTrue(all(len(f["s"]) == 3 for e, dt in events if e == "snn" for f in dt["frames"]))
                det = events[-1][1]
                self.assertEqual((det["acted"], det["reason"], det["theta"]), (False, "auto-dodge off", 0.85))
                self.assertGreaterEqual(det["y"], 0.85)
                self.assertTrue(set(det) >= {"t", "k", "y", "theta", "acted"})

                code, s = post(port, "/api/auto", {"enabled": True})
                self.assertEqual((code, s["ros"]["auto"], s["ros"]["auto_source"]), (200, True, "web"))
                code, s = post(port, "/api/theta", {"theta": 0.6})
                self.assertEqual((code, s["model"]["theta"]), (200, 0.6))
                code, s = post(port, "/api/color", {"enabled": True})
                self.assertEqual((code, s["color"]["enabled"]), (200, True))
                self.assertEqual(post(port, "/api/auto", {"enabled": "x"})[0], 400)
                self.assertEqual(post(port, "/api/theta", {"theta": None})[0], 400)
                # a page that connects later gets the recent SNN history first
                events = read_sse(port, 1.0)
                self.assertEqual([e for e, _ in events][:2], ["status", "snn"])
                self.assertGreater(len(events[1][1]["frames"]), 50)
            finally:
                b.close()
            self.assertFalse(any(th.is_alive() for th in b.threads))

    def test_acting_only_when_auto_fresh_and_ros(self):
        b = B.Bridge(B.Config({"snn": "mock"}))
        now = time.monotonic()
        b._detection(10, 0.9, now)
        self.assertEqual(b.last_detection["reason"], "auto-dodge off")
        b.set_auto(True, "test")
        b._detection(200, 0.9, now)
        self.assertEqual((b.last_detection["acted"], b.last_detection["reason"]), (False, "no ROS"))
        ros = b.ros = FakeRos()
        b._detection(400, 0.9, time.monotonic() - 1.0)                       # frame from 1 s ago
        self.assertEqual((ros.dashes, b.last_detection["reason"]), (0, "stale frame"))
        b._detection(600, 0.9, time.monotonic())
        self.assertEqual((ros.dashes, b.last_detection["acted"], len(ros.detections)), (1, True, 2))
        self.assertTrue(b.set_auto(False, "Y button"))
        self.assertFalse(b.set_auto(False, "Y button"))                      # no change, nothing published
        self.assertEqual(ros.autos, [False])
        b._detection(800, 0.9, time.monotonic())
        self.assertEqual(ros.dashes, 1)
        self.assertEqual(b.snapshot()["counters"]["detections"], 5)

    def test_robot_events(self):
        b = B.Bridge(B.Config({"snn": "mock"}))
        b.ros = FakeRos()
        self.assertEqual(b.snapshot()["robot"]["state"], "unknown")           # nothing listens to /reflex/dash
        b.robot_listeners = 1
        self.assertEqual(b.snapshot()["robot"]["state"], "idle")
        b.on_robot_event({"event": "dash", "source": "button X (lowstate)", "wall_time": 1.0})
        b.on_robot_event({"event": "hold", "out_m": 0.42})
        s = b.snapshot()
        self.assertEqual((s["robot"]["state"], s["robot"]["out_m"], s["counters"]["dodges"]), ("hold", 0.42, 1))
        self.assertEqual(s["robot"]["last"], {"event": "hold", "out_m": 0.42})
        self.assertEqual(s["recent"][0]["source"], "X button")
        b.on_robot_event({"event": "ignored", "source": "/reflex/dash", "state": "hold"})
        self.assertEqual(b.snapshot()["robot"]["state"], "hold")
        b.on_robot_event({"event": "idle", "reason": "back at the start"})
        self.assertEqual(b.snapshot()["robot"]["state"], "idle")

    def test_fpga_loop_against_fake_server(self):
        srv = FakeSpleat()
        srv.start()
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, "rec.bin")
            synthetic_recording(rec, seconds=2.0, throw_at=0.5)
            b = B.Bridge(B.Config({"camera": "replay:" + rec, "snn": "fpga", "spleat_port": srv.port,
                                   "web_port": 0, "web_host": "127.0.0.1"}))
            b.start()
            try:
                self.assertTrue(self.wait(lambda: b.snapshot()["fpga"]["state"] == "connected"))
                self.assertTrue(self.wait(lambda: b.detections >= 1, timeout=4))
                s = b.snapshot()
                self.assertEqual((s["model"]["mock"], s["fpga"]["latency_ms"]), (False, 1.23))
                self.assertGreater(s["fpga"]["fps"], 50)
                self.assertTrue(srv.shapes_ok)
                srv.close()                                                   # board gone
                self.assertTrue(self.wait(lambda: b.snapshot()["fpga"]["state"] == "offline", timeout=4))
                self.assertEqual(b.readout.value, 0.0)                         # readout reset
            finally:
                b.close()
                srv.close()

    def test_unreachable_fpga_is_offline(self):
        free = socket.create_server(("127.0.0.1", 0))
        port = free.getsockname()[1]
        free.close()
        b = B.Bridge(B.Config({"camera": "listen", "listen_port": 0, "snn": "fpga", "spleat_port": port,
                               "web_port": 0, "web_host": "127.0.0.1"}))
        b.start()
        try:
            self.assertTrue(self.wait(lambda: b.fpga_error != ""))
            s = b.snapshot()
            self.assertEqual((s["fpga"]["state"], s["camera"]["state"]), ("offline", "waiting"))
        finally:
            b.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

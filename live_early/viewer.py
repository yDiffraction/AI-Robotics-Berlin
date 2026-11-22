#!/usr/bin/env python3
"""Live viewer for the early throw-motion detector (GOAL B, runs/early_ctx v2).

GenX320 on the Kria -> events to this laptop (despot_debug_camera, started over SSH) -> binary 10 ms frames
(80x80 cells, 2 polarities, cell = 1 if any event; the training binning) -> SNN on the FPGA (SPLEAT server,
TCP 3333) and/or the float model on this laptop -> host readout on output channel 0
(y = 0.85 * y + 0.15 * spike, alarm on an upward crossing of theta, 1 s refractory) -> browser page with the
event image, the readout trace, output spikes and alarms.

  python live_early/viewer.py                                   # camera, FPGA, http://127.0.0.1:8770
  python live_early/viewer.py --backend both                    # FPGA and float model side by side
  python live_early/viewer.py --replay live_early/test/x.npy --backend float      # no board needed
  python live_early/viewer.py --replay x.npy --backend both --check               # headless FPGA vs float check

Stop other users of the camera on the board first (despot_send_and_forget, record_tool, live_view).
The theta in model/readout.json is PROVISIONAL (picked on held-out test sessions); recalibrate on the new session.
"""
import argparse
import base64
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SENSOR, GRID, FRAME_US = 320, 80, 10_000
EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])   # camera stream and .bin files
HISTORY = 600   # frames kept for the trace (6 s)


# ----------------------------------------------------------------------------- frame sources
def start_camera(host, port):
    """Start despot_debug_camera on the board; it connects back to this laptop (SSH client address) on `port`."""
    command = f"""set -- $SSH_CONNECTION
despot_debug_camera --host "$1" --port {port} >/tmp/early_viewer_camera.log 2>&1 &
camera_pid=$!
trap 'kill "$camera_pid" 2>/dev/null; wait "$camera_pid" 2>/dev/null' EXIT
trap 'exit' HUP INT TERM
read -r stop
"""
    return subprocess.Popen(["ssh", "-4", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", host, command],
                            stdin=subprocess.PIPE)


def stop_camera(proc):
    try:
        proc.communicate(input=b"\n", timeout=5)
    except Exception:
        proc.kill()


def camera_frames(a, stop):
    proc = None
    with socket.create_server(("", a.listen_port)) as server:
        server.settimeout(1.0)
        if not a.no_start:
            proc = start_camera(a.ssh, a.listen_port)
        print(f"waiting for despot_debug_camera on port {a.listen_port}", flush=True)
        try:
            while not stop.is_set():
                try:
                    conn, addr = server.accept()
                    break
                except socket.timeout:
                    if proc is not None and proc.poll() is not None:
                        raise SystemExit("camera start over SSH failed (see /tmp/early_viewer_camera.log on the board)")
            else:
                return
            print(f"camera connected from {addr[0]}", flush=True)
            frame = np.zeros(2 * GRID * GRID, np.float32)
            cur = None
            with conn:
                conn.settimeout(2.0)
                while not stop.is_set():
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
                    p = np.minimum(ev["p"], 1).astype(np.intp)
                    cell = (p * GRID + y * GRID // SENSOR) * GRID + x * GRID // SENSOR
                    num = ev["t"].astype(np.int64) // FRAME_US
                    for k in np.unique(num):
                        if cur is None:
                            cur = k
                        if k > cur:
                            yield frame.reshape(2, GRID, GRID)
                            for _ in range(min(int(k - cur - 1), 20)):   # pauses without events
                                yield np.zeros((2, GRID, GRID), np.float32)
                            frame = np.zeros(2 * GRID * GRID, np.float32)
                            cur = k
                        frame[cell[num == k]] = 1.0
        finally:
            if proc is not None:
                stop_camera(proc)


def replay_frames(a, stop):
    """.npy (bit-packed frames from runs/early_ctx/prep.py) or .bin (record_tool: 4-byte header + events)."""
    path = Path(a.replay)
    while True:
        if path.suffix == ".npy":
            packed = np.load(path)
            gen = (np.unpackbits(packed[k]).reshape(2, GRID, GRID).astype(np.float32) for k in range(len(packed)))
        else:
            raw = path.read_bytes()
            ev = np.frombuffer(raw[4:4 + (len(raw) - 4) // EVENT_DTYPE.itemsize * EVENT_DTYPE.itemsize], dtype=EVENT_DTYPE)
            t = ev["t"].astype(np.int64)
            n = (t - t.min()) // FRAME_US
            order = np.argsort(n, kind="stable")
            n, ev = n[order], ev[order]
            cell = (np.minimum(ev["p"], 1).astype(np.int64) * GRID + np.minimum(ev["y"], SENSOR - 1) * GRID // SENSOR) * GRID \
                + np.minimum(ev["x"], SENSOR - 1) * GRID // SENSOR
            bounds = np.searchsorted(n, np.arange(int(n[-1]) + 2))

            def gen_bin():
                for k in range(int(n[-1]) + 1):
                    f = np.zeros(2 * GRID * GRID, np.float32)
                    f[cell[bounds[k]:bounds[k + 1]]] = 1.0
                    yield f.reshape(2, GRID, GRID)
            gen = gen_bin()
        t0 = time.perf_counter()
        for k, f in enumerate(gen):
            if stop.is_set():
                return
            yield f
            if not a.fast:
                time.sleep(max(0.0, t0 + (k + 1) * FRAME_US / 1e6 - time.perf_counter()))
        if not a.loop:
            return


# ----------------------------------------------------------------------------- backends
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


class Board:
    """SPLEAT server on the Kria: one CBOR request {"id", "payload": float32[2,80,80]} per frame (as tools/ball_live.py)."""

    def __init__(self, host, port=3333):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(2.0)
        self.next_id = 0

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
                return np.frombuffer(msg["payload"], "<f4").copy(), msg.get("latency", -1)


class Float:
    """The float checkpoint on this laptop, stepped frame by frame with its state (no resets)."""

    def __init__(self, ckpt):
        import torch
        sys.path.insert(0, str(HERE))
        import speedpatch  # noqa: F401
        from snn_model import ThrowSNN, arch_from_config
        torch.set_num_threads(2)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = ThrowSNN(1, arch=arch_from_config(ck["config"]))
        m.load_state_dict(ck["model"])
        m.eval()
        self.torch = torch
        self.layers = list(m.backbone) + [m.head]
        self.leak = [float(1 - 1 / L.lif.tau) for L in self.layers]
        self.th = [float(L.lif.v_threshold) for L in self.layers]
        self.V = [None] * len(self.layers)

    def infer(self, frame):
        torch = self.torch
        with torch.no_grad():
            h = torch.from_numpy(np.ascontiguousarray(frame, np.float32))[None, None]
            for i, L in enumerate(self.layers):
                cur = (L.bn(L.conv(h)) if getattr(L, "bn", None) is not None else L.conv(h))[:, 0]
                v = cur if self.V[i] is None else self.V[i] * self.leak[i] + cur
                s = (v >= self.th[i]).float()
                self.V[i] = (1 - s) * v
                h = s[:, None]
        return np.array([float(h.flatten()[0])], np.float32), 0


# ----------------------------------------------------------------------------- readout
class Readout:
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


# ----------------------------------------------------------------------------- web page
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Throw Motion Live</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--mute:#677085;--line:#dde1e8;--ok:#1f8a4c;--alarm:#d1342f;--board:#2f6fdf;--float:#d98a1c;--th:#8a5cd1}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--panel:#1b2029;--ink:#e6e9ef;--mute:#9aa3b5;--line:#2c3340;--ok:#35b66d;--alarm:#ff5a52;--board:#6ea0ff;--float:#f2b14c;--th:#b48cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:minmax(0,480px) minmax(0,1fr)}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
h1{font-size:16px;margin:0 0 10px}#img{width:100%;aspect-ratio:1;image-rendering:pixelated;background:#000;border-radius:6px;display:block}
#status{font-size:26px;font-weight:700;text-align:center;padding:14px;border-radius:8px;color:#fff;background:var(--ok);transition:background .1s}
#status.alarm{background:var(--alarm)}#trace{width:100%;height:260px;display:block}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:10px}
.stat{border:1px solid var(--line);border-radius:8px;padding:8px}.stat b{display:block;font-size:18px}.stat span{color:var(--mute);font-size:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mute);font-size:12px;margin-top:6px}.sw{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:4px}
.note{color:var(--mute);font-size:12px;margin-top:8px}
</style></head><body><main>
<section class="card"><h1>Camera (binary 10 ms frames, last 30 ms)</h1><canvas id="img" width="80" height="80"></canvas>
<div class="legend"><span><i class="sw" style="background:#fff"></i>ON events</span><span><i class="sw" style="background:#3d8bff"></i>OFF events</span></div></section>
<section class="card"><div id="status">no throw motion</div>
<h1 style="margin-top:14px">Model output (channel 0) &mdash; last 6 s</h1><canvas id="trace"></canvas>
<div class="legend" id="legend"></div>
<div class="stats"><div class="stat"><span>readout y</span><b id="sy">&ndash;</b></div><div class="stat"><span>&theta; (provisional)</span><b id="sth">&ndash;</b></div>
<div class="stat"><span>alarms</span><b id="sal">0</b></div><div class="stat"><span>frames/s</span><b id="sfps">&ndash;</b></div>
<div class="stat"><span>board round trip</span><b id="srtt">&ndash;</b></div><div class="stat"><span>frame</span><b id="sk">&ndash;</b></div></div>
<div class="note" id="note"></div></section></main>
<script>
const N=600, css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let H={y:[],s:[],a:[],yf:[],sf:[]}, cfg=null, nAl=0;
const ic=document.getElementById('img').getContext('2d'), im=ic.createImageData(80,80);
const tc=document.getElementById('trace'), tx=tc.getContext('2d');
function fit(){const r=tc.getBoundingClientRect();tc.width=r.width*devicePixelRatio;tc.height=r.height*devicePixelRatio}
addEventListener('resize',fit);fit();
function drawImg(b64){const b=atob(b64);for(let i=0;i<6400;i++){const v=b.charCodeAt(i),o=i*4;
 let r=0,g=0,bl=0;if(v&2){r=g=bl=255}else if(v&1){r=61;g=139;bl=255}im.data[o]=r;im.data[o+1]=g;im.data[o+2]=bl;im.data[o+3]=255}ic.putImageData(im,0,0)}
function line(arr,col,W,Hh,pad,lw){tx.strokeStyle=col;tx.lineWidth=lw;tx.beginPath();const n=arr.length;
 for(let i=0;i<n;i++){const x=W-(n-1-i)*(W/N),y=pad+(1-arr[i])*(Hh-2*pad);i?tx.lineTo(x,y):tx.moveTo(x,y)}tx.stroke()}
function drawTrace(){const W=tc.width,Hh=tc.height,p=10*devicePixelRatio;tx.clearRect(0,0,W,Hh);
 tx.strokeStyle=css('--line');tx.lineWidth=1;for(const v of [0,.25,.5,.75,1]){const y=p+(1-v)*(Hh-2*p);tx.beginPath();tx.moveTo(0,y);tx.lineTo(W,y);tx.stroke()}
 if(cfg){const y=p+(1-cfg.theta)*(Hh-2*p);tx.setLineDash([6,4]);tx.strokeStyle=css('--th');tx.lineWidth=1.5*devicePixelRatio;tx.beginPath();tx.moveTo(0,y);tx.lineTo(W,y);tx.stroke();tx.setLineDash([])}
 const n=H.a.length;tx.fillStyle=css('--alarm');for(let i=0;i<n;i++)if(H.a[i]){const x=W-(n-1-i)*(W/N);tx.fillRect(x-1.5*devicePixelRatio,0,3*devicePixelRatio,Hh)}
 const main=cfg&&cfg.primary=='float'?'--float':'--board';
 tx.globalAlpha=.5;for(let i=0;i<n;i++){if(H.s[i]){const x=W-(n-1-i)*(W/N);tx.fillStyle=css(main);tx.fillRect(x,Hh-6*devicePixelRatio,1.5*devicePixelRatio,6*devicePixelRatio)}}tx.globalAlpha=1;
 if(cfg&&cfg.both)line(H.yf,css('--float'),W,Hh,p,1.5*devicePixelRatio);line(H.y,css(main),W,Hh,p,2.2*devicePixelRatio)}
const es=new EventSource('/stream');
es.onmessage=e=>{const d=JSON.parse(e.data);if(!cfg){cfg=d.cfg;document.getElementById('sth').textContent=cfg.theta.toFixed(2);
 const L=document.getElementById('legend');L.innerHTML=`<span><i class="sw" style="background:var(${cfg.primary=='float'?'--float':'--board'})"></i>${cfg.primary=='float'?'float model (laptop)':'FPGA'} readout y</span>`+(cfg.both?'<span><i class="sw" style="background:var(--float)"></i>float model (laptop)</span>':'')+'<span><i class="sw" style="background:var(--th)"></i>&theta;</span><span><i class="sw" style="background:var(--alarm)"></i>alarm</span><span>ticks = output spikes</span>';
 document.getElementById('note').textContent=cfg.note}
 for(const r of d.rows){H.y.push(r[0]);H.s.push(r[1]);H.a.push(r[2]);H.yf.push(r[3]);H.sf.push(r[4]);if(r[2])nAl++}
 for(const k in H)if(H[k].length>N)H[k].splice(0,H[k].length-N);
 if(d.img)drawImg(d.img);drawTrace();
 const st=document.getElementById('status'),on=d.alarm_age>=0&&d.alarm_age<100;st.className=on?'alarm':'';st.textContent=on?'THROW MOTION':'no throw motion';
 document.getElementById('sy').textContent=H.y.length?H.y[H.y.length-1].toFixed(2):'–';document.getElementById('sal').textContent=nAl;
 document.getElementById('sfps').textContent=d.fps.toFixed(0);document.getElementById('srtt').textContent=d.rtt_ms>=0?d.rtt_ms.toFixed(1)+' ms':'–';document.getElementById('sk').textContent=d.k};
es.onerror=()=>{document.getElementById('status').textContent='viewer stopped';document.getElementById('status').className=''};
</script></body></html>"""


class State:
    def __init__(self, cfg):
        self.lock = threading.Lock()
        self.cfg = cfg
        self.rows = deque(maxlen=HISTORY)
        self.seq = 0
        self.img = np.zeros((GRID, GRID), np.uint8)
        self.k, self.fps, self.rtt, self.last_alarm = 0, 0.0, -1.0, None
        self.done = False


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen, first = None, True
            try:
                while not state.done:
                    with state.lock:
                        seq = state.seq
                        n_new = len(state.rows) if seen is None else min(len(state.rows), seq - seen)
                        rows = list(state.rows)[len(state.rows) - n_new:] if n_new else []
                        msg = {"rows": [[round(r[0], 4), r[1], r[2], round(r[3], 4), r[4]] for r in rows],
                               "img": base64.b64encode(state.img.tobytes()).decode(), "k": state.k,
                               "fps": state.fps, "rtt_ms": state.rtt,
                               "alarm_age": -1 if state.last_alarm is None else state.k - state.last_alarm}
                    seen = seq
                    if first:
                        msg["cfg"] = state.cfg
                        first = False
                    self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
    return H


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "model"), help="folder with readout.json and model.pt (float)")
    ap.add_argument("--backend", choices=["board", "float", "both"], default="board")
    ap.add_argument("--host", default="192.168.77.1", help="SPLEAT server (board)")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--ssh", default="root@192.168.77.1", help="SSH target for starting despot_debug_camera")
    ap.add_argument("--listen-port", type=int, default=3334)
    ap.add_argument("--no-start", action="store_true", help="despot_debug_camera is started by hand")
    ap.add_argument("--replay", help=".npy (packed frames) or .bin recording instead of the camera")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fast", action="store_true", help="replay as fast as possible (no 100 Hz pacing)")
    ap.add_argument("--theta", type=float, default=None, help="override the readout threshold")
    ap.add_argument("--http-port", type=int, default=8770)
    ap.add_argument("--check", action="store_true", help="headless: replay through FPGA and float, print agreement")
    ap.add_argument("--save", help="npz with per-frame outputs")
    a = ap.parse_args()

    ro = json.loads((Path(a.model) / "readout.json").read_text())
    theta = a.theta if a.theta is not None else float(ro["theta"])
    board = Board(a.host, a.port) if a.backend in ("board", "both") else None
    flt = Float(Path(a.model) / "model.pt") if a.backend in ("float", "both") else None
    primary = "board" if board else "float"
    r_main, r_flt = Readout(0.85, theta), Readout(0.85, theta)
    cfg = {"theta": theta, "primary": primary, "both": a.backend == "both",
           "note": f"model {Path(a.model).resolve().name}: alarm when y = EMA(0.85) of output spikes crosses theta; "
                   "theta is provisional until calibrated on the new session."}
    state = State(cfg)
    stop = threading.Event()
    if not a.check:
        srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(state))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"viewer: http://127.0.0.1:{a.http_port}/", flush=True)

    frames = replay_frames(a, stop) if a.replay else camera_frames(a, stop)
    recent = deque(maxlen=3)
    outs, outs_f, rtts, t_fps, n_fps = [], [], [], time.perf_counter(), 0
    try:
        for k, f in enumerate(frames):
            recent.append(f)
            s_b = s_f = 0.0
            rtt = -1.0
            if board:
                t0 = time.perf_counter()
                ob, _ = board.infer(f)
                rtt = (time.perf_counter() - t0) * 1e3
                rtts.append(rtt)
                outs.append(ob)
                s_b = float(ob[0] > 0)
            if flt:
                of, _ = flt.infer(f)
                outs_f.append(of)
                s_f = float(of[0] > 0)
            y, al = r_main.step(k, s_b if board else s_f)
            yf, _ = r_flt.step(k, s_f)
            n_fps += 1
            now = time.perf_counter()
            with state.lock:
                state.rows.append((y, int(s_b if board else s_f), int(al), yf, int(s_f)))
                state.seq += 1
                state.k = k
                if al:
                    state.last_alarm = k
                    print(json.dumps({"alarm": k, "t_s": round(k / 100, 2), "y": round(y, 3)}), flush=True)
                if k % 4 == 0:
                    acc = np.zeros((2, GRID, GRID), bool)
                    for g in recent:
                        acc |= g > 0
                    state.img = (acc[1].astype(np.uint8) * 2 | acc[0].astype(np.uint8))
                if now - t_fps >= 1.0:
                    state.fps, state.rtt = n_fps / (now - t_fps), (float(np.median(rtts[-100:])) if rtts else -1.0)
                    t_fps, n_fps = now, 0
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        state.done = True
    summary = {"frames": len(outs) or len(outs_f)}
    if outs:
        O = np.array(outs)
        summary.update(board_ch0_spikes=int((O[:, 0] > 0).sum()), board_other_ch_spikes=int((O[:, 1:] > 0).sum()),
                       rtt_ms_median=round(float(np.median(rtts)), 2), rtt_ms_p99=round(float(np.percentile(rtts, 99)), 2))
    if outs_f:
        summary["float_ch0_spikes"] = int((np.array(outs_f)[:, 0] > 0).sum())
    if outs and outs_f:
        b, fl = np.array(outs)[:, 0] > 0, np.array(outs_f)[:, 0] > 0
        summary["ch0_frame_agreement"] = round(float((b == fl).mean()), 4)
        yb, yfl = Readout(0.85, theta), Readout(0.85, theta)
        ab = [k for k, v in enumerate(b) if yb.step(k, float(v))[1]]
        af = [k for k, v in enumerate(fl) if yfl.step(k, float(v))[1]]
        summary.update(alarms_board=len(ab), alarms_float=len(af),
                       alarms_matched_50ms=sum(any(abs(x - z) <= 5 for z in ab) for x in af))
    print(json.dumps(summary), flush=True)
    if a.save:
        np.savez_compressed(a.save, board=np.array(outs), float=np.array(outs_f), rtt_ms=np.array(rtts), theta=theta)


if __name__ == "__main__":
    main()

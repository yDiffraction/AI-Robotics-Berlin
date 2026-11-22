#!/usr/bin/env python3
"""Record a whole take from the event camera and mark the throws by key while it runs.

  python tools/record_tool.py                              # -> dataset/04/samples
  python tools/record_tool.py --out dataset/05/samples --host root@durance-kria.local
then open http://127.0.0.1:8766

Buttons: Start, Stop.   Keys while a take is recorded:
  Space      real throw, pressed about when the ball leaves the hand
  k          fake throw (feint), pressed about when the ball would have left the hand
  Backspace  remove the last hint

The board streams raw events to this laptop (despot_debug_camera, as `river-tools capture`
does); stop despot_send_and_forget first, both apps need the camera.

Written to --out:
  sample_<date>_<time>.bin  the whole take from Start to Stop, format as in tools/events_tool.py
  hints.csv                 one row per key press: file, label, hint_s, wall_time, note
    label   throw | feint
    hint_s  seconds from the first event of the file, the clock of decision_s in labels.csv. It is
            only as exact as the key press; the decision moment is the event-rate peak next to it.

The take and hints.csv are written to disk while the take runs. A take cut off by a crash is made
readable the next time the tool starts with the same --out.
"""
import argparse
import json
import os
import signal
import socket
import struct
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import events_tool as ET
from find_swings import read_labels, write_labels

ROOT = Path(__file__).resolve().parent.parent
COLS = ["file", "label", "hint_s", "wall_time", "note"]
LABELS = ("throw", "feint")
HEADER = struct.Struct("!I")
PX = ET.SENSOR * ET.SENSOR
MAX_BYTES = 0xFFFFFFFF // ET.EVENT_DTYPE.itemsize * ET.EVENT_DTYPE.itemsize  # the header is 32 bit
LOCK = threading.RLock()


class Take:
    """One recording: the .bin file and its hints."""

    def __init__(self, folder):
        self.path = folder / f"sample_{datetime.now():%Y%m%d_%H%M%S_%f}.bin"
        self.fh = open(self.path, "xb")
        self.fh.write(HEADER.pack(0))
        self.size = 0
        self.t0 = None  # timestamp of the first event in the file
        self.hints = []
        self.synced = time.monotonic()

    def write(self, data, first_us):
        if self.t0 is None:
            self.t0 = first_us
        self.fh.write(data)
        self.size += len(data)
        if time.monotonic() - self.synced > 1:
            self.sync()

    def sync(self):
        self.fh.seek(0)
        self.fh.write(HEADER.pack(self.size))
        self.fh.seek(0, 2)
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.synced = time.monotonic()

    def close(self):
        self.sync()
        self.fh.close()

    def save_hints(self):
        path = self.path.parent / "hints.csv"
        rows = [r for r in read_labels(path)[1] if r["file"] != self.path.name] if path.exists() else []
        write_labels(path, COLS, rows + [dict(h, file=self.path.name) for h in self.hints])


def repair(folder):
    """Takes cut off by a crash have a header that is behind the file: set it to the events there."""
    n = ET.EVENT_DTYPE.itemsize
    for path in sorted(folder.glob("sample_*.bin")):
        size = (path.stat().st_size - HEADER.size) // n * n
        if size < 0:
            continue
        with open(path, "r+b") as fh:
            if HEADER.unpack(fh.read(HEADER.size))[0] != size:
                fh.seek(0)
                fh.write(HEADER.pack(size))
                fh.truncate(HEADER.size + size)
                print(f"repaired unfinished take {path.name}: {size // n} events")


class Recorder:
    """State shared by the camera thread and the web page."""

    def __init__(self, folder):
        self.folder = folder
        self.camera = "waiting"  # waiting | connected | ended
        self.take = None
        self.done = []  # takes finished in this run
        self.message = ""
        self.acc = np.zeros(2 * PX, np.uint8)  # events per polarity and pixel since the last /frame
        self.last_us = 0  # timestamp of the newest event received ...
        self.last_wall = 0.0  # ... and when it arrived
        self.counted = []  # (arrival time, events) of the batches of the last second

    def now_us(self):
        """Camera clock now: the newest event plus the time since it arrived."""
        return self.last_us + (time.monotonic() - self.last_wall) * 1e6

    def feed(self, data):
        ev = np.frombuffer(data, dtype=ET.EVENT_DTYPE)
        idx = (ev["polarity"].astype(np.int64).clip(0, 1) * ET.SENSOR + ev["y"]) * ET.SENSOR + ev["x"]
        hits = np.bincount(idx[idx < 2 * PX], minlength=2 * PX)
        with LOCK:
            self.acc = np.minimum(self.acc + hits, 3).astype(np.uint8)
            self.last_us, self.last_wall = int(ev["timestamp"][-1]), time.monotonic()
            self.counted = [c for c in self.counted if self.last_wall - c[0] < 1] + [(self.last_wall, len(ev))]
            if self.take is not None:
                if self.take.size + len(data) > MAX_BYTES:
                    self.stop("Recording stopped: the file reached the maximum size of 4 GB.")
                else:
                    self.take.write(data, int(ev["timestamp"][0]))

    def frame(self):
        with LOCK:
            acc, self.acc = self.acc, np.zeros(2 * PX, np.uint8)
        return (acc[:PX] | (acc[PX:] << 2)).tobytes()

    def start(self):
        with LOCK:
            if self.take is None and self.camera == "connected":
                self.folder.mkdir(parents=True, exist_ok=True)
                self.take = Take(self.folder)
                self.message = ""

    def stop(self, message=""):
        with LOCK:
            take, self.take = self.take, None
            if take is None:
                return
            take.close()
            if not take.size:
                take.path.unlink()
                self.message = message or "Recording discarded: no events arrived."
                return
            self.message = message
            self.done.insert(0, {"file": take.path.name, "seconds": (self.last_us - take.t0) / 1e6,
                                 "throw": sum(h["label"] == "throw" for h in take.hints),
                                 "feint": sum(h["label"] == "feint" for h in take.hints)})

    def hint(self, label):
        with LOCK:
            take = self.take
            if take is None or label not in LABELS:
                return
            t = 0.0 if take.t0 is None else max(0.0, (self.now_us() - take.t0) / 1e6)
            take.hints.append({"label": label, "hint_s": f"{t:.2f}",
                               "wall_time": datetime.now().isoformat(timespec="milliseconds"), "note": "key"})
            take.save_hints()

    def undo(self):
        with LOCK:
            if self.take is not None and self.take.hints:
                self.take.hints.pop()
                self.take.save_hints()

    def state(self):
        with LOCK:
            now = time.monotonic()
            self.counted = [c for c in self.counted if now - c[0] < 1]
            take = self.take
            out = {"camera": self.camera, "rate": sum(n for _, n in self.counted), "message": self.message,
                   "folder": str(self.folder), "done": self.done, "recording": take is not None}
            if take is not None:
                out.update(file=take.path.name, bytes=take.size,
                           seconds=0.0 if take.t0 is None else max(0.0, (self.now_us() - take.t0) / 1e6),
                           hints=[{"label": h["label"], "t": float(h["hint_s"])} for h in take.hints])
            return out


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


def receive(args, rec, stop):
    """Take the event batches from the board: 4-byte size, then that many bytes of events."""
    camera = None
    try:
        with socket.create_server(("", args.listen_port)) as server:
            server.settimeout(0.5)
            if not args.no_camera:
                from river_tools.capture.remote import start_camera, stop_camera
                camera = start_camera(args.host, args.listen_port)
            while not stop.is_set():
                if camera is not None and camera.poll() is not None:
                    with LOCK:
                        rec.camera = "ended"
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with LOCK:
                    rec.camera = "connected"
                with conn:
                    conn.settimeout(0.5)
                    while True:
                        header = recv_exact(conn, HEADER.size, stop)
                        data = header and recv_exact(conn, HEADER.unpack(header)[0], stop)
                        if data is None:
                            break
                        data = data[:len(data) // ET.EVENT_DTYPE.itemsize * ET.EVENT_DTYPE.itemsize]
                        if data:  # the board also sends empty batches
                            rec.feed(data)
                with LOCK:
                    rec.camera = "waiting"
                    rec.stop("Recording stopped: the connection to the camera was lost.")
    finally:
        rec.stop()
        if camera is not None:
            stop_camera(camera)


class Handler(BaseHTTPRequestHandler):
    rec = None

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/frame":
            self.send(self.rec.frame(), "application/octet-stream")
        elif self.path == "/state":
            self.send(json.dumps(self.rec.state()).encode(), "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/start":
            self.rec.start()
        elif self.path == "/stop":
            self.rec.stop()
        elif self.path == "/hint":
            self.rec.hint(d.get("label"))
        elif self.path == "/undo":
            self.rec.undo()
        else:
            return self.send_error(404)
        self.send(json.dumps(self.rec.state()).encode(), "application/json")

    def log_message(self, *args):
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Throw recording</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c313a; --text:#e6e8ec; --dim:#8b93a1;
          --throw:#ff6b4a; --feint:#4aa8ff; --ok:#5fd38d; --rec:#e5484d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif;
         display:grid; grid-template-columns:1fr 340px; height:100vh; }
  main { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;
         padding:16px; min-width:0; }
  #wrap { position:relative; width:min(100%, calc(100vh - 90px)); aspect-ratio:1; }
  #view { width:100%; height:100%; image-rendering:pixelated; background:#000; border:4px solid #333;
          display:block; }
  body.rec #view { border-color:var(--rec); }
  #flash { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
           font-size:44px; font-weight:700; opacity:0; transition:opacity .5s; pointer-events:none;
           text-shadow:0 2px 12px #000; }
  #flash.on { opacity:1; transition:none; }
  #cam { color:var(--dim); }
  aside { background:var(--panel); border-left:1px solid var(--line); padding:16px; overflow-y:auto;
          display:flex; flex-direction:column; gap:14px; }
  #clock { font-size:40px; font-variant-numeric:tabular-nums; font-weight:600; line-height:1; }
  #dot { display:inline-block; width:14px; height:14px; border-radius:50%; background:#444; margin-right:10px; }
  body.rec #dot { background:var(--rec); animation:blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity:.25; } }
  #file { color:var(--dim); font-size:12px; word-break:break-all; min-height:1.5em; }
  #buttons { display:flex; gap:10px; }
  button { flex:1; font:600 17px system-ui,sans-serif; padding:14px 0; border-radius:8px; border:0;
           color:#0c0d10; cursor:pointer; }
  #start { background:var(--ok); } #stop { background:var(--rec); color:#fff; }
  #buttons button:disabled { background:#333842; color:#6b7280; cursor:default; }
  #counts { display:flex; gap:10px; }
  .count { flex:1; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  .count b { display:block; font-size:26px; font-variant-numeric:tabular-nums; line-height:1.1; }
  .count.throw b { color:var(--throw); } .count.feint b { color:var(--feint); }
  kbd { background:#2a2f39; border:1px solid #3a404c; border-radius:4px; padding:0 6px; color:var(--text);
        font:12px ui-monospace,monospace; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); margin:0 0 4px;
       font-weight:600; }
  .row { display:flex; gap:8px; font-variant-numeric:tabular-nums; padding:2px 0; }
  .row .t { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .tag { font-size:11px; padding:1px 7px; border-radius:9px; color:#0c0d10; font-weight:600; }
  .tag.throw { background:var(--throw); } .tag.feint { background:var(--feint); }
  #hints { max-height:22vh; overflow-y:auto; }
  #message { color:var(--rec); font-weight:600; }
  .dim { color:var(--dim); }
  @media (max-width:820px) { body { grid-template-columns:1fr; height:auto; }
    #wrap { width:min(100%, 70vh); } aside { border-left:0; border-top:1px solid var(--line); } }
</style></head><body>
<main>
  <div id="wrap"><canvas id="view" width="320" height="320"></canvas><div id="flash"></div></div>
  <div id="cam"></div>
</main>
<aside>
  <div><div id="clock"><span id="dot"></span><span id="time">0:00.0</span></div><div id="file"></div></div>
  <div id="buttons"><button id="start" disabled>Start</button><button id="stop" disabled>Stop</button></div>
  <div id="message"></div>
  <div id="counts">
    <div class="count throw"><b id="nthrow">0</b>real throw <kbd>Space</kbd></div>
    <div class="count feint"><b id="nfeint">0</b>feint <kbd>k</kbd></div>
  </div>
  <div class="dim">Press the key at roughly the moment the ball leaves the hand, or would leave it.
    <kbd>Backspace</kbd> deletes the last hint.</div>
  <div><h2>Hints in this recording</h2><div id="hints"></div></div>
  <div><h2>Fertige Aufnahmen</h2><div id="done"></div><div id="folder" class="dim"></div></div>
</aside>
<script>
const PX = 320 * 320, LUT = [0, 150, 205, 255], NAMES = {throw: "THROW", feint: "FEINT"};
const KEYS = {" ": "throw", "k": "feint"};
const $ = id => document.getElementById(id);
const ctx = $("view").getContext("2d"), img = ctx.createImageData(320, 320), glow = new Float32Array(2 * PX);
let st = {}, synced = 0, lost = false;

const clock = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

function show(s) {
  st = s; synced = performance.now(); lost = false;
  document.body.classList.toggle("rec", s.recording);
  $("start").disabled = s.recording || s.camera !== "connected";
  $("stop").disabled = !s.recording;
  $("cam").textContent = s.camera === "connected" ? `Camera connected · ${Math.round(s.rate / 1000)} k Events/s`
    : s.camera === "ended" ? "The SSH session to the camera has ended – restart the tool (is despot_send_and_forget still running?)"
    : "Waiting for the camera …";
  $("file").textContent = s.recording ? `${s.file} · ${(s.bytes / 1e6).toFixed(1)} MB` : "not recording";
  $("message").textContent = s.message;
  const hints = s.hints || [];
  $("nthrow").textContent = hints.filter(h => h.label === "throw").length;
  $("nfeint").textContent = hints.filter(h => h.label === "feint").length;
  $("hints").innerHTML = hints.map((h, i) => `<div class="row"><span class="t">${i + 1} · ${h.t.toFixed(2)} s</span>` +
    `<span class="tag ${h.label}">${h.label}</span></div>`).reverse().join("") || '<span class="dim">none yet</span>';
  $("done").innerHTML = s.done.map(d => `<div class="row"><span class="t">${d.file.slice(7, -4)} · ${clock(d.seconds)}</span>` +
    `<span class="tag throw">${d.throw}</span><span class="tag feint">${d.feint}</span></div>`).join("") ||
    '<span class="dim">none yet</span>';
  $("folder").textContent = "Folder: " + s.folder;
}

async function post(path, body) {
  try {
    const r = await fetch(path, {method: "POST", body: JSON.stringify(body || {})});
    if (!r.ok) throw new Error(r.status);
    show(await r.json());
    return true;
  } catch (e) { offline(); return false; }
}

function offline() {
  lost = true;
  $("message").textContent = "No connection to the tool – is tools/record_tool.py still running?";
}

function flash(text, color) {
  const f = $("flash");
  f.textContent = text; f.style.color = color; f.classList.add("on");
  requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove("on")));
}

async function frames() {
  try {
    const b = new Uint8Array(await (await fetch("/frame")).arrayBuffer()), d = img.data;
    for (let p = 0; p < PX; p++) {
      const v = b[p];
      const off = glow[p] = Math.max(glow[p] * 0.6, LUT[v & 3]);
      const on = glow[PX + p] = Math.max(glow[PX + p] * 0.6, LUT[v >> 2]);
      d[4 * p] = off; d[4 * p + 1] = on; d[4 * p + 2] = off; d[4 * p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  } catch (e) { offline(); }
  setTimeout(frames, 33);
}

async function poll() {
  try { show(await (await fetch("/state")).json()); } catch (e) { offline(); }
  setTimeout(poll, 250);
}

function tick() {
  const s = st.recording && !lost ? st.seconds + (performance.now() - synced) / 1000 : 0;
  $("time").textContent = clock(s);
  requestAnimationFrame(tick);
}

$("start").addEventListener("click", e => { e.target.blur(); post("/start"); });
$("stop").addEventListener("click", e => { e.target.blur(); post("/stop"); });

document.addEventListener("keydown", async e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (!KEYS[k] && k !== "Backspace") return;
  e.preventDefault();  // Space must not press a focused button or scroll the page
  if (e.repeat) return;
  const label = KEYS[k];  // the tool decides whether a take is running, the page can be a moment behind
  if (!await post(label ? "/hint" : "/undo", {label})) return;
  if (!st.recording) flash("Not recording", "#e6e8ec");
  else if (label) flash(NAMES[label], label === "throw" ? "#ff6b4a" : "#4aa8ff");
  else flash("deleted", "#e6e8ec");
});
document.addEventListener("keyup", e => { if (e.key === " ") e.preventDefault(); });

poll(); frames(); tick();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "04" / "samples", help="folder for the takes")
    ap.add_argument("--host", default="root@192.168.77.1", help="SSH target of the board")
    ap.add_argument("--listen-port", type=int, default=3334, help="port the board sends the events to")
    ap.add_argument("--no-camera", action="store_true", help="despot_debug_camera is started by hand")
    ap.add_argument("--port", type=int, default=8766, help="port of the web page")
    args = ap.parse_args()

    repair(args.out)
    rec = Handler.rec = Recorder(args.out)
    stop = threading.Event()
    worker = threading.Thread(target=receive, args=(args, rec, stop))
    worker.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"takes go to {args.out} -> http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join()


if __name__ == "__main__":
    main()

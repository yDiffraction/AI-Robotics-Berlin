#!/usr/bin/env python3
"""Label arm swings by hotkey: every swing from the labels.csv files plays in a loop in the browser.

  python tools/label_tool.py                               # all dataset/*/samples/labels.csv
  python tools/label_tool.py dataset/03/samples/labels.csv
then open http://127.0.0.1:8765

Keys:  1 real throw   2 fake throw (feint)   3 no swing (idle)   x unusable
       Space/Enter confirm the proposed label    <- -> previous / next    u mark as unverified
       a d  move the decision moment by 20 ms (with Shift 100 ms)
       s speed   p pause   q e single frames   c wider context   r reload the list

Each key press is written to the labels.csv straight away (columns `label`, decision moment,
`verified`: 0 = open, 1 = confirmed in this tool, a name = labelled frame by frame by that one).
The clip shown is what tools/build_dataset.py cuts: 0.7 s before to 0.5 s after the decision
moment; the frame turns red from the decision moment on.
"""
import argparse
import json
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import events_tool as ET
from find_swings import decision_key, read_labels, seconds, write_labels

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("throw", "feint", "idle", "invalid", "?")
FRAME_MS = 20
LOCK = threading.Lock()


class Store:
    """The swing rows of the labels.csv files. The files are read again for every request, so rows
    that find_swings.py adds while the tool is open show up after a reload (key r)."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def items(self):
        out = []
        for src, path in enumerate(self.paths):
            cols, rows = read_labels(path)
            key = decision_key(cols)
            out += [{"src": src, "set": path.parent.parent.name, "file": r["file"], "key": r[key],
                     "label": r["label"], "decision": float(r[key]), "verified": r.get("verified") or "0",
                     "note": r.get("note") or r.get("reason") or ""} for r in rows if r["label"] in LABELS]
        return out

    def update(self, src, file, key, label, decision, verified):
        """Change the row (file, decision moment `key` as written in the csv); returns the new key."""
        cols, rows = read_labels(self.paths[src])
        if "verified" not in cols:
            cols.append("verified")
        k = decision_key(cols)
        r = next(r for r in rows if r["file"] == file and r[k] == key)
        if abs(float(r[k]) - decision) > 1e-6:
            r[k] = f"{decision:.2f}"
        r["label"], r["verified"] = label, verified
        write_labels(self.paths[src], cols, rows)
        return r[k]


@lru_cache(maxsize=4)
def recording(path):
    ev = ET.load(path)
    return ev, seconds(ev)


def frames(path, t0, t1):
    """uint8 [n, 320, 320]: OFF count in bits 0-1, ON count in bits 2-3 (both capped at 3)."""
    ev, t = recording(path)
    n = round((t1 - t0) * 1000 / FRAME_MS)
    lo, hi = np.searchsorted(t, [t0, t1])
    e = ev[lo:hi]
    k = ((t[lo:hi] - t0) * 1000 / FRAME_MS).astype(int).clip(0, n - 1)
    acc = np.zeros((n, 2, ET.SENSOR, ET.SENSOR), np.uint16)
    np.add.at(acc, (k, e["polarity"].astype(int).clip(0, 1), e["y"].astype(int), e["x"].astype(int)), 1)
    acc = acc.clip(0, 3).astype(np.uint8)
    return (acc[:, 0] | (acc[:, 1] << 2)).tobytes()


class Handler(BaseHTTPRequestHandler):
    store = None

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        with LOCK:
            if url.path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/items":
                self.send(json.dumps(self.store.items()).encode(), "application/json")
            elif url.path == "/frames":
                path = self.store.paths[int(q["src"])].parent / Path(q["file"]).name
                self.send(frames(path, float(q["t0"]), float(q["t1"])), "application/octet-stream")
            else:
                self.send_error(404)

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/label" or d["label"] not in LABELS:
            return self.send_error(400)
        with LOCK:
            key = self.store.update(int(d["src"]), d["file"], d["key"], d["label"], float(d["decision"]),
                                    str(d["verified"]))
        self.send(json.dumps({"key": key}).encode(), "application/json")

    def log_message(self, *args):
        pass


PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Wurf-Label-Tool</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c313a; --text:#e6e8ec; --dim:#8b93a1;
          --throw:#ff6b4a; --feint:#4aa8ff; --idle:#9aa3b2; --invalid:#c9a227; --ok:#5fd38d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif;
         display:grid; grid-template-columns:270px 1fr; height:100vh; }
  #list { overflow-y:auto; border-right:1px solid var(--line); background:var(--panel);
          scroll-padding:70px 0 60px; }
  #list h1 { font-size:13px; margin:0; padding:10px 12px; border-bottom:1px solid var(--line);
             position:sticky; top:0; background:var(--panel); font-weight:600; }
  .it { display:flex; gap:8px; align-items:center; padding:4px 12px; cursor:pointer;
        font-variant-numeric:tabular-nums; color:var(--dim); }
  .it:hover { background:#252932; }
  .it.cur { background:#2b3140; color:var(--text); }
  .it .t { flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .it .ok { color:var(--ok); width:12px; }
  .tag { font-size:11px; padding:1px 7px; border-radius:9px; color:#0c0d10; font-weight:600; }
  .throw { background:var(--throw); } .feint { background:var(--feint); } .idle { background:var(--idle); }
  .invalid { background:var(--invalid); } .q { background:#fff; }
  main { overflow:auto; padding:18px 22px; display:flex; flex-direction:column; align-items:center; gap:10px; }
  #head { display:flex; gap:14px; align-items:center; flex-wrap:wrap; justify-content:center; }
  #big { font-size:22px; padding:4px 18px; border-radius:18px; }
  #state { font-size:15px; }
  #view { width:min(640px, 100%, calc(100vh - 290px)); aspect-ratio:1; image-rendering:pixelated;
          background:#000; border:4px solid #333; }
  #view.after { border-color:#e5484d; }
  #bar { width:min(640px, 100%, calc(100vh - 290px)); height:16px; }
  #info { color:var(--dim); text-align:center; min-height:2.9em; }
  #error { color:#e5484d; font-weight:600; }
  #keys { display:flex; flex-wrap:wrap; gap:6px 16px; justify-content:center; color:var(--dim); max-width:760px; }
  kbd { background:#2a2f39; border:1px solid #3a404c; border-radius:4px; padding:0 6px; color:var(--text);
        font:12px ui-monospace,monospace; }
  @media (max-width:760px) { body { grid-template-columns:1fr; height:auto; } #list { max-height:30vh; } }
</style></head><body>
<div id="list"><h1 id="count"></h1><div id="rows"></div></div>
<main>
  <div id="head"><span id="big" class="tag"></span><span id="state"></span></div>
  <canvas id="view" width="320" height="320"></canvas>
  <canvas id="bar" width="640" height="16"></canvas>
  <div id="info"></div><div id="error"></div>
  <div id="keys">
    <span><kbd>1</kbd> richtiger Wurf</span><span><kbd>2</kbd> Fake-Wurf</span>
    <span><kbd>3</kbd> keine Wurfbewegung</span><span><kbd>x</kbd> unbrauchbar</span>
    <span><kbd>Leertaste</kbd> Vorschlag bestätigen</span><span><kbd>←</kbd> <kbd>→</kbd> blättern</span>
    <span><kbd>a</kbd> <kbd>d</kbd> Entscheidungsmoment ∓20 ms (<kbd>Shift</kbd> 100 ms)</span>
    <span><kbd>s</kbd> Tempo</span><span><kbd>p</kbd> Pause</span><span><kbd>q</kbd> <kbd>e</kbd> Einzelbild</span>
    <span><kbd>c</kbd> mehr Kontext</span><span><kbd>u</kbd> wieder offen</span>
    <span><kbd>r</kbd> Liste neu laden</span>
  </div>
</main>
<script>
const FRAME_MS = __FRAME_MS__, PX = 320 * 320, PRE = 0.7, POST = 0.5;
const NAMES = {throw: "richtiger Wurf", feint: "Fake-Wurf", idle: "keine Wurfbewegung", invalid: "unbrauchbar", "?": "noch kein Label"};
const KEYS = {"1": "throw", "2": "feint", "3": "idle", "x": "invalid"};
const LUT = [0, 150, 205, 255], SPEEDS = [1, 0.5, 0.25];
const $ = id => document.getElementById(id);
const ctx = $("view").getContext("2d"), bar = $("bar").getContext("2d"), img = ctx.createImageData(320, 320);
let items = [], cur = 0, buf = null, n = 0, fi = 0, speed = 0, paused = false, wide = false, last = 0, acc = 0, req = 0;

const win = () => wide ? [PRE + 0.8, POST + 0.7] : [PRE, POST];
const cls = l => l === "?" ? "q" : l;

async function load() {
  const it = items[cur], [pre, post] = win(), id = ++req;
  const r = await fetch(`/frames?src=${it.src}&file=${encodeURIComponent(it.file)}&t0=${(it.decision - pre).toFixed(3)}&t1=${(it.decision + post).toFixed(3)}`);
  const b = new Uint8Array(await r.arrayBuffer());
  if (id !== req) return;
  buf = b; n = b.length / PX; fi = 0; acc = 0;
}

function draw() {
  if (!buf) return;
  const o = fi * PX, d = img.data;
  for (let p = 0; p < PX; p++) {
    const v = buf[o + p], off = LUT[v & 3];
    d[4 * p] = off; d[4 * p + 1] = LUT[v >> 2]; d[4 * p + 2] = off; d[4 * p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const [pre, post] = win(), t = fi * FRAME_MS / 1000 - pre, w = 640, x = s => (s + pre) / (pre + post) * w;
  $("view").classList.toggle("after", t >= 0 && t < POST);
  bar.fillStyle = "#23272f"; bar.fillRect(0, 0, w, 16);
  bar.fillStyle = "#39404d"; bar.fillRect(x(-PRE), 0, x(POST) - x(-PRE), 16);
  bar.fillStyle = "#e5484d"; bar.fillRect(x(0) - 1, 0, 3, 16);
  bar.fillStyle = "#fff"; bar.fillRect(x(t), 0, 2, 16);
  const it = items[cur];
  $("info").innerHTML = `Datensatz ${it.set} · ${it.file} · Entscheidungsmoment ${it.decision.toFixed(2)} s · ` +
    `Bild ${t >= 0 ? "+" : "−"}${Math.abs(t).toFixed(2)} s · Tempo ${SPEEDS[speed]}×${paused ? " · Pause" : ""}` +
    (it.note ? `<br>${it.note.replace(/</g, "&lt;")}` : "");
}

function tick(now) {
  if (!paused && n) {
    acc += now - last;
    const step = FRAME_MS / SPEEDS[speed];
    while (acc >= step) { acc -= step; fi = (fi + 1) % n; }
  }
  last = now; draw(); requestAnimationFrame(tick);
}

function list() {
  const done = items.filter(i => i.verified !== "0").length;
  $("count").textContent = `${done} von ${items.length} geprüft`;
  $("rows").innerHTML = items.map((it, i) =>
    `<div class="it${i === cur ? " cur" : ""}" data-i="${i}"><span class="ok">${it.verified !== "0" ? "✓" : ""}</span>` +
    `<span class="t">${it.set} · ${it.file.split("_")[2]} · ${it.decision.toFixed(2)} s</span>` +
    `<span class="tag ${cls(it.label)}">${it.label}</span></div>`).join("");
  const it = items[cur];
  $("big").className = "tag " + cls(it.label); $("big").textContent = NAMES[it.label];
  $("state").textContent = it.verified === "1" ? "✓ geprüft" : it.verified !== "0" ? `✓ gelabelt von ${it.verified}`
    : it.label === "?" ? "" : "Vorschlag, noch nicht geprüft";
  $("state").style.color = it.verified !== "0" ? "var(--ok)" : "var(--dim)";
  const el = $("rows").children[cur];
  if (el) el.scrollIntoView({block: "nearest"});
}

function go(i) { cur = (i + items.length) % items.length; list(); load(); }

let queue = Promise.resolve();
function save(change, advance) {
  const it = Object.assign(items[cur], change);
  queue = queue.then(async () => {  // one write at a time, in the order of the key presses
    const r = await fetch("/label", {method: "POST", body: JSON.stringify(it)});
    if (!r.ok) throw new Error(r.status);
    it.key = (await r.json()).key;
  }).catch(() => { $("error").textContent = "Speichern fehlgeschlagen – läuft tools/label_tool.py noch?"; });
  if (advance) {
    const open = items.map((x, j) => x.verified === "0" ? j : -1).filter(j => j >= 0);
    go(open.find(j => j > cur) ?? open[0] ?? cur);
  } else { list(); }
}

document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey || !items.length) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key, it = items[cur];
  if (KEYS[k]) save({label: KEYS[k], verified: "1"}, true);
  else if (k === " " || k === "Enter") { if (it.label !== "?") save({verified: "1"}, true); }
  else if (k === "u") save({verified: "0"}, false);
  else if (k === "ArrowRight") go(cur + 1);
  else if (k === "ArrowLeft") go(cur - 1);
  else if (k === "a" || k === "d") {
    const step = (e.shiftKey ? 0.1 : 0.02) * (k === "a" ? -1 : 1);
    save({decision: Math.max(0, Math.round((it.decision + step) * 100) / 100)}, false); load();
  }
  else if (k === "s") speed = (speed + 1) % SPEEDS.length;
  else if (k === "p") paused = !paused;
  else if (k === "q" || k === "e") { paused = true; fi = (fi + (k === "e" ? 1 : n - 1)) % n; }
  else if (k === "c") { wide = !wide; load(); }
  else if (k === "r") reload(false);
  else return;
  e.preventDefault();
});
$("rows").addEventListener("click", e => { const el = e.target.closest(".it"); if (el) go(+el.dataset.i); });

async function reload(first) {
  const id = it => it.src + "|" + it.file + "|" + it.key, old = items[cur];
  items = await (await fetch("/items")).json();
  if (!items.length) { $("count").textContent = "keine Einträge"; return; }
  const same = old ? items.findIndex(x => id(x) === id(old)) : -1, open = items.findIndex(x => x.verified === "0");
  go(same >= 0 ? same : open >= 0 ? open : 0);
  if (first) requestAnimationFrame(t => { last = t; tick(t); });
}
reload(true);
</script></body></html>
""".replace("__FRAME_MS__", str(FRAME_MS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels", nargs="*", help="labels.csv files (default: dataset/*/samples/labels.csv)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    paths = args.labels or sorted((ROOT / "dataset").glob("[0-9]*/samples/labels.csv"))
    Handler.store = Store(paths)
    n = len(Handler.store.items())
    print(f"{n} swings from {len(paths)} label files -> http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local browser viewer for the event datasets used by the early-throw trainer.

Shows every clip as the network sees it (2 x 80 x 80 binary cells, 10 ms frames), with the
release marker, an activity trace and all metadata. Two kinds of source are understood:

  bank   sparse .npz from prep_sources.py / extpos/to_bank.py (codes, offsets, n_frames, meta)
         or pretrain/prep_dvsgesture.py (codes, offsets, n_frames, labels, users)
  clips  a match.py output folder: labels.csv + <folder>/<file>.bin (GenX320 events, 320x320).
         These can also be shown at native 320 x 320 resolution.

With no arguments it finds the sources used in the last runs (22 Nov 2026):
  training/early/banks/*.npz, data/extpos/banks/*.npz, data/extpos/clips_*/, data/dvsg_*.npz
Add more (e.g. a noah.npz copied from the A100) as arguments.

  python training/early/viewer/viewer.py                 # then open http://127.0.0.1:8765
  python training/early/viewer/viewer.py path/to/noah.npz --port 8800

Banks are memory-mapped, so even the 340 MB mb.npz opens instantly. The server only listens
on 127.0.0.1: some sources (motion_buddy) must not be published.
"""
import argparse
import csv
import json
import mmap
import os
import re
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HW, SENSOR, FRAME_US = 80, 320, 10_000
BIN_DT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
DVSG_NAMES = ["hand_clap", "rh_wave", "lh_wave", "rh_cw", "rh_ccw", "lh_cw", "lh_ccw",
              "arm_roll", "air_drums", "air_guitar", "other"]
DEFAULT_GLOBS = ["training/early/banks/*.npz", "data/extpos/banks/*.npz",
                 "data/extpos/clips_*/labels.csv", "data/dvsg_*.npz"]


# ------------------------------------------------------------------ npz member memory-mapping
def npz_arrays(path):
    """{name: array} for an .npz; uncompressed members are memory-mapped, others loaded."""
    out, fh = {}, open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            if info.compress_type != zipfile.ZIP_STORED:
                out[name] = np.load(z.open(info), allow_pickle=False)
                continue
            # local file header: 30 bytes + name + extra, then the raw .npy
            hdr = mm[info.header_offset:info.header_offset + 30]
            n_name, n_extra = int.from_bytes(hdr[26:28], "little"), int.from_bytes(hdr[28:30], "little")
            start = info.header_offset + 30 + n_name + n_extra
            npy = np.lib.format
            f = _MMFile(mm, start)
            version = npy.read_magic(f)
            shape, fortran, dtype = npy._read_array_header(f, version)
            if dtype.hasobject or dtype.kind == "U" and shape == ():
                out[name] = np.load(z.open(info), allow_pickle=False)
                continue
            out[name] = np.ndarray(shape, dtype, buffer=mm, offset=f.pos,
                                   order="F" if fortran else "C")
    return out


class _MMFile:
    def __init__(self, mm, pos):
        self.mm, self.pos = mm, pos

    def read(self, n):
        b = self.mm[self.pos:self.pos + n]
        self.pos += n
        return b


# ------------------------------------------------------------------ sources
class Bank:
    kind = "bank"

    def __init__(self, path):
        self.path = Path(path)
        a = npz_arrays(path)
        self.codes, self.off, self.nfr = a["codes"], a["offsets"], a["n_frames"]
        if "meta" in a:
            self.meta = json.loads(str(a["meta"]))
        else:  # DVS Gesture bank from pretrain/prep_dvsgesture.py
            self.meta = [dict(src="dvsg", label="neg", sub=DVSG_NAMES[int(l)], group=f"dvsg:U{int(u)}",
                              rel=-1) for l, u in zip(a["labels"], a["users"])]
        for i, m in enumerate(self.meta):
            m["n_frames"] = int(self.nfr[i])
            m["n_codes"] = int(self.off[i + 1] - self.off[i])

    def clip(self, i, res):
        return np.asarray(self.codes[self.off[i]:self.off[i + 1]], np.int64), HW, int(self.nfr[i])


class Clips:
    kind = "clips"

    def __init__(self, labels_csv):
        self.dir = Path(labels_csv).parent
        self.path = self.dir
        self.rows = list(csv.DictReader(open(labels_csv, newline="")))
        self.meta = []
        for r in self.rows:
            m = {k: v for k, v in r.items() if v not in ("", None)}
            for k in ("params", "stats"):
                if k in m:
                    try:
                        m[k] = json.loads(m[k])
                    except ValueError:
                        pass
            rel = r.get("release_us") or ""
            m["rel"] = int(float(rel)) // FRAME_US if rel not in ("", "-1") else -1
            m["label"] = r.get("role") or r.get("label") or "?"
            m["sub"] = r.get("cls") or r.get("label") or "?"
            m["group"] = r.get("group") or r.get("folder") or "?"
            self.meta.append(m)

    def clip(self, i, res):
        r = self.rows[i]
        raw = (self.dir / r["folder"] / r["file"]).read_bytes()
        ev = np.frombuffer(raw[4:], dtype=BIN_DT)
        if len(ev) == 0:
            return np.zeros(0, np.int64), res, 120
        t = ev["t"].astype(np.int64)
        if t.min() > 5_000_000:  # absolute timestamps: make clip-relative
            t = t - t.min()
        f = t // FRAME_US
        x = ev["x"].astype(np.int64) * res // SENSOR
        y = ev["y"].astype(np.int64) * res // SENSOR
        p = (ev["p"] > 0).astype(np.int64)
        c = np.unique(((f * 2 + p) * res + y) * res + x)
        return c, res, max(120, int(c.max() // (2 * res * res)) + 1)


def load_sources(extra):
    paths = []
    for g in DEFAULT_GLOBS:
        paths += sorted(ROOT.glob(g))
    paths += [Path(p).resolve() for p in extra]
    srcs = {}
    for p in paths:
        if p.is_dir() and (p / "labels.csv").exists():
            p = p / "labels.csv"
        try:
            s = Clips(p) if p.name == "labels.csv" else Bank(p)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        rel = os.path.relpath(s.path, ROOT).replace("\\", "/")
        name = re.sub(r"[^\w./-]", "_", rel)
        srcs[name] = s
        print(f"{s.kind:5s} {rel}: {len(s.meta)} clips")
    return srcs


# ------------------------------------------------------------------ http
def make_handler(srcs):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/sources":
                    body = [dict(name=n, kind=s.kind, n=len(s.meta)) for n, s in srcs.items()]
                    return self.send(json.dumps(body).encode(), "application/json")
                if u.path == "/api/meta":
                    return self.send(json.dumps(srcs[q["src"]].meta).encode(), "application/json")
                if u.path == "/api/clip":
                    s = srcs[q["src"]]
                    res = int(q.get("res", HW)) if s.kind == "clips" else HW
                    codes, res, nfr = s.clip(int(q["i"]), res)
                    hdr = np.array([res, nfr], "<u4").tobytes()
                    return self.send(hdr + codes.astype("<u4").tobytes(), "application/octet-stream")
                self.send(b"not found", "text/plain", 404)
            except (KeyError, IndexError, ValueError) as e:
                self.send(str(e).encode(), "text/plain", 400)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extra", nargs="*", help="more banks (.npz) or clip folders (with labels.csv)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srcs = load_sources(a.extra)
    if not srcs:
        sys.exit("no sources found")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(srcs))
    print(f"\nviewer: http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

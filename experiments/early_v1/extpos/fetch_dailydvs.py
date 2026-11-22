#!/usr/bin/env python3
"""Fetch selected DailyDVS-200 recordings as *raw* events (no cropping, no anchoring).

DailyDVS-200 (ECCV 2024, MIT licence): 22,046 DVXplorer-Lite (320x240) recordings of 200 actions
by 47 people. The Hugging Face mirror is one 91 GB zip split into 9 parts; this reads the zip
central directory with HTTP range requests and downloads only the selected members
(same approach as tools/dailydvs_import.py on the throw-detector branch).

Unlike that importer, every class (positive, negative, ambiguous) is stored untouched as
raw/<group>/<stem>.npz (t, x, y, p, native resolution). All geometry, density, noise and
timing decisions happen later in match.py, identically for positives and negatives, so the
preprocessing itself cannot become a label cue.

  python fetch_dailydvs.py index --out data/extpos/dailydvs
  python fetch_dailydvs.py fetch --out data/extpos/dailydvs --limit 10 --workers 8
(needs dv-processing; the v2e env has it)
"""
import argparse
import csv
import io
import json
import struct
import tempfile
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from labels import DD_CLASSES

REPO = "https://huggingface.co/datasets/Charlieqaq/DailyDVS-200/resolve/main/"
PART_SIZE = 10737418240
TOTAL = 91144068855
NPARTS = 9


class Remote:
    def __init__(self):
        self.urls = [f"{REPO}DailyDvs-200.zip.part-{i + 1:03d}" for i in range(NPARTS)]

    def read(self, start, length):
        out = bytearray()
        while length > 0:
            part, off = divmod(start, PART_SIZE)
            n = min(length, PART_SIZE - off)
            req = urllib.request.Request(self.urls[part], headers={"Range": f"bytes={off}-{off + n - 1}"})
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) == n:
                        break
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(2 ** attempt)
            out += data
            start += n
            length -= n
        return bytes(out)


class RemoteFile(io.RawIOBase):
    def __init__(self, remote, block=8 << 20):
        self.r, self.pos, self.block, self.cache = remote, 0, block, {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else TOTAL + off
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), TOTAL - self.pos)
        got = bytearray()
        while len(got) < n:
            blk = (self.pos + len(got)) // self.block
            if blk not in self.cache:
                s = blk * self.block
                self.cache[blk] = self.r.read(s, min(self.block, TOTAL - s))
            data = self.cache[blk]
            o = self.pos + len(got) - blk * self.block
            got += data[o:o + n - len(got)]
        b[:n] = got
        self.pos += n
        return n


META = ("ID", "Scene", "Move", "PersonNum", "Perspective", "Distance", "Height", "Indoor/Outdoor",
        "Daytime/Nighttime", "Background Complexity", "Standing/Sitting")


def cmd_index(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(io.BufferedReader(RemoteFile(Remote()), buffer_size=1 << 20))
    labels = {Path(d["FileName"]).name: d
              for d in json.load(urllib.request.urlopen(REPO + "all_data_label.json"))}
    rows = []
    for zi in zf.infolist():
        name = Path(zi.filename).name
        if not name.endswith(".aedat4"):
            continue
        meta = labels.get(name, {})
        action = int(meta.get("Action", name[1:].split("P")[0]))
        rows.append({"member": zi.filename, "offset": zi.header_offset, "csize": zi.compress_size,
                     "size": zi.file_size, "ctype": zi.compress_type, "action": action,
                     **{k: meta.get(k, "") for k in META}})
    with (out / "index_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sel = [r for r in rows if r["action"] in DD_CLASSES]
    print(f"{len(rows)} recordings indexed, {len(sel)} in the selected classes, "
          f"{sum(r['csize'] for r in sel) / 1e9:.1f} GB to fetch")


def read_aedat4(blob):
    import dv_processing as dv
    with tempfile.TemporaryDirectory() as d:
        fn = Path(d) / "x.aedat4"
        fn.write_bytes(blob)
        rec = dv.io.MonoCameraRecording(str(fn))
        res = rec.getEventResolution()
        chunks = []
        while rec.isRunning():
            b = rec.getNextEventBatch()
            if b is not None and len(b) > 0:
                chunks.append(b.numpy())
        del rec
    if not chunks:
        return None
    a = np.concatenate(chunks)
    return (a["timestamp"].astype(np.int64), a["x"].astype(np.int16), a["y"].astype(np.int16),
            a["polarity"].astype(np.uint8), (int(res[0]), int(res[1])))


def cmd_fetch(args):
    out = Path(args.out)
    rows = [r for r in csv.DictReader((out / "index_all.csv").open()) if int(r["action"]) in DD_CLASSES]
    if args.actions:
        keep = {int(a) for a in args.actions.split(",")}
        rows = [r for r in rows if int(r["action"]) in keep]
    if args.limit:
        rng = np.random.default_rng(0)
        by = {}
        for r in rows:
            by.setdefault(r["action"], []).append(r)
        rows = [r for a in sorted(by) for r in
                [by[a][i] for i in rng.permutation(len(by[a]))[:args.limit]]]
    remote, lock = Remote(), threading.Lock()
    mpath = out / "raw_manifest.csv"
    done = {r["member"] for r in csv.DictReader(mpath.open())} if mpath.exists() else set()
    todo = [r for r in rows if r["member"] not in done]
    print(f"{len(todo)} to fetch ({len(done)} done), {sum(int(r['csize']) for r in todo) / 1e9:.2f} GB")

    def job(r):
        off, csize = int(r["offset"]), int(r["csize"])
        nlen, xlen = struct.unpack("<HH", remote.read(off, 30)[26:30])
        raw = remote.read(off + 30 + nlen + xlen, csize)
        blob = zlib.decompress(raw, -15) if int(r["ctype"]) == zipfile.ZIP_DEFLATED else raw
        dec = read_aedat4(blob)
        if dec is None:
            return
        t, x, y, p, (w, h) = dec
        a = int(r["action"])
        name, role = DD_CLASSES[a]
        stem = Path(r["member"]).stem
        dst = out / "raw" / role / f"{stem}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, t=t - t[0], x=x, y=y, p=p, wh=np.array([w, h]))
        row = {"member": r["member"], "path": str(dst.relative_to(out)), "action": a, "action_name": name,
               "role": role, "person": r["ID"], "events": len(t), "dur_us": int(t[-1] - t[0]),
               "w": w, "h": h, **{k.replace("/", "_").replace(" ", "_").lower(): r[k] for k in META[1:]}}
        with lock:
            new = not mpath.exists()
            with mpath.open("a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    wr.writeheader()
                wr.writerow(row)

    def safe(r):
        try:
            job(r)
        except Exception as e:
            print("failed", r["member"], e, flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(safe, todo)):
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "fetch"])
    ap.add_argument("--out", default="data/extpos/dailydvs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="at most N recordings per action (0 = all)")
    ap.add_argument("--actions", default="", help="comma-separated action ids (default: all in labels.py)")
    a = ap.parse_args()
    {"index": cmd_index, "fetch": cmd_fetch}[a.cmd](a)

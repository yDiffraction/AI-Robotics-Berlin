#!/usr/bin/env python3
"""Import selected DailyDVS-200 classes as GenX320-style training clips.

DailyDVS-200 (ECCV 2024, MIT licence, https://github.com/QiWang233/DailyDVS-200) holds 22,046
real event recordings (DVXplorer Lite, 320x240) of 200 everyday actions by 47 people. The
Hugging Face mirror (Charlieqaq/DailyDVS-200) is one 91 GB zip split into 9 byte parts. This
tool never downloads the whole archive: it reads the zip central directory with HTTP range
requests and then fetches only the members of the selected classes.

Groups (output sub-folders, see GROUPS below):
  throw          a ball / object is thrown away from the body
  toss           something is thrown, but up / down / into a bin (not at a target)
  arm_no_release arm swings, punches, waves without anything leaving the hand (feint-like)
  idle           everyday actions with little arm swing

Each recording becomes one 1.2 s clip (0.7 s before / 0.5 s after the anchor). The anchor is
the event-rate peak (20 ms bins, same convention as the real recordings) for throw, toss and
arm_no_release, and a random window for idle. The 320x240 frame is cropped to a 240x240
square around the activity and scaled to 320x320 (sub-pixel jitter avoids empty rows).

  uv run --with dv-processing tools/dailydvs_import.py index --out dataset/external/dailydvs
  uv run --with dv-processing tools/dailydvs_import.py fetch --out dataset/external/dailydvs --workers 16
"""

import argparse
import csv
import io
import json
import struct
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from event_sim import DTYPE  # noqa: E402

REPO = "https://huggingface.co/datasets/Charlieqaq/DailyDVS-200/resolve/main/"
PART_SIZE = 10737418240
TOTAL = 91144068855
NPARTS = 9

# DailyDVS action id -> (group, name). Ids from resource/action_description.csv.
GROUPS = {
    147: ("throw", "throwing_ball"), 187: ("throw", "pitch"), 37: ("throw", "throw_things"),
    69: ("toss", "throw_hats_in_the_air"), 70: ("toss", "throw_rubbish"),
    171: ("toss", "bounce_volleyball_on_hand"),
    146: ("arm_no_release", "swing_badminton_racket"), 127: ("arm_no_release", "punch"),
    49: ("arm_no_release", "hit_people_with_things"), 13: ("arm_no_release", "wave_hands"),
    14: ("arm_no_release", "point_to_something"), 84: ("arm_no_release", "rotate_arm"),
    54: ("arm_no_release", "stretch"), 21: ("arm_no_release", "surrender"),
    56: ("arm_no_release", "salute"), 176: ("arm_no_release", "bounce_ball"),
    0: ("idle", "yawn"), 4: ("idle", "put_hands_in_pockets"), 15: ("idle", "cross_arms"),
    31: ("idle", "use_mobile_phone"), 51: ("idle", "nod"), 52: ("idle", "shake_head"),
    90: ("idle", "check_time"), 107: ("idle", "walk"), 122: ("idle", "touch_nose"),
    165: ("idle", "stand_up"),
}
LABEL = {"throw": "throw", "toss": "throw", "arm_no_release": "feint", "idle": "idle"}
CLASS_ID = {"throw": 1, "feint": 2, "idle": 3}
PRE_US, POST_US = 700_000, 500_000


class Remote:
    """Random access to the concatenated split archive via HTTP range requests."""

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
                    time.sleep(2 ** attempt)  # HF answers 429 when hit too hard
            out += data
            start += n
            length -= n
        return bytes(out)


class RemoteFile(io.RawIOBase):
    """Seekable file object (read-ahead cache) so zipfile can parse the central directory."""

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


def cmd_index(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(io.BufferedReader(RemoteFile(Remote()), buffer_size=1 << 20))
    labels = {Path(d["FileName"]).name: d for d in json.load(urllib.request.urlopen(REPO + "all_data_label.json"))}
    rows = []
    for zi in zf.infolist():
        name = Path(zi.filename).name
        if not name.endswith(".aedat4"):
            continue
        meta = labels.get(name, {})
        action = int(meta.get("Action", name[1:].split("P")[0]))
        if action not in GROUPS:
            continue
        rows.append({"member": zi.filename, "offset": zi.header_offset, "csize": zi.compress_size,
                     "size": zi.file_size, "ctype": zi.compress_type, "action": action,
                     "group": GROUPS[action][0], "action_name": GROUPS[action][1],
                     **{k: meta.get(k, "") for k in ("ID", "Scene", "Move", "PersonNum", "Perspective", "Distance",
                                                     "Height", "Indoor/Outdoor", "Daytime/Nighttime",
                                                     "Background Complexity", "Standing/Sitting")}})
    with (out / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(zf.infolist())} members, {len(rows)} selected, {sum(r['csize'] for r in rows) / 1e9:.1f} GB")


def read_aedat4(blob):
    import dv_processing as dv
    with tempfile.NamedTemporaryFile(suffix=".aedat4") as f:
        f.write(blob)
        f.flush()
        rec = dv.io.MonoCameraRecording(f.name)
        chunks = []
        while rec.isRunning():
            b = rec.getNextEventBatch()
            if b is not None and len(b) > 0:
                chunks.append(b.numpy())
    if not chunks:
        return None
    a = np.concatenate(chunks)
    return a["timestamp"].astype(np.int64), a["x"].astype(np.int64), a["y"].astype(np.int64), a["polarity"].astype(np.uint8)


def peak_anchor(t, lo, hi):
    """Event-rate peak (20 ms bins, 3-tap smoothing, bin centre) inside [lo, hi]."""
    edges = np.arange(lo, hi + 20_000, 20_000)
    h = np.convolve(np.histogram(t, edges)[0], np.ones(3) / 3, mode="same")
    return int(edges[int(np.argmax(h))] + 10_000)


def to_clip(t, x, y, p, group, rng):
    t = t - t[0]
    dur = int(t[-1])
    if dur < PRE_US + POST_US:
        return None, None
    if group == "idle":
        a0 = int(rng.uniform(0, dur - PRE_US - POST_US))
        anchor = None
    else:
        anchor = peak_anchor(t, PRE_US, dur - POST_US)
        a0 = anchor - PRE_US
    m = (t >= a0) & (t < a0 + PRE_US + POST_US)
    t, x, y, p = t[m] - a0, x[m], y[m], p[m]
    if len(t) < 500:
        return None, None
    # 240x240 square around the activity (horizontal centre of the events), scaled to 320x320
    cx = int(np.clip(np.median(x), 120, 200))
    keep = (x >= cx - 120) & (x < cx + 120)
    t, x, y, p = t[keep], x[keep] - (cx - 120), y[keep], p[keep]
    s = 320 / 240
    ev = np.empty(len(t), DTYPE)
    ev["x"] = np.clip((x + rng.random(len(x))) * s, 0, 319).astype(np.uint16)
    ev["y"] = np.clip((y + rng.random(len(y))) * s, 0, 319).astype(np.uint16)
    ev["polarity"] = p
    ev["timestamp"] = t
    return ev, (None if anchor is None else PRE_US)


def save_bin(path, ev):
    body = ev.tobytes()
    path.write_bytes(struct.pack("!I", len(body)) + body)


def cmd_fetch(args):
    out = Path(args.out)
    rows = list(csv.DictReader((out / "index.csv").open()))
    if args.limit:
        rows = [r for g in {r["action"] for r in rows} for r in [x for x in rows if x["action"] == g][:args.limit]]
    remote, lock = Remote(), threading.Lock()
    manifest_rows, done = [], set()
    mpath = out / "manifest.csv"
    if mpath.exists():
        done = {r["source_member"] for r in csv.DictReader(mpath.open())}
    todo = [r for r in rows if r["member"] not in done]
    print(f"{len(todo)} to fetch ({len(done)} already done)")

    def job(r):
        rng = np.random.default_rng(abs(hash(r["member"])) % 2**32)
        off, csize = int(r["offset"]), int(r["csize"])
        head = remote.read(off, 30)
        nlen, xlen = struct.unpack("<HH", head[26:30])
        raw = remote.read(off + 30 + nlen + xlen, csize)
        blob = zlib.decompress(raw, -15) if int(r["ctype"]) == zipfile.ZIP_DEFLATED else raw
        dec = read_aedat4(blob)
        if dec is None:
            return None
        ev, rel = to_clip(*dec, r["group"], rng)
        if ev is None:
            return None
        label = LABEL[r["group"]]
        stem = Path(r["member"]).stem
        name = f"gesture_{CLASS_ID[label]:02d}-{label}-dailydvs-{r['action_name']}-{stem}.bin"
        (out / r["group"]).mkdir(exist_ok=True)
        save_bin(out / r["group"] / name, ev)
        row = {"file": name, "label": label, "group": r["group"], "action": r["action"],
               "action_name": r["action_name"], "source": "dailydvs", "parent": stem, "person": r["ID"],
               "release_us": "" if rel is None else rel, "release_source": "" if rel is None else "rate_peak",
               "events": len(ev), "duration_us": PRE_US + POST_US, "perspective": r["Perspective"],
               "distance": r["Distance"], "height": r["Height"], "camera_move": r["Move"],
               "persons": r["PersonNum"], "indoor": r["Indoor/Outdoor"], "night": r["Daytime/Nighttime"],
               "background": r["Background Complexity"], "sitting": r["Standing/Sitting"],
               "source_member": r["member"]}
        with lock:
            new = not mpath.exists()
            with mpath.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        return row

    n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda r: _safe(job, r), todo):
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done")


def _safe(fn, r):
    try:
        return fn(r)
    except Exception as e:  # keep going on a broken member
        print(f"failed {r['member']}: {e}", flush=True)
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["index", "fetch"])
    p.add_argument("--out", default="dataset/external/dailydvs")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="fetch at most N recordings per action (0 = all)")
    args = p.parse_args()
    {"index": cmd_index, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()

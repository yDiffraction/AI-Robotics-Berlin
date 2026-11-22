#!/usr/bin/env python3
"""Fetch HMDB51 clips of the classes in labels.HMDB_CLASSES from the Hugging Face mirror
divm/hmdb51 (one mp4 per clip, ~0.9 GB for all 51 classes).

HMDB51 file names carry per-clip tags, kept in manifest.csv and used by v2e_convert.py:
  body  f full / u upper / h head / l lower      cam  cm camera moves / nm static camera
  np    number of people                          view fr / ba / le / ri
  qual  goo / med / bad
`video` (the source movie / web video) is the CV group: clips of one video show the same person.

  python fetch_hmdb.py --out data/extpos/hmdb [--static-only] [--limit 20]
"""
import argparse
import csv
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from labels import HMDB_CLASSES

API = "https://huggingface.co/api/datasets/divm/hmdb51/tree/main/{split}?limit=1000"
RAW = "https://huggingface.co/datasets/divm/hmdb51/resolve/main/{path}"
TAGS = re.compile(r"_(f|h|u|l)_(cm|nm)_np(\d+)_(fr|ba|le|ri)_(goo|med|bad)_(\d+)\.mp4$")


def list_split(split):
    url, files = API.format(split=split), []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            files += [x for x in json.load(r) if x["type"] == "file"]
            nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = nxt.group(1) if nxt else None
    return files


def parse(name):
    m = TAGS.search(name)
    if not m:
        return None
    for c in sorted(HMDB_CLASSES, key=len, reverse=True):
        if name.startswith(c + "_"):
            core = name[len(c) + 1:m.start()]
            video = core[:-(len(c) + 1)] if core.endswith("_" + c) else core
            return dict(cls=c, role=HMDB_CLASSES[c], video=video, body=m[1], cam=m[2], np=int(m[3]),
                        view=m[4], qual=m[5], idx=int(m[6]))
    return None


def get(url, dst):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            dst.write_bytes(data)
            return True
        except Exception:
            time.sleep(2 ** attempt)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/extpos/hmdb")
    ap.add_argument("--static-only", action="store_true", help="only clips tagged nm (static camera)")
    ap.add_argument("--limit", type=int, default=0, help="at most N clips per class")
    ap.add_argument("--classes", default="", help="comma-separated subset of labels.HMDB_CLASSES")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "validation", "test"):
        for f in list_split(split):
            name = f["path"].split("/")[-1]
            m = parse(name)
            if m is None:
                continue
            if a.classes and m["cls"] not in a.classes.split(","):
                continue
            if a.static_only and m["cam"] != "nm":
                continue
            rows.append(dict(file=name, hf_path=f["path"], split=split, size=f["size"], **m))
    if a.limit:
        by = {}
        for r in sorted(rows, key=lambda r: (r["cam"] != "nm", r["qual"] == "bad", r["file"])):
            by.setdefault(r["cls"], []).append(r)
        rows = [r for c in by.values() for r in c[:a.limit]]
    print(f"{len(rows)} clips, {sum(r['size'] for r in rows) / 1e6:.0f} MB")
    todo = [r for r in rows if not (out / "videos" / r["file"]).exists()]
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(lambda r: get(RAW.format(path=r["hf_path"]), out / "videos" / r["file"]), todo))
    print(f"downloaded {sum(ok)}/{len(todo)}")
    rows = [r for r in rows if (out / "videos" / r["file"]).exists()]
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

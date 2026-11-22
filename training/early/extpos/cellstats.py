"""Shared I/O and statistics on the board input grid (2 x 80 x 80 binary cells, 10 ms).

Everything that decides whether an external clip "looks like ours" is measured here, on the
same grid the SNN sees, so a statistic that differs here is a statistic the network could use.
"""
import json
from pathlib import Path

import numpy as np

SENSOR = 320
HW = 80
BLOCK = SENSOR // HW
FRAME_US = 10_000
CLIP_US = 1_200_000
REL_US = 700_000
DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])


# --------------------------------------------------------------------------- I/O
def read_bin(path):
    raw = Path(path).read_bytes()
    return np.frombuffer(raw[4:], dtype=DTYPE).copy()


def write_bin(path, x, y, p, t):
    order = np.argsort(t, kind="stable")
    ev = np.empty(len(t), DTYPE)
    ev["x"], ev["y"], ev["p"], ev["t"] = x[order], y[order], p[order], t[order]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.uint32(len(ev)).astype(">u4").tobytes())
        f.write(ev.tobytes())


def to_cells(ev_or_xypt, n_frames=CLIP_US // FRAME_US, t0=0, src=SENSOR):
    """Binary [T, 2, 80, 80] grid exactly like prep_sources.encode / the board."""
    if isinstance(ev_or_xypt, np.ndarray):
        x, y, p, t = (ev_or_xypt[k].astype(np.int64) for k in ("x", "y", "p", "t"))
    else:
        x, y, p, t = (np.asarray(a, np.int64) for a in ev_or_xypt)
    f = (t - t0) // FRAME_US
    k = (f >= 0) & (f < n_frames)
    g = np.zeros((n_frames, 2, HW, HW), np.uint8)
    g[f[k], (p[k] > 0).astype(np.int64), y[k] * HW // src, x[k] * HW // src] = 1
    return g


# ---------------------------------------------------------------------- statistics
def clip_stats(g, rel_frame=None, win=(-50, 30)):
    """Per-clip statistics. g: [T,2,H,W] binary. Motion statistics use the swing window
    around rel_frame (default -500..+300 ms); without a release the busiest 800 ms."""
    T = g.shape[0]
    cnt = g.reshape(T, -1).sum(1).astype(np.float64)
    if rel_frame is None or rel_frame < 0:
        L = win[1] - win[0]
        cs = np.convolve(cnt, np.ones(L), "valid")
        a = int(np.argmax(cs)) if len(cs) else 0
        sl = slice(a, a + L)
    else:
        sl = slice(max(0, rel_frame + win[0]), min(T, rel_frame + win[1]))
    w = g[sl]
    occ = w.any(1)                                  # [t, H, W]
    tot = w.sum()
    s = {}
    s["cells_mean"] = float(cnt[sl].mean())
    s["cells_p90"] = float(np.percentile(cnt[sl], 90))
    s["cells_quiet"] = float(np.percentile(cnt, 10))           # noise floor
    s["on_frac"] = float(w[:, 1].sum() / max(tot, 1))
    # spatial extent of the motion: percentiles of active-cell coordinates
    ys, xs = np.nonzero(occ.sum(0))
    wts = occ.sum(0)[ys, xs].astype(np.float64)
    if len(ys) >= 4:
        def wp(v, q):
            o = np.argsort(v); c = np.cumsum(wts[o]); return float(v[o][np.searchsorted(c, q * c[-1])])
        s["y05"], s["y95"], s["x05"], s["x95"] = wp(ys, .05), wp(ys, .95), wp(xs, .05), wp(xs, .95)
        s["ext_h"], s["ext_w"] = s["y95"] - s["y05"], s["x95"] - s["x05"]
        s["cx"] = float((xs * wts).sum() / wts.sum()); s["cy"] = float((ys * wts).sum() / wts.sum())
    # isolated cells (no other active cell in the 3x3 neighbourhood, same or adjacent frame)
    o = g.any(1).astype(np.int32)
    pad = np.pad(o, ((1, 1), (1, 1), (1, 1)))
    nb = sum(pad[1 + dt:1 + dt + T, 1 + dy:1 + dy + HW, 1 + dx:1 + dx + HW]
             for dt in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - o
    s["isolated_frac"] = float(((o == 1) & (nb == 0)).sum() / max(o.sum(), 1))
    # temporal texture: lag-1 autocorrelation of the per-frame count and the share of empty
    # frames inside the motion window (30 fps video without interpolation gives a comb)
    c = cnt[sl] - cnt[sl].mean()
    s["ac1"] = float((c[1:] * c[:-1]).sum() / max((c * c).sum(), 1e-9))
    s["ac3"] = float((c[3:] * c[:-3]).sum() / max((c * c).sum(), 1e-9))
    s["empty_frac"] = float((cnt[sl] < max(2, 0.05 * cnt[sl].mean())).mean())
    # edge thickness: active cells per occupied row segment (how "fat" moving edges are)
    runs = np.diff(np.pad(occ, ((0, 0), (0, 0), (1, 1))).astype(np.int8), axis=2)
    nseg = (runs == 1).sum()
    s["run_len"] = float(occ.sum() / max(nseg, 1))
    # peakiness of the activity curve
    s["peak_ratio"] = float(cnt[sl].max() / max(np.median(cnt[sl]), 1))
    s["touched"] = float(occ.any(0).mean())
    return s


def summarize(rows, keys=None):
    keys = keys or [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], np.float64)
        if len(v):
            out[k] = [round(float(np.percentile(v, q)), 3) for q in (10, 50, 90)]
    return out


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))

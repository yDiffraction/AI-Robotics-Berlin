"""Label-preserving transforms and synthesis helpers for GenX320 event streams.

All functions take and return numpy structured arrays with fields
x (uint16), y (uint16), polarity (uint8), timestamp (int64, microseconds), sorted by time.
The sensor is 320 x 320.
"""

import numpy as np

SENSOR = 320
DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("polarity", "u1"), ("timestamp", "<i8")])


def as_native(ev):
    """Copy any event array (e.g. big-endian from file) into native, writable form."""
    out = np.empty(len(ev), DTYPE)
    for name in DTYPE.names:
        out[name] = ev[name]
    return out


def _from_float(x, y, p, t):
    keep = (x >= 0) & (x < SENSOR) & (y >= 0) & (y < SENSOR)
    out = np.empty(int(keep.sum()), DTYPE)
    out["x"] = x[keep].astype(np.int64)
    out["y"] = y[keep].astype(np.int64)
    out["polarity"] = p[keep]
    out["timestamp"] = t[keep].astype(np.int64)
    return out[np.argsort(out["timestamp"], kind="stable")]


def window(ev, t0, t1, rebase=True):
    """Events in [t0, t1) (absolute microseconds); optionally shift so the window starts at 0."""
    sel = ev[(ev["timestamp"] >= t0) & (ev["timestamp"] < t1)].copy()
    if rebase:
        sel["timestamp"] -= t0
    return sel


def geometric(ev, rng, flip=None, shift=40, scale=(0.75, 1.25), rotate_deg=10.0):
    """Random mirror / scale / rotation / translation around the image centre."""
    x = ev["x"].astype(np.float64)
    y = ev["y"].astype(np.float64)
    if flip if flip is not None else rng.random() < 0.5:
        x = SENSOR - 1 - x
    s = rng.uniform(*scale)
    a = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg))
    c = SENSOR / 2
    xc, yc = x - c, y - c
    xr = s * (np.cos(a) * xc - np.sin(a) * yc) + c + rng.uniform(-shift, shift)
    yr = s * (np.sin(a) * xc + np.cos(a) * yc) + c + rng.uniform(-shift, shift)
    # scaling > 1 leaves holes between pixels; jitter fills them like a denser sensor
    if s > 1:
        xr += rng.uniform(-s / 2, s / 2, len(xr))
        yr += rng.uniform(-s / 2, s / 2, len(yr))
    return _from_float(np.floor(xr), np.floor(yr), ev["polarity"], ev["timestamp"].astype(np.float64))


def time_stretch(ev, factor, anchor_us):
    """Play the motion faster (factor < 1) or slower (factor > 1) around an anchor time."""
    out = ev.copy()
    out["timestamp"] = (anchor_us + (ev["timestamp"] - anchor_us) * factor).astype(np.int64)
    return out[np.argsort(out["timestamp"], kind="stable")]


def dropout(ev, rng, keep=(0.6, 1.0), polarity_bias=0.15):
    """Drop events uniformly (contrast / lighting) with a slight ON/OFF imbalance."""
    k = rng.uniform(*keep)
    bias = rng.uniform(-polarity_bias, polarity_bias)
    p = np.where(ev["polarity"] == 1, k * (1 + bias), k * (1 - bias))
    return ev[rng.random(len(ev)) < p]


def add_noise(ev, rng, duration_us, rate_hz=(0.0, 60_000.0), hot_pixels=(0, 12), hot_rate_hz=200.0):
    """Uniform background activity plus a few hot pixels."""
    n = rng.poisson(rng.uniform(*rate_hz) * duration_us / 1e6)
    noise = np.empty(n, DTYPE)
    noise["x"] = rng.integers(0, SENSOR, n)
    noise["y"] = rng.integers(0, SENSOR, n)
    noise["polarity"] = rng.integers(0, 2, n)
    noise["timestamp"] = rng.integers(0, max(1, duration_us), n)
    parts = [ev, noise]
    for _ in range(rng.integers(*hot_pixels)):
        m = rng.poisson(hot_rate_hz * duration_us / 1e6)
        hot = np.empty(m, DTYPE)
        hot["x"], hot["y"] = rng.integers(0, SENSOR), rng.integers(0, SENSOR)
        hot["polarity"] = rng.integers(0, 2, m)
        hot["timestamp"] = rng.integers(0, max(1, duration_us), m)
        parts.append(hot)
    return merge(*parts)


def merge(*streams):
    out = np.concatenate([as_native(s) for s in streams])
    return out[np.argsort(out["timestamp"], kind="stable")]


def reverse(ev, t0, t1):
    """Time-mirror the events of [t0, t1): motion runs backwards, so ON and OFF swap."""
    seg = ev[(ev["timestamp"] >= t0) & (ev["timestamp"] < t1)].copy()
    seg["timestamp"] = t0 + (t1 - 1 - seg["timestamp"])
    seg["polarity"] = 1 - seg["polarity"]
    return seg[np.argsort(seg["timestamp"], kind="stable")]


def feint_from_throw(ev, release_us, rng, stop_before=(0.04, 0.22), back=(0.25, 0.5), hold=(0.0, 0.15)):
    """Turn a real throw into a feint: cut before release, hold, then replay the wind-up backwards.

    stop_before: how long before the release the arm stops (seconds)
    back:        length of the wind-up segment replayed backwards (seconds)
    hold:        pause at the turning point (seconds); only sparse events during the pause
    Returns (events, stop_time_us).
    """
    stop = int(release_us - rng.uniform(*stop_before) * 1e6)
    seg_len = int(rng.uniform(*back) * 1e6)
    pause = int(rng.uniform(*hold) * 1e6)
    before = ev[ev["timestamp"] < stop]
    tail = reverse(ev, stop - seg_len, stop)
    tail["timestamp"] += seg_len + pause
    # during the pause the scene is almost still: keep a thin sample of the last 30 ms
    still = ev[(ev["timestamp"] >= stop - 30_000) & (ev["timestamp"] < stop)].copy()
    still = still[rng.random(len(still)) < 0.1]
    if pause > 0 and len(still):
        still["timestamp"] = stop + rng.integers(0, pause, len(still))
    else:
        still = still[:0]
    # the background (static scene) continues after the cut: reuse late-clip sparse events
    return merge(before, still, tail), stop


def overlay(base, distractor, rng, weight=(0.3, 0.8), shift=80):
    """Mix events of another recording (people moving in the background) into a clip."""
    d = distractor[rng.random(len(distractor)) < rng.uniform(*weight)]
    d = geometric(d, rng, shift=shift, scale=(0.5, 0.9), rotate_deg=5)
    return merge(base, d)

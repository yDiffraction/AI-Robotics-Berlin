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


def geometric(ev, rng, flip=None, shift=15, scale=(0.9, 1.1), rotate_deg=5.0):
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


def dropout(ev, rng, keep=(0.85, 1.0), polarity_bias=0.08):
    """Drop events uniformly (contrast / lighting) with a slight ON/OFF imbalance."""
    k = rng.uniform(*keep)
    bias = rng.uniform(-polarity_bias, polarity_bias)
    p = np.where(ev["polarity"] == 1, k * (1 + bias), k * (1 - bias))
    return ev[rng.random(len(ev)) < p]


def add_noise(ev, rng, duration_us, rate_hz=(0.0, 500.0), hot_pixels=(0, 3), hot_rate_hz=(20.0, 100.0)):
    """Uniform background activity plus a few hot pixels."""
    n = rng.poisson(rng.uniform(*rate_hz) * duration_us / 1e6)
    noise = np.empty(n, DTYPE)
    noise["x"] = rng.integers(0, SENSOR, n)
    noise["y"] = rng.integers(0, SENSOR, n)
    noise["polarity"] = rng.integers(0, 2, n)
    noise["timestamp"] = rng.integers(0, max(1, duration_us), n)
    parts = [ev, noise]
    for _ in range(rng.integers(*hot_pixels)):
        m = rng.poisson(rng.uniform(*hot_rate_hz) * duration_us / 1e6)
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


def ease_out(ev, t0, t1, extend_us, power=1.5):
    """Time-warp [t0, t1) so the motion decelerates and lasts `extend_us` longer; later events shift."""
    out = ev.copy()
    t = out["timestamp"].astype(np.float64)
    seg = (t >= t0) & (t < t1)
    u = (t[seg] - t0) / (t1 - t0)
    out["timestamp"][seg] = (t0 + (t1 - t0 + extend_us) * u ** power).astype(np.int64)
    out["timestamp"][t >= t1] += int(extend_us)
    return out[np.argsort(out["timestamp"], kind="stable")]


def feint_splice(throw, throw_rel, donor, donor_rel, rng, stop_before=(0.0, 0.05), ease=0.06,
                 extend=(0.0, 0.03), fade=0.03):
    """Synthetic feint: real wind-up of a throw, decelerated, cross-faded into the return phase
    of a REAL feint (donor). Replaces the earlier cut-and-reverse recipe whose rate trough was a
    100 %-separable shortcut (see dataset/README.md).

    throw/donor: native event arrays; *_rel: decision moment (us) in each stream.
    Returns (events, decision_us).
    """
    stop = int(throw_rel - rng.uniform(*stop_before) * 1e6)
    ext = int(rng.uniform(*extend) * 1e6)
    a = ease_out(throw[throw["timestamp"] < stop], stop - int(ease * 1e6), stop, ext)
    decision = stop + ext
    f = int(fade * 1e6)
    # donor: everything after its decision moment, shifted so it starts at our decision and
    # moved so the donor's body sits where the thrower's body is (takes differ in position)
    def body_centre(ev, t_end):
        sel = ev[(ev["timestamp"] >= t_end - 300_000) & (ev["timestamp"] < t_end)]
        return (np.median(sel["x"]), np.median(sel["y"])) if len(sel) > 100 else (np.nan, np.nan)
    tx, ty = body_centre(throw, stop)
    dx, dy = body_centre(donor, donor_rel)
    d = donor[donor["timestamp"] >= donor_rel - f].copy()
    d["timestamp"] = d["timestamp"] - donor_rel + decision
    if np.isfinite([tx, ty, dx, dy]).all():
        d = _from_float(d["x"] + (tx - dx), d["y"] + (ty - dy), d["polarity"], d["timestamp"].astype(np.float64))
    # linear cross-fade over `fade`: thin the throw tail and the donor head
    ta = a["timestamp"].astype(np.float64)
    keep_a = rng.random(len(a)) < np.clip((decision - ta) / f, 0, 1)
    td = d["timestamp"].astype(np.float64)
    keep_d = rng.random(len(d)) < np.clip((td - decision + f) / f, 0, 1)
    return merge(a[keep_a | (ta < decision - f)], d[keep_d]), decision


def refractory(ev, dead_us):
    """Drop events that follow the previous event of the same pixel within `dead_us`."""
    if len(ev) == 0 or dead_us <= 0:
        return ev
    key = ev["y"].astype(np.int64) * 320 + ev["x"].astype(np.int64)
    order = np.lexsort((ev["timestamp"], key))
    k, t = key[order], ev["timestamp"][order].astype(np.int64)
    keep = np.ones(len(ev), bool)
    same = k[1:] == k[:-1]
    keep[1:] = ~(same & (t[1:] - t[:-1] < dead_us))
    return ev[np.sort(order[keep])]


def ego_motion(ev, rng, amplitude_px=(1.0, 3.0), slice_us=20_000):
    """Small coherent camera shake (robot body motion): one smooth 2-D offset per time slice."""
    if len(ev) == 0:
        return ev
    t = ev["timestamp"].astype(np.float64)
    amp = rng.uniform(*amplitude_px)
    fx, fy = rng.uniform(0.5, 3.0, 2)
    px, py = rng.uniform(0, 2 * np.pi, 2)
    ts = np.floor(t / slice_us) * slice_us / 1e6
    dx = amp * np.sin(2 * np.pi * fx * ts + px)
    dy = amp * np.sin(2 * np.pi * fy * ts + py)
    return _from_float(np.round(ev["x"] + dx), np.round(ev["y"] + dy), ev["polarity"], t)


def overlay(base, distractor, rng, budget=(0.1, 0.3), shift=80, duration_us=None):
    """Mix a random time crop of another recording (a person moving in the background) into a
    clip, thinned so it adds `budget` x the base event count."""
    duration_us = duration_us or int(base["timestamp"].max() + 1 if len(base) else 1_000_000)
    span = int(distractor["timestamp"].max()) if len(distractor) else 0
    t0 = int(rng.uniform(0, max(1, span - duration_us)))
    d = window(distractor, t0, t0 + duration_us)
    if len(d) == 0:
        return base
    target = rng.uniform(*budget) * max(len(base), 1)
    d = d[rng.random(len(d)) < min(1.0, target / len(d))]
    d = geometric(d, rng, shift=shift, scale=(0.5, 0.9), rotate_deg=5)
    return merge(base, d)

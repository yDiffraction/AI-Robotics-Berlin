#!/usr/bin/env python3
"""Raw external events -> GenX320-look 1.2 s clips (positives, negatives and ambiguous alike).

Every external recording, whatever its label, goes through exactly the same steps, so no step
can become a label cue. Targets come from our own clips (`own_profile.json`, made by
`profile_own.py`) and are *sampled per clip* from our distribution rather than fixed.

 1. foreign-noise removal: a background-activity filter (an event needs another event in its
    3x3 pixel / +-3 ms neighbourhood) removes the source sensor's noise and hot pixels, and the
    flicker v2e turns video compression artefacts into.
 2. anchor: event-rate peak (20 ms bins, 3-tap smoothing) in the allowed range, the same rule
    as our own labels (`release_us` = 700 ms). Negatives are anchored the same way (their
    busiest moment), so "a burst of motion at 700 ms" carries no label. `--neg-random` adds an
    unanchored window per negative as well.
 3. geometry: the motion region (5-95 % of active cells in the swing window) is scaled to a
    height drawn from our clips and moved to a centre drawn from our clips (+-jitter). Sub-pixel
    jitter avoids empty rows when enlarging. If the source frame edge would cut the body
    inside the image, the clip is moved down so the cut lies at the bottom border, as in ours.
 4. density: a per-pixel contrast-threshold increase (keep every k-th event of each pixel and
    polarity, fractional k) brings the swing-window active cells per 10 ms frame to a level
    drawn from our clips. This thins weak edges first, like a less sensitive sensor; random
    dropout would leave speckle (more isolated cells than ours).
 5. polarity: the ON share of cells is moved to a value drawn from our clips by thinning
    the dominant polarity.
 6. noise: the per-cell noise rate map of our own sensor (measured on our empty-scene idle
    clips, including its hot cells) is added with a random gain.

Video sources (HMDB): `--time-jitter-us 33333` spreads timestamps over one video frame period;
without it the activity still pulses at 30 fps (30 ms autocorrelation 0.14 against our 0.78).

Output: <out>/<role>/<file>.bin (GenX320 format, t from 0, 1.2 s) + <out>/labels.csv in the
schema of our dataset (plus src / role / cls / group / variant / params) + previews.

  python match.py --src dd   --raw data/extpos/dailydvs --profile own_profile.json --out data/extpos/clips_dd
  python match.py --src hmdb --raw data/extpos/hmdb/raw --profile own_profile.json --out data/extpos/clips_hmdb
  add --naive for the baseline conversion (crop + rescale only, like tools/dailydvs_import.py)
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import cellstats as cs

PRE_US, POST_US = 700_000, 500_000
CLASS_ID = {"pos": "01", "neg": "03", "amb": "04"}


# ------------------------------------------------------------------- loading sources
def load_dd(raw_dir, args):
    rows = list(csv.DictReader(open(Path(raw_dir) / "raw_manifest.csv")))
    out = []
    for r in rows:
        if not args.keep_moving and r.get("move", "0") == "1":
            continue
        if args.frontal_only and r.get("perspective") != "Frontal":
            continue
        out.append(dict(path=str(Path(raw_dir) / r["path"]), role=r["role"], cls=r["action_name"],
                        group=f"dd:P{r['person']}", parent=Path(r["path"]).stem, src="dd",
                        note=f"{r['perspective']}/{r['distance']}/{r['scene']}"))
    return out


def load_hmdb(raw_dir, args):
    out = []
    for f in sorted(Path(raw_dir).glob("*/*.npz")):
        m = json.loads(str(np.load(f)["meta"]))
        out.append(dict(path=str(f), role=m["role"], cls=m["cls"], group=f"hmdb:{m['video']}",
                        parent=f.stem, src="hmdb", note=f"{m['body']}/{m['cam']}/{m['view']}/{m['qual']}"))
    return out


def read_raw(path):
    d = np.load(path)
    t = d["t"].astype(np.int64)
    o = np.argsort(t, kind="stable")
    return t[o] - t[o][0] if len(t) else t, d["x"][o].astype(np.int32), d["y"][o].astype(np.int32), \
        d["p"][o].astype(np.uint8), tuple(int(v) for v in d["wh"])


# ------------------------------------------------------------------------- steps
def ba_filter(t, x, y, p, wh, dt_us=3000, px=1, support=2):
    """Keep events with at least `support` other events within +-px pixels and +-dt_us (bins)."""
    if len(t) == 0:
        return np.zeros(0, bool)
    W, H = wh
    tb = (t // dt_us).astype(np.int64)
    key = (tb * H + y) * W + x
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    total = np.zeros(len(u), np.int64)
    for dtb in (-1, 0, 1):
        for dy in range(-px, px + 1):
            for dx in range(-px, px + 1):
                k = u + (dtb * H + dy) * W + dx
                i = np.searchsorted(u, k)
                i[i >= len(u)] = 0
                hit = u[i] == k
                total += np.where(hit, cnt[i], 0)
    # total includes the event's own bin: needs one more event than itself
    return total[inv] >= 1 + support


def rate_peak(t, lo, hi):
    edges = np.arange(lo, hi + 20_000, 20_000)
    if len(edges) < 2:
        return None
    h = np.convolve(np.histogram(t, edges)[0], np.ones(3) / 3, mode="same")
    return int(edges[int(np.argmax(h))] + 10_000)


def motion_box(x, y, t, rel_us, W, H):
    """5-95 % extent and centre of active 4x4 cells in the swing window (-500..+300 ms)."""
    m = (t >= rel_us - 500_000) & (t < rel_us + 300_000)
    cx_, cy_ = x[m] // cs.BLOCK, y[m] // cs.BLOCK
    if m.sum() < 50:
        return None
    k = np.unique(((t[m] - rel_us) // cs.FRAME_US) * 1_000_000 + cy_ * 1000 + cx_)
    yy, xx = (k % 1_000_000) // 1000, k % 1000
    y05, y95 = np.percentile(yy, [5, 95])
    return dict(h=float(y95 - y05) * cs.BLOCK, cx=float(xx.mean()) * cs.BLOCK, cy=float(yy.mean()) * cs.BLOCK,
                bottom=float(np.percentile(yy, 99)) * cs.BLOCK)


def threshold_thin(x, y, p, t, k, rng):
    """Keep the events where the per-(pixel, polarity) running count crosses a multiple of k."""
    if k <= 1.0:
        return np.ones(len(t), bool)
    key = (y.astype(np.int64) * 1024 + x) * 2 + p
    o = np.lexsort((t, key))
    ks = key[o]
    start = np.r_[True, ks[1:] != ks[:-1]]
    grp = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    n = np.arange(len(ks)) - first[grp] + 1                 # 1-based count within pixel/polarity
    phase = rng.uniform(0, k, size=len(first))[grp]         # random phase per pixel
    keep_sorted = np.floor((n + phase) / k) > np.floor((n - 1 + phase) / k)
    keep = np.zeros(len(t), bool)
    keep[o] = keep_sorted
    return keep


def cells_of(t, x, y, p):
    return cs.to_cells((x, y, p, t), n_frames=cs.CLIP_US // cs.FRAME_US, t0=0)


def add_noise(g, rate, gain, rng):
    """OR in Bernoulli noise cells from our sensor's per-cell rate map [2,80,80]."""
    T = g.shape[0]
    nz = rng.random((T,) + rate.shape) < np.clip(rate * gain, 0, 1)
    return g | nz.astype(np.uint8), nz


def cells_to_events(cells_extra, rng):
    """Turn noise cells into events at a random pixel / time inside the cell."""
    f, pp, cy, cx = np.nonzero(cells_extra)
    n = len(f)
    t = f.astype(np.int64) * cs.FRAME_US + rng.integers(0, cs.FRAME_US, n)
    return t, cx * cs.BLOCK + rng.integers(0, cs.BLOCK, n), cy * cs.BLOCK + rng.integers(0, cs.BLOCK, n), pp.astype(np.uint8)


# ----------------------------------------------------------------------- per clip
def process(job):
    item, prof, args, variant = job
    rng = np.random.default_rng((abs(hash(item["parent"])) + 7919 * variant) % (1 << 32))
    t, x, y, p, (W, H) = read_raw(item["path"])
    if args.time_jitter_us > 0 and len(t):
        # video sources: events bunch at the real video frames (30 fps) even after interpolation;
        # a uniform jitter of one frame period cancels that periodic component exactly
        t = t + rng.integers(-args.time_jitter_us // 2, args.time_jitter_us // 2 + 1, len(t))
        o = np.argsort(t, kind="stable")
        t, x, y, p = t[o] - t[o][0], x[o], y[o], p[o]
    dur = int(t[-1]) if len(t) else 0
    if dur < PRE_US + POST_US or len(t) < 2000:
        return None
    S = cs.SENSOR
    info = dict(n_raw=len(t))
    if not args.naive:
        k = ba_filter(t, x, y, p, (W, H))
        info["ba_kept"] = round(float(k.mean()), 3)
        t, x, y, p = t[k], x[k], y[k], p[k]
    # ---- anchor
    neg_random = item["role"] == "neg" and variant >= args.variants_neg
    if neg_random:
        a = int(rng.uniform(PRE_US, dur - POST_US))
    else:
        a = rate_peak(t, PRE_US, dur - POST_US)
        if a is None:
            return None
    m = (t >= a - PRE_US) & (t < a + POST_US)
    t, x, y, p = t[m] - (a - PRE_US), x[m], y[m], p[m]
    if len(t) < 500:
        return None
    rel = PRE_US
    if args.naive:
        # baseline = what tools/dailydvs_import.py does: square crop around the median x, rescale
        side = min(W, H)
        cx = int(np.clip(np.median(x), side // 2, W - side // 2))
        keep = (x >= cx - side // 2) & (x < cx + side // 2)
        t, x, y, p = t[keep], x[keep] - (cx - side // 2), y[keep], p[keep]
        s = S / side
        x = np.clip((x + rng.random(len(x))) * s, 0, S - 1).astype(np.int32)
        y = np.clip((y + rng.random(len(y))) * s, 0, S - 1).astype(np.int32)
        info["naive"] = 1
        return finish(item, variant, t, x, y, p, rel, info, args, neg_random)
    # ---- actor mask (source coordinates): only the actor is taken from the foreign recording; the
    # background (noise) comes from our sensor. Same for every role.
    kc = clean_cells(t, x, y, p, args.min_nb, (W, H))
    t, x, y, p = t[kc], x[kc], y[kc], p[kc]
    am, share = actor_mask(t, x, y, (W, H))
    info["actor_share"] = round(share, 3)
    if item["role"] == "pos" and share < args.min_actor_share:
        return None                      # diffuse / no clear actor: not a usable positive
    k = am[y // cs.BLOCK, x // cs.BLOCK]
    t, x, y, p = t[k], x[k], y[k], p[k]
    # ---- geometry
    box = motion_box(x, y, t, rel, W, H)
    if box is None or box["h"] < 8:
        return None
    ref = prof["own_pos"][int(rng.integers(len(prof["own_pos"])))]
    tgt_h = ref["ext_h"] * cs.BLOCK * rng.uniform(0.85, 1.15)
    s = float(np.clip(tgt_h / max(box["h"], 1.0), 0.3, 2.5))
    tcx = (ref["cx"] + rng.uniform(-6, 6)) * cs.BLOCK
    tcy = (ref["cy"] + rng.uniform(-3, 3)) * cs.BLOCK
    ox, oy = tcx - box["cx"] * s, tcy - box["cy"] * s
    xf = (x + rng.random(len(x))) * s + ox
    yf = (y + rng.random(len(y))) * s + oy
    keep = (xf >= 0) & (xf < S) & (yf >= 0) & (yf < S)
    # source frame edges that land inside the image would show as straight cut lines through the
    # body: fade events out towards such an edge (over 12 % of the source size, random depth)
    fade = np.ones(len(t))
    L = 0.12 * min(W, H)
    for inside, dist in ((ox > 1, x.astype(float)), (ox + W * s < S - 1, W - 1 - x.astype(float)),
                         (oy > 1, y.astype(float)), (oy + H * s < S - 1, H - 1 - y.astype(float))):
        if inside:
            fade *= np.clip(dist / (L * rng.uniform(0.6, 1.4)), 0, 1) ** 1.5
    keep &= rng.random(len(t)) < fade
    t, x, y, p = t[keep], xf[keep].astype(np.int32), yf[keep].astype(np.int32), p[keep]
    info.update(scale=round(s, 3), shift=[round(ox), round(oy)])
    if len(t) < 300:
        return None
    sw = slice((rel - 500_000) // cs.FRAME_US, (rel + 300_000) // cs.FRAME_US)

    def bisect(measure, thin, target, lo=1.0, hi=64.0, tol=0.03):
        """Geometric bisection on the threshold factor k; returns the keep-mask closest to target."""
        best, best_err, k_best = np.ones(len(t), bool), abs(measure(np.ones(len(t), bool)) - target), 1.0
        for _ in range(14):
            mid = (lo * hi) ** 0.5
            kk = thin(mid)
            v = measure(kk)
            if abs(v - target) < best_err:
                best, best_err, k_best = kk, abs(v - target), mid
            if best_err <= tol * abs(target):
                break
            lo, hi = (mid, hi) if v > target else (lo, mid)
        return best, k_best

    # ---- polarity balance first (DVXplorer is ON-heavy): raise the dominant polarity's threshold
    def on_share(keep):
        g = cells_of(t[keep], x[keep], y[keep], p[keep])[sw]
        return g[:, 1].sum() / max(g.sum(), 1)

    on = on_share(np.ones(len(t), bool))
    tgt_on = float(np.clip(ref["on_frac"] + rng.normal(0, 0.01), 0.40, 0.55))
    if abs(on - tgt_on) > 0.01:
        dom = 1 if on > tgt_on else 0
        sub = np.random.default_rng(int(rng.integers(1 << 30)))
        sign = 1 if dom == 1 else -1
        keep, k_on = bisect(lambda kk: sign * on_share(kk),
                            lambda k: (p != dom) | threshold_thin(x, y, p, t, k, sub),
                            sign * tgt_on, tol=0.01 / tgt_on)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
        info["on_fix"] = [round(float(on), 3), round(tgt_on, 3), round(k_on, 2)]

    # ---- density: per-pixel threshold increase on both polarities
    def dens(keep):
        g = cells_of(t[keep], x[keep], y[keep], p[keep])[sw]
        return g.reshape(g.shape[0], -1).sum(1).mean()

    target = ref["cells_mean"] * rng.uniform(0.85, 1.2)
    base = dens(np.ones(len(t), bool))
    info["cells_before"], info["cells_target"] = round(float(base), 1), round(float(target), 1)
    if base > target:
        sub = np.random.default_rng(int(rng.integers(1 << 30)))
        keep, k = bisect(dens, lambda k: threshold_thin(x, y, p, t, k, sub), target)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
        info["k"] = round(k, 2)
    return finish(item, variant, t, x, y, p, rel, info, args, neg_random, prof, rng)


def box2d(a, r):
    c = np.pad(a, r).cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def occupancy(t, x, y, wh):
    """[T, H/4, W/4] cell occupancy (both polarities) on the source grid."""
    gw, gh = -(-wh[0] // cs.BLOCK), -(-wh[1] // cs.BLOCK)
    T = cs.CLIP_US // cs.FRAME_US
    o = np.zeros((T, gh, gw), np.int16)
    f = np.clip(t // cs.FRAME_US, 0, T - 1)
    o[f, y // cs.BLOCK, x // cs.BLOCK] = 1
    return o


def actor_mask(t, x, y, wh, r=3, rel_thr=0.08, grow=2):
    """Cells around the main activity: 7x7-smoothed activity map above 8 % of its maximum,
    largest connected region, grown by 2 cells. Returns the mask and the share of active cells
    inside it."""
    occ = occupancy(t, x, y, wh).sum(0).astype(np.float64)
    sm = box2d(occ, r)
    m = sm >= rel_thr * sm.max()
    # largest 4-connected component (flood fill from the maximum)
    lab = np.zeros_like(m, bool)
    stack = [tuple(np.unravel_index(np.argmax(sm), sm.shape))]
    while stack:
        i, j = stack.pop()
        if 0 <= i < m.shape[0] and 0 <= j < m.shape[1] and m[i, j] and not lab[i, j]:
            lab[i, j] = True
            stack += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    lab = box2d(lab.astype(np.float64), grow) > 0
    return lab, float(occ[lab].sum() / max(occ.sum(), 1))


def clean_cells(t, x, y, p, min_nb, wh):
    """Drop events in cells with fewer than min_nb active neighbour cells (3x3, +-1 frame): the
    foreign sensor's residual noise, which otherwise outlines the source field of view."""
    o = occupancy(t, x, y, wh)
    T, gh, gw = o.shape
    pad = np.pad(o, 1)
    nb = sum(pad[1 + a:1 + a + T, 1 + b:1 + b + gh, 1 + c:1 + c + gw]
             for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)) - o
    f = np.clip(t // cs.FRAME_US, 0, T - 1)
    return nb[f, y // cs.BLOCK, x // cs.BLOCK] >= min_nb


def finish(item, variant, t, x, y, p, rel, info, args, neg_random, prof=None, rng=None):
    rng = rng or np.random.default_rng(0)
    if prof is not None:
        rate = np.array(prof["noise_rate"], np.float64)
        gain = float(np.exp(rng.normal(args.noise_log_gain, 0.5)))
        g = cells_of(t, x, y, p)
        _, nz = add_noise(np.zeros_like(g), rate, gain, rng)
        nz &= ~g.astype(bool)
        nt, nx, ny, npol = cells_to_events(nz, rng)
        t, x, y, p = (np.r_[t, nt], np.r_[x, nx], np.r_[y, ny], np.r_[p, npol])
        info["noise_gain"] = round(gain, 2)
    tag = f"v{variant}" + ("r" if neg_random else "")
    name = f"gesture_{CLASS_ID[item['role']]}-{item['role']}-{item['src']}_{item['cls']}-{item['parent']}-{tag}.bin"
    out = Path(args.out) / item["role"] / name
    cs.write_bin(out, x.astype(np.uint16), y.astype(np.uint16), p.astype(np.uint8), t.astype(np.int64))
    st = cs.clip_stats(cells_of(t, x, y, p), None if neg_random else rel // cs.FRAME_US)
    return dict(file=name, folder=item["role"], label="throw_motion" if item["role"] == "pos" else item["role"],
                class_id=CLASS_ID[item["role"]], dataset=item["src"], source=Path(item["path"]).name,
                parent=item["parent"], start_s="", decision_s="",
                release_us="" if neg_random else rel, events=len(t), verified="auto", note=item["note"],
                src=item["src"], role=item["role"], cls=item["cls"], group=item["group"], variant=tag,
                params=json.dumps(info), stats=json.dumps({k: round(v, 3) for k, v in st.items()}))


def preview(rows, out, n=24):
    """One row per clip, 12 frames of 100 ms (ON green, OFF magenta), like our dataset previews."""
    from PIL import Image
    rows = rows[:n]
    if not rows:
        return
    tile = 80
    im = np.zeros((len(rows) * tile, 12 * tile, 3), np.uint8)
    for i, r in enumerate(rows):
        g = cs.to_cells(cs.read_bin(Path(out) / r["folder"] / r["file"]))
        for j in range(12):
            blk = g[j * 10:(j + 1) * 10].any(0)
            im[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, 1] = blk[1] * 255
            im[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, 0] = blk[0] * 255
            im[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile, 2] = blk[0] * 255
        im[i * tile, :, :] = 60
    return Image.fromarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", choices=["dd", "hmdb"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants-pos", type=int, default=3, help="variants per positive recording")
    ap.add_argument("--variants-amb", type=int, default=1)
    ap.add_argument("--variants-neg", type=int, default=1)
    ap.add_argument("--neg-random", action="store_true", help="one extra unanchored window per negative")
    ap.add_argument("--keep-moving", action="store_true", help="DailyDVS: keep recordings with Move=1")
    ap.add_argument("--naive", action="store_true", help="baseline conversion only (for the probe)")
    ap.add_argument("--limit", type=int, default=0, help="at most N recordings per (role, class)")
    ap.add_argument("--time-jitter-us", type=int, default=0,
                    help="uniform timestamp jitter (video sources: one frame period, e.g. 33333)")
    ap.add_argument("--min-nb", type=int, default=2, help="cell cleaning: min active neighbour cells")
    ap.add_argument("--min-actor-share", type=float, default=0.75,
                    help="positives whose activity is less concentrated on one actor are dropped")
    ap.add_argument("--noise-log-gain", type=float, default=0.7, help="median noise gain = exp(this)")
    ap.add_argument("--frontal-only", action="store_true", help="DailyDVS: only Perspective=Frontal")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    prof = json.load(open(a.profile))
    items = (load_dd if a.src == "dd" else load_hmdb)(a.raw, a)
    if a.limit:
        by = {}
        for it in items:
            by.setdefault((it["role"], it["cls"]), []).append(it)
        items = [it for v in by.values() for it in v[:a.limit]]
    jobs = []
    for it in items:
        nv = {"pos": a.variants_pos, "amb": a.variants_amb, "neg": a.variants_neg}[it["role"]]
        nv += int(a.neg_random and it["role"] == "neg")
        jobs += [(it, prof, a, v) for v in range(nv)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, r in enumerate(ex.map(process, jobs, chunksize=4)):
            if r:
                rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["role"], r["file"]))
    with (Path(a.out) / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    for role in ("pos", "neg", "amb"):
        rr = [r for r in rows if r["role"] == role]
        if rr:
            preview([rr[i] for i in rng.permutation(len(rr))], a.out).save(Path(a.out) / f"preview_{role}.png")
    print(f"{len(rows)} clips from {len(items)} recordings in {time.time() - t0:.0f}s:",
          {r: sum(x['role'] == r for x in rows) for r in ('pos', 'neg', 'amb')})


if __name__ == "__main__":
    main()

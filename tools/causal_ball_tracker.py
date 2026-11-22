#!/usr/bin/env python3
"""Causal classical ball-release detector on binary 80x80 event frames (10 ms, 4x4 blocks).

Processes frames strictly in order: step(frame) only uses the current and past frames and
returns True on the frame where it fires. Per frame (every gate can be switched off for ablations):

 1. activity     armed: the 5-frame mean activity reached wind-up level (>= ARM cells) within the
                 last ARM_WIN frames; ego-motion: a frame with > EGO active cells (the whole image
                 moves) blocks triggers for EGO_HOLD frames.
 2. denoise      optional hysteresis (Zhu et al. 2023): seed = cells active now and within +-1 cell
                 in the previous frame; keep active cells within +-GROW cells of a seed (HYST=1).
 3. blobs        8-connected components of SMIN..SMAX cells. A blob is "free" if no cell of the
                 largest component (body/arm, if larger than SMAX) lies within BODY_GAP cells of
                 its bounding box and no other component of >= ISO_MIN cells within ISO cells.
 4. linking      each blob extends the best chain of a blob in the previous frame (or GAP frames
                 earlier) whose top row lies DY_MIN..DY_MAX cells per frame lower and whose centre
                 is within +-DX_MAX columns per frame (speed gate); unlinked blobs start a chain,
                 chains not extended for GAP + 1 frames end (pruning).
 5. trigger      the last FREE_LAST blobs are free, the current blob is compact (<= MAX_H rows)
                 with at most ABOVE active cells above it within +-4 columns (the ball flies above
                 everything), and some window of the last WIN..MAX_WIN chain frames passes: rise of
                 the top row >= MIN_RISE cells, least-squares rise velocity VMIN..VMAX cells/frame,
                 rms residual of that constant-velocity fit <= MAX_RES, |column velocity| <= MAX_VX,
                 mean newness (share of blob cells inactive 2 frames earlier, cf. Falanga et al.
                 2020) >= MIN_NEW.

usage:  tr = CausalBallTracker(); for f in frames: fired = tr.step(f)
        first_trigger(frames) -> (frame index or -1, chain)
        python3 causal_tracker.py clip.bin [...]
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)

DEFAULTS = dict(
    SMIN=3, SMAX=45,          # blob size (cells)
    BODY_GAP=3,               # free from the body: largest component not within this many cells (0 = off)
    ISO=3, ISO_MIN=3,         # free from clutter: no other component >= ISO_MIN cells within ISO cells (0 = off)
    FREE_LAST=2,              # the last FREE_LAST blobs of the chain must be free (0 = off)
    DY_MIN=0, DY_MAX=5,       # rise of the blob top between linked frames (cells)
    DX_MAX=3,                 # column change between linked frames (cells)
    GAP=0,                    # a chain may skip this many frames without a blob (ranges scale with the gap)
    MAXLEN=12,                # chain memory (frames)
    WIN=5,                    # shortest tested window = minimum chain length (frames, 4 links)
    MAX_WIN=12,               # longest tested window (frames); every window WIN..MAX_WIN is tested
    MIN_RISE=6,               # rise of the top row over the window (cells)
    VMIN=0.8, VMAX=3.5,       # least-squares rise velocity (cells/frame)
    MAX_RES=0.6,              # rms residual of the constant-velocity fit (cells, 0 = off)
    MAX_VX=1.0,               # |column velocity| (cells/frame)
    MIN_NEW=0.4,              # mean newness over the window (0 = off)
    MAX_H=8,                  # compact: height of the current blob <= MAX_H rows (0 = off)
    ABOVE=0,                  # max active cells above the current blob within +-4 columns (-1 = off)
    ARM=35, ARM_WIN=100,      # armed: 5-frame mean activity >= ARM within the last ARM_WIN frames (0 = off)
    EGO=600, EGO_HOLD=10,     # ego-motion gate (0 = off)
    HYST=0, GROW=2,           # hysteresis denoising (1 = on)
)


@dataclass
class Blob:
    top: int
    h: int
    x: float
    size: int
    new: float
    free: bool
    above: int
    chain: list = field(default_factory=list)   # [(t, top, x, size, new, free)], oldest first


class CausalBallTracker:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown parameters {unknown}")
        self.p = {**DEFAULTS, **params}
        self.reset()

    def reset(self):
        self.t = -1
        self.prev = []            # last two raw frames (newness, hysteresis)
        self.hist = []            # blobs of the previous GAP + 1 frames, with their chains
        self.act = []             # recent activity (cells per frame)
        self.ego_until = -1
        self.last_chain = None    # chain that fired

    # ---- per-frame pieces -----------------------------------------------------------------------
    def _armed(self):
        p = self.p
        if not p["ARM"]:
            return True
        a = np.asarray(self.act[-(p["ARM_WIN"] + 4):], float)
        if len(a) < 5:
            return False
        return bool((np.convolve(a, np.ones(5) / 5, mode="valid") >= p["ARM"]).any())

    def _denoise(self, f):
        p = self.p
        if not p["HYST"] or not self.prev:
            return f
        seed = f & ndi.binary_dilation(self.prev[-1], structure=EIGHT)
        if not seed.any():
            return np.zeros_like(f)
        return f & ndi.binary_dilation(seed, structure=EIGHT, iterations=p["GROW"])

    def _blobs(self, f):
        p = self.p
        lab, n = ndi.label(f, structure=EIGHT)
        if n == 0:
            return []
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        big = int(np.argmax(sizes))
        body_gap = p["BODY_GAP"] if sizes[big] > p["SMAX"] else 0
        old = self.prev[-2] if len(self.prev) >= 2 else None
        H, W = f.shape
        out = []
        for k, sl in enumerate(ndi.find_objects(lab), start=1):
            if not p["SMIN"] <= sizes[k] <= p["SMAX"]:
                continue
            (y0, y1), (x0, x1) = (sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop)

            def near(r):
                return lab[max(y0 - r, 0):min(y1 + r, H), max(x0 - r, 0):min(x1 + r, W)]
            free = True
            if body_gap:
                free = not (near(body_gap) == big).any()
            if free and p["ISO"]:
                others = np.unique(near(p["ISO"]))
                free = not any(sizes[j] >= p["ISO_MIN"] for j in others if j not in (0, k))
            ys, xs = np.nonzero(lab[sl] == k)
            ys, xs = ys + y0, xs + x0
            new = float((~old[ys, xs]).mean()) if old is not None else 1.0
            above = int(f[:y0, max(int(xs.mean()) - 4, 0):int(xs.mean()) + 5].sum()) if p["ABOVE"] >= 0 else 0
            out.append(Blob(y0, y1 - y0, float(xs.mean()), int(sizes[k]), new, free, above))
        return out

    def _link(self, b):
        p = self.p
        best, best_key = None, None
        for gap, blobs in enumerate(reversed(self.hist)):       # gap 0 = previous frame
            k = gap + 1
            for q in blobs:
                dy, dx = q.top - b.top, b.x - q.x
                if not (p["DY_MIN"] * k <= dy <= p["DY_MAX"] * k and abs(dx) <= p["DX_MAX"] * k):
                    continue
                c = q.chain
                v = (c[-2][1] - c[-1][1]) / (c[-1][0] - c[-2][0]) if len(c) >= 2 else dy / k
                key = (min(len(c), p["WIN"]), -gap, -abs(dy / k - v))
                if best_key is None or key > best_key:
                    best, best_key = q, key
        obs = (self.t, b.top, b.x, b.size, b.new, b.free)
        b.chain = (best.chain[-(p["MAXLEN"] - 1):] if best else []) + [obs]

    def _window_ok(self, c):
        """Constant-velocity rise test on chain observations c (array [n, 6])."""
        p = self.p
        t, y, x = c[:, 0], c[:, 1], c[:, 2]
        if y[0] - y[-1] < p["MIN_RISE"]:
            return False
        A = np.stack([t - t.mean(), np.ones_like(t)], 1)
        sol_y = np.linalg.lstsq(A, y, rcond=None)[0]
        if not p["VMIN"] <= -sol_y[0] <= p["VMAX"]:
            return False
        if p["MAX_RES"] and np.sqrt(np.mean((y - A @ sol_y) ** 2)) > p["MAX_RES"]:
            return False
        if abs(np.linalg.lstsq(A, x, rcond=None)[0][0]) > p["MAX_VX"]:
            return False
        if p["MIN_NEW"] and c[:, 4].mean() < p["MIN_NEW"]:
            return False
        return True

    def _fires(self, b):
        p = self.p
        if len(b.chain) < p["WIN"]:
            return False
        if p["FREE_LAST"] and not all(o[5] for o in b.chain[-p["FREE_LAST"]:]):
            return False
        if p["ABOVE"] >= 0 and b.above > p["ABOVE"]:
            return False
        if p["MAX_H"] and b.h > p["MAX_H"]:
            return False
        c = np.asarray(b.chain, float)
        return any(self._window_ok(c[-w:]) for w in range(p["WIN"], min(len(c), p["MAX_WIN"]) + 1))

    # ---- public ---------------------------------------------------------------------------------
    def step(self, frame):
        """Feed one binary 80x80 frame; True if the detector fires on this frame."""
        p = self.p
        self.t += 1
        f = np.asarray(frame, bool)
        a = int(f.sum())
        self.act = (self.act + [a])[-(p["ARM_WIN"] + 4):]
        if p["EGO"] and a > p["EGO"]:
            self.ego_until = self.t + p["EGO_HOLD"]
        blobs = self._blobs(self._denoise(f))
        for b in blobs:
            self._link(b)
        self.hist = (self.hist + [blobs])[-(p["GAP"] + 1):]
        self.prev = (self.prev + [f])[-2:]
        if self.t <= self.ego_until or not self._armed():
            return False
        for b in blobs:
            if self._fires(b):
                self.last_chain = b.chain
                return True
        return False


def first_trigger(frames, **params):
    """(index of the first frame where the detector fires or -1, firing chain or None)."""
    tr = CausalBallTracker(**params)
    for t, f in enumerate(frames):
        if tr.step(f):
            return t, tr.last_chain
    return -1, None


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clips as C
    for path in sys.argv[1:]:
        t, chain = first_trigger(C.frames(path))
        print(f"{path}: " + ("no trigger" if t < 0 else f"fires at frame {t} ({t * 10} ms)"))

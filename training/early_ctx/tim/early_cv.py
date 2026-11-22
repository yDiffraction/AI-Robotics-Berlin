#!/usr/bin/env python3
"""Early throw detector: one output neuron that fires as early as possible before a throw
(or the throw-like wind-up of a feint) and stays silent for everything else, running
indefinitely on the chip.

Data (banks from prep_sources.py, sparse 80x80 / 10 ms binary codes):
  positives  noah   own GenX320 throws + feints (release at frame 70 of 120)
  negatives  noah   idle clips from the same camera/scene (kills "our scene = throw")
             mb     motion_buddy (Durance, GenX320): sit / lay-down / hello / dance / no-action
             dd     DailyDVS-200 (DVXplorer): arm_no_release + idle actions
             dvsg   DVS Gesture (DVS128): 11 hand/arm gestures
  heldpos    dd     DailyDVS throw / toss actions: never trained on (unless --dd-pos); an
                    unseen-camera positive test of whether the net learned the motion or the source

Training samples are multi-second STREAMS stitched from 1 s clips of random sources
(each stream: one positive with prob --p-pos, otherwise only negatives, plus quiet gaps).
Against source shortcuts every segment gets: density normalisation (thinned to a random
common events/frame level), a faint "bed" layer of foreign negative clips under the whole
stream, polarity swap, flip, shift/zoom, time stretch, sensor noise and hot pixels; Noah
idle clips enter the stream exactly like positives do (same scene transitions).

Target per frame (release R): 1 on [R-onset, R+post], ignored on [R-onset-dc, R-onset) and
after R+post until the clip ends, 0 elsewhere.  Readout = EMA of the output spikes
(alpha, host side); loss = weighted MSE on that readout, early frames weighted up.
PLIF leaks are capped (--tau-max) so no neuron can become a slow integrator.

Evaluation per held-out fold (groups = recording session / person / user per source):
  * 60 s negative stream per source + a mixed one + a quiet (noise only) stream ->
    false alarms per minute at every threshold, output/layer drift first vs last 10 s
  * each held-out positive placed after 3 s of held-out negatives -> detection rate,
    latency (first crossing - release), early false trigger, per threshold
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "robust"):
    sys.path.insert(0, str(p))
import speedpatch  # noqa: F401,E402
from snn_model import ARCHS, ThrowSNN, arch_from_config  # noqa: E402

HW, CELLS, PIX = 80, 2 * 80 * 80, 80 * 80
SRCS = ["noah", "mb", "dd", "dvsg", "hmdb"]   # banks that are missing are skipped
NS = len(SRCS)
THETAS = np.round(np.r_[np.arange(0.05, 0.96, 0.05), 0.97, 0.98, 0.99], 2)


# ----------------------------------------------------------------------------- bank
class Bank:
    def __init__(self, root, dev, max_frames=300):
        codes, fptr, meta, nfr = [], [], [], []
        base = 0
        for s in SRCS:
            if not (Path(root) / f"{s}.npz").exists():
                continue
            d = np.load(Path(root) / f"{s}.npz")
            m = json.loads(str(d["meta"]))
            c_all, off, nf = d["codes"], d["offsets"], d["n_frames"]
            for i in range(len(m)):
                c = c_all[off[i]:off[i + 1]]
                n = int(min(nf[i], max_frames))
                fr = c // CELLS
                c = c[fr < n]
                fptr.append(np.searchsorted(fr[fr < n], np.arange(n + 1)) + base)
                codes.append(c); nfr.append(n); meta.append(m[i]); base += len(c)
        self.meta = meta
        self.codes = torch.from_numpy(np.concatenate(codes).astype(np.int32)).to(dev)
        self.fp = torch.from_numpy(np.concatenate(fptr).astype(np.int64)).to(dev)
        self.fps = np.concatenate([[0], np.cumsum([n + 1 for n in nfr])[:-1]]).astype(np.int64)
        self.nfr = np.array(nfr)
        self.src = np.array([SRCS.index(m["src"]) for m in meta])
        self.lab = np.array([{"neg": 0, "pos": 1, "heldpos": 2}[m["label"]] for m in meta])
        self.rel = np.array([m["rel"] for m in meta])
        self.group = np.array([m["group"] for m in meta])
        self.sub = np.array([m.get("kind", m["sub"]) for m in meta])
        self.dens = np.array([max(1.0, len(c) / n) for c, n in zip(codes, nfr)])
        self.dev = dev
        print(f"bank: {len(meta)} clips, {self.codes.numel() / 1e6:.0f}M codes", flush=True)


def assign_folds(bank, k, seed):
    """Group-level folds per source.  Groups holding positives are balanced by positive count;
    negative-only Noah groups (one long idle recording) are cut into k contiguous time blocks."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(bank.meta), -1)
    for s in range(NS):
        idx = np.flatnonzero(bank.src == s)
        if len(idx) == 0:
            continue
        groups = sorted(set(bank.group[idx]))
        if s == 0:
            pos_groups = [g for g in groups if (bank.lab[idx][bank.group[idx] == g] == 1).any()]
            for g in groups:
                if g not in pos_groups:
                    gi = idx[bank.group[idx] == g]
                    order = gi[np.argsort([bank.meta[i].get("start", 0) for i in gi])]
                    fold[order] = np.arange(len(order)) * k // len(order)
            groups, weight = pos_groups, lambda g: (bank.lab[idx][bank.group[idx] == g] == 1).sum()
        else:
            weight = lambda g: (bank.group[idx] == g).sum()  # noqa: E731
        rng.shuffle(groups)
        load = np.zeros(k)
        for g in sorted(groups, key=weight, reverse=True):
            f = int(np.argmin(load))
            fold[idx[bank.group[idx] == g]] = f
            load[f] += weight(g)
    return fold


# ----------------------------------------------------------------------------- stream rendering
def render(bank, S, B, T, a, rng, train=True):
    """S: dict of per-segment numpy arrays -> dense binary input [B, T, 2, 80, 80] (float32)."""
    dev = bank.dev
    out = torch.zeros(B * T * CELLS, dtype=torch.float16, device=dev)
    n = len(S["clip"])
    if n:
        g = lambda k, dt=torch.float32: torch.as_tensor(S[k], device=dev, dtype=dt)  # noqa: E731
        f0 = g("f0", torch.int64)
        fpi = torch.as_tensor(bank.fps[S["clip"]], device=dev) + f0
        st, en = bank.fp[fpi], bank.fp[fpi + g("f1", torch.int64) - f0]
        ln = en - st
        seg = torch.repeat_interleave(torch.arange(n, device=dev), ln)
        idx = st[seg] + torch.arange(seg.numel(), device=dev) - (torch.cumsum(ln, 0) - ln)[seg]
        m = torch.rand(seg.numel(), device=dev) < g("keep")[seg]
        idx, seg = idx[m], seg[m]
        c = bank.codes[idx].long()
        f = (c // CELLS - f0[seg]).float()
        r = c % CELLS
        p, y, x = r // PIX, (r % PIX) // HW, r % HW
        dith = torch.rand_like(f) if train else 0.5
        tt = g("t0", torch.int64)[seg] + torch.floor((f + dith) * g("s")[seg]).long()
        xf = torch.where(g("flip", torch.bool)[seg], 79 - x, x).float()
        z = g("zoom")[seg]
        xi = ((xf - 39.5) * z + 39.5 + g("dx")[seg]).round().long()
        yi = ((y.float() - 39.5) * z + 39.5 + g("dy")[seg]).round().long()
        p = p ^ g("pswap", torch.int64)[seg]
        b = g("b", torch.int64)[seg]
        ok = (xi >= 0) & (xi < HW) & (yi >= 0) & (yi < HW) & (tt >= 0) & (tt < T)
        out[((((b * T + tt) * 2 + p) * HW + yi) * HW + xi)[ok]] = 1
    if train and a.noise > 0:   # sensor noise + hot pixels
        nu = rng.uniform(0, a.noise, B)
        cnt = rng.poisson(nu * T * CELLS)
        bb = np.repeat(np.arange(B), cnt)
        flat = bb * T * CELLS + rng.integers(0, T * CELLS, len(bb))
        hot = []
        for bi in range(B):
            for _ in range(rng.integers(0, a.hot + 1)):
                cell = rng.integers(0, CELLS)
                fr = np.flatnonzero(rng.random(T) < rng.uniform(0.05, 0.5))
                hot.append((bi * T + fr) * CELLS + cell)
        flat = np.concatenate([flat] + hot)
        out[torch.as_tensor(flat, device=dev)] = 1
    return out.view(B, T, 2, HW, HW).float()


def seg_params(rng, bank, clip, a, train=True, bed=False):
    """Random per-segment augmentation (identity when train=False)."""
    if not train:
        return dict(keep=1.0, s=1.0, flip=False, dx=0.0, dy=0.0, zoom=1.0, pswap=0)
    if bed:
        keep = rng.uniform(0.03, a.bed_keep) * min(1.0, 300 / bank.dens[clip])
    elif a.dnorm:
        td = math.exp(rng.uniform(math.log(a.d_lo), math.log(a.d_hi)))
        keep = float(np.clip(td / bank.dens[clip], 0.03, 1.0))
    else:
        keep = rng.uniform(a.thin_min, 1.0)
    return dict(keep=keep, s=math.exp(rng.uniform(-1, 1) * math.log(a.stretch)),
                flip=bool(rng.random() < 0.5), dx=rng.uniform(-a.shift_x, a.shift_x),
                dy=rng.uniform(-a.shift_y, a.shift_y),
                zoom=math.exp(rng.uniform(-1, 1) * math.log(a.zoom)),
                pswap=int(rng.random() < 0.5 * a.pswap))


class Planner:
    def __init__(self, bank, a, pos, neg_by_src, rng):
        self.b, self.a, self.pos, self.neg, self.rng = bank, a, pos, neg_by_src, rng
        w = np.array([getattr(a, f"w_{s}") for s in SRCS]) * np.array([len(neg_by_src[s]) > 0 for s in range(NS)])
        self.w = w / w.sum()
        self.all_neg = np.concatenate([neg_by_src[s] for s in range(NS)])
        # positives: optional per-source shares (e.g. noah:0.5,dd:0.35,hmdb:0.15), else uniform over clips
        self.pos_by_src, pw = {}, {}
        for item in filter(None, a.pos_w_src.split(",")):
            k_, v_ = item.split(":")
            pw[SRCS.index(k_)] = float(v_)
        for s in range(NS):
            ps = pos[bank.src[pos] == s]
            if len(ps) and (not pw or pw.get(s, 0) > 0):
                self.pos_by_src[s] = ps
        if pw:
            ks_ = sorted(self.pos_by_src)
            self.pos_src, self.pos_p = ks_, np.array([pw[s_] for s_ in ks_]) / sum(pw[s_] for s_ in ks_)
        else:
            self.pos_src = None

    def batch(self, B, T):
        a, rng, bank = self.a, self.rng, self.b
        S = {k: [] for k in ["clip", "f0", "f1", "t0", "b", "keep", "s", "flip", "dx", "dy", "zoom", "pswap"]}
        tgt = np.zeros((B, T), np.float32)
        w = np.ones((B, T), np.float32)
        srcmap = np.full((B, T), -1, np.int64)   # source of the main-layer segment per frame (DANN)
        on, dc, post = a.onset // 10, a.dc // 10, a.post // 10

        def add(bi, clip, f0, f1, t0, prm):
            for k, v in prm.items():
                S[k].append(v)
            S["clip"].append(clip); S["f0"].append(f0); S["f1"].append(f1); S["t0"].append(t0); S["b"].append(bi)

        for bi in range(B):
            occ = None
            if rng.random() < a.p_pos:
                if self.pos_src is not None:
                    i = int(rng.choice(self.pos_by_src[int(rng.choice(self.pos_src, p=self.pos_p))]))
                else:
                    i = int(rng.choice(self.pos))
                prm = seg_params(rng, bank, i, a)
                rel = int(bank.rel[i])
                f0 = int(rng.integers(0, max(1, rel - on - dc - 5)))
                R = int(rng.integers(int(0.4 * T), T - post - 5))
                t0 = R - int(round((rel - f0) * prm["s"]))
                t1 = t0 + int((bank.nfr[i] - f0) * prm["s"])
                add(bi, i, f0, int(bank.nfr[i]), t0, prm)
                tgt[bi, max(0, R - on):R + post + 1] = 1
                w[bi, max(0, R - on):R + post + 1] = a.pos_w * np.linspace(a.early_w, 1.0, R + post + 1 - max(0, R - on))
                w[bi, max(0, R - on - dc):max(0, R - on)] = 0
                w[bi, R + post + 1:min(T, t1 + 5)] = 0
                srcmap[bi, max(0, t0):max(0, min(T, t1))] = bank.src[i]
                occ = (t0, t1)
            t = -int(rng.integers(0, 40))
            while t < T:
                if occ and occ[0] - 8 <= t < occ[1]:
                    t = occ[1] - int(rng.integers(0, 10))
                    continue
                if rng.random() < a.p_gap:
                    t += int(rng.integers(20, a.gap_max))
                    continue
                s = rng.choice(NS, p=self.w)
                i = int(rng.choice(self.neg[s]))
                n = int(bank.nfr[i])
                L = int(rng.integers(min(40, n), n + 1))
                f0 = int(rng.integers(0, n - L + 1))
                prm = seg_params(rng, bank, i, a)
                ln = int(L * prm["s"])
                if occ and t < occ[0] and t + ln > occ[0] + 8:
                    ln = occ[0] + int(rng.integers(0, 10)) - t
                    L = max(5, int(ln / prm["s"]))
                add(bi, i, f0, f0 + L, t, prm)
                srcmap[bi, max(0, t):max(0, min(T, t + ln))] = s
                t += max(5, ln - int(rng.integers(0, 10)))
            if rng.random() < a.p_bed:   # faint foreign layer under the whole stream
                t = -int(rng.integers(0, 60))
                while t < T:
                    i = int(rng.choice(self.all_neg))
                    prm = seg_params(rng, bank, i, a, bed=True)
                    add(bi, i, 0, int(bank.nfr[i]), t, prm)
                    t += int(bank.nfr[i] * prm["s"])
        S = {k: np.array(v) for k, v in S.items()}
        x = render(bank, S, B, T, a, rng)
        d = bank.dev
        return x, torch.as_tensor(tgt, device=d), torch.as_tensor(w, device=d), torch.as_tensor(srcmap, device=d)


# ----------------------------------------------------------------------------- model
def build_model(a, dev):
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        model = ThrowSNN(ck["config"]["num_classes"], arch=arch_from_config(ck["config"])).to(dev)
        model.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith("head.")}, strict=False)
        model.reset_head(1)
    else:
        model = ThrowSNN(1, arch=ARCHS[a.arch]).to(dev)
    return model


def cap_tau(model, tau_max):
    if tau_max <= 0:
        return
    lo = -math.log(tau_max - 1.0)   # tau = 1 + exp(-logit)  ->  tau <= tau_max  <=>  logit >= lo
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "leak_logit") and isinstance(m.leak_logit, torch.nn.Parameter):
                m.leak_logit.clamp_(min=lo)


def ema_readout(spikes, alpha, L=60):
    """spikes [B, T] -> causal EMA [B, T] (what the host computes on the chip's output)."""
    k = (1 - alpha) * alpha ** torch.arange(L, device=spikes.device, dtype=spikes.dtype)
    return F.conv1d(F.pad(spikes[:, None], (L - 1, 0)), k.flip(0).view(1, 1, L))[:, 0]


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


@torch.no_grad()
def run_long(model, x, chunk=16):
    """Forward without grad; returns output spikes [B, T] and per-layer mean rates [L, B, T]."""
    rates, outs = [], []
    for k in range(0, x.shape[0], chunk):
        h, rr = x[k:k + chunk], []
        for layer in model.backbone:
            h = layer(h)
            rr.append(h.mean((2, 3, 4)))
        outs.append(model.head(h).flatten(2)[..., 0])
        rates.append(torch.stack(rr))
    return torch.cat(outs), torch.cat(rates, 1)


# ----------------------------------------------------------------------------- evaluation
def concat_plan(bank, clips, T, rng, B_index=0, start=0):
    """Deterministic abutting concatenation of whole clips (no augmentation) up to T frames."""
    S = {k: [] for k in ["clip", "f0", "f1", "t0", "b", "keep", "s", "flip", "dx", "dy", "zoom", "pswap"]}
    t, j = start, 0
    order = list(clips)
    while t < T and order:
        i = int(order[j % len(order)]); j += 1
        prm = seg_params(rng, bank, i, None, train=False)
        for k, v in prm.items():
            S[k].append(v)
        f1 = int(min(bank.nfr[i], T - t))   # never spill past T (no overlap with what follows)
        S["clip"].append(i); S["f0"].append(0); S["f1"].append(f1); S["t0"].append(t); S["b"].append(B_index)
        t += f1
    return S


def merge(Ss):
    return {k: np.concatenate([np.array(S[k]) for S in Ss]) if Ss else np.array([]) for k in Ss[0]}


def fa_stats(y, theta, refractory=100):
    """y [T] numpy -> number of alarm onsets above theta with a refractory period."""
    n, last = 0, -10 ** 9
    for t in np.flatnonzero(y >= theta):
        if t - last >= refractory:   # a sustained alarm counts once per refractory period
            n += 1
            last = t
    return n


def evaluate(model, bank, fold, k, a, train_neg_pool):
    rng = np.random.default_rng(1234 + k)
    dev = bank.dev
    model.eval()
    res = {"neg": {}, "drift": {}, "pos": []}
    T = a.eval_T
    test_neg = {s: rng.permutation(np.flatnonzero((bank.src == s) & (bank.lab == 0) & (fold == k))) for s in range(NS)}
    # --- long negative streams
    streams = {SRCS[s]: [test_neg[s]] for s in range(NS) if len(test_neg[s])}
    mixed = np.array([v for tup in zip(*[np.resize(test_neg[s], 60) for s in range(NS) if len(test_neg[s])]) for v in tup])
    streams["mixed"] = [mixed]
    for name, (clips,) in streams.items():
        S = concat_plan(bank, clips, T, rng)
        x = render(bank, {kk: np.array(v) for kk, v in S.items()}, 1, T, a, rng, train=False)
        sp, rates = run_long(model, x)
        y = ema_readout(sp, a.alpha)[0].cpu().numpy()
        mins = T / 6000
        res["neg"][name] = {"fa_per_min": [fa_stats(y, th) / mins for th in THETAS], "mean_y": float(y.mean()),
                            "y_first10s": float(y[:1000].mean()), "y_last10s": float(y[-1000:].mean())}
        del x
    # --- quiet stream: sensor noise only (does anything build up over time?)
    q = torch.zeros(1, T, 2, HW, HW, device=dev)
    g = torch.Generator(device=dev).manual_seed(7)
    q[torch.rand(q.shape, device=dev, generator=g) < 2e-4] = 1
    sp, rates = run_long(model, q)
    y = ema_readout(sp, a.alpha)[0].cpu().numpy()
    r = rates[:, 0].cpu().numpy()
    res["drift"] = {"fa_per_min": [fa_stats(y, th) / (T / 6000) for th in THETAS],
                    "out_spikes": float(sp.sum()), "y_first10s": float(y[:1000].mean()), "y_last10s": float(y[-1000:].mean()),
                    "layer_rate_first10s": r[:, :1000].mean(1).tolist(), "layer_rate_last10s": r[:, -1000:].mean(1).tolist()}
    del q
    # --- positives after 3 s of held-out negative context
    pos = np.flatnonzero(((bank.lab == 1) | (bank.lab == 2)) & (fold == k))
    ctx_pool = np.concatenate([test_neg[s] for s in range(NS)])
    Tp, ctx = 520, 300
    for c0 in range(0, len(pos), 16):
        chunk = pos[c0:c0 + 16]
        Ss = []
        for bi, i in enumerate(chunk):
            ctx_clips = rng.choice(ctx_pool, 4)
            if rng.random() < 0.5:   # half of the contexts end in our own scene (idle)
                noah = test_neg[0] if len(test_neg[0]) else ctx_pool
                ctx_clips[-1] = rng.choice(noah)
            S = concat_plan(bank, ctx_clips, ctx, rng, B_index=bi)
            # trim the context so the positive starts exactly at ctx (overlap is fine: union)
            S2 = concat_plan(bank, [i], ctx + int(bank.nfr[i]), rng, B_index=bi, start=ctx)
            S3 = concat_plan(bank, rng.choice(ctx_pool, 2), Tp, rng, B_index=bi, start=ctx + int(bank.nfr[i]))
            Ss += [S, S2, S3]
        x = render(bank, merge(Ss), len(chunk), Tp, a, rng, train=False)
        sp, _ = run_long(model, x)
        y = ema_readout(sp, a.alpha).cpu().numpy()
        for bi, i in enumerate(chunk):
            R = ctx + int(bank.rel[i])
            yy = y[bi]
            row = {"i": int(i), "src": SRCS[bank.src[i]], "sub": str(bank.sub[i]), "group": str(bank.group[i]),
                   "lab": int(bank.lab[i]), "pre_max": float(yy[:R - 80].max()),
                   "first_cross": [], "peak": float(yy[R - 80:R + 31].max())}
            for th in THETAS:
                hit = np.flatnonzero(yy[R - 80:R + 31] >= th)
                row["first_cross"].append(int(hit[0]) * 10 - 800 if len(hit) else None)
            res["pos"].append(row)
    model.train()
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default="banks")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--out", default="runs_early")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--final", action="store_true", help="train on everything, save final.pt")
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # model / optimisation
    ap.add_argument("--init", default=str(Path.home() / "snn_pretrain/runs/sweep_v1/best.pt"))
    ap.add_argument("--arch", default="v1", choices=list(ARCHS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--T", type=int, default=300, help="training stream length (frames of 10 ms)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--dann", type=float, default=0.0, help="source-adversarial weight on negative frames (0 = off)")
    ap.add_argument("--tau-max", type=float, default=8.0, help="cap PLIF time constants (frames); 0 = off")
    ap.add_argument("--alpha", type=float, default=0.85, help="EMA readout coefficient per 10 ms frame")
    # targets
    ap.add_argument("--onset", type=int, default=300, help="target on from this many ms before release")
    ap.add_argument("--dc", type=int, default=300, help="ignored ms before the onset")
    ap.add_argument("--post", type=int, default=300, help="target stays on this many ms after release")
    ap.add_argument("--pos-w", type=float, default=3.0)
    ap.add_argument("--early-w", type=float, default=2.0, help="weight multiplier at onset (linear to 1 at R+post)")
    # stream composition
    ap.add_argument("--p-pos", type=float, default=0.5)
    ap.add_argument("--dd-pos", action="store_true", help="also train on DailyDVS throw/toss of the training folds")
    ap.add_argument("--w-noah", type=float, default=0.3)
    ap.add_argument("--w-mb", type=float, default=0.25)
    ap.add_argument("--w-dd", type=float, default=0.25)
    ap.add_argument("--w-dvsg", type=float, default=0.2)
    ap.add_argument("--w-hmdb", type=float, default=0.0, help="negative share of HMDB51 (v2e) clips")
    ap.add_argument("--pos-w-src", default="", help="positive shares per source, e.g. noah:0.5,dd:0.35,hmdb:0.15")
    ap.add_argument("--pos-srcs", default="", help="only these sources' positives are trained on (default all)")
    ap.add_argument("--p-gap", type=float, default=0.15)
    ap.add_argument("--gap-max", type=int, default=150)
    ap.add_argument("--p-bed", type=float, default=0.7)
    ap.add_argument("--bed-keep", type=float, default=0.25)
    # augmentation
    ap.add_argument("--dnorm", type=int, default=1, help="density-normalising thinning")
    ap.add_argument("--d-lo", type=float, default=80)
    ap.add_argument("--d-hi", type=float, default=500)
    ap.add_argument("--thin-min", type=float, default=0.4)
    ap.add_argument("--pswap", type=float, default=1.0)
    ap.add_argument("--stretch", type=float, default=1.25)
    ap.add_argument("--zoom", type=float, default=1.25)
    ap.add_argument("--shift-x", type=float, default=20)
    ap.add_argument("--shift-y", type=float, default=6)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--hot", type=int, default=3)
    ap.add_argument("--eval-T", type=int, default=6000)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bank = Bank(a.banks, dev)
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    fold = assign_folds(bank, a.folds, a.fold_seed)
    ks = [None] if a.final else [k for k in range(a.folds) if a.only_fold < 0 or k == a.only_fold]
    for k in ks:
        tr = np.ones(len(bank.meta), bool) if k is None else fold != k
        seed = a.seed * 100 + (k or 0)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(tr & ((bank.lab == 1) | ((bank.lab == 2) & a.dd_pos)))
        if a.pos_srcs:
            pos = pos[np.isin(bank.src[pos], [SRCS.index(s_) for s_ in a.pos_srcs.split(",")])]
        neg = [np.flatnonzero(tr & (bank.lab == 0) & (bank.src == s)) for s in range(NS)]
        plan = Planner(bank, a, pos, neg, rng)
        model = build_model(a, dev)
        cap_tau(model, a.tau_max)
        # conditional DANN: on negative frames a source classifier must fail on the backbone features
        dom = torch.nn.Sequential(torch.nn.Linear(model.final_ch, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, len(SRCS))).to(dev)
        ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
        opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                                 {"params": model.head.parameters(), "lr": a.head_lr},
                                 {"params": dom.parameters(), "lr": a.head_lr}], weight_decay=a.wd)
        warm = max(1, a.steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        log, acc, t0 = [], [], time.time()
        for step in range(a.steps):
            x, tgt, w, srcmap = plan.batch(a.bs, a.T)
            h = model.backbone(x)
            sp = model.head(h).flatten(2)[..., 0]
            y = ema_readout(sp, a.alpha)
            loss = (w * (y - tgt) ** 2).sum() / w.sum()
            dacc = float("nan")
            if a.dann > 0:
                B_, T_, C_ = h.shape[:3]
                feat = ema_readout(h.mean((3, 4)).transpose(1, 2).reshape(B_ * C_, T_), a.alpha)
                feat = feat.view(B_, C_, T_).transpose(1, 2)                       # [B, T, C]
                lam = a.dann * (2 / (1 + math.exp(-10 * step / a.steps)) - 1)
                logits = dom(GradReverse.apply(feat, lam))
                m = (tgt == 0) & (w > 0) & (srcmap >= 0)
                m[:, :20] = False
                if m.any():
                    loss = loss + F.cross_entropy(logits[m], srcmap[m])
                    dacc = (logits[m].argmax(-1) == srcmap[m]).float().mean().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            cap_tau(model, a.tau_max)
            ema.update_parameters(model)
            with torch.no_grad():
                pm, nm = (tgt > 0) & (w > 0), (tgt == 0) & (w > 0)
                acc.append([loss.item(), y[pm].mean().item() if pm.any() else float("nan"),
                            y[nm].mean().item(), (y[nm] > 0.5).float().mean().item(), dacc])
            if (step + 1) % 100 == 0:
                m = np.nanmean(acc, 0)
                log.append({"step": step + 1, "loss": round(m[0], 4), "y_pos": round(m[1], 3), "y_neg": round(m[2], 4),
                            "neg_gt.5": round(m[3], 4), "dom_acc": round(m[4], 3), "min": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                acc = []
        net = ema.module
        torch.optim.swa_utils.update_bn((plan.batch(a.bs, a.T)[0] for _ in range(20)), ema)
        taus = [float(m.tau) for m in net.modules() if hasattr(m, "leak_logit")]
        cfg = {"arch": net.arch, "num_classes": 1, "classes": ["throw"], "readout": "ema", "alpha": a.alpha,
               "frame_ms": 10, "input": [2, 80, 80], "taus": taus, "args": vars(a)}
        if k is None:
            torch.save({"model": net.state_dict(), "config": cfg}, out / "final.pt")
            print(json.dumps({"final": str(out / "final.pt"), "log": log[-3:]}))
            continue
        torch.save({"model": net.state_dict(), "config": cfg}, out / f"model_f{k}.pt")
        res = evaluate(net, bank, fold, k, a, neg)
        res.update({"fold": k, "log": log, "taus": taus, "thetas": THETAS.tolist(), "args": vars(a)})
        (out / f"result_f{k}.json").write_text(json.dumps(res))
        print("fold", k, "done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train / cross-validate the binary throw-motion detector SNN (board format, riverops).

Input: binary [T, 2, 80, 80] frames of 10 ms (throwdet/prep.py table). Output: two spiking
neurons (0 = no throw motion, 1 = throw motion). Live readout: score_t = EMA(spikes_1 - spikes_0)
over ~100 ms, alarm when score_t > threshold. The network state is never reset in live use.

Per-frame targets: motion clips are 1 from `anchor - pre` to `anchor + post` frames (the swing),
other frames of motion clips are ignored; clips without throw motion are 0 from frame `warm` on.

Shortcut control:
  * balanced sampling: every (domain, target) cell gets a fixed share of each batch, so the
    domain (own camera / DailyDVS / synthetic) carries no information about the target
  * density matching: dense external clips are thinned to a random density of our camera
  * mirror, scale, shift, time shift, time stretch, cell dropout, noise
  * background compositing: our own idle clips are OR-ed into external clips (our scene/noise)

Evaluation (no model selection on held-out data, the model after the last step is used):
  * own camera, grouped K-fold by recording (the idle recording 04/211821 is split into
    contiguous time chunks); clips derived from held-out recordings are dropped from training
  * DailyDVS held-out persons (official val/test persons)
  * live stream: the held-out idle chunk of 04/211821 played as one continuous stream
    -> false alarms per minute

  python throwdet/train.py --tag base --arch v1 --init ~/throwdet/pretrained/dvsg_v1.pt
  python throwdet/train.py --tag base_final --folds 0   # train on everything, save model
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import speedpatch  # noqa: F401,E402
from model import ARCHS, ThrowSNN, arch_from_config  # noqa: E402

HW, T, CELLS = 80, 120, 2 * 80 * 80
DD_HELDOUT = {"3", "4", "5", "7", "10", "11", "16", "24", "27", "31", "33", "37", "41", "42", "43", "45"}


# ----------------------------------------------------------------------------- data
class Table:
    def __init__(self, path):
        z = np.load(path)
        self.codes, off = z["codes"], z["offsets"]
        self.start, self.end = off[:-1], off[1:]
        for k in ("y", "anchor", "kind", "domain", "group", "parent", "src", "start_s", "fname"):
            setattr(self, k, z[k])
        self.move = z["move"] if "move" in z else np.full(len(self.y), "")
        self.n = len(self.y)
        self.density = (self.end - self.start) / T

    def dense(self, idx):
        buf = np.zeros((len(idx), T * CELLS), np.uint8)
        for b, i in enumerate(idx):
            buf[b, self.codes[self.start[i]:self.end[i]].astype(np.int64)] = 1
        return torch.from_numpy(buf).view(len(idx), T, 2, HW, HW)


def own_folds(tb, k, seed=0):
    """fold id per clip for the own domain (-1 = not own / not evaluated)."""
    fold = np.full(tb.n, -1)
    if k <= 0:
        return fold
    own = tb.domain == "own"
    groups = tb.group.copy()
    # split the long idle recording into contiguous time chunks so every fold has negatives
    idle_rec = own & (np.char.find(tb.group.astype(str), "211821") >= 0)
    if idle_rec.any():
        s = tb.start_s[idle_rec].astype(float)
        edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
        groups[idle_rec] = np.array([f"idlechunk{np.searchsorted(edges, v)}" for v in s])
    rng = np.random.default_rng(seed)
    ug = sorted(set(groups[own]))
    rng.shuffle(ug)
    # greedy: biggest groups first, into the fold with the fewest clips of that group's majority target
    stats = {g: (np.sum(own & (groups == g) & (tb.y == 1)), np.sum(own & (groups == g) & (tb.y == 0))) for g in ug}
    load = np.zeros((k, 2))
    gf = {}
    for g in ug:
        if g.startswith("idlechunk"):
            gf[g] = int(g[len("idlechunk"):])
            load[gf[g]] += stats[g]
    for g in sorted([g for g in ug if g not in gf], key=lambda g: -sum(stats[g])):
        p, n = stats[g]
        cost = load + np.array([p, n])
        f = int(np.argmin(cost.max(1) + 0.01 * cost.sum(1)))
        gf[g] = f
        load[f] += (p, n)
    for i in np.flatnonzero(own):
        fold[i] = gf[groups[i]]
    return fold, groups


class Sampler:
    """Draws batches with fixed shares per (domain, target) cell.

    neg_pow > 0: inside negative cells, clips are drawn with probability ~ density**neg_pow
    (hard negatives: people walking through the view instead of an empty scene)."""

    def __init__(self, tb, train_idx, shares, rng, neg_pow=0.0):
        self.rng, self.cells = rng, []
        for (dom, y), w in shares.items():
            doms = dom.split("+")
            idx = train_idx[np.isin(tb.domain[train_idx], doms) & (tb.y[train_idx] == y)]
            if len(idx) and w > 0:
                p = None
                if y == 0 and neg_pow > 0:
                    p = (tb.density[idx] + 2.0) ** neg_pow
                    p = p / p.sum()
                self.cells.append((idx, w, p))
        tot = sum(c[1] for c in self.cells)
        self.p = np.array([c[1] / tot for c in self.cells])

    def draw(self, n):
        which = self.rng.choice(len(self.cells), n, p=self.p)
        return np.array([self.rng.choice(self.cells[c][0], p=self.cells[c][2]) for c in which])


def prefetch(fn, n=4):
    q = queue.Queue(n)

    def work():
        while True:
            q.put(fn())
    for _ in range(2):
        threading.Thread(target=work, daemon=True).start()
    while True:
        yield q.get()


# ----------------------------------------------------------------------------- augmentation
def augment(x, tb, idx, bg_pool, args, rng):
    """x: [B,T,2,H,W] float (GPU). Returns augmented x and per-sample time shift (frames)."""
    B, dev = x.shape[0], x.device
    # density matching for dense external / synthetic clips
    if args.density_match:
        dens = torch.tensor(tb.density[idx], device=dev, dtype=torch.float32)
        ext = torch.tensor(np.isin(tb.domain[idx], ["dd", "sim", "own_aug"]), device=dev)
        target = torch.exp(torch.empty(B, device=dev).uniform_(math.log(args.dens_lo), math.log(args.dens_hi)))
        keep = torch.where(ext, (target / dens.clamp(min=1)).clamp(max=1.0), torch.ones_like(dens))
        x = x * (torch.rand_like(x) < keep.view(B, 1, 1, 1, 1))
    # time stretch (speed of the movement) and shift; returns the frame map for the targets
    stretch = np.exp(rng.uniform(-args.stretch, args.stretch, B)) if args.stretch > 0 else np.ones(B)
    shift = rng.integers(-args.tshift, args.tshift + 1, B) if args.tshift > 0 else np.zeros(B, int)
    src = np.zeros((B, T), np.int64)
    for b in range(B):
        src[b] = np.round((np.arange(T) - shift[b] - 60) / stretch[b] + 60).astype(np.int64)
    valid = torch.tensor((src >= 0) & (src < T), device=dev)
    srct = torch.tensor(np.clip(src, 0, T - 1), device=dev)
    x = torch.gather(x, 1, srct.view(B, T, 1, 1, 1).expand(-1, -1, 2, HW, HW)) * valid.view(B, T, 1, 1, 1)
    # spatial: mirror, scale, shift
    flip = torch.rand(B, device=dev) < 0.5
    scale = torch.exp(torch.empty(B, device=dev).uniform_(-args.scale, args.scale))
    sh = (torch.rand(B, 2, device=dev) * 2 - 1) * torch.tensor(
        [args.shift_x if args.shift_x >= 0 else args.shift, args.shift], device=dev) / (HW / 2)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = torch.where(flip, -1.0, 1.0) / scale
    theta[:, 1, 1] = 1.0 / scale
    theta[:, :, 2] = sh
    grid = F.affine_grid(theta, (B, 2 * T, HW, HW), align_corners=False)
    x = F.grid_sample(x.flatten(1, 2), grid, mode="nearest", padding_mode="zeros",
                      align_corners=False).view(B, T, 2, HW, HW)
    if args.drop > 0:
        x = x * (torch.rand_like(x) >= torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0, args.drop))
    if args.noise > 0:
        x = torch.maximum(x, (torch.rand_like(x) < args.noise * torch.rand(B, 1, 1, 1, 1, device=dev)).float())
    # background compositing: OR an own idle clip into external clips
    if args.bg > 0 and len(bg_pool):
        ext = np.isin(tb.domain[idx], ["dd", "sim"]) & (rng.random(B) < args.bg)
        if ext.any():
            bgi = rng.choice(bg_pool, ext.sum())
            bgx = tb.dense(bgi).to(dev).float()
            x[torch.tensor(ext, device=dev)] = torch.maximum(x[torch.tensor(ext, device=dev)], bgx)
    return x, src


def targets(tb, idx, src, args):
    """Per-frame target (0/1) and mask [B, T] given the frame map src (output frame -> input frame)."""
    B = len(idx)
    tgt = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T), np.float32)
    for b, i in enumerate(idx):
        s = src[b]
        if tb.y[i] == 1:
            a = tb.anchor[i] if tb.anchor[i] >= 0 else 70
            on = (s >= a - args.pre) & (s <= a + args.post)
            tgt[b, on] = 1
            mask[b, on] = 1
        else:
            mask[b, np.arange(T) >= args.warm] = 1
    return torch.tensor(tgt), torch.tensor(mask)


# ----------------------------------------------------------------------------- model / readout
def ema(spikes, alpha):
    """Causal EMA over time of [B, T, K] -> [B, T, K] (same as the live host readout)."""
    out, s = [], torch.zeros_like(spikes[:, 0])
    for t in range(spikes.shape[1]):
        s = (1 - alpha) * s + alpha * spikes[:, t]
        out.append(s)
    return torch.stack(out, 1)


def score_of(spikes, alpha):
    r = ema(spikes.float(), alpha)
    return r[..., 1] - r[..., 0]


def build_model(args, dev):
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model = ThrowSNN(ck["config"]["num_classes"], arch=arch_from_config(ck["config"]), tau=args.tau)
        model.load_state_dict(ck["model"])
        model.reset_head(2)
    else:
        model = ThrowSNN(2, arch=ARCHS[args.arch], tau=args.tau)
    return model.to(dev)


@torch.no_grad()
def run_clips(model, tb, idx, dev, bs=24):
    model.eval()
    outs = []
    for k in range(0, len(idx), bs):
        outs.append(model(tb.dense(idx[k:k + bs]).to(dev).float()).cpu())
    model.train()
    return torch.cat(outs) if outs else torch.zeros(0, T, 2)


@torch.no_grad()
def run_stream(model, tb, idx, dev, chunk=1000, overlap=60):
    """Play clips back to back as one stream -> spikes [N*T, 2]. Processed in chunks; each chunk
    starts `overlap` frames early so the (short-memory) neuron state is warmed up as in live use."""
    model.eval()
    x = tb.dense(idx).view(len(idx) * T, 2, HW, HW)
    out = []
    for s in range(0, len(x), chunk):
        a = max(0, s - overlap)
        o = model(x[a:s + chunk][None].to(dev).float())[0].cpu()
        out.append(o[s - a:])
    model.train()
    return torch.cat(out)


def clip_scores(spikes, tb, idx, args):
    sc = score_of(spikes, args.alpha).numpy()
    res = []
    for b, i in enumerate(idx):
        if tb.y[i] == 1:
            a = tb.anchor[i] if tb.anchor[i] >= 0 else 70
            w = sc[b, max(a - args.pre, args.warm):min(a + args.post + 1, T)]
            first = np.flatnonzero(sc[b, args.warm:] > args.thr)
            lat = (first[0] + args.warm - a) * 10 if len(first) else None
        else:
            w = sc[b, args.warm:]
            lat = None
        res.append({"i": int(i), "y": int(tb.y[i]), "score": float(w.max()), "kind": str(tb.kind[i]),
                    "domain": str(tb.domain[i]), "src": str(tb.src[i]), "fname": str(tb.fname[i]),
                    "latency_ms": lat, "curve": [round(float(v), 3) for v in sc[b, ::5]]})
    return res


def metrics(res, thr):
    if not res:
        return {}
    y = np.array([r["y"] for r in res]); s = np.array([r["score"] for r in res])
    out = {"n": len(y), "n_pos": int(y.sum())}
    if 0 < y.sum() < len(y):
        pos, neg = s[y == 1], s[y == 0]
        out["auc"] = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        tpr, tnr = float((pos > thr).mean()), float((neg <= thr).mean())
        out.update(tpr=tpr, tnr=tnr, bal_acc=(tpr + tnr) / 2)
        # best achievable balanced accuracy (threshold chosen on this set; optimistic, for reference)
        ths = np.unique(s)
        out["bal_acc_best"] = float(max(((pos > t).mean() + (neg <= t).mean()) / 2 for t in ths))
    lats = [r["latency_ms"] for r in res if r["y"] == 1 and r["latency_ms"] is not None]
    if lats:
        out["latency_ms_median"] = float(np.median(lats))
    kinds = {}
    for r in res:
        kinds.setdefault(r["kind"], []).append(r["score"] > thr)
    out["detect_rate_by_kind"] = {k: round(float(np.mean(v)), 3) for k, v in kinds.items()}
    return out


def false_alarms(stream_spikes, args, thr):
    sc = score_of(stream_spikes[None], args.alpha)[0].numpy()
    above = sc > thr
    alarms, last = 0, -10**9
    for t in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
        if t - last >= 100:          # 1 s refractory, like a dodge that takes a second
            alarms += 1
            last = t
    minutes = len(sc) / 100 / 60
    return {"minutes": round(minutes, 2), "alarms": alarms, "per_min": round(alarms / max(minutes, 1e-9), 2),
            "frac_above": round(float(above.mean()), 4)}


# ----------------------------------------------------------------------------- training
def qat_config(model, weight_bits=8, potential_bits=16, potential_frac=12):
    """Same fixed-point formats as throwdet/export.py (weight frac bits from the BN-fused weights)."""
    from riverops.nn import FixedPointFormat, QuantConfig, fuse_model
    fused = fuse_model(copy.deepcopy(model.export_module()).eval().cpu())
    max_w = max((m.conv if hasattr(m, "conv") else m).weight.abs().max().item() for m in fused)
    wf = weight_bits - 1 - max(0, math.ceil(math.log2(max_w + 1e-12)))
    return QuantConfig(weight_format=FixedPointFormat(weight_bits, wf, True),
                       potential_format=FixedPointFormat(potential_bits, potential_frac, True),
                       enable_weight_quant=True, enable_potential_quant=True, qat_mode=True)


def train_one(tb, train_idx, args, dev, rng, log, model=None, steps=None, lr=None):
    model = build_model(args, dev) if model is None else model
    steps = steps or args.steps
    lr = lr or args.lr
    bb = [p for n, p in model.named_parameters() if not n.startswith("head")]
    hd = [p for n, p in model.named_parameters() if n.startswith("head")]
    opt = torch.optim.Adam([{"params": bb, "lr": lr * args.bb_lr}, {"params": hd, "lr": lr}],
                           weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * args.bb_lr, lr],
                                                total_steps=steps, pct_start=0.1)
    shares = {}
    for spec in args.shares.split(","):
        k, w = spec.split("=")
        dom, y = k.rsplit(":", 1)
        shares[(dom, int(y))] = float(w)
    sampler = Sampler(tb, train_idx, shares, rng, args.neg_pow)
    bg_pool = train_idx[(tb.domain[train_idx] == "own") & (tb.y[train_idx] == 0)]
    gen = prefetch(lambda: (lambda i: (i, tb.dense(i)))(sampler.draw(args.batch)))
    t0 = time.time()
    for step in range(1, steps + 1):
        idx, xb = next(gen)
        x, src = augment(xb.to(dev, non_blocking=True).float(), tb, idx, bg_pool, args, rng)
        tgt, mask = targets(tb, idx, src, args)
        if args.crop < T:
            c = int(rng.integers(0, T - args.crop + 1))
            x, tgt, mask = x[:, c:c + args.crop], tgt[:, c:c + args.crop], mask[:, c:c + args.crop].clone()
            mask[:, :10] = 0            # neuron state / EMA warm-up of the crop
        tgt, mask = tgt.to(dev), mask.to(dev)
        spikes = model(x)
        r = ema(spikes, args.alpha)
        logits = (r[..., 1] - r[..., 0]) * args.temp
        loss = (F.binary_cross_entropy_with_logits(logits, tgt, reduction="none") * mask).sum() / mask.sum()
        loss = loss + args.rate_reg * spikes.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            log(f"  step {step}/{steps} loss {loss.item():.4f} out-rate {spikes.mean().item():.3f} "
                f"({(time.time() - t0) / step:.2f}s/step)")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=str(Path.home() / "throwdet/data/table.npz"))
    ap.add_argument("--out", default=str(Path.home() / "throwdet/runs"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="v1", choices=list(ARCHS))
    ap.add_argument("--init", default="", help="DVS-Gesture checkpoint (arch taken from it)")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--folds", type=int, default=4, help="own-camera CV folds; 0 = train on everything")
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=80, help="random training window length (frames)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bb-lr", type=float, default=1.0, help="backbone lr factor")
    ap.add_argument("--shares", default="own:1=0.2,own:0=0.2,own_aug:1=0.05,own_aug:0=0.05,"
                                         "sim:1=0.1,sim:0=0.1,dd:1=0.15,dd:0=0.15")
    ap.add_argument("--dd-kinds", default="throw,toss,arm,idle", help="DailyDVS kinds used for training")
    ap.add_argument("--neg-pow", type=float, default=0.0, help="hard-negative sampling exponent (0 = uniform)")
    ap.add_argument("--dd-static", type=int, default=0, help="1 = only DailyDVS clips with a static camera")
    ap.add_argument("--pre", type=int, default=25, help="frames before the anchor labelled as motion")
    ap.add_argument("--post", type=int, default=30, help="frames after the anchor labelled as motion")
    ap.add_argument("--warm", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.1, help="EMA factor of the live readout (per 10 ms)")
    ap.add_argument("--temp", type=float, default=8.0)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--rate-reg", type=float, default=0.0)
    ap.add_argument("--density-match", type=int, default=1)
    ap.add_argument("--dens-lo", type=float, default=30)
    ap.add_argument("--dens-hi", type=float, default=200)
    ap.add_argument("--stretch", type=float, default=0.2)
    ap.add_argument("--tshift", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--shift", type=float, default=8)
    ap.add_argument("--shift-x", type=float, default=-1, help="horizontal shift (px of 80); -1 = same as --shift")
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.002)
    ap.add_argument("--bg", type=float, default=0.5, help="probability of OR-ing own idle background into external clips")
    ap.add_argument("--qat-steps", type=int, default=0, help="quantization-aware fine-tuning steps after training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda")
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    log(f"== {args.tag} {json.dumps(vars(args))}")
    tb = Table(args.table)
    dd_kinds = set(args.dd_kinds.split(","))
    usable = ~((tb.domain == "dd") & ~np.isin(tb.kind, list(dd_kinds)))
    person = np.array([g.split(":", 1)[1] if g.startswith("dd:") else "" for g in tb.group.astype(str)])
    dd_test = np.flatnonzero((tb.domain == "dd") & np.isin(person, list(DD_HELDOUT)))
    dd_train_mask = (tb.domain == "dd") & ~np.isin(person, list(DD_HELDOUT)) & usable
    if args.dd_static:
        dd_train_mask &= tb.move != "1"

    if args.folds > 0:
        fold, groups = own_folds(tb, args.folds, 0)   # fixed split for all runs
    else:
        fold, groups = np.full(tb.n, -1), tb.group
    results = {"args": vars(args), "folds": {}}
    all_res, all_streams = [], []
    folds = range(args.folds) if args.folds > 0 else [None]
    for k in folds:
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        rng = np.random.default_rng(args.seed * 100 + (k or 0))
        torch.manual_seed(args.seed * 100 + (k or 0))
        if k is None:
            held_groups = set()
            test_own = np.array([], int)
        else:
            held_groups = set(groups[fold == k])
            test_own = np.flatnonzero(fold == k)
        own_train = (tb.domain == "own") & (fold != k) if k is not None else (tb.domain == "own")
        aug_train = (tb.domain == "own_aug") & ~np.isin(tb.group, list(set(tb.group[fold == k]) if k is not None else []))
        train_idx = np.flatnonzero(own_train | aug_train | (tb.domain == "sim") | dd_train_mask)
        log(f"fold {k}: train {len(train_idx)} clips, held-out own {len(test_own)} "
            f"(pos {int(tb.y[test_own].sum())}), dd test {len(dd_test)}")
        model = train_one(tb, train_idx, args, dev, rng, log)
        if args.qat_steps > 0:
            from riverops.nn import prepare_qat
            qcfg = qat_config(model)
            log(f"  QAT: weights {qcfg.weight_format}, potentials {qcfg.potential_format}")
            model = prepare_qat(model, qcfg, fused_bn=True).to(dev).train()
            model = train_one(tb, train_idx, args, dev, rng, log, model=model, steps=args.qat_steps,
                              lr=args.lr * 0.05)
        name = "final" if k is None else f"fold{k}"
        torch.save({"model": model.state_dict(), "arch": model.arch, "args": vars(args),
                    "readout": {"alpha": args.alpha, "thr": args.thr, "classes": ["none", "throw_motion"]}},
                   out / f"{name}.pt")
        fr = {}
        if len(test_own):
            res = clip_scores(run_clips(model, tb, test_own, dev), tb, test_own, args)
            all_res += res
            fr["own"] = metrics(res, args.thr)
            # live stream: held-out chunk of the idle recording, in time order
            st = test_own[(np.char.find(tb.group[test_own].astype(str), "211821") >= 0)]
            if len(st):
                st = st[np.argsort(tb.start_s[st].astype(float))]
                sp = run_stream(model, tb, st, dev)
                all_streams.append(sp)
                fr["idle_stream"] = false_alarms(sp, args, args.thr)
        if k in (None, 0) or args.only_fold >= 0:
            dres = clip_scores(run_clips(model, tb, dd_test, dev), tb, dd_test, args)
            fr["dd_test"] = metrics(dres, args.thr)
            results.setdefault("dd_res", []).extend([{kk: v for kk, v in r.items() if kk != "curve"} for r in dres])
        log(f"fold {k}: {json.dumps(fr)}")
        results["folds"][name] = fr
        del model
        torch.cuda.empty_cache()
    if all_res:
        results["own_oof"] = metrics(all_res, args.thr)
        results["own_res"] = all_res
        thr_grid = {}
        for thr in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            m = metrics(all_res, thr)
            fa = [false_alarms(s, args, thr)["alarms"] for s in all_streams]
            mins = sum(len(s) for s in all_streams) / 6000
            thr_grid[thr] = {"tpr": m.get("tpr"), "tnr": m.get("tnr"), "bal_acc": m.get("bal_acc"),
                             "stream_fa_per_min": round(sum(fa) / max(mins, 1e-9), 2) if all_streams else None}
        results["own_thr_grid"] = thr_grid
        log(f"OWN out-of-fold: {json.dumps(results['own_oof'])}")
        log(f"threshold grid: {json.dumps(thr_grid)}")
    json.dump(results, open(out / "result.json", "w"),
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()

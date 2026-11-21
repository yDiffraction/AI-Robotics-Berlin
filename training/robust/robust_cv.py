#!/usr/bin/env python3
"""Grouped CV of throw-vs-no-throw SNNs (no throw = feint + idle) on very few real recordings, with data generated
FROM the real recordings (no procedural synthesis) and shortcut-aware training.

Why: with 24 recordings, throws and feints were recorded in separate blocks and the auto
label uses the post/pre activity ratio, so the *pre-release* activity level separates the
classes (~75 % from the first 10 ms frame). A model that fits that cue looks fine in CV
and fails on new people. Everything here is about making the post-release motion the only
usable signal and measuring whether the model uses it.

Windows are cut on the fly (GPU) from full recordings (prep_raw.py) around the swing peak:
frames [peak - pre, peak + post) of 10 ms, binary 2x80x80 after augmentation.

Generation (all from real recordings, --gen comma list):
  thin     random global event thinning (keep 30-100 %): activity level stops being a label
  swap     context swap: frames before a cut 120-450 ms before release come from ANOTHER
           recording (either class), spatially aligned, so the prefix carries no label
  overlay  OR-in a thinned, shifted idle chunk (distractor / level noise; never throw/feint
           material, which could carry a visible ball into the other class)
  mix      EventMix-style space-time cutmix between samples with mixed soft labels
  extra    offline windows from prep_raw --extra (e.g. aug_train) at --extra-frac per batch
always: release jitter, speed 0.85-1.15, mirror, scale, shift, cell dropout, noise cells.

Losses (--loss):
  rate     MSE(output rate, one-hot)                         (previous pipeline)
  ts       CE on the running spike rate at every frame after release+--post-from; idle at
           every frame after --warmup
  --sym L  prefix symmetry: for throw/feint clips the throw and feint running logits must be
           equal before release (the decision is not observable yet) -> stops early fitting

Classes: 2 outputs, throw / no throw. The bank keeps sub-types (throw, feint, idle): batches draw
--p-throw throws, the rest feints (--feint-share) and idle; metrics are reported per sub-type.

Metrics (pooled over held-out recordings, never used for selection):
  bal_acc    mean of throw recall and no-throw specificity (feint and idle weighted equally)
  recall_throw, fp_feint, fp_idle   throw recall and false-trigger rates at window end
  acc_tf     throw vs feint only, balanced (the hard part)
  acc_tf@+200ms / acc_tf@+0ms  running 2-way accuracy after release
  prefix_tf@-600/-400/-200ms   2-way accuracy from the prefix only (shortcut use; ideal 50 %)
  swap_tf    2-way accuracy when the clip's first part (up to -200 ms) is replaced by an
             opposite-class training recording (follows the motion? ideal = acc_tf)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import speedpatch  # noqa: F401
from snn_model import ARCHS, ThrowSNN, arch_from_config

K = 2          # outputs: 0 = throw, 1 = no throw (feint + idle merged)
SUB = 3        # bank sub-types: 0 throw, 1 feint, 2 idle (kept for sampling and metrics)
HW = 80


# ----------------------------------------------------------------------------- data
class Bank:
    def __init__(self, path, dev):
        d = torch.load(path)
        self.recs = d["recs"]
        L = max(len(r["counts"]) for r in self.recs)
        self.counts = torch.zeros(len(self.recs), L, 2, HW, HW, dtype=torch.uint8)
        for i, r in enumerate(self.recs):
            self.counts[i, :len(r["counts"])] = r["counts"]
        self.counts = self.counts.to(dev)
        self.len = np.array([len(r["counts"]) for r in self.recs])
        self.label = np.array([r["label"] for r in self.recs])
        self.peak = np.array([r["peak"] for r in self.recs])
        self.windup = np.array([r["windup"] for r in self.recs])
        self.names = [r["name"] for r in self.recs]
        self.clip = bool(self.recs[0].get("clip", False))
        self.group = np.array([r.get("group", r["name"]) for r in self.recs])
        # quiet stretches for idle: before the wind-up and after the follow-through
        self.idle = []
        for i in range(len(self.recs)):
            if self.clip:                       # clip dataset: idle clips are whole idle windows
                if self.label[i] == 2:
                    self.idle.append((i, 0, int(self.len[i])))
                continue
            for a, b in ((0, self.windup[i] - 10), (self.peak[i] + 45, self.len[i])):
                if b - a >= 30:
                    self.idle.append((i, int(a), int(b)))
        ex = d.get("extra") or []
        self.extra = None
        if ex:
            self.extra = torch.stack([e["counts"] for e in ex]).to(dev)
            self.ex_label = np.array([e["label"] for e in ex])
            self.ex_rel = np.array([e["release"] for e in ex])
            self.ex_parent = [e["parent"] for e in ex]
            self.ex_donor = [e["donor"] for e in ex]
        self.dev = dev

    def gather(self, rec, idx):
        """rec [B], idx [B,T] frame indices (out of range -> empty frame) -> float counts [B,T,2,H,W]."""
        rec = torch.as_tensor(rec, device=self.dev)
        idx = torch.as_tensor(idx, device=self.dev)
        lens = torch.as_tensor(self.len, device=self.dev)[rec]
        ok = (idx >= 0) & (idx < lens[:, None])
        x = self.counts[rec[:, None], idx.clamp(0, self.counts.shape[1] - 1)].float()
        return x * ok[..., None, None, None]

    def event_idx(self, i, T, pre, speed=1.0, jitter=0):
        """Frame indices of a window around rec i's peak; jitter is clamped to stay inside."""
        rel = (np.arange(T) - pre) * speed
        lo, hi = -(self.peak[i] + rel[0]), self.len[i] - 1 - (self.peak[i] + rel[-1])
        if self.clip:                           # clips have no context: shift with empty frames
            return np.round(self.peak[i] + rel + int(jitter)).astype(np.int64)
        j = int(np.clip(jitter, min(lo, hi), max(lo, hi))) if lo <= hi else int(jitter)
        return np.round(self.peak[i] + rel + j).astype(np.int64)

    def idle_idx(self, k, T, rng):
        i, a, b = self.idle[k]
        n = b - a
        p = (rng.integers(0, 2 * n) + np.arange(T)) % (2 * n)          # ping-pong over the stretch
        return i, a + np.where(p < n, p, 2 * n - 1 - p)


def centroid(x):
    """x [B,T,2,H,W] -> activity centroid [B,2] (y, x) in pixels."""
    m = x.sum((1, 2))                                                  # [B,H,W]
    w = m.sum((1, 2)).clamp(min=1e-6)
    ar = torch.arange(HW, device=x.device, dtype=x.dtype)
    return torch.stack([(m.sum(2) * ar).sum(1) / w, (m.sum(1) * ar).sum(1) / w], 1)


def thin_binary(c, keep):
    """counts -> binary with each event kept with prob keep [B] (cell on if any event survives)."""
    p = 1 - (1 - keep.view(-1, 1, 1, 1, 1)) ** c
    return (torch.rand_like(c) < p).float()


def geometric(x, a):
    B, T = x.shape[:2]
    dev = x.device
    flip = torch.rand(B, device=dev) < a.flip
    scale = torch.empty(B, device=dev).uniform_(1 - a.scale, 1 + a.scale)
    sx = a.shift if a.shift_x < 0 else a.shift_x      # anisotropic: wide x, small y keeps a ball near the top edge
    sy = a.shift if a.shift_y < 0 else a.shift_y
    shift = (torch.rand(B, 2, device=dev) * 2 - 1) * (torch.tensor([sx, sy], device=dev) / (HW / 2))
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = torch.where(flip, -1.0, 1.0) / scale
    theta[:, 1, 1] = 1.0 / scale
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, (B, 2 * T, HW, HW), align_corners=False)
    x = F.grid_sample(x.flatten(1, 2), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x.view(B, T, 2, HW, HW)


class Sampler:
    def __init__(self, bank, train_recs, a, rng):
        self.b, self.a, self.rng = bank, a, rng
        self.tf = [i for i in train_recs if bank.label[i] < 2]
        self.by_class = {c: [i for i in self.tf if bank.label[i] == c] for c in (0, 1)}
        tr = set(train_recs)
        self.idle = [k for k, (i, _, _) in enumerate(bank.idle) if i in tr]
        self.gen = set(a.gen.split(",")) if a.gen else set()
        self.extra = []
        if bank.extra is not None and "extra" in self.gen:
            names = {bank.names[i] for i in train_recs}
            held = set(bank.names) - names
            self.extra = [k for k in range(len(bank.ex_label))
                          if bank.ex_parent[k] not in held and bank.ex_donor[k] not in held]

    def window(self, i, speed, jitter):
        return self.b.event_idx(i, self.a.T, self.a.pre, speed, jitter)

    def batch(self):
        a, b, rng = self.a, self.b, self.rng
        B, T = a.bs, a.T
        # sub-types: throw with p_throw, else feint / idle (feints are the hard negatives)
        u = rng.random(B)
        ys = np.where(u < a.p_throw, 0, np.where((rng.random(B) < a.feint_share) | (not self.idle), 1, 2))
        if not self.idle:
            ys = np.where(ys == 2, 1, ys)
        recs, idxs, rels = [], [], []
        for y in ys:
            speed = rng.uniform(1 - a.speed, 1 + a.speed)
            if y < 2:
                i = int(rng.choice(self.by_class[y]))
                recs.append(i); idxs.append(self.window(i, speed, rng.integers(-a.tjit, a.tjit + 1)))
                rels.append(a.pre)
            else:
                i, ix = b.idle_idx(int(rng.choice(self.idle)), T, rng)
                recs.append(i); idxs.append(ix); rels.append(-1)
        x = b.gather(np.array(recs), np.stack(idxs))
        y = torch.as_tensor(ys, device=b.dev)
        rel = torch.as_tensor(rels, device=b.dev)

        if self.extra and a.extra_frac > 0:
            n = int(round(B * a.extra_frac))
            ks = rng.choice(self.extra, n)
            sh = rng.integers(-a.tjit, a.tjit + 1, n)
            ix = torch.as_tensor(np.arange(T)[None] - a.pre + 70 + sh[:, None], device=b.dev)
            ok = (ix >= 0) & (ix < 120)
            xe = b.extra[torch.as_tensor(ks, device=b.dev)[:, None], ix.clamp(0, 119)].float()
            x[:n] = xe * ok[..., None, None, None]
            y[:n] = torch.as_tensor(b.ex_label[ks], device=b.dev)
            rel[:n] = torch.as_tensor(np.where(b.ex_label[ks] < 2, a.pre, -1), device=b.dev)

        if "swap" in self.gen:
            for j in range(B):
                if y[j] < 2 and rel[j] >= 0 and rng.random() < a.p_swap:
                    d = int(rng.choice(self.tf))
                    c = a.pre - int(rng.integers(a.swap_min, a.swap_max + 1))
                    xd = b.gather(np.array([d]), self.window(d, 1.0, 0)[None])
                    x[j] = splice(x[j:j + 1], xd, c)[0]

        if "overlay" in self.gen:
            m = torch.rand(B, device=b.dev) < a.p_overlay
            if m.any():
                n = int(m.sum())
                # idle material only: a throw chunk could paste a ball streak into a feint
                pairs = [b.idle_idx(int(rng.choice(self.idle)), T, rng) for _ in range(n)]
                xo = b.gather(np.array([p[0] for p in pairs]), np.stack([p[1] for p in pairs]))
                xo = thin_binary(xo, torch.empty(n, device=b.dev).uniform_(0.05, 0.3))
                xo = torch.roll(xo, (int(rng.integers(-20, 21)), int(rng.integers(-20, 21))), (3, 4))
                x[m] = x[m] + xo * 1.0

        keep = torch.ones(B, device=b.dev)
        if "thin" in self.gen:
            keep = torch.empty(B, device=b.dev).uniform_(a.thin_min, 1.0)
        x = thin_binary(x, keep * (1 - a.drop))
        x = geometric(x, a)
        x = torch.maximum(x, (torch.rand_like(x) < a.noise).float())
        target = F.one_hot((y != 0).long(), K).float()          # throw vs no throw

        if "mix" in self.gen and rng.random() < a.p_mix:
            perm = torch.randperm(B, device=b.dev)
            lam = rng.uniform(0.3, 0.7)
            h = int(HW * np.sqrt(lam)); y0, x0 = rng.integers(0, HW - h + 1, 2)
            box = torch.zeros(1, 1, 1, HW, HW, device=b.dev)
            box[..., y0:y0 + h, x0:x0 + h] = 1
            mixed = x * (1 - box) + x[perm] * box
            na = (x * (1 - box)).sum((1, 2, 3, 4)); nb = (x[perm] * box).sum((1, 2, 3, 4))
            wb = (nb / (na + nb).clamp(min=1)).view(-1, 1)
            target = target * (1 - wb) + target[perm] * wb
            rel = torch.where(rel >= 0, rel, rel[perm])
            x = mixed
        return x, target, rel


def splice(x, xd, c, fade=2):
    """Frames < c from donor xd (shifted onto x's body position), >= c from x; union across the seam."""
    cy = centroid(x[:, c:c + 15]); cd = centroid(xd[:, max(0, c - 15):c])
    dy, dx = (cy - cd).round().int()[0].tolist()
    xd = torch.roll(xd, (dy, dx), (3, 4))
    out = x.clone()
    out[:, :c] = xd[:, :c]
    lo, hi = max(0, c - fade), min(x.shape[1], c + fade)
    out[:, lo:hi] = torch.maximum(x[:, lo:hi], xd[:, lo:hi])
    return out


# ----------------------------------------------------------------------------- model / loss
def build_model(a, dev):
    hw = HW // a.pool
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        model = ThrowSNN(ck["config"]["num_classes"], arch=arch_from_config(ck["config"]), hw=hw).to(dev)
        sd = {k: v for k, v in ck["model"].items() if not k.startswith("head.")}
        model.load_state_dict(sd, strict=False)
        model.reset_head(K)
    else:
        model = ThrowSNN(K, arch=ARCHS[a.arch], hw=hw).to(dev)
    for layer in list(model.backbone)[:a.freeze]:
        for p in layer.parameters():
            p.requires_grad_(False)
    return model


def prep_input(x, a):
    return F.max_pool3d(x.flatten(1, 2), (1, a.pool, a.pool)).view(x.shape[0], x.shape[1], 2, HW // a.pool, HW // a.pool) \
        if a.pool > 1 else x


def loss_fn(spikes, target, rel, a):
    B, T, _ = spikes.shape
    if a.loss == "rate":
        loss = F.mse_loss(spikes.mean(1), target)
    else:
        t = torch.arange(T, device=spikes.device)
        logits = spikes.cumsum(1) / (t + 1).view(1, -1, 1) * a.temp           # [B,T,K]
        start = torch.where(rel >= 0, rel + a.post_from, torch.full_like(rel, a.warmup))
        w = (t[None] >= start[:, None]).float()                               # [B,T]
        ce = -(target[:, None] * F.log_softmax(logits, -1)).sum(-1)
        loss = (ce * w).sum() / w.sum().clamp(min=1)
    if a.sym > 0:
        t = torch.arange(T, device=spikes.device)
        logits = spikes.cumsum(1) / (t + 1).view(1, -1, 1) * a.temp
        lo = torch.clamp(rel[:, None] - a.sym_win, min=a.warmup) if a.sym_win > 0 else a.warmup
        m = ((t[None] >= lo) & (t[None] < rel[:, None]) & (rel[:, None] >= 0)).float()
        d = (logits[..., 0] - logits[..., 1]) ** 2
        loss = loss + a.sym * (d * m).sum() / m.sum().clamp(min=1)
    return loss


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, bank, test_recs, train_recs, a):
    model.eval()
    T, pre = a.T, a.pre
    out = []
    tf = [i for i in test_recs if bank.label[i] < 2]

    def run(x):
        res = []
        for k in range(0, len(x), 16):
            res.append(model(prep_input(x[k:k + 16], a)).float().cumsum(1).cpu())
        return torch.cat(res)

    if tf:
        x = (bank.gather(np.array(tf), np.stack([bank.event_idx(i, T, pre) for i in tf])) > 0).float()
        cum = run(x)
        # swap: prefix (up to -200 ms) from opposite-class training recordings
        donors = {c: [i for i in train_recs if bank.label[i] == c] for c in (0, 1)}
        sw_x, sw_owner = [], []
        rng = np.random.default_rng(123)
        for j, i in enumerate(tf):
            for d in rng.choice(donors[1 - bank.label[i]], min(a.swap_eval, len(donors[1 - bank.label[i]])), replace=False):
                xd = (bank.gather(np.array([d]), bank.event_idx(d, T, pre)[None]) > 0).float()
                sw_x.append(splice(x[j:j + 1], xd, pre - 20)); sw_owner.append(j)
        sw = run(torch.cat(sw_x)) if sw_x else None
        # stress tests: unseen-condition perturbations of the held-out clips (end-of-window counts)
        g = torch.Generator(device=x.device).manual_seed(0)
        stress = {
            "thin50": x * (torch.rand(x.shape, generator=g, device=x.device) < 0.5),
            "noise": torch.maximum(x, (torch.rand(x.shape, generator=g, device=x.device) < 2e-3).float()),
            "shift10": torch.roll(x, 10, 4),
            "t+100ms": torch.cat([torch.zeros_like(x[:, :10]), x[:, :-10]], 1),
            "t-100ms": torch.cat([x[:, 10:], torch.zeros_like(x[:, :10])], 1),
            "mirror": torch.flip(x, [4]),
        }
        st = {k: run(v)[:, -1] for k, v in stress.items()}
        for j, i in enumerate(tf):
            row = {"rec": bank.names[i], "group": str(bank.group[i]), "y": int(bank.label[i]), "end": cum[j, -1].tolist()}
            for ms in (-600, -400, -200, 0, 100, 200):
                f = min(T - 1, pre + ms // 10)
                row[f"t{ms:+d}"] = cum[j, f].tolist()
            row["stress"] = {k: v[j].tolist() for k, v in st.items()}
            row["swap_end"] = [sw[k, -1].tolist() for k, o in enumerate(sw_owner) if o == j] if sw is not None else []
            out.append(row)
    idle = [k for k, (i, _, _) in enumerate(bank.idle) if i in set(test_recs)]
    if idle:
        rng = np.random.default_rng(7)
        pairs = [(bank.idle[k][0], np.arange(T)) if bank.idle[k][2] - bank.idle[k][1] == T
                 else bank.idle_idx(k, T, rng) for k in idle]
        x = (bank.gather(np.array([p[0] for p in pairs]), np.stack([p[1] for p in pairs])) > 0).float()
        cum = run(x)
        for j, k in enumerate(idle):
            out.append({"rec": bank.names[bank.idle[k][0]], "group": str(bank.group[bank.idle[k][0]]), "y": 2,
                        "end": cum[j, -1].tolist()})
    model.train()
    return out


def summarize(rows):
    """Rows carry the sub-type y (0 throw, 1 feint, 2 idle) and output counts [throw, no_throw].
    A clip is called 'throw' when throw > no_throw; ties count half."""
    def hit(v, y):         # correct binary decision; tie = 0.5
        return 0.5 if v[0] == v[1] else float((v[0] > v[1]) == (y == 0))

    def trig(v):           # predicted throw
        return 0.5 if v[0] == v[1] else float(v[0] > v[1])
    s = {"n_throw": sum(r["y"] == 0 for r in rows), "n_feint": sum(r["y"] == 1 for r in rows),
         "n_idle": sum(r["y"] == 2 for r in rows)}

    def bal_tf(pairs):     # balanced over throw / feint
        per = [np.mean([v for v, y in pairs if y == c]) for c in (0, 1) if any(y == c for _, y in pairs)]
        return float(np.mean(per)) if per else float("nan")
    tf = [r for r in rows if r["y"] < 2]
    th = [r for r in rows if r["y"] == 0]
    fe = [r for r in rows if r["y"] == 1]
    idl = [r for r in rows if r["y"] == 2]
    s["recall_throw"] = float(np.mean([trig(r["end"]) for r in th])) if th else float("nan")
    s["fp_feint"] = float(np.mean([trig(r["end"]) for r in fe])) if fe else float("nan")
    s["fp_idle"] = float(np.mean([trig(r["end"]) for r in idl])) if idl else float("nan")
    neg = [1 - v for v in (s["fp_feint"], s["fp_idle"]) if v == v]
    s["bal_acc"] = float(np.mean([s["recall_throw"], np.mean(neg)])) if neg and th else float("nan")
    if tf:
        s["acc_tf"] = bal_tf([(hit(r["end"], r["y"]), r["y"]) for r in tf])
        for ms in (-600, -400, -200):
            s[f"prefix_tf@{ms}ms"] = bal_tf([(hit(r[f"t{ms:+d}"], r["y"]), r["y"]) for r in tf])
        for ms in (0, 100, 200):
            s[f"acc_tf@+{ms}ms"] = bal_tf([(hit(r[f"t{ms:+d}"], r["y"]), r["y"]) for r in tf])
        sw = [(hit(v, r["y"]), r["y"]) for r in tf for v in r["swap_end"]]
        s["swap_tf"] = bal_tf(sw) if sw else float("nan")
        for k in (tf[0].get("stress") or {}):
            s[f"stress_{k}"] = bal_tf([(hit(r["stress"][k], r["y"]), r["y"]) for r in tf])
    return {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in s.items()}


def folds_of(bank, n, seed, group="rec"):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(bank.recs), int)
    if group == "session":   # whole recording sessions per fold, balanced by throw/feint clip count
        gs = sorted(set(bank.group), key=lambda g: (-(bank.label[bank.group == g] < 2).sum(), g))
        load = np.zeros(n)
        start = np.array([r.get("start", 0.0) for r in bank.recs])
        for g in [g for g in gs if (bank.label[bank.group == g] == 2).all()]:
            # idle-only recording (all idle from one take): contiguous time blocks, one per fold,
            # so every training set has idle and every test fold has unseen idle
            idx = np.flatnonzero(bank.group == g)
            for k, part in enumerate(np.array_split(idx[np.argsort(start[idx])], n)):
                fold[part] = k
        gs = [g for g in gs if not (bank.label[bank.group == g] == 2).all()]
        for g in gs:
            k = int(np.argmin(load)); fold[bank.group == g] = k; load[k] += (bank.label[bank.group == g] < 2).sum()
        return fold
    if bank.clip:            # clip-level stratified folds (optimistic: same sessions in train and test)
        for c in range(SUB):
            for j, i in enumerate(rng.permutation(np.flatnonzero(bank.label == c))):
                fold[i] = j % n
        return fold
    k = 0
    for c in (0, 1):
        for i in rng.permutation(np.flatnonzero(bank.label == c)):
            fold[i] = k % n; k += 1
    return fold


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="runs_robust")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--cv-seeds", default="0,1", help="fold assignments (each = full CV)")
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--group", default="rec", choices=["rec", "session"], help="CV grouping (clip banks)")
    ap.add_argument("--final", action="store_true", help="train on all recordings, save model, no eval")
    # input
    ap.add_argument("--pre", type=int, default=70, help="frames before release in the window")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--pool", type=int, default=1, help="2 -> 40x40 input (OR-pooling)")
    # model
    ap.add_argument("--init", default="")
    ap.add_argument("--arch", default="v1", choices=list(ARCHS))
    ap.add_argument("--freeze", type=int, default=0)
    # training
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--loss", default="ts", choices=["rate", "ts"])
    ap.add_argument("--temp", type=float, default=10.0)
    ap.add_argument("--post-from", type=int, default=5, help="ts loss starts this many frames after release")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sym", type=float, default=0.0, help="prefix symmetry weight")
    ap.add_argument("--sym-win", type=int, default=0, help="symmetry only this many frames before release (0 = from warmup)")
    ap.add_argument("--p-throw", type=float, default=0.5, help="share of throw clips per batch")
    ap.add_argument("--feint-share", type=float, default=0.5, help="share of feints among the no-throw clips")
    ap.add_argument("--label-smooth", type=float, default=0.0)
    # augmentation / generation
    ap.add_argument("--gen", default="")
    ap.add_argument("--tjit", type=int, default=8)
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--flip", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=8)
    ap.add_argument("--shift-x", type=float, default=-1, help="override --shift for x (pixels)")
    ap.add_argument("--shift-y", type=float, default=-1, help="override --shift for y (pixels)")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--thin-min", type=float, default=0.3)
    ap.add_argument("--p-swap", type=float, default=0.5)
    ap.add_argument("--swap-min", type=int, default=12)
    ap.add_argument("--swap-max", type=int, default=45)
    ap.add_argument("--p-overlay", type=float, default=0.3)
    ap.add_argument("--p-mix", type=float, default=0.3)
    ap.add_argument("--extra-frac", type=float, default=0.25)
    ap.add_argument("--swap-eval", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    bank = Bank(a.bank, dev)
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    tf_recs = np.arange(len(bank.recs)) if bank.clip else np.flatnonzero(bank.label < 2)
    jobs = []
    if a.final:
        jobs.append((None, None, list(tf_recs), []))
    else:
        for cs in [int(s) for s in a.cv_seeds.split(",")]:
            fold = folds_of(bank, a.folds, cs, a.group)
            for k in range(a.folds):
                if a.only_fold >= 0 and k != a.only_fold:
                    continue
                test = [i for i in tf_recs if fold[i] == k]
                jobs.append((cs, k, [i for i in tf_recs if fold[i] != k], test))

    rows_all, t_start = [], time.time()
    for cs, k, train, test in jobs:
        seed = a.seed * 1000 + (cs or 0) * 100 + (k or 0)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = build_model(a, dev)
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else model
        head = [p for p in model.head.parameters()]
        body = [p for p in model.backbone.parameters() if p.requires_grad]
        lr_body = a.lr if a.init else 2 * a.lr
        opt = torch.optim.AdamW([{"params": body, "lr": lr_body}, {"params": head, "lr": a.head_lr}], weight_decay=a.wd)
        warm = max(1, a.steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
        smp = Sampler(bank, train, a, rng)
        log = []
        run_l, run_a = [], []
        for step in range(a.steps):
            x, target, rel = smp.batch()
            if a.label_smooth > 0:
                target = target * (1 - a.label_smooth) + a.label_smooth / K
            sp = model(prep_input(x, a))
            loss = loss_fn(sp, target, rel, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if a.ema > 0:
                ema.update_parameters(model)
            run_l.append(loss.item()); run_a.append((sp.sum(1).argmax(1) == target.argmax(1)).float().mean().item())
            if (step + 1) % 100 == 0:
                log.append({"step": step + 1, "loss": round(float(np.mean(run_l)), 4), "train_acc": round(float(np.mean(run_a)), 3)})
                run_l, run_a = [], []
        net = ema.module if a.ema > 0 else model
        if a.ema > 0:   # BN running stats of the EMA copy: refresh on training batches
            torch.optim.swa_utils.update_bn(((prep_input(smp.batch()[0], a),) for _ in range(20)), ema)
        if a.final:
            cfg = {"num_classes": K, "classes": ["throw", "no_throw"], "arch": [list(x) for x in net.arch],
                   "T": a.T, "pre": a.pre, "frame_ms": 10, "hw": HW // a.pool, "args": vars(a)}
            torch.save({"model": net.state_dict(), "config": cfg}, out / "final.pt")
            print(json.dumps({"final": str(out / "final.pt"), "log": log[-3:]}), flush=True)
            continue
        rows = evaluate(net, bank, test, train, a)
        for r in rows:
            r.update(cv_seed=cs, fold=k)
        rows_all += rows
        print(json.dumps({"cv_seed": cs, "fold": k, "train_last": log[-1] if log else {}, "test": summarize(rows),
                          "min": round((time.time() - t_start) / 60, 1)}), flush=True)
    if not a.final:
        res = {"tag": a.tag, "args": vars(a), "summary": summarize(rows_all),
               "per_cv_seed": {str(cs): summarize([r for r in rows_all if r["cv_seed"] == cs])
                               for cs in sorted({r["cv_seed"] for r in rows_all})},
               "rows": rows_all, "minutes": round((time.time() - t_start) / 60, 1)}
        name = "result.json" if a.only_fold < 0 else f"result_f{a.only_fold}.json"
        (out / name).write_text(json.dumps(res, indent=1))
        print("RESULT " + json.dumps({"tag": a.tag, **res["summary"], "min": res["minutes"]}), flush=True)


if __name__ == "__main__":
    main()

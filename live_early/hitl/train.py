#!/usr/bin/env python3
"""Fine-tune the board model on human button presses (sessions from collect.py), then export a board folder.

The board only runs inference, so learning happens here on a float replica of the network (BPTT, ATan surrogate),
starting from v3 (training/early_ctx ctx2.py recipe) and keeping what made v3 precise:
  * targets as ctx2: y = 1 from press - pre (500 ms) to press + post (300 ms), ignored dc before / post_ign after,
    earlier frames weighted up; a press is treated like a release time in labels.json
  * hard negatives: windows around the board's own alarms that no press claimed (= its false alarms while recording)
    and around idle/invalid events of the early_ctx recordings, weighted hn_w
  * tonic-drive penalty, tau cap 8, EMA of the weights, ctx2 augmentation (flip, shift, thinning, noise)
  * BatchNorm statistics stay frozen (the export folds them; they are v3's clean-window statistics)
  * optional replay of the early_ctx whole recordings (--ctx-frames), so the old data keeps the precision

Losses (--loss, comma-separated):
  sup  ctx2's weighted MSE on the readout y = EMA(0.85) of output spikes (frame level)
  mil  event level, aimed at recall: every press window must push max(y) above theta + margin, every scored
       negative frame must stay below theta - margin (squared hinges). v3 fails mostly here: on held-out
       sessions its readout reaches 0.7 on 56 % of throws but crosses its theta on only 19 %
  pg   REINFORCE (K sampled output spike trains, leave-one-out baseline), reward per alarm: +1 (+early bonus) for
       the first alarm in a press window, -fa_cost outside, -miss_cost for a window without one. Optimises the
       exact alarm rule; high variance, see README

Validation: the last --val-frac of every human session (or --val sessions), streamed with carried state like the
board. Reports recall / false alarms per min / alarm-minus-press over a theta grid for the starting model, for the
board's recorded spikes (what really ran) and the fine-tuned model; theta* = the lowest theta with <= --fa-target
false alarms per minute. Exports <out>/board/ (weights.txt, readout.json with theta*, model.pt).

  python live_early/hitl/train.py live_early/hitl/sessions/* --out runs/hitl/r1 --minutes 20
  python live_early/hitl/train.py sessions/* --out runs/hitl/r1 --ctx-frames ~/early_ctx/frames --ctx-frac 0.5
  python live_early/hitl/train.py --estimate-mem --bs 16 --T 360                 # memory of one training step
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from riverops.nn.surrogate import ATan  # noqa: E402
from export_board import export, load_ckpt  # noqa: E402

SPIKE = ATan()
GRID, FPS = 80, 100
SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


# ----------------------------------------------------------------------------- data
def load_session(path: Path):
    chunks = sorted(path.glob("chunk_*.npz"))
    if not chunks:
        return None
    cols = {k: [] for k in ("frames", "spike", "held", "alarm", "watch")}
    for c in chunks:
        try:
            z = np.load(c)
            z["frames"].shape
        except Exception:      # e.g. a chunk still being uploaded: keep the complete prefix (frame indices stay right)
            break
        for k in cols:
            # watch = 1: the person was "just watching" (no labels), those frames are not training data
            cols[k].append(z[k] if k in z.files else np.zeros(len(z["frames"]), np.uint8))
    if not cols["frames"]:
        return None
    s = {k: np.concatenate(v) for k, v in cols.items()}
    downs, n = [], len(s["frames"])
    ev = path / "events.jsonl"
    if ev.exists():
        for line in ev.read_text().splitlines():
            e = json.loads(line)
            if e["ev"] == "down" and e["k"] < n:
                downs.append(e["k"])
            elif e["ev"] == "undo" and downs:
                downs.pop()
    s["presses"] = np.array(downs, np.int64)
    s["name"] = path.name
    return s


def load_ctx(frames_dir: Path):
    """early_ctx whole recordings: frames/<stem>.npy + labels.json; releases act as presses."""
    labels = json.loads((REPO / "training" / "early_ctx" / "labels.json").read_text())
    out = []
    for name, meta in labels.items():
        f = frames_dir / (Path(name).stem + ".npy")
        if not f.exists():
            continue
        fr = np.load(f)
        n = min(len(fr), int(round(meta["end_s"] * FPS)))
        ev = meta["events"]
        out.append(dict(name="ctx:" + Path(name).stem, frames=fr[:n], spike=np.full(n, -1, np.int8), held=np.zeros(n, np.uint8),
                        presses=np.array([int(round(e.get("release_s", e["t"]) * FPS)) for e in ev
                                          if e["label"] in ("throw", "feint")], np.int64),
                        hn=[int(round(e["t"] * FPS)) for e in ev if e["label"] in ("idle", "invalid")], lag=0))
    return out


def make_targets(n, presses, held, a, lag, hn=()):
    """Per-frame target (0/1), weight (0 = ignored) and the positive windows (ctx.make_targets, with presses)."""
    tgt, w = np.zeros(n, np.float32), np.ones(n, np.float32)
    if a.label == "hold" and held.any():
        idx = np.flatnonzero(held) - lag
        idx = idx[(idx >= 0) & (idx < n)]
        br = np.flatnonzero(np.diff(idx) > 1)
        windows = list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]])) if len(idx) else []
    else:
        windows = [(p - lag - a.pre, p - lag + a.post) for p in presses]
    windows = [(max(0, int(s)), min(n - 1, int(e))) for s, e in windows if e >= 0 and s < n]
    posz, hard = np.zeros(n, bool), np.zeros(n, bool)
    for c in hn:
        hard[max(0, c - a.hn_half):max(0, c + a.hn_half)] = True
    for s, e in windows:
        tgt[s:e + 1] = 1
        w[s:e + 1] = a.pos_w * np.linspace(a.early_w, 1.0, e - s + 1)
        posz[max(0, s - a.dc):min(n, e + 1 + a.post_ign)] = True
        w[max(0, s - a.dc):s] = 0
        w[e + 1:min(n, e + 1 + a.post_ign)] = 0
    w[hard & ~posz] = a.hn_w
    return tgt, w, windows


def unclaimed_alarms(alarm, presses, a, lag):
    """Board alarms during recording that no press window claims = the board's false alarms -> hard negatives."""
    al = np.flatnonzero(alarm)
    return [int(t) for t in al if not any(p - lag - a.pre - a.dc <= t <= p - lag + a.post + a.post_ign for p in presses)]


def unpack(packed):
    """uint8 [..., 1600] (np.packbits order) -> float [..., 2, 80, 80]."""
    bits = (packed[..., None] >> SHIFTS.to(packed.device)) & 1
    return bits.reshape(*packed.shape[:-1], 2, GRID, GRID).float()


class Sampler:
    """Windows of warm + T frames: around press windows (p_pos), around hard negatives (p_hn), else uniform;
    uniform picks draw ctx recordings with probability ctx_frac."""

    def __init__(self, segs, a, dev, rng):
        self.segs, self.a, self.dev, self.rng = segs, a, dev, rng
        self.frames = [torch.from_numpy(s["frames"]).to(dev) for s in segs]
        self.tgt = [torch.from_numpy(s["tgt"]).to(dev) for s in segs]
        self.w = [torch.from_numpy(s["w"]).to(dev) for s in segs]
        self.len = np.array([len(s["frames"]) for s in segs])
        is_ctx = np.array([s["name"].startswith("ctx:") for s in segs])
        span = np.maximum(self.len - a.T - a.warm, 1).astype(float)
        p = np.zeros(len(segs))
        for grp, share in ((is_ctx, a.ctx_frac), (~is_ctx, 1 - a.ctx_frac)):
            if grp.any():
                p[grp] = share * span[grp] / span[grp].sum()
        self.p_seg = p / p.sum()
        self.pos = [(i, e) for i, s in enumerate(segs) for _, e in s["windows"]]
        self.hn = [(i, c) for i, s in enumerate(segs) for c in s["hn"]]

    def batch(self, B, aug=True):
        a, T, W = self.a, self.a.T, self.a.warm
        xs, ts, ws, wins = [], [], [], []
        for _ in range(B):
            u = self.rng.random()
            if u < a.p_pos and self.pos:
                i, e = self.pos[self.rng.integers(len(self.pos))]
                st = e - int(self.rng.integers(int(0.4 * T), T - 20)) - W      # window end inside the scored part
            elif u < a.p_pos + a.p_hn and self.hn:
                i, c = self.hn[self.rng.integers(len(self.hn))]
                st = c - int(self.rng.integers(int(0.2 * T), int(0.9 * T))) - W
            else:
                i = int(self.rng.choice(len(self.segs), p=self.p_seg))
                st = int(self.rng.integers(0, max(1, self.len[i] - T - W)))
            st = int(np.clip(st, 0, max(0, self.len[i] - T - W)))
            sl = slice(st, st + T + W)
            w = self.w[i][sl].clone()
            w[:W] = 0                                   # state starts at zero here; the board's never does
            xs.append(self.frames[i][sl]); ts.append(self.tgt[i][sl]); ws.append(w)
            wins.append([(s - st, e - st) for s, e in self.segs[i]["windows"] if e >= st + W and s < st + T + W])
        x = unpack(torch.stack(xs))
        if aug:                                         # ctx2 augmentation
            for b in range(B):
                if self.rng.random() < 0.5:
                    x[b] = x[b].flip(-1)
                dx = int(self.rng.integers(-a.shift, a.shift + 1))
                if dx:
                    x[b] = torch.roll(x[b], dx, -1)
                    if dx > 0:
                        x[b, ..., :dx] = 0
                    else:
                        x[b, ..., dx:] = 0
                if self.rng.random() < 0.3:
                    x[b] = x[b].flip(1)
            keep = torch.empty(B, 1, 1, 1, 1, device=self.dev).uniform_(a.thin_min, 1.0)
            x = x * (torch.rand_like(x) < keep)
            x = torch.maximum(x, (torch.rand_like(x) < 3e-4).float())
        return x, torch.stack(ts), torch.stack(ws), wins


# ----------------------------------------------------------------------------- network (float replica of the board)
def layers_of(model):
    return list(model.backbone) + [model.head]


def run(model, x, state=None, head_cur=False):
    """x [B, T, 2, 80, 80] -> output spikes [B, T, K], head input current [B, T, K] (optional), new state.
    Same math as the board / ctx2.stream_batch: v = leak * V + I; s = (v >= th); hard reset; ATan surrogate grad."""
    Ls = layers_of(model)
    state = list(state) if state is not None else [None] * len(Ls)
    B, T = x.shape[:2]
    h, cur_head = x, None
    for i, L in enumerate(Ls):
        cur = L.conv(h)                                # riverops wrappers take [B, T, C, H, W]
        if getattr(L, "bn", None) is not None:
            cur = L.bn(cur)
        if i == len(Ls) - 1 and head_cur:
            cur_head = cur.flatten(2)                  # [B, T, K]
        leak, th = 1.0 - 1.0 / L.lif.tau, L.lif.v_threshold
        V, out = state[i], []
        for t in range(T):
            v = cur[:, t] if V is None else V * leak + cur[:, t]
            s = SPIKE(v - th)
            V = (1.0 - s) * v
            out.append(s)
        state[i] = V
        h = torch.stack(out, 1)
    return h.flatten(2), cur_head, state                # spikes [B, T, K]


def cap_tau(model, tau_max):
    lo = -math.log(tau_max - 1.0)                      # tau = 1 + exp(-logit)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "leak_logit") and isinstance(m.leak_logit, torch.nn.Parameter):
                m.leak_logit.clamp_(min=lo)


def tonic_excess(model, margin=0.9):
    """ctx2: the BN-folded constant drive of every channel must stay below margin * v_th / tau."""
    tot = 0.0
    for L in layers_of(model):
        conv, bn = L.conv, getattr(L, "bn", None)
        b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
        c0 = b if bn is None else bn.bias + bn.weight * (b - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
        tot = tot + (torch.relu(c0 - margin * float(L.lif.v_threshold) / float(L.lif.tau)) ** 2).sum()
    return tot


def mil_loss(y, tgt, w, wins, a):
    """Event-level hinge: max y over each press window >= theta + margin; scored negatives <= theta - margin."""
    pos = []
    for b, ws in enumerate(wins):
        for s, e in ws:
            s, e = max(a.warm, s), min(y.shape[1] - 1, e)
            if e >= s:
                pos.append(F.relu(a.mil_theta + a.mil_margin - y[b, s:e + 1].max()) ** 2)
    neg_m = (tgt == 0) & (w > 0)
    neg = (w * F.relu(y - (a.mil_theta - a.mil_margin)) ** 2 * neg_m).sum() / neg_m.sum().clamp(min=1)
    return (torch.stack(pos).mean() if pos else y.sum() * 0) + a.mil_neg * neg


def pg_loss(model, cur, tw, wins, a):
    """REINFORCE on the output neuron, K sampled spike trains per window, leave-one-out baseline.
    The head LIF is re-run per sample so its resets follow the sampled spikes."""
    L = model.head
    cur = cur[..., :a.spec.n_used]                     # [B, T, U]: the channels the readout uses
    B, T = cur.shape[:2]
    K = a.pg_samples
    leak, th = 1.0 - 1.0 / L.lif.tau, L.lif.v_threshold
    c = cur.repeat_interleave(K, 0)
    V, logp, acts = None, [], []
    for t in range(T):
        v = c[:, t] if V is None else V * leak + c[:, t]
        p = torch.sigmoid(a.pg_beta * (v - th)).clamp(1e-4, 1 - 1e-4)
        s = torch.bernoulli(p.detach())
        logp.append((s * p.log() + (1 - s) * (1 - p).log()).sum(-1))
        acts.append(s)
        V = (1.0 - s) * v
    logp, acts = torch.stack(logp, 1), torch.stack(acts, 1).cpu().numpy()
    tw = tw.repeat_interleave(K, 0).cpu().numpy()       # 0 = ignored, 1 = negative, 2 = positive
    R = np.zeros((B * K, T), np.float32)
    for j in range(B * K):
        al = a.spec.alarms(acts[j], a.mil_theta)
        claimed = set()
        for s, e in wins[j // K]:
            s_, e_ = max(0, s), min(T - 1, e)
            hit = [t for t in al if s_ <= t <= e_]
            if hit:
                R[j, hit[0]] += 1.0 + a.early_bonus * (e_ - hit[0]) / max(1, e_ - s_)
                claimed.update(hit)
            elif e < T and s >= a.warm:
                R[j, e_] -= a.miss_cost
        for t in al:
            if t not in claimed and t >= a.warm and tw[j, t] == 1:
                R[j, t] -= a.fa_cost
    G, acc = np.zeros_like(R), np.zeros(B * K, np.float32)
    for t in range(T - 1, -1, -1):
        acc = R[:, t] + a.pg_gamma * acc
        G[:, t] = acc
    G = torch.from_numpy(G).to(cur.device).view(B, K, T)
    adv = (G - (G.sum(1, keepdim=True) - G) / max(1, K - 1)).view(B * K, T)
    adv[:, :a.warm] = 0
    return -(adv * logp).sum() / (B * K * T), float(R.sum() / (B * K))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def stream_spikes(model, packed, dev, chunk=500):
    state, out = None, []
    for k in range(0, len(packed), chunk):
        x = unpack(torch.from_numpy(packed[k:k + chunk]).to(dev))[None]
        s, _, state = run(model, x, state)
        out.append((s[0] > 0.5).cpu().numpy())         # [T, K]
    return np.concatenate(out) if out else np.zeros(0, bool)


def score(spikes_by_seg, segs, theta, a):
    hits = misses = fas = 0
    lat, minutes = [], 0.0
    for sp, s in zip(spikes_by_seg, segs):
        al = a.spec.alarms(sp, theta)
        minutes += len(sp) / (60 * FPS)
        claimed = set()
        for st, en in s["windows"]:
            h = [t for t in al if st <= t <= en]
            if h:
                hits += 1
                claimed.add(h[0])
                lat.append((h[0] - (en - a.post)) * 10)        # relative to the press (minus lag)
            else:
                misses += 1
        fas += sum(1 for t in al if t not in claimed and s["tgt"][t] == 0 and s["w"][t] > 0)
    n = hits + misses
    return {"theta": float(theta), "hits": hits, "presses": n, "recall": round(hits / n, 3) if n else None,
            "fa_per_min": round(fas / max(minutes, 1e-9), 2), "fas": fas,
            "alarm_minus_press_ms": float(np.median(lat)) if lat else None}


def sweep(spikes_by_seg, segs, a):
    lo, hi = a.spec.theta_range
    rows = [score(spikes_by_seg, segs, th, a) for th in np.round(np.linspace(lo, hi, 25), 3)]
    ok = [r for r in rows if r["fa_per_min"] <= a.fa_target]
    return rows, (ok[0] if ok else rows[-1])


# ----------------------------------------------------------------------------- main
class Trainer:
    """Optimizer + EMA weights + one training step (shared by train.py and online_train.py)."""

    def __init__(self, model, a, losses):
        self.model, self.a, self.losses = model, a, losses
        for i, L in enumerate(layers_of(model)):
            if i < a.freeze:
                for p in L.parameters():
                    p.requires_grad_(False)
        head_ids = {id(p) for p in model.head.parameters()}
        bb = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        hd = [p for p in model.head.parameters() if p.requires_grad]
        self.params = bb + hd
        self.opt = torch.optim.AdamW([{"params": bb, "lr": a.lr}, {"params": hd, "lr": 3 * a.lr}], weight_decay=a.wd)
        self.avg = copy.deepcopy(model).eval()

    def step(self, sampler):
        model, a, losses = self.model, self.a, self.losses
        model.train()
        for m in model.modules():                      # BN statistics frozen
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        x, tgt, w, wins = sampler.batch(a.bs)
        sp, cur, _ = run(model, x, head_cur="pg" in losses)
        y = a.spec.filter_torch(sp)
        ty = a.spec.y_neg + tgt * (a.spec.y_pos - a.spec.y_neg)     # sup targets in readout units
        loss, parts = a.tonic_w * tonic_excess(model), {}
        if "sup" in losses:
            sup = (w * (y - ty) ** 2).sum() / w.sum().clamp(min=1)
            loss, parts["sup"] = loss + sup, round(sup.item(), 4)
        if "mil" in losses:
            mil = mil_loss(y, tgt, w, wins, a)
            loss, parts["mil"] = loss + a.mil_w * mil, round(mil.item(), 4)
        if "pg" in losses:
            tw = torch.where(w == 0, 0, torch.where(tgt > 0, 2, 1))
            pg, ret = pg_loss(model, cur, tw, wins, a)
            loss, parts["return"] = loss + a.pg_w * pg, round(ret, 3)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        cap_tau(model, a.tau_max)
        with torch.no_grad():                          # EMA of the weights (ctx2); buffers (BN stats) unchanged
            for pa, pm in zip(self.avg.parameters(), model.parameters()):
                pa.lerp_(pm, 1 - a.ema)
        parts["out_rate"] = round(float(sp.detach()[:, a.warm:, a.spec.pos_channel].mean()), 4)
        return parts


def segment(s, lo, hi, lag, hn, a):
    """Slice [lo, hi) of a loaded session/recording into a training segment with targets.
    Frames marked "just watching" get weight 0 and carry no press windows or hard negatives."""
    watch = s["watch"][lo:hi].astype(bool) if "watch" in s else np.zeros(hi - lo, bool)
    hn = [c - lo for c in hn if lo <= c < hi and not watch[c - lo]]
    tgt, w, wins = make_targets(hi - lo, s["presses"] - lo, s["held"][lo:hi], a, lag, hn)
    if watch.any():
        w[watch] = 0
        wins = [(st, en) for st, en in wins if not watch[st:en + 1].any()]
    return dict(name=s["name"], frames=s["frames"][lo:hi], spike=s["spike"][lo:hi], tgt=tgt, w=w, windows=wins, hn=hn)


def setup(a):
    """Resolve the model folder: readout rule (a.spec), init checkpoint, default theta."""
    from readout_spec import Spec
    a.model_dir = Path(a.model_dir)
    a.spec = Spec(a.model_dir)
    a.init = a.init or str(a.model_dir / "model.pt")
    if a.mil_theta is None:
        a.mil_theta = a.spec.theta


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--model-dir", default=str(HERE.parent / "model_d_neg11"),
                    help="board folder to start from: model.pt, readout.json (rule), config.yaml/model.json (firmware)")
    ap.add_argument("--init", default=None, help="checkpoint to start from (default <model-dir>/model.pt)")
    ap.add_argument("--out", default="runs/hitl/run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("labels (frames of 10 ms; defaults = ctx2: onset 500 ms, post 300, dc 200, post_ign 500)")
    g.add_argument("--label", choices=["tap", "hold"], default="tap")
    g.add_argument("--lag", type=int, default=0, help="human reaction delay in frames, subtracted from presses")
    g.add_argument("--pre", type=int, default=50)
    g.add_argument("--post", type=int, default=30)
    g.add_argument("--dc", type=int, default=20)
    g.add_argument("--post-ign", type=int, default=50)
    g.add_argument("--pos-w", type=float, default=1.5)
    g.add_argument("--early-w", type=float, default=2.0)
    g.add_argument("--hn-w", type=float, default=2.0)
    g.add_argument("--hn-half", type=int, default=100)
    g = ap.add_argument_group("data")
    g.add_argument("--ctx-frames", default="", help="early_ctx frames/ dir (whole recordings) to replay alongside")
    g.add_argument("--ctx-frac", type=float, default=0.5, help="share of uniform windows drawn from the ctx recordings")
    g.add_argument("--val", default="", help="comma-separated human session names held out; default last --val-frac")
    g.add_argument("--val-frac", type=float, default=0.2)
    g = ap.add_argument_group("optimisation")
    g.add_argument("--loss", default="sup,mil", help="comma list of sup, mil, pg")
    g.add_argument("--bs", type=int, default=16)
    g.add_argument("--T", type=int, default=300)
    g.add_argument("--warm", type=int, default=60)
    g.add_argument("--p-pos", type=float, default=0.4)
    g.add_argument("--p-hn", type=float, default=0.2)
    g.add_argument("--shift", type=int, default=10)
    g.add_argument("--thin-min", type=float, default=0.6)
    g.add_argument("--lr", type=float, default=3e-4, help="backbone; head gets 3x (ctx2)")
    g.add_argument("--wd", type=float, default=1e-4)
    g.add_argument("--freeze", type=int, default=0, help="freeze the first N layers")
    g.add_argument("--tonic-w", type=float, default=1.0)
    g.add_argument("--tau-max", type=float, default=8.0)
    g.add_argument("--ema", type=float, default=0.99)
    g.add_argument("--steps", type=int, default=100000)
    g.add_argument("--minutes", type=float, default=20.0)
    g.add_argument("--eval-every", type=int, default=200)
    g = ap.add_argument_group("mil / pg")
    g.add_argument("--mil-w", type=float, default=1.0)
    g.add_argument("--mil-theta", type=float, default=None, help="theta the event losses / RL reward use (default: readout.json)")
    g.add_argument("--mil-margin", type=float, default=0.05)
    g.add_argument("--mil-neg", type=float, default=1.0)
    g.add_argument("--pg-w", type=float, default=1.0)
    g.add_argument("--pg-samples", type=int, default=4)
    g.add_argument("--pg-beta", type=float, default=5.0)
    g.add_argument("--pg-gamma", type=float, default=0.97)
    g.add_argument("--fa-cost", type=float, default=1.0)
    g.add_argument("--miss-cost", type=float, default=1.0)
    g.add_argument("--early-bonus", type=float, default=0.5)
    g = ap.add_argument_group("evaluation")
    g.add_argument("--alpha", type=float, default=0.85)
    g.add_argument("--fa-target", type=float, default=1.0, help="theta* = lowest theta with <= this many false alarms/min")
    g.add_argument("--estimate-mem", action="store_true", help="measure memory/time of one training step and exit")
    return ap


def main():
    a = build_parser().parse_args()
    losses = set(a.loss.split(","))
    assert losses <= {"sup", "mil", "pg"}, a.loss

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    import speedpatch  # noqa: F401  (only touches the riverops forward, not run())

    setup(a)
    model, ck = load_ckpt(a.init)
    model.to(dev)
    if a.estimate_mem:
        return estimate_mem(model, a, dev, losses)

    human = [s for s in (load_session(Path(p)) for p in a.sessions if Path(p).is_dir()) if s is not None]
    ctx_recs = load_ctx(Path(a.ctx_frames).expanduser()) if a.ctx_frames else []
    if not human and not ctx_recs:
        raise SystemExit("no sessions (chunk_*.npz) and no --ctx-frames recordings found")
    val_names = set(filter(None, a.val.split(",")))
    train_segs, val_segs = [], []

    def seg(s, lo, hi, lag, hn):
        return segment(s, lo, hi, lag, hn, a)

    for s in human:
        n = len(s["frames"])
        hn = unclaimed_alarms(s["alarm"], s["presses"], a, a.lag)
        if val_names:
            (val_segs if s["name"] in val_names else train_segs).append(seg(s, 0, n, a.lag, hn))
            continue
        cut = int(n * (1 - a.val_frac))
        for lo, hi, dst in ((0, cut, train_segs), (cut, n, val_segs)):
            if hi - lo >= a.T + a.warm:
                dst.append(seg(s, lo, hi, a.lag, hn))
    for s in ctx_recs:
        train_segs.append(seg(s, 0, len(s["frames"]), 0, s["hn"]))
    if not train_segs:
        raise SystemExit(f"sessions too short: need > {a.T + a.warm} frames of training data")
    info = {"human_sessions": [s["name"] for s in human], "ctx_recordings": len(ctx_recs),
            "train_min": round(sum(len(s["frames"]) for s in train_segs) / 6000, 2),
            "val_min": round(sum(len(s["frames"]) for s in val_segs) / 6000, 2),
            "train_windows": sum(len(s["windows"]) for s in train_segs), "train_hard_neg": sum(len(s["hn"]) for s in train_segs),
            "val_windows": sum(len(s["windows"]) for s in val_segs)}
    print(json.dumps(info), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def evaluate(net):
        net.eval()
        return sweep([stream_spikes(net, s["frames"], dev) for s in val_segs], val_segs, a)

    report = {"args": vars(a), "data": info}
    if val_segs:
        report["init"] = evaluate(model)[1]
        if all((s["spike"] >= 0).all() for s in val_segs):
            report["board_recorded"] = sweep([s["spike"] > 0 for s in val_segs], val_segs, a)[1]
        print(json.dumps({k: report[k] for k in ("init", "board_recorded") if k in report}), flush=True)

    trainer = Trainer(model, a, losses)
    avg = trainer.avg
    sampler = Sampler(train_segs, a, dev, rng)
    t0, log, best, best_key, step = time.time(), [], None, None, 0
    for step in range(1, a.steps + 1):
        if time.time() - t0 > a.minutes * 60:
            step -= 1
            break
        parts = trainer.step(sampler)
        if step % a.eval_every == 0 and val_segs:
            rows, sel = evaluate(avg)
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1), **parts, "val": sel}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            key = (sel["hits"], -sel["fas"])
            if best_key is None or key > best_key:
                best_key, best = key, {"step": step, "sel": sel, "rows": rows,
                                       "state": {k: v.detach().cpu().clone() for k, v in avg.state_dict().items()}}
        elif step % 20 == 0:
            print(json.dumps({"step": step, "min": round((time.time() - t0) / 60, 2), **parts}), flush=True)

    if best is None:                                   # no validation data, or no eval reached: last EMA weights
        rows, sel = evaluate(avg) if val_segs else ([], {"theta": a.mil_theta})
        best = {"step": step, "sel": sel, "rows": rows, "state": {k: v.detach().cpu() for k, v in avg.state_dict().items()}}
    model.load_state_dict(best["state"])
    model.cpu().eval()
    cfg = dict(ck["config"])
    cfg.update(taus=[float(torch.as_tensor(L.lif.tau).detach()) for L in layers_of(model)],
               hitl={"init": a.init, "sessions": info["human_sessions"], "ctx_recordings": len(ctx_recs),
                     "step": best["step"], "loss": a.loss})
    ckpt = {"model": model.state_dict(), "config": cfg}
    torch.save(ckpt, out / "model.pt")
    theta = float(best["sel"]["theta"])
    md5, _ = export(model, out / "board", theta, {"hitl_val": best["sel"]}, model_dir=a.model_dir)
    torch.save(ckpt, out / "board" / "model.pt")
    report.update(steps=step, best_step=best["step"], selected=best["sel"], sweep=best["rows"], log=log, weights_md5=md5)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"export": str(out / "board"), "md5": md5, "theta": theta, "steps": step, "val": best["sel"],
                      "init": report.get("init"), "board_recorded": report.get("board_recorded")}), flush=True)


def estimate_mem(model, a, dev, losses):
    """Bytes autograd keeps for one training step (activations) and the step time."""
    saved = [0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        return t

    B, T = a.bs, a.T + a.warm
    x = (torch.rand(B, T, 2, GRID, GRID, device=dev) < 0.03).float()
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        sp, cur, _ = run(model, x, head_cur=True)
        loss = a.spec.filter_torch(sp).mean() + (cur.mean() if "pg" in losses else 0)
    loss.backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    res = {"bs": B, "frames": T, "device": str(dev), "saved_activations_GB": round(saved[0] / 1e9, 2),
           "input_GB": round(x.numel() * 4 / 1e9, 2), "step_s": round(time.time() - t0, 2),
           "params": sum(p.numel() for p in model.parameters())}
    if dev.type == "cuda":
        res["cuda_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()

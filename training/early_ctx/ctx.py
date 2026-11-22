#!/usr/bin/env python3
"""GOAL B early throw detector trained and evaluated on WHOLE recordings (runs/early_ctx).

  train   windows cut from whole recordings of the training sessions (pauses, walking, ball pick-up and
          camera handling are real negatives; throws AND feints positive), optional foreign negatives
          from Tim's banks (mb/dd/dvsg), Tim's v1 arch + DVS-Gesture init, 1 output, EMA readout loss.
  stream  run a checkpoint over whole recordings -> output spikes per 10 ms frame (npz)
  score   corrected whole-recording eval (union of masked time, idle/invalid = hard negatives,
          only fresh alarm onsets count as hits), theta chosen on the TRAINING sessions of each fold.

Frames come from frames/*.npy (prep.py, bit-packed, binning identical to tools/ball_live.frames_from_file).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FPS = 100
FOLDS = {0: ["170441", "201703", "215305"],
         1: ["195808", "200001", "200828", "201452", "201622"]}
IDLE_REC = "211821"   # 04: idle-only recording (other, busier scene): training negative only, never scored


def rec_key(name):
    return name.split("_")[2]


def labels():
    return json.loads((HERE / "labels.json").read_text())


def fold_of(name):
    k = rec_key(name)
    for f, ks in FOLDS.items():
        if k in ks:
            return f
    return None


# ----------------------------------------------------------------------------- targets
def make_targets(meta, onset=0.5, dc=0.2, post=0.3, post_ign=0.5, hn_w=2.0, hn_half=1.0, pos_w=1.5, early_w=2.0):
    N = int(round(meta["end_s"] * FPS))
    tgt = np.zeros(N, np.float32)
    w = np.ones(N, np.float32)
    hard = np.zeros(N, bool)
    posz = np.zeros(N, bool)
    for e in meta["events"]:
        if e["label"] in ("idle", "invalid"):
            a, b = int((e["t"] - hn_half) * FPS), int((e["t"] + hn_half) * FPS)
            hard[max(0, a):max(0, b)] = True
    for e in meta["events"]:
        if e["label"] not in ("throw", "feint"):
            continue
        R = int(round(e.get("release_s", e["t"]) * FPS))
        a, b = max(0, R - int(onset * FPS)), min(N, R + int(post * FPS) + 1)
        if b <= a:
            continue
        tgt[a:b] = 1
        w[a:b] = pos_w * np.linspace(early_w, 1.0, b - a)
        posz[max(0, a - int(dc * FPS)):min(N, b + int(post_ign * FPS))] = True
        w[max(0, a - int(dc * FPS)):a] = 0
        w[b:min(N, b + int(post_ign * FPS))] = 0
    w[hard & ~posz] = hn_w
    return tgt, w


# ----------------------------------------------------------------------------- train
def unpack(x):
    """uint8 [..., 1600] (packbits of 12800 bools, big-endian bit order) -> float [..., 2, 80, 80]."""
    import torch
    bits = (x[..., None] >> torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)) & 1
    return bits.reshape(*x.shape[:-1], 2, 80, 80).float()


def train(a):
    import torch
    sys.path.insert(0, str(HERE / "tim"))
    import speedpatch  # noqa: F401
    from early_cv import build_model, cap_tau, ema_readout
    dev = "cuda"
    torch.manual_seed(a.seed * 100 + a.fold)
    rng = np.random.default_rng(a.seed * 100 + a.fold)
    lab = labels()
    recs = []
    for name, meta in lab.items():
        f = fold_of(name)
        if rec_key(name) == IDLE_REC:
            if not a.use_idle:
                continue
        elif f is None or f == a.fold:
            continue
        fr = np.load(HERE / "frames" / (Path(name).stem + ".npy"))
        tgt, w = make_targets(meta, onset=a.onset / 1000, hn_w=a.hn_w, pos_w=a.pos_w, early_w=a.early_w)
        n = min(len(fr), len(tgt))
        recs.append(dict(name=name, x=torch.from_numpy(fr[:n]).to(dev), tgt=tgt[:n], w=w[:n],
                         pos=[int(round(e.get("release_s", e["t"]) * FPS)) for e in meta["events"]
                              if e["label"] in ("throw", "feint") and e.get("release_s", e["t"]) * FPS < n]))
    print("train recs:", [(r["name"], len(r["x"]), len(r["pos"])) for r in recs], flush=True)
    pos_list = [(i, R) for i, r in enumerate(recs) for R in r["pos"]]
    lens = np.array([len(r["x"]) for r in recs], float)
    bank_plan = None
    if a.foreign > 0:
        sys.argv = ["x"]
        import early_cv
        from early_cv import Bank, Planner
        ap = argparse.Namespace(**{**vars(early_cv_defaults()), "p_pos": 0.0, "w_noah": 0.0, "w_mb": 0.4,
                                   "w_dd": 0.4, "w_dvsg": 0.2, "w_hmdb": 0.0, "p_bed": 0.3, "dnorm": 0,
                                   "noise": 3e-4, "hot": 1, "thin_min": 0.6})
        bank = Bank(a.banks, dev)
        neg = [np.flatnonzero((bank.lab == 0) & (bank.src == s)) for s in range(len(early_cv.SRCS))]
        neg[0] = np.array([], int)
        bank_plan = Planner(bank, ap, np.array([], int), neg, rng)
    T, W = a.T, a.warm
    model = build_model(argparse.Namespace(init=a.init, arch="v1"), dev)
    if a.resume:   # continue from a finished run of this fold (full state incl. head)
        model.load_state_dict(torch.load(Path(a.resume) / f"model_f{a.fold}.pt", map_location=dev, weights_only=False)["model"])
    cap_tau(model, 8.0)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": 3 * a.lr}], weight_decay=1e-2)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    def batch(B):
        xs, ts, ws = [], [], []
        for _ in range(B):
            if rng.random() < a.p_pos and pos_list:
                i, R = pos_list[rng.integers(len(pos_list))]
                s = R - int(rng.integers(int(0.4 * T), T - 40)) - W
            else:
                i = int(rng.choice(len(recs), p=lens / lens.sum()))
                s = int(rng.integers(-W, len(recs[i]["x"]) - T))
            r = recs[i]
            s = int(np.clip(s, 0, len(r["x"]) - T - W))
            xs.append(r["x"][s:s + T + W])
            t = np.zeros(T + W, np.float32); ww = np.zeros(T + W, np.float32)
            t[:] = r["tgt"][s:s + T + W]; ww[:] = r["w"][s:s + T + W]
            ww[:W] = 0   # state warm-up frames carry no loss
            ts.append(t); ws.append(ww)
        x = unpack(torch.stack(xs))                       # [B, T+W, 2, 80, 80]
        # light augmentation: mirror, x shift, event thinning, polarity swap, sensor noise
        for b in range(B):
            if rng.random() < 0.5:
                x[b] = x[b].flip(-1)
            dx = int(rng.integers(-a.shift, a.shift + 1))
            if dx:
                x[b] = torch.roll(x[b], dx, -1)
                if dx > 0:
                    x[b, ..., :dx] = 0
                else:
                    x[b, ..., dx:] = 0
            if rng.random() < 0.3:
                x[b] = x[b].flip(1)
        keep = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(a.thin_min, 1.0)
        x = x * (torch.rand_like(x) < keep)
        x = torch.maximum(x, (torch.rand_like(x) < 3e-4).float())
        return x, torch.as_tensor(np.stack(ts), device=dev), torch.as_tensor(np.stack(ws), device=dev)

    log, acc, t0 = [], [], time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for step in range(a.steps):
        nb = int(round(a.bs * a.foreign)) if bank_plan else 0
        x, tgt, w = batch(a.bs - nb)
        if nb:
            xb, tb, wb, _ = bank_plan.batch(nb, T + W)
            wb[:, :W] = 0
            x = torch.cat([x, xb]); tgt = torch.cat([tgt, tb]); w = torch.cat([w, wb])
        sp = model.head(model.backbone(x)).flatten(2)[..., 0]
        y = ema_readout(sp, a.alpha)
        loss = (w * (y - tgt) ** 2).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        cap_tau(model, 8.0)
        ema.update_parameters(model)
        with torch.no_grad():
            pm, nm = (tgt > 0) & (w > 0), (tgt == 0) & (w > 0)
            acc.append([loss.item(), y[pm].mean().item() if pm.any() else float("nan"), y[nm].mean().item(),
                        (y[nm] > 0.5).float().mean().item()])
        if (step + 1) % 50 == 0:
            m = np.nanmean(acc, 0)
            log.append({"step": step + 1, "loss": round(m[0], 4), "y_pos": round(m[1], 3), "y_neg": round(m[2], 4),
                        "neg_gt.5": round(m[3], 4), "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
            acc = []
    net = ema.module
    torch.optim.swa_utils.update_bn((batch(a.bs)[0] for _ in range(10)), ema)
    taus = [float(m.tau) for m in net.modules() if hasattr(m, "leak_logit")]
    cfg = {"arch": net.arch, "num_classes": 1, "classes": ["throw"], "readout": "ema", "alpha": a.alpha,
           "frame_ms": 10, "input": [2, 80, 80], "taus": taus, "args": vars(a)}
    torch.save({"model": net.state_dict(), "config": cfg}, out / f"model_f{a.fold}.pt")
    (out / f"trainlog_f{a.fold}.json").write_text(json.dumps(log))
    print("saved", out / f"model_f{a.fold}.pt", flush=True)


def early_cv_defaults():
    import early_cv
    import argparse as _ap
    # re-create Tim's argparse defaults without touching his file
    src = Path(early_cv.__file__).read_text()
    p = _ap.ArgumentParser()
    for line in src.splitlines():
        line = line.strip()
        if line.startswith("ap.add_argument("):
            try:
                eval(line.replace("ap.add_argument(", "p.add_argument(", 1), {"p": p, "Path": Path, "ARCHS": early_cv.ARCHS})
            except Exception:
                pass
    return p.parse_args([])


# ----------------------------------------------------------------------------- stream
def layer_c0(L):
    """BN-folded constant drive per output channel (what the neuron integrates with zero input)."""
    import torch
    bn = getattr(L, "bn", None)
    b = L.conv.bias if getattr(L.conv, "bias", None) is not None else torch.zeros(L.conv.out_channels, device=L.conv.weight.device)
    if bn is None:
        return b.detach(), None
    sd = torch.sqrt(bn.running_var + bn.eps)
    return (bn.bias + bn.weight * (b - bn.running_mean) / sd).detach(), sd


def tonic_report(m):
    out = []
    for i, L in enumerate(list(m.backbone) + [m.head]):
        c0, _ = layer_c0(L)
        thr = float(L.lif.v_threshold) / float(L.lif.tau)
        out.append(int((c0 > thr).sum()))
    return out


def clamp_tonic(m, margin=0.9):
    """Post-hoc: lower bn.bias (or conv bias) so that no channel's constant drive exceeds margin * v_th / tau,
    i.e. no neuron fires without input (kills free-running tonic oscillators that carry state for minutes)."""
    import torch
    with torch.no_grad():
        for L in list(m.backbone) + [m.head]:
            c0, sd = layer_c0(L)
            lim = margin * float(L.lif.v_threshold) / float(L.lif.tau)
            ex = torch.clamp(c0 - lim, min=0)
            if sd is None:
                L.conv.bias -= ex
            else:
                L.bn.bias -= ex


def stateful_forward(m, fr, dev, chunk=2000):
    """One uninterrupted run over the whole recording (state carried across chunks, single reset at file start).
    Manual PLIF loop as in the reviewers' fullstream.py (checked against a full-sequence forward below)."""
    import torch
    layers = list(m.backbone) + [m.head]
    V = [None] * len(layers)
    out = []
    with torch.no_grad():
        for s in range(0, len(fr), chunk):
            h = unpack(torch.from_numpy(fr[s:s + chunk]).to(dev))[None]
            for i, L in enumerate(layers):
                cur = L.bn(L.conv(h)) if getattr(L, "bn", None) is not None else L.conv(h)
                leak = 1 - 1 / L.lif.tau
                v = torch.zeros_like(cur[:, 0]) if V[i] is None else V[i]
                sp = []
                for t in range(cur.shape[1]):
                    v = v * leak + cur[:, t]
                    s_ = (v >= L.lif.v_threshold).float()
                    v = (1 - s_) * v
                    sp.append(s_)
                V[i] = v
                h = torch.stack(sp, 1)
            out.append(h.flatten(2)[0, :, 0].cpu().numpy())
    return np.concatenate(out)


def stream(a):
    import torch
    sys.path.insert(0, str(HERE / "tim"))
    import speedpatch  # noqa: F401
    from snn_model import ThrowSNN, arch_from_config
    dev = "cuda"
    lab = labels()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    info = {}
    for f in (0, 1):
        ck = torch.load(Path(a.model_dir) / f"model_f{f}.pt", map_location=dev, weights_only=False)
        cfg = ck["config"]
        m = ThrowSNN(1, arch=arch_from_config(cfg)).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        info[f"tonic_f{f}_before"] = tonic_report(m)
        if a.clamp:
            clamp_tonic(m)
        info[f"tonic_f{f}"] = tonic_report(m)
        if f == 0:   # sanity: stateful loop == full-sequence forward (short recording)
            fr = np.load(HERE / "frames" / "sample_20261121_201703_662612.npy")
            with torch.no_grad():
                ref = m(unpack(torch.from_numpy(fr).to(dev))[None])[0, :, 0].float().cpu().numpy()
            info["sanity_mismatch_frames"] = int((stateful_forward(m, fr, dev, 400) != ref).sum())
            info["sanity_frames"] = len(fr)
        # drift test: 60 s of sensor noise only (2e-4 per cell, as Tim's eval) -> output spikes
        g = np.random.default_rng(7)
        q = np.packbits(g.random((6000, 12800)) < 2e-4, axis=1)
        info[f"drift_out_spikes_f{f}"] = int(stateful_forward(m, q, dev).sum())
        for name in lab:
            if rec_key(name) == IDLE_REC:
                continue
            fr = np.load(HERE / "frames" / (Path(name).stem + ".npy"))
            out = stateful_forward(m, fr, dev)
            np.savez_compressed(Path(a.out) / f"f{f}_{rec_key(name)}.npz", spikes=out, alpha=float(cfg.get("alpha", 0.85)))
            print(a.model_dir, f, name, int(out.sum()), flush=True)
    print(json.dumps(info), flush=True)
    (Path(a.out) / "info.json").write_text(json.dumps(info))


# ----------------------------------------------------------------------------- score
def ema(sp, alpha):
    from scipy.signal import lfilter
    return lfilter([1 - alpha], [1, -alpha], sp).astype(np.float32)


def readout(sp, alpha, mode, base_s=1.5):
    if mode == "slow":   # longer host-side smoothing (alpha 0.93 per 10 ms frame, ~140 ms)
        return ema(sp, 0.93)
    y = ema(sp, alpha)
    if mode == "abs":
        return y
    # relative rise: y minus its own mean over the preceding base_s (excluding the last 200 ms)
    L, gap = int(base_s * FPS), 20
    c = np.concatenate([[0.0], np.cumsum(y, dtype=np.float64)])
    k = np.arange(len(y))
    hi = np.maximum(k - gap, 0)
    lo = np.maximum(k - gap - L, 0)
    m = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return (y - m).astype(np.float32)


def alarm_onsets(y, theta, refr):
    """Fresh alarms only: upward crossing of theta (y[k-1] < theta <= y[k]) and >= refr frames after the last alarm."""
    up = np.flatnonzero((y[1:] >= theta) & (y[:-1] < theta)) + 1
    out, last = [], -10 ** 9
    for k in up:
        if k - last >= refr:
            out.append(k)
            last = k
    return np.array(out, int)


def score_rec(y, meta, theta, refr=100, pre=1.0, post=0.3, ev_half=1.5, hn_half=1.0):
    """Every fresh alarm onset is exactly one of: hit (first unused alarm in a swing's window [t-pre, t+post],
    greedy one-to-one), hn_fa (inside idle/invalid +-hn_half, outside all hit windows), near_fa (any other alarm
    within +-ev_half of a swing, incl. duplicates), pause_fa (everything else).  Times: alarm at frame k is
    decided at the end of that frame -> (k+1)/FPS.  lead = alarm - reference (negative = early)."""
    N = min(len(y), int(round(meta["end_s"] * FPS)))
    y = y[:N]
    al = alarm_onsets(y, theta, refr)
    ta = (al + 1) / FPS
    evz = np.zeros(N, bool); hnz = np.zeros(N, bool); hw = np.zeros(N, bool)
    swings = sorted([e for e in meta["events"] if e["label"] in ("throw", "feint") and e["t"] * FPS < N], key=lambda e: e["t"])
    hns = [e for e in meta["events"] if e["label"] in ("idle", "invalid") and e["t"] * FPS < N]
    for e in swings:
        evz[max(0, int((e["t"] - ev_half) * FPS)):max(0, int((e["t"] + ev_half) * FPS))] = True
        hw[max(0, int((e["t"] - pre) * FPS)):max(0, int((e["t"] + post) * FPS) + 1)] = True
    for e in hns:
        hnz[max(0, int((e["t"] - hn_half) * FPS)):max(0, int((e["t"] + hn_half) * FPS))] = True
    hnz &= ~hw
    pause = ~evz & ~hnz
    near = evz & ~hnz
    used = np.zeros(len(al), bool)
    hits = []
    for e in swings:
        cand = np.flatnonzero(~used & (ta >= e["t"] - pre) & (ta <= e["t"] + post))
        a = int(round((e["t"] - pre) * FPS))
        ref = e.get("release_s")
        row = {"label": e["label"], "hit": bool(len(cand)), "pre_active": bool(y[max(0, a - 1)] >= theta),
               "has_release": ref is not None}
        if len(cand):
            used[cand[0]] = True
            row["lead_rel_ms"] = round(1000 * (ta[cand[0]] - ref)) if ref is not None else None
            row["lead_peak_ms"] = round(1000 * (ta[cand[0]] - e["t"]))
        hits.append(row)
    rest = al[~used]
    return {"hits": hits, "pause_fa": int(pause[rest].sum()), "pause_min": pause.sum() / FPS / 60,
            "hn_fa": int(hnz[rest].sum()), "hn_min": hnz.sum() / FPS / 60, "n_hn": len(hns),
            "near_fa": int(near[rest].sum()), "near_min": near.sum() / FPS / 60, "n_alarms": len(al), "tot_min": N / FPS / 60}


def poisson_up(n, conf=0.95):
    from scipy.stats import chi2
    return chi2.ppf(conf, 2 * n + 2) / 2


def aggregate(rs):
    H = [h for r in rs for h in r["hits"]]
    pf = sum(r["pause_fa"] for r in rs); pm = sum(r["pause_min"] for r in rs)
    hf = sum(r["hn_fa"] for r in rs); hm = sum(r["hn_min"] for r in rs)
    out = {}
    for lab in ("throw", "feint", "all"):
        hh = [h for h in H if lab == "all" or h["label"] == lab]
        out[lab] = [sum(h["hit"] for h in hh), len(hh)]
    lr = [h["lead_rel_ms"] for h in H if h["hit"] and h.get("lead_rel_ms") is not None]
    lp = [h["lead_peak_ms"] for h in H if h["hit"]]
    out.update(recall=out["all"][0] / max(1, out["all"][1]), pause_fa=pf, pause_min=round(pm, 2),
               fa_per_min=pf / max(pm, 1e-9), fa_up95=poisson_up(pf) / max(pm, 1e-9),
               hn_fa=hf, hn_min=round(hm, 2), hn_per_min=hf / max(hm, 1e-9), n_hn=sum(r["n_hn"] for r in rs),
               near_fa=sum(r["near_fa"] for r in rs), near_min=round(sum(r["near_min"] for r in rs), 2),
               n_alarms=sum(r["n_alarms"] for r in rs), tot_min=round(sum(r["tot_min"] for r in rs), 2),
               nonhit_per_min=(sum(r["n_alarms"] for r in rs) - out["all"][0]) / max(sum(r["tot_min"] for r in rs), 1e-9),
               lead_rel_med=float(np.median(lr)) if lr else None, n_lead_rel=len(lr),
               lead_peak_med=float(np.median(lp)) if lp else None,
               pre_active=sum(h["pre_active"] for h in H))
    return out


THETAS = {"slow": np.round(np.r_[np.arange(0.3, 0.96, 0.05), 0.97, 0.98, 0.99], 3), "abs": np.round(np.r_[np.arange(0.3, 0.96, 0.05), 0.97, 0.98, 0.99], 3),
          "rel": np.round(np.arange(0.1, 0.91, 0.05), 3)}


def score(a):
    lab = labels()
    res = {}
    for mdir in a.streams:
        name = Path(mdir).name
        ys = {}
        for f in (0, 1):
            for rec in lab:
                if rec_key(rec) == IDLE_REC:
                    continue
                z = np.load(Path(mdir) / f"f{f}_{rec_key(rec)}.npz")
                ys[(f, rec)] = (z["spikes"], float(z["alpha"]))
        res[name] = {}
        for mode in ("abs", "rel", "slow"):
            Y = {k: readout(sp, al, mode) for k, (sp, al) in ys.items()}
            # theta per fold from that fold's TRAINING sessions (02/03 only, 215305 excluded: keypress labels)
            # rule: max recall s.t. pause FA <= a.fa_target /min; ties -> higher theta
            chosen = {}
            for f in (0, 1):
                tr = [r for r in lab if fold_of(r) not in (None, f) and rec_key(r) != "215305"]
                best = None
                for th in THETAS[mode]:
                    ag = aggregate([score_rec(Y[(f, r)], lab[r], th) for r in tr])
                    ok = ag["fa_per_min"] <= a.fa_target
                    key = (ok, ag["recall"] if ok else -ag["fa_per_min"], th)
                    if best is None or key > best[0]:
                        best = (key, th, ag)
                chosen[f] = float(best[1])
            test, test215 = [], []
            per_fold = {}
            for f in (0, 1):
                te = [r for r in lab if fold_of(r) == f]
                rr = {r: score_rec(Y[(f, r)], lab[r], chosen[f]) for r in te}
                test += [v for r, v in rr.items() if rec_key(r) != "215305"]
                test215 += [v for r, v in rr.items() if rec_key(r) == "215305"]
                per_fold[f] = aggregate([v for r, v in rr.items() if rec_key(r) != "215305"])
            # theta sweep on the held-out 02/03 sessions, reported only (never used for selection)
            sweep = {}
            for th in THETAS[mode]:
                rr = [score_rec(Y[(f, r)], lab[r], th) for f in (0, 1) for r in lab
                      if fold_of(r) == f and rec_key(r) != "215305"]
                ag = aggregate(rr)
                sweep[float(th)] = {k: ag[k] for k in ("recall", "fa_per_min", "hn_per_min", "near_fa", "lead_rel_med")}
            te_all = [(f, r) for f in (0, 1) for r in lab if fold_of(r) == f and rec_key(r) != "215305"]
            # random baseline: same alarms, all labelled event times of a recording circularly shifted by one
            # random offset (>= 5 s from identity) per recording and draw; theta = that fold's chosen theta
            Ys_, Ms_ = [], []
            for f, r in te_all:
                Ys_.append(Y[(f, r)]); Ms_.append(lab[r])
            ch_hits = []
            rng_c = np.random.default_rng(0)
            for _ in range(a.n_chance):
                h = 0
                for y_, m_ in zip(Ys_, Ms_):
                    end = min(len(y_) / FPS, m_["end_s"])
                    u = rng_c.uniform(5.0, end - 5.0)
                    ev = [{k: v for k, v in e.items() if k != "release_s"} | {"t": (e["t"] + u) % end}
                          for e in m_["events"] if e["label"] != "end"]
                    th_ = chosen[[ff for ff, rr in te_all if lab[rr] is m_][0]]
                    h += sum(x["hit"] for x in score_rec(y_, dict(m_, events=ev), th_)["hits"])
                ch_hits.append(h)
            chance = {"mean": float(np.mean(ch_hits)), "p95": float(np.percentile(ch_hits, 95)), "n": a.n_chance}
            res[name][mode] = {"theta": chosen, "test_0203": aggregate(test), "per_fold": per_fold, "chance": chance,
                               "test_215305": aggregate(test215) if test215 else None, "sweep_test": sweep}
            t = res[name][mode]["test_0203"]
            print(f"{name:16s} {mode} th={chosen} | throw {t['throw'][0]}/{t['throw'][1]} feint {t['feint'][0]}/{t['feint'][1]}"
                  f" all {t['all'][0]}/{t['all'][1]} ({100 * t['recall']:.0f}%) | pause FA {t['pause_fa']} in {t['pause_min']} min"
                  f" = {t['fa_per_min']:.2f}/min (up95 {t['fa_up95']:.2f}) | hard-neg FA {t['hn_fa']} in {t['hn_min']} min ({t['hn_per_min']:.2f}/min) near-swing FA {t['near_fa']}"
                  f" | all non-hit alarms {t['nonhit_per_min']:.2f}/min of {t['tot_min']} min"
                  f" | lead rel {t['lead_rel_med']} (n={t['n_lead_rel']}) peak {t['lead_peak_med']} | pre-active {t['pre_active']}"
                  f" | CHANCE {chance['mean']:.1f} (p95 {chance['p95']:.0f})",
                  flush=True)
            t = res[name][mode]["test_215305"]
            if t:
                print(f"{'':16s}   215305: throw {t['throw'][0]}/{t['throw'][1]} feint {t['feint'][0]}/{t['feint'][1]}"
                      f" | pause FA {t['pause_fa']} in {t['pause_min']} min = {t['fa_per_min']:.2f}/min | lead peak {t['lead_peak_med']}")
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=800)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--T", type=int, default=300)
    t.add_argument("--warm", type=int, default=60)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--alpha", type=float, default=0.85)
    t.add_argument("--onset", type=int, default=500)
    t.add_argument("--p-pos", type=float, default=0.5)
    t.add_argument("--hn-w", type=float, default=2.0)
    t.add_argument("--pos-w", type=float, default=1.5)
    t.add_argument("--early-w", type=float, default=2.0)
    t.add_argument("--resume", default="")
    t.add_argument("--shift", type=int, default=10)
    t.add_argument("--thin-min", type=float, default=0.6)
    t.add_argument("--use-idle", type=int, default=1)
    t.add_argument("--foreign", type=float, default=0.0, help="share of each batch from Tim's foreign negative banks")
    t.add_argument("--banks", default=str(Path.home() / "early/banks"))
    t.add_argument("--init", default=str(Path.home() / "snn_pretrain/runs/sweep_v1/best.pt"))
    s = sub.add_parser("stream")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--clamp", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("--streams", nargs="+", required=True)
    c.add_argument("--fa-target", type=float, default=1.0)
    c.add_argument("--n-chance", type=int, default=100)
    c.add_argument("--json", default=str(HERE / "score.json"))
    a = ap.parse_args()
    {"train": train, "stream": stream, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()

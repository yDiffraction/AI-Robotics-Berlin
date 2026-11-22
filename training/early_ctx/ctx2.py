#!/usr/bin/env python3
"""GOAL B v2 (runs/early_ctx): fixes on top of ctx.py (ctx.py itself is unchanged, old results stay reproducible).

  train   * batch 32 by default: the PLIF time loop is launch-bound, a 4x bigger batch costs ~15 % more time per step
          * time-budgeted cosine LR (--minutes), so a run fits a wall-clock slot
          * tonic-drive penalty in the loss (BN-folded constant drive of every channel must stay below
            margin * v_th / tau) instead of only the post-hoc clamp
          * hard-negative window sampling around idle/invalid labels (walking, ball pick-up, camera handling)
          * BatchNorm statistics refreshed on CLEAN windows (no augmentation, no foreign data) = what the live
            path sees; ctx.py refreshed them on augmented windows
          * optional tau cap (--tau-max), optional resume
  stream  all recordings of a fold in ONE batch (same math as ctx.stateful_forward, checked against the
          full-sequence forward), drift test, tonic report
  score   ctx scoring plus: finer theta grid at the top (with alpha 0.85, y >= 0.99 needs ~29 output spikes in a
          row, and 0.99 was the top of the old grid), 'cnt' readout (output spikes in the last 300 ms / 30),
          and a theta-free model-quality number: best held-out recall at pause FA <= 1/2/3 per min with its chance
          baseline. That number picks theta on the test set, so it compares MODELS, it is not an operating point.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctx  # noqa: E402
from ctx import FPS, IDLE_REC, rec_key, labels, fold_of, make_targets, unpack  # noqa: E402


# ----------------------------------------------------------------------------- model helpers
def _layers(m):
    return list(m.backbone) + [m.head]


def tonic_excess(m, margin=0.9):
    """Differentiable: sum over channels of relu(BN-folded constant drive - margin * v_th / tau)^2."""
    import torch
    tot = 0.0
    for L in _layers(m):
        conv, bn = L.conv, getattr(L, "bn", None)
        b = conv.bias if getattr(conv, "bias", None) is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
        c0 = b if bn is None else bn.bias + bn.weight * (b - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
        lim = margin * float(L.lif.v_threshold) / float(L.lif.tau)
        tot = tot + (torch.relu(c0 - lim) ** 2).sum()
    return tot


def stream_batch(m, frs, dev, chunk=1000):
    """Stateful run over several whole recordings at once (padded at the end; causal, so padding changes nothing).
    Same PLIF math as ctx.stateful_forward (hard reset, single reset at file start)."""
    import torch
    layers = _layers(m)
    n, Lmax = len(frs), max(len(f) for f in frs)
    V = [None] * len(layers)
    out = np.zeros((n, Lmax), np.float32)
    with torch.no_grad():
        for s in range(0, Lmax, chunk):
            e = min(Lmax, s + chunk)
            xs = np.zeros((n, e - s, frs[0].shape[1]), np.uint8)
            for k, f in enumerate(frs):
                seg = f[s:e]
                xs[k, :len(seg)] = seg
            h = unpack(torch.from_numpy(xs).to(dev))
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
            out[:, s:e] = h.flatten(2)[:, :, 0].cpu().numpy()
    return [out[k, :len(f)] for k, f in enumerate(frs)]


# ----------------------------------------------------------------------------- train
def train(a):
    import torch
    sys.path.insert(0, str(HERE / "tim"))
    import speedpatch  # noqa: F401
    from early_cv import build_model, cap_tau, ema_readout
    dev = "cuda"
    if a.mem_frac < 1:   # share the GPU: hard cap on this process's memory
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
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
                              if e["label"] in ("throw", "feint") and e.get("release_s", e["t"]) * FPS < n],
                         hn=[int(round(e["t"] * FPS)) for e in meta["events"]
                             if e["label"] in ("idle", "invalid") and e["t"] * FPS < n]))
    print("train recs:", [(r["name"], len(r["x"]), len(r["pos"]), len(r["hn"])) for r in recs], flush=True)
    pos_list = [(i, R) for i, r in enumerate(recs) for R in r["pos"]]
    hn_list = [(i, C) for i, r in enumerate(recs) for C in r["hn"]]
    lens = np.array([len(r["x"]) for r in recs], float)
    bank_plan = None
    if a.foreign > 0:
        sys.argv = ["x"]
        import early_cv
        from early_cv import Bank, Planner
        ap = argparse.Namespace(**{**vars(ctx.early_cv_defaults()), "p_pos": 0.0, "w_noah": 0.0, "w_mb": 0.4,
                                   "w_dd": 0.4, "w_dvsg": 0.2, "w_hmdb": 0.0, "p_bed": 0.3, "dnorm": 0,
                                   "noise": 3e-4, "hot": 1, "thin_min": 0.6})
        bank = Bank(a.banks, dev)
        neg = [np.flatnonzero((bank.lab == 0) & (bank.src == s)) for s in range(len(early_cv.SRCS))]
        neg[0] = np.array([], int)
        bank_plan = Planner(bank, ap, np.array([], int), neg, rng)
    T, W = a.T, a.warm
    model = build_model(argparse.Namespace(init=a.init, arch="v1"), dev)
    if a.resume:
        model.load_state_dict(torch.load(Path(a.resume) / f"model_f{a.fold}.pt", map_location=dev, weights_only=False)["model"])
    cap_tau(model, a.tau_max)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))
    groups = [{"params": model.backbone.parameters(), "lr": a.lr, "base": a.lr},
              {"params": model.head.parameters(), "lr": 3 * a.lr, "base": 3 * a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)

    def batch(B, aug=True):
        xs, ts, ws = [], [], []
        for _ in range(B):
            u = rng.random()
            if u < a.p_pos and pos_list:
                i, R = pos_list[rng.integers(len(pos_list))]
                s = R - int(rng.integers(int(0.4 * T), T - 40)) - W
            elif u < a.p_pos + a.p_hn and hn_list:
                i, C = hn_list[rng.integers(len(hn_list))]
                s = C - int(rng.integers(int(0.2 * T), int(0.9 * T))) - W
            else:
                i = int(rng.choice(len(recs), p=lens / lens.sum()))
                s = int(rng.integers(-W, len(recs[i]["x"]) - T))
            r = recs[i]
            s = int(np.clip(s, 0, len(r["x"]) - T - W))
            xs.append(r["x"][s:s + T + W])
            t = r["tgt"][s:s + T + W].copy()
            ww = r["w"][s:s + T + W].copy()
            ww[:W] = 0
            ts.append(t); ws.append(ww)
        x = unpack(torch.stack(xs))
        if aug:
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

    if a.fast:   # fused Triton PLIF loop, checked against the riverops path on a real batch first
        import lif_triton
        xc, _, _ = batch(4)
        chk = lif_triton.check_batch(model, xc)
        print("fast-path check:", chk, flush=True)
        assert chk["spike_mismatch"] == 0 and chk["max_rel_grad_err"] < 1e-2, chk
        fwd = lambda x: lif_triton.fast_forward(model, x)[..., 0]  # noqa: E731
    else:
        fwd = lambda x: model.head(model.backbone(x)).flatten(2)[..., 0]  # noqa: E731

    log, acc, t0 = [], [], time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = a.minutes * 60
    step = 0
    while step < a.steps:
        p = (time.time() - t0) / budget if budget > 0 else step / a.steps
        if p >= 1:
            break
        f_lr = min(1.0, (step + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * f_lr
        nb = int(round(a.bs * a.foreign)) if bank_plan else 0
        x, tgt, w = batch(a.bs - nb)
        if nb:
            xb, tb, wb, _ = bank_plan.batch(nb, T + W)
            wb[:, :W] = 0
            x = torch.cat([x, xb]); tgt = torch.cat([tgt, tb]); w = torch.cat([w, wb])
        sp = fwd(x)
        y = ema_readout(sp, a.alpha)
        loss = (w * (y - tgt) ** 2).sum() / w.sum()
        ton = tonic_excess(model) if a.tonic_w > 0 else torch.zeros((), device=dev)
        opt.zero_grad(set_to_none=True)
        (loss + a.tonic_w * ton).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cap_tau(model, a.tau_max)
        ema.update_parameters(model)
        step += 1
        with torch.no_grad():
            pm, nm = (tgt > 0) & (w > 0), (tgt == 0) & (w > 0)
            acc.append([loss.item(), y[pm].mean().item() if pm.any() else float("nan"), y[nm].mean().item(),
                        (y[nm] > 0.5).float().mean().item(), float(ton)])
        if step % 50 == 0:
            m = np.nanmean(acc, 0)
            log.append({"step": step, "loss": round(m[0], 4), "y_pos": round(m[1], 3), "y_neg": round(m[2], 4),
                        "neg_gt.5": round(m[3], 4), "tonic": round(m[4], 4), "lr_f": round(f_lr, 3),
                        "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
            acc = []
    net = ema.module
    # BN statistics from clean own windows (no augmentation, no foreign data) = the live input distribution
    torch.optim.swa_utils.update_bn((batch(a.bs, aug=False)[0] for _ in range(a.bn_batches)), ema)
    taus = [float(mm.tau) for mm in net.modules() if hasattr(mm, "leak_logit")]
    cfg = {"arch": net.arch, "num_classes": 1, "classes": ["throw"], "readout": "ema", "alpha": a.alpha,
           "frame_ms": 10, "input": [2, 80, 80], "taus": taus, "steps_done": step, "args": vars(a)}
    torch.save({"model": net.state_dict(), "config": cfg}, out / f"model_f{a.fold}.pt")
    (out / f"trainlog_f{a.fold}.json").write_text(json.dumps(log))
    print("saved", out / f"model_f{a.fold}.pt", "steps", step, "tonic after clean BN", ctx.tonic_report(net), flush=True)


# ----------------------------------------------------------------------------- stream
def stream(a):
    import torch
    sys.path.insert(0, str(HERE / "tim"))
    import speedpatch  # noqa: F401
    from snn_model import ThrowSNN, arch_from_config
    dev = "cuda"
    if a.mem_frac < 1:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    lab = labels()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    info = {}
    names = [n for n in lab if rec_key(n) != IDLE_REC]
    frs = [np.load(HERE / "frames" / (Path(n).stem + ".npy")) for n in names]
    g = np.random.default_rng(7)
    noise = np.packbits(g.random((6000, 12800)) < 2e-4, axis=1)
    for f in (0, 1):
        ck = torch.load(Path(a.model_dir) / f"model_f{f}.pt", map_location=dev, weights_only=False)
        cfg = ck["config"]
        m = ThrowSNN(1, arch=arch_from_config(cfg)).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        info[f"tonic_f{f}_before"] = ctx.tonic_report(m)
        if a.clamp:
            ctx.clamp_tonic(m)
        info[f"tonic_f{f}"] = ctx.tonic_report(m)
        outs = stream_batch(m, frs + [noise], dev, chunk=a.chunk)
        info[f"drift_out_spikes_f{f}"] = int(outs[-1].sum())
        if f == 0:
            k = [rec_key(n) for n in names].index("201703")
            with torch.no_grad():
                ref = m(unpack(torch.from_numpy(frs[k]).to(dev))[None])[0, :, 0].float().cpu().numpy()
            info["sanity_mismatch_frames"] = int((outs[k] != ref).sum())
            info["sanity_frames"] = len(ref)
        for n, o in zip(names, outs[:-1]):
            np.savez_compressed(Path(a.out) / f"f{f}_{rec_key(n)}.npz", spikes=o, alpha=float(cfg.get("alpha", 0.85)))
        print(a.model_dir, f, "spikes", int(sum(o.sum() for o in outs[:-1])), flush=True)
    print(json.dumps(info), flush=True)
    (Path(a.out) / "info.json").write_text(json.dumps(info))


# ----------------------------------------------------------------------------- score
TOP = [0.97, 0.98, 0.99, 0.993, 0.996, 0.998, 0.999]
THETAS = {"abs": np.round(np.r_[np.arange(0.3, 0.96, 0.05), TOP], 3),
          "slow": np.round(np.r_[np.arange(0.3, 0.96, 0.05), TOP], 3),
          "rel": np.round(np.arange(0.1, 0.91, 0.05), 3),
          "cnt": np.round(np.arange(3, 30) / 30, 4)}


def readout(sp, alpha, mode):
    if mode == "cnt":   # output spikes in the last 300 ms / 30
        c = np.concatenate([[0.0], np.cumsum(sp, dtype=np.float64)])
        k = np.arange(1, len(sp) + 1)
        return ((c[k] - c[np.maximum(k - 30, 0)]) / 30).astype(np.float32)
    return ctx.readout(sp, alpha, mode)


def chance_hits(items, n, seed=0):
    """items: list of (y, meta, theta). Same alarms, labels circularly shifted per recording and draw."""
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(n):
        h = 0
        for y, m, th in items:
            end = min(len(y) / FPS, m["end_s"])
            u = rng.uniform(5.0, end - 5.0)
            ev = [{k: v for k, v in e.items() if k != "release_s"} | {"t": (e["t"] + u) % end}
                  for e in m["events"] if e["label"] != "end"]
            h += sum(x["hit"] for x in ctx.score_rec(y, dict(m, events=ev), th)["hits"])
        hits.append(h)
    return {"mean": float(np.mean(hits)), "p95": float(np.percentile(hits, 95)), "n": n}


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
        te_all = [(f, r) for f in (0, 1) for r in lab if fold_of(r) == f and rec_key(r) != "215305"]
        for mode in a.modes:
            Y = {k: readout(sp, al, mode) for k, (sp, al) in ys.items()}
            chosen = {}
            for f in (0, 1):   # pre-registered rule: theta on the fold's TRAINING sessions (02/03)
                tr = [r for r in lab if fold_of(r) not in (None, f) and rec_key(r) != "215305"]
                best = None
                for th in THETAS[mode]:
                    ag = ctx.aggregate([ctx.score_rec(Y[(f, r)], lab[r], th) for r in tr])
                    ok = ag["fa_per_min"] <= a.fa_target
                    key = (ok, ag["recall"] if ok else -ag["fa_per_min"], th)
                    if best is None or key > best[0]:
                        best = (key, th, ag)
                chosen[f] = float(best[1])
            # second rule, out-of-sample: theta for fold f = the same rule applied to the OTHER fold model's
            # held-out outputs on fold f's training sessions (labels of fold f's training sessions only; the
            # in-sample rule above sees outputs the model was trained on and drifts to the top of the grid).
            # Caveat: model 1-f was trained on fold f's test sessions, so this is a recipe-level calibration.
            chosen_x = {}
            for f in (0, 1):
                tr = [r for r in lab if fold_of(r) == 1 - f and rec_key(r) != "215305"]
                best = None
                for th in THETAS[mode]:
                    ag = ctx.aggregate([ctx.score_rec(Y[(1 - f, r)], lab[r], th) for r in tr])
                    ok = ag["fa_per_min"] <= a.fa_target
                    key = (ok, ag["recall"] if ok else -ag["fa_per_min"], th)
                    if best is None or key > best[0]:
                        best = (key, th, ag)
                chosen_x[f] = float(best[1])
            test_x = [ctx.score_rec(Y[(f, r)], lab[r], chosen_x[f]) for f, r in te_all]
            chance_x = chance_hits([(Y[k], lab[k[1]], chosen_x[k[0]]) for k in te_all], a.n_chance_q)
            test = [ctx.score_rec(Y[(f, r)], lab[r], chosen[f]) for f, r in te_all]
            t215 = [ctx.score_rec(Y[(f, r)], lab[r], chosen[f]) for f in (0, 1) for r in lab
                    if fold_of(r) == f and rec_key(r) == "215305"]
            chance = chance_hits([(Y[k], lab[k[1]], chosen[k[0]]) for k in te_all], a.n_chance)
            sweep = {}
            for th in THETAS[mode]:
                ag = ctx.aggregate([ctx.score_rec(Y[k], lab[k[1]], th) for k in te_all])
                sweep[float(th)] = {k: ag[k] for k in ("recall", "fa_per_min", "hn_per_min", "near_fa", "lead_rel_med", "all")}
            quality = {}
            for B in (1.0, 2.0, 3.0):   # theta-free model comparison (theta on TEST: not an operating point)
                ok = [(v["recall"], th) for th, v in sweep.items() if v["fa_per_min"] <= B]
                if not ok:
                    continue
                rc, th = max(ok)
                ch = chance_hits([(Y[k], lab[k[1]], th) for k in te_all], a.n_chance_q)
                quality[B] = {"theta": th, "hits": sweep[th]["all"][0], "chance": ch["mean"], "chance_p95": ch["p95"],
                              "above_chance": sweep[th]["all"][0] - ch["mean"], "fa_per_min": sweep[th]["fa_per_min"],
                              "hn_per_min": sweep[th]["hn_per_min"], "lead_rel_med": sweep[th]["lead_rel_med"]}
            res[name][mode] = {"theta": chosen, "test_0203": ctx.aggregate(test), "chance": chance,
                               "test_215305": ctx.aggregate(t215) if t215 else None, "sweep_test": sweep,
                               "quality": quality, "theta_xfold": chosen_x, "test_0203_xfold": ctx.aggregate(test_x),
                               "chance_xfold": chance_x}
            tx = res[name][mode]["test_0203_xfold"]
            print(f"{name:18s} {mode:4s} XFOLD th={chosen_x[0]:.3f}/{chosen_x[1]:.3f} | all {tx['all'][0]}/{tx['all'][1]}"
                  f" chance {chance_x['mean']:.0f} -> {tx['all'][0] - chance_x['mean']:+.0f} | pauseFA {tx['fa_per_min']:.2f}/min"
                  f" (<= {tx['fa_up95']:.1f}) hnFA {tx['hn_per_min']:.1f}/min | lead rel {tx['lead_rel_med']} (n={tx['n_lead_rel']})", flush=True)
            t = res[name][mode]["test_0203"]
            q = " ".join(f"FA<={int(B)}: {v['hits']}-{v['chance']:.0f}={v['above_chance']:+.0f} (th {v['theta']}, lead {v['lead_rel_med']})"
                         for B, v in quality.items())
            print(f"{name:18s} {mode:4s} th={chosen[0]:.3f}/{chosen[1]:.3f} | all {t['all'][0]}/{t['all'][1]}"
                  f" chance {chance['mean']:.0f} -> {t['all'][0] - chance['mean']:+.0f} | pauseFA {t['fa_per_min']:.2f}/min"
                  f" hnFA {t['hn_per_min']:.1f}/min | lead rel {t['lead_rel_med']} (n={t['n_lead_rel']})"
                  f" || QUALITY {q}", flush=True)
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=100000)
    t.add_argument("--minutes", type=float, default=20)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--T", type=int, default=300)
    t.add_argument("--warm", type=int, default=60)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--alpha", type=float, default=0.85)
    t.add_argument("--onset", type=int, default=500)
    t.add_argument("--p-pos", type=float, default=0.4)
    t.add_argument("--p-hn", type=float, default=0.2)
    t.add_argument("--hn-w", type=float, default=2.0)
    t.add_argument("--pos-w", type=float, default=1.5)
    t.add_argument("--early-w", type=float, default=2.0)
    t.add_argument("--tonic-w", type=float, default=1.0)
    t.add_argument("--tau-max", type=float, default=8.0)
    t.add_argument("--bn-batches", type=int, default=20)
    t.add_argument("--fast", type=int, default=1, help="fused Triton PLIF loop for training (lif_triton.py)")
    t.add_argument("--mem-frac", type=float, default=1.0, help="cap this process at this fraction of GPU memory")
    t.add_argument("--resume", default="")
    t.add_argument("--shift", type=int, default=10)
    t.add_argument("--thin-min", type=float, default=0.6)
    t.add_argument("--use-idle", type=int, default=1)
    t.add_argument("--foreign", type=float, default=0.25)
    t.add_argument("--banks", default=str(Path.home() / "early/banks"))
    t.add_argument("--init", default=str(Path.home() / "snn_pretrain/runs/sweep_v1/best.pt"))
    s = sub.add_parser("stream")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--clamp", type=int, default=0)
    s.add_argument("--chunk", type=int, default=1000, help="frames per streaming chunk (lower for big archs)")
    s.add_argument("--mem-frac", type=float, default=1.0)
    c = sub.add_parser("score")
    c.add_argument("--streams", nargs="+", required=True)
    c.add_argument("--modes", nargs="+", default=["abs", "rel", "slow", "cnt"])
    c.add_argument("--fa-target", type=float, default=1.0)
    c.add_argument("--n-chance", type=int, default=100)
    c.add_argument("--n-chance-q", type=int, default=30)
    c.add_argument("--json", default=str(HERE / "score2.json"))
    a = ap.parse_args()
    {"train": train, "stream": stream, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Evals 2-4 on a trained throw model: does synthetic data look / act like real data *for the SNN*?

Run on checkpoints of the synthetic-only run (tstr_*), so that both the real windows (all 38) and the
synthetic test clips (150, fixed split) are unseen by the model.

  2  activity: firing rate per layer, dead / saturated neurons, real vs synthetic
  3  domain separability on SNN features (per-channel spike counts per layer, log1p):
       C2ST  logistic regression real vs class-matched synthetic, grouped CV -> AUC (+ permutation null)
       MMD   RBF, median heuristic, permutation p-value, with a real-vs-real split-half baseline
  4  gradients: per-layer cosine between mean gradients
       baseline  real half vs other real half (split by recording)
       test      real half vs class-matched synthetic sample of the same size
     plus per-sample cosine of each synthetic clip with the mean real gradient

usage: python eval_model.py ../runs_throw/tstr_s1050/{step0,step100,step300,final}.pt --out ../runs_throw/eval_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import speedpatch  # noqa: F401
from model import ThrowSNN, arch_from_config, rate_readout
from train_throw import Windows, synth_split

K, T = 3, 120


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    m = ThrowSNN(ck["config"]["num_classes"], arch=arch_from_config(ck["config"])).to(dev)
    m.load_state_dict(ck["model"])
    return m.eval()


def layers_of(model):
    return list(model.backbone) + [model.head]


@torch.no_grad()
def activity(model, ds, idx, dev, bs=8):
    """Per window: mean firing rate per layer; per-channel spike counts per layer; per-neuron rates."""
    store = {}
    hooks = [l.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(k, o))
             for k, l in enumerate(layers_of(model))]
    L = len(hooks)
    rate, chan, neuron = [[] for _ in range(L)], [[] for _ in range(L)], [None] * L
    for s in range(0, len(idx), bs):
        model(ds.batch([(int(i), 0) for i in idx[s:s + bs]], T).to(dev).float())
        for k in range(L):
            o = store[k].float()
            if o.dim() == 3:
                o = o[..., None, None]
            rate[k].append(o.mean((1, 2, 3, 4)).cpu())
            chan[k].append(o.sum((1, 3, 4)).cpu())
            nr = o.mean(1)                                     # [B, C, H, W]
            neuron[k] = nr.sum(0) if neuron[k] is None else neuron[k] + nr.sum(0)
    for h in hooks:
        h.remove()
    rate = [torch.cat(r).numpy() for r in rate]
    chan = [torch.log1p(torch.cat(c)).numpy() for c in chan]
    neuron = [(n / len(idx)).flatten().cpu().numpy() for n in neuron]
    return rate, chan, neuron


def matched(ys_pool, y_target, rng):
    """Synthetic indices with the same class counts as y_target."""
    return np.concatenate([rng.choice(np.flatnonzero(ys_pool == c), int((y_target == c).sum()), replace=False)
                           for c in range(K)])


def c2st(Xr, Xs, yr, ys, groups_r, rng, reps=40):
    aucs, null = [], []
    for rep in range(reps):
        j = matched(ys, yr, rng)
        X = np.vstack([Xr, Xs[j]])
        lab = np.r_[np.ones(len(Xr)), np.zeros(len(j))]
        grp = np.r_[groups_r, [f"s{i}" for i in j]]
        for target, store in ((lab, aucs), (rng.permutation(lab), null)):
            p = np.zeros(len(X))
            for tr, te in GroupKFold(5).split(X, target, grp):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                clf.fit(X[tr], target[tr])
                p[te] = clf.predict_proba(X[te])[:, 1]
            store.append(roc_auc_score(target, p))
    return {"auc_median": float(np.median(aucs)), "auc_iqr": [float(np.percentile(aucs, 25)), float(np.percentile(aucs, 75))],
            "null_auc_p95": float(np.percentile(null, 95)), "frac_above_null_p95": float((np.array(aucs) > np.percentile(null, 95)).mean())}


def mmd2(X, Y, gamma):
    def k(a, b):
        d = ((a[:, None] - b[None]) ** 2).sum(-1)
        return np.exp(-gamma * d)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
            - 2 * kxy.mean())


def mmd_test(Xr, Xs, yr, ys, parents, rng, perms=300, reps=20):
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-6
    Xr, Xs = (Xr - mu) / sd, (Xs - mu) / sd
    d = ((Xr[:, None] - Xr[None]) ** 2).sum(-1)
    gamma = 1.0 / np.median(d[d > 0])
    obs, pvals = [], []
    for _ in range(reps):
        Y = Xs[matched(ys, yr, rng)]
        o = mmd2(Xr, Y, gamma)
        pool = np.vstack([Xr, Y])
        nul = []
        for _ in range(perms):
            q = rng.permutation(len(pool))
            nul.append(mmd2(pool[q[:len(Xr)]], pool[q[len(Xr):]], gamma))
        obs.append(o)
        pvals.append(float((np.array(nul) >= o).mean()))
    up = np.unique(parents)
    base = []
    for _ in range(100):
        h = set(rng.permutation(up)[:len(up) // 2])
        m = np.array([p in h for p in parents])
        base.append(mmd2(Xr[m], Xr[~m], gamma))
    return {"mmd2_real_synth": float(np.median(obs)), "p_median": float(np.median(pvals)),
            "mmd2_real_real_halves": float(np.median(base)),
            "note": "real-vs-real uses halves (~19 vs 19); real-vs-synth uses 38 vs 38"}


def per_sample_grads(model, ds, idx, dev):
    """{layer: [N, P_layer]} per-sample gradients (eval-mode BN, MSE-on-rate loss as in training)."""
    names = [n for n, _ in model.named_parameters()]
    group = {n: ".".join(n.split(".")[:2]) if n.startswith("backbone") else "head" for n in names}
    layers = list(dict.fromkeys(group.values()))
    out = {L: [] for L in layers}
    for i in idx:
        x = ds.batch([(int(i), 0)], T).to(dev).float()
        y = torch.tensor([int(ds.labels[i])], device=dev)
        model.zero_grad(set_to_none=True)
        F.mse_loss(rate_readout(model(x)), F.one_hot(y, K).float()).backward()
        per = {L: [] for L in layers}
        for n, p in model.named_parameters():
            per[group[n]].append(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=dev))
        for L in layers:
            out[L].append(torch.cat(per[L]).detach())
    return {L: torch.stack(v) for L, v in out.items()}


def cosines(Gr, Gs, yr, ys, parents, rng, reps=60):
    base_layers = list(Gr)
    layers = base_layers + ["all"]
    cat = lambda G: torch.cat([G[L] for L in base_layers], 1)
    Gr = {**Gr, "all": cat(Gr)}
    Gs = {**Gs, "all": cat(Gs)}
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
    up = np.unique(parents)
    base = {L: [] for L in layers}
    test = {L: [] for L in layers}
    for _ in range(reps):
        h = set(rng.permutation(up)[:len(up) // 2])
        m = np.array([p in h for p in parents])
        a, b = np.flatnonzero(m), np.flatnonzero(~m)
        j = matched(ys, yr[b], rng)
        for L in layers:
            base[L].append(cos(Gr[L][a].mean(0), Gr[L][b].mean(0)))
            test[L].append(cos(Gr[L][a].mean(0), Gs[L][j].mean(0)))
    res = {L: {"real_real_median": float(np.median(base[L])), "real_real_iqr": np.percentile(base[L], [25, 75]).tolist(),
               "real_synth_median": float(np.median(test[L])), "real_synth_iqr": np.percentile(test[L], [25, 75]).tolist()}
           for L in layers}
    # per-sample: each synthetic clip vs the mean real gradient OF ITS CLASS; real clips leave-one-recording-out
    g = Gr["all"]
    ps_s, ps_r = [], []
    for c in range(K):
        rc = np.flatnonzero(yr == c)
        gR = g[rc].mean(0)
        ps_s += [(int(i), cos(Gs["all"][i], gR)) for i in np.flatnonzero(ys == c)]
        for i in rc:
            others = [k for k in rc if parents[k] != parents[i]]
            if others:
                ps_r.append(cos(g[i], g[others].mean(0)))
    # per class (removes the effect of which class dominates the loss) + gradient norms
    res["per_class_all"] = {}
    for c in range(K):
        rc = np.flatnonzero(yr == c)
        sc = np.flatnonzero(ys == c)
        bb, tt = [], []
        for _ in range(reps):
            pc = np.unique(parents[rc])
            h = set(rng.permutation(pc)[:len(pc) // 2])
            m = np.array([parents[k] in h for k in rc])
            a, b = rc[m], rc[~m]
            if len(a) == 0 or len(b) == 0:
                continue
            bb.append(cos(g[a].mean(0), g[b].mean(0)))
            tt.append(cos(g[a].mean(0), Gs["all"][rng.choice(sc, len(b), replace=False)].mean(0)))
        res["per_class_all"][str(c)] = {
            "real_real_median": float(np.median(bb)), "real_synth_median": float(np.median(tt)),
            "grad_norm_real": float(g[rc].norm(dim=1).median()), "grad_norm_synth": float(Gs["all"][sc].norm(dim=1).median())}
    s_vals = np.array([v for _, v in ps_s])
    res["per_sample"] = {"synth_median": float(np.median(s_vals)), "synth_frac_negative": float((s_vals < 0).mean()),
                         "real_loo_median": float(np.median(ps_r)), "real_loo_frac_negative": float((np.array(ps_r) < 0).mean())}
    return res, ps_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-grads", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    real, synth = Windows(Path(args.data) / "throw_real.npz"), Windows(Path(args.data) / "throw_synth.npz")
    s_test, _ = synth_split(synth)
    r_idx = np.arange(len(real))
    yr, ys = real.labels[r_idx], synth.labels[s_test]
    parents = real.parent[r_idx]
    out = {}
    for path in args.ckpts:
        rng = np.random.default_rng(0)
        name = Path(path).stem
        model = load_model(path, dev)
        ra, ca, na = activity(model, real, r_idx, dev)
        sa, csa, nsa = activity(model, synth, s_test, dev)
        L = len(ra)
        r = {"layers": [f"backbone.{k}" for k in range(L - 1)] + ["head"]}
        r["firing_rate"] = [{"real": float(ra[k].mean()), "synth": float(sa[k].mean()),
                             "real_by_class": [float(ra[k][yr == c].mean()) for c in range(K)],
                             "synth_by_class": [float(sa[k][ys == c].mean()) for c in range(K)]} for k in range(L)]
        # dead / saturated neurons with matched sample sizes (38 real vs 38 class-matched synth, avg over draws)
        dead_s, sat_s = [], []
        for _ in range(10):
            j = matched(ys, yr, rng)
            _, _, nj = activity(model, synth, s_test[j], dev)
            dead_s.append([float((n == 0).mean()) for n in nj])
            sat_s.append([float((n > 0.5).mean()) for n in nj])
        r["dead_frac"] = {"real": [float((n == 0).mean()) for n in na], "synth": np.mean(dead_s, 0).tolist()}
        r["saturated_frac"] = {"real": [float((n > 0.5).mean()) for n in na], "synth": np.mean(sat_s, 0).tolist()}
        feats = {f"L{k}": (ca[k], csa[k]) for k in range(L)}
        feats["all"] = (np.hstack(ca), np.hstack(csa))
        r["c2st"] = {k: c2st(a, b, yr, ys, parents, rng) for k, (a, b) in feats.items()}
        r["mmd"] = {k: mmd_test(feats[k][0], feats[k][1], yr, ys, parents, rng) for k in (f"L{L - 2}", "all")}
        if not args.no_grads:
            Gr = per_sample_grads(model, real, r_idx, dev)
            Gs = per_sample_grads(model, synth, s_test, dev)
            r["grad_cos"], ps = cosines(Gr, Gs, yr, ys, parents, rng)
            worst = sorted(ps, key=lambda t: t[1])[:10]
            r["grad_worst_synth"] = [{"file": str(synth.file[s_test[i]]), "cos": round(v, 3)} for i, v in worst]
            del Gr, Gs
            torch.cuda.empty_cache()
        out[name] = r
        print(f"== {name}", flush=True)
        print(" firing rate real/synth:", [(round(f["real"], 4), round(f["synth"], 4)) for f in r["firing_rate"]], flush=True)
        print(" dead real/synth:", [round(x, 3) for x in r["dead_frac"]["real"]], [round(x, 3) for x in r["dead_frac"]["synth"]])
        print(" C2ST AUC:", {k: round(v["auc_median"], 3) for k, v in r["c2st"].items()},
              "null p95 ~", round(r["c2st"]["all"]["null_auc_p95"], 3), flush=True)
        print(" MMD:", {k: (round(v["mmd2_real_synth"], 4), round(v["mmd2_real_real_halves"], 4), v["p_median"])
                        for k, v in r["mmd"].items()}, flush=True)
        if "grad_cos" in r:
            print(" grad cos (real-real | real-synth):",
                  {k: (round(v["real_real_median"], 2), round(v["real_synth_median"], 2))
                   for k, v in r["grad_cos"].items() if k not in ("per_sample", "per_class_all")}, flush=True)
            print(" per-sample:", r["grad_cos"]["per_sample"], flush=True)
            print(" per-class (real-real, real-synth, |g| real, |g| synth):", {c: tuple(round(x, 3) for x in v.values()) for c, v in r["grad_cos"]["per_class_all"].items()}, flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

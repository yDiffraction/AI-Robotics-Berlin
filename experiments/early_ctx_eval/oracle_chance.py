# chance baseline for a FIXED theta (report-only oracle point)
import sys, numpy as np
from pathlib import Path
import ctx
mdir, mode, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
lab = ctx.labels()
te = [(f, r) for f in (0, 1) for r in lab if ctx.fold_of(r) == f and ctx.rec_key(r) != "215305"]
Y = {}
for f, r in te:
    z = np.load(Path(mdir) / f"f{f}_{ctx.rec_key(r)}.npz"); Y[(f, r)] = ctx.readout(z["spikes"], float(z["alpha"]), mode)
ag = ctx.aggregate([ctx.score_rec(Y[k], lab[k[1]], th) for k in te])
rng = np.random.default_rng(0); hits = []
for _ in range(100):
    h = 0
    for f, r in te:
        m = lab[r]; y = Y[(f, r)]; end = min(len(y) / 100, m["end_s"]); u = rng.uniform(5, end - 5)
        ev = [{k: v for k, v in e.items() if k != "release_s"} | {"t": (e["t"] + u) % end} for e in m["events"] if e["label"] != "end"]
        h += sum(x["hit"] for x in ctx.score_rec(y, dict(m, events=ev), th)["hits"])
    hits.append(h)
print(mdir, mode, th, ag["all"], "pauseFA %.2f hn %.2f near %d lead %s | chance %.1f p95 %.0f" % (ag["fa_per_min"], ag["hn_per_min"], ag["near_fa"], ag["lead_rel_med"], np.mean(hits), np.percentile(hits, 95)))

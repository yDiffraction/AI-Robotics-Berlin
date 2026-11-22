import sys, numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]
import ctx
from pathlib import Path
lab=ctx.labels(); mdir=sys.argv[1]; mode=sys.argv[2]
for f in (0,1):
    tr=[r for r in lab if ctx.fold_of(r) not in (None,f) and ctx.rec_key(r)!="215305"]
    Y={r: ctx.readout(*(lambda z:(z["spikes"],float(z["alpha"])))(np.load(Path(mdir)/f"f{f}_{ctx.rec_key(r)}.npz")),mode) for r in tr}
    for th in ctx.THETAS[mode][::2]:
        ag=ctx.aggregate([ctx.score_rec(Y[r],lab[r],th) for r in tr])
        print(f, th, "rec %.2f fa %.2f (%d in %.1f min) hn %.2f"%(ag["recall"],ag["fa_per_min"],ag["pause_fa"],ag["pause_min"],ag["hn_per_min"]))

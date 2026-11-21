#!/usr/bin/env python3
"""Quality report for an imported external event dataset (e.g. tools/dailydvs_import.py output).

For every group / action it reports clip count, event rate, ON fraction, peak prominence (rate in
the 60 ms around the anchor / median rate) and recording attributes, and compares them with our
own GenX320 recordings (real_train/val/test). It then runs two probes on the shortcut_probe
features (logistic regression, grouped by person):
  realism       external <label> vs real <label>: ~50 % = indistinguishable from our camera
  separability  external throw vs external arm_no_release: high = the clips carry the throw cue
and writes a preview reel per group.

  uv run tools/external_report.py --ext dataset/external/dailydvs --real dataset/build --out report.md
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import events_tool as ET  # noqa: E402
from shortcut_probe import features, logreg  # noqa: E402


def stats(ev, anchor):
    t = ev["timestamp"].astype(np.int64)
    dur = max(int(t[-1] - t[0]), 1)
    h = np.histogram(t, np.arange(t[0], t[-1] + 20_000, 20_000))[0]
    a = (anchor - int(t[0])) // 20_000
    peak = h[max(a - 1, 0):a + 2].mean() if anchor is not None else h.max()
    return {"rate_k": len(ev) / dur * 1e3, "on": float((ev["polarity"] == 1).mean()),
            "prom": float(peak / (np.median(h) + 1))}


def probe(Xa, ga, Xb, gb, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    groups = np.array(list(ga) + list(gb))
    uniq = sorted(set(groups)); rng.shuffle(uniq)
    fold = np.array([{g: i % folds for i, g in enumerate(uniq)}[g] for g in groups])
    pred = np.full(len(y), np.nan)
    for k in range(folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        w, b = logreg((X[tr] - mu) / sd, y[tr])
        pred[te] = ((X[te] - mu) / sd @ w + b) > 0
    ok = ~np.isnan(pred)
    return ((pred[ok & (y == 1)] == 1).mean() + (pred[ok & (y == 0)] == 0).mean()) / 2


def load_rows(folder, manifest, sub_of):
    out = []
    for r in csv.DictReader(open(manifest)):
        p = Path(folder) / sub_of(r) / r["file"]
        if not p.exists():
            continue
        ev = ET.load(p)
        if len(ev) < 100:
            continue
        anchor = int(r["release_us"]) if r.get("release_us") not in ("", None) else None
        f_anchor = anchor if anchor is not None else int(ev["timestamp"][0] + ev["timestamp"][-1]) // 2
        out.append({**r, **stats(ev, anchor), "feat": features(ev, f_anchor), "path": str(p)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ext", required=True, help="imported folder with <group>/ sub-folders and manifest.csv")
    ap.add_argument("--real", default="dataset/build")
    ap.add_argument("--out", default=None, help="markdown report (default: <ext>/REPORT.md)")
    ap.add_argument("--reel", type=int, default=12, help="clips per group in the preview reel (0 = none)")
    args = ap.parse_args()
    ext = Path(args.ext)
    rows = load_rows(ext, ext / "manifest.csv", lambda r: r["group"])
    real = [r for r in load_rows(args.real, Path(args.real) / "manifest.csv", lambda r: r["folder"])
            if r["folder"].startswith("real_")]
    L = []
    L.append(f"# Quality report: {ext.name}\n")
    L.append(f"{len(rows)} clips imported. Reference: {len(real)} clips of our own GenX320 recordings.\n")

    def med(xs, k):
        v = [x[k] for x in xs]
        return f"{np.median(v):.0f}" if k == "rate_k" else f"{np.median(v):.2f}"

    L.append("## Per action\n")
    L.append("| group | action | label | clips | persons | k ev/s (median) | ON frac | peak prominence | frontal | near | camera moves |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        by[(r["group"], r["action_name"])].append(r)
    for (g, a), xs in sorted(by.items()):
        frac = lambda k, v: f"{np.mean([x[k] == v for x in xs]):.0%}"
        L.append(f"| {g} | {a} | {xs[0]['label']} | {len(xs)} | {len({x['person'] for x in xs})} | {med(xs, 'rate_k')} | "
                 f"{med(xs, 'on')} | {med(xs, 'prom')} | {frac('perspective', 'Frontal')} | {frac('distance', 'Near')} | "
                 f"{frac('camera_move', '1')} |")
    L.append("\n**Our GenX320 recordings for comparison:**\n")
    L.append("| label | clips | k ev/s (median) | ON frac | peak prominence |")
    L.append("|---|---|---|---|---|")
    for lab in ("throw", "feint", "idle"):
        xs = [r for r in real if r["label"] == lab]
        if xs:
            L.append(f"| {lab} | {len(xs)} | {med(xs, 'rate_k')} | {med(xs, 'on')} | {med(xs, 'prom')} |")

    L.append("\n## Probes (logistic regression on shortcut_probe features, grouped by person)\n")
    L.append("Realism: 50 % = a simple classifier cannot tell the clip from our camera. Separability: "
             "high = throw and arm_no_release differ in timing/shape, i.e. the clips carry the cue.\n")
    L.append("| probe | A | B | n A | n B | balanced accuracy |")
    L.append("|---|---|---|---|---|---|")

    def P(name, A, B):
        if len(A) < 5 or len(B) < 5:
            return
        acc = probe(np.array([x["feat"] for x in A]), [x["person"] or x["file"] for x in A],
                    np.array([x["feat"] for x in B]), [x.get("parent") or x["file"] for x in B])
        L.append(f"| {name} | {A[0].get('group', A[0]['label'])} | {B[0].get('group', B[0]['label'])} | {len(A)} | {len(B)} | {acc:.1%} |")

    for g, lab in (("throw", "throw"), ("toss", "throw"), ("arm_no_release", "feint"), ("idle", "idle")):
        P("realism", [r for r in rows if r["group"] == g], [r for r in real if r["label"] == lab])
    fr = [r for r in rows if r["perspective"] == "Frontal"]
    P("realism (frontal only)", [r for r in fr if r["group"] == "throw"], [r for r in real if r["label"] == "throw"])
    for g in ("throw", "toss"):
        A = [r for r in rows if r["group"] == g]
        B = [r for r in rows if r["group"] == "arm_no_release"]
        if A and B:
            Xa = np.array([x["feat"] for x in A]); Xb = np.array([x["feat"] for x in B])
            acc = probe(Xa, [x["person"] for x in A], Xb, [x["person"] for x in B])
            L.append(f"| separability | {g} | arm_no_release | {len(A)} | {len(B)} | {acc:.1%} |")
    A = [r for r in rows if r["group"] == "throw" and r["action_name"] == "throwing_ball"]
    B = [r for r in rows if r["group"] == "arm_no_release" and r["action_name"] in ("swing_badminton_racket", "punch", "hit_people_with_things")]
    if A and B:
        acc = probe(np.array([x["feat"] for x in A]), [x["person"] for x in A],
                    np.array([x["feat"] for x in B]), [x["person"] for x in B])
        L.append(f"| separability | throwing_ball | racket/punch/hit | {len(A)} | {len(B)} | {acc:.1%} |")

    if args.reel:
        import events_video
        rng = np.random.default_rng(0)
        L.append("\n## Preview reels\n")
        for g in sorted({r["group"] for r in rows}):
            xs = [r for r in rows if r["group"] == g]
            pick = [xs[i] for i in rng.choice(len(xs), min(args.reel, len(xs)), replace=False)]
            path = ext / "videos" / f"reel_{g}.mp4"
            path.parent.mkdir(exist_ok=True)
            events_video.write_reel([x["path"] for x in pick], path, duration_s=1.2,
                                    releases={Path(x["path"]).name: x["release_us"] for x in pick})
            L.append(f"- `{path.relative_to(ext)}`")
    out = Path(args.out) if args.out else ext / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

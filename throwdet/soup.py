#!/usr/bin/env python3
"""Weight averaging ("model soup") of runs that share the same configuration and pretrained init.

All runs start from the same DVS-Gesture backbone, so their fine-tuned weights can be averaged
into ONE network of the same size (it still fits the board). Averaging reduces the large
seed-to-seed variance of single runs.

  # per fold, to measure it out of fold (then run eval_readout.py on the output dir):
  python throwdet/soup.py --runs e10_nodens_negheavy,e20_neg_seed1 --out e10_soup2
  # final models (trained with --folds 0):
  python throwdet/soup.py --runs f_a,f_b,f_c --out f_soup --final
"""
import argparse
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(Path.home() / "throwdet/runs"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    runs = [root / r for r in args.runs.split(",")]
    out = root / args.out
    out.mkdir(exist_ok=True)
    names = ["final"] if args.final else [f"fold{k}" for k in range(args.folds)]
    for name in names:
        cks = [torch.load(r / f"{name}.pt", map_location="cpu", weights_only=False) for r in runs]
        avg = {}
        for k, v in cks[0]["model"].items():
            if v.is_floating_point():
                avg[k] = sum(c["model"][k] for c in cks) / len(cks)
            else:
                avg[k] = v
        ck = dict(cks[0])
        ck["model"] = avg
        ck["soup_of"] = [r.name for r in runs]
        torch.save(ck, out / f"{name}.pt")
    # eval_readout.py takes the training arguments from the first log line
    first = open(runs[0] / "log.txt").readline()
    (out / "log.txt").write_text(first.replace(f"== {runs[0].name} ", f"== {args.out} ", 1)
                                 + f"soup of {[r.name for r in runs]}\nOWN out-of-fold: (soup, see readout.txt)\n")
    print(f"wrote {out}: {names} averaged over {len(runs)} runs")


if __name__ == "__main__":
    main()

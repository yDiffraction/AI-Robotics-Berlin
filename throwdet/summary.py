#!/usr/bin/env python3
"""One table over all evaluated runs (readout_sweep.json): clip AUC on our own camera (out of fold)
and motion recall at false-alarm budgets of 0.5 / 1 / 2 / 4 alarms per minute on the idle stream."""
import json
from pathlib import Path

rows = []
for p in sorted(Path.home().glob("throwdet/runs/*/readout_sweep.json")):
    d = json.load(open(p))
    if isinstance(d, list):
        continue
    by = {r["budget_fa_per_min"]: r for r in d["rows"]}
    rows.append((p.parent.name, d["auc"], by))
rows.sort(key=lambda r: -(r[2].get(2.0, {}).get("recall", 0)))
print("| run | AUC own (OOF) | recall @0.5 FA/min | recall @1 | recall @2 | recall @4 | readout @2 (alpha, hold, thr) |")
print("|---|---|---|---|---|---|---|")
for tag, auc, by in rows:
    rc = lambda b: f"{by[b]['recall']:.2f}" if b in by else "-"
    r2 = by.get(2.0)
    ro = f"{r2['alpha']}, {r2['hold']}, {r2['thr']:.2f}" if r2 else "-"
    print(f"| {tag} | {auc:.3f} | {rc(0.5)} | {rc(1.0)} | {rc(2.0)} | {rc(4.0)} | {ro} |")

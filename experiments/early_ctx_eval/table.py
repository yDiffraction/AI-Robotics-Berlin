#!/usr/bin/env python3
"""score*.json -> markdown table rows (held-out whole recordings 02/03, theta chosen on training sessions)."""
import json, sys
rows = []
for f in sys.argv[1:]:
    r = json.load(open(f))
    for m, v in r.items():
        for mode, d in v.items():
            t, s = d["test_0203"], d["test_215305"]
            th = d["theta"]
            lr = t["lead_rel_med"]
            rows.append(f"| {m} | {mode} | {th['0']:.2f}/{th['1']:.2f} | {t['throw'][0]}/{t['throw'][1]} | {t['feint'][0]}/{t['feint'][1]} | "
                        f"**{t['all'][0]}/{t['all'][1]} ({100*t['recall']:.0f} %)** | {d['chance']['mean']:.0f} (p95 {d['chance']['p95']:.0f}) | {t['pause_fa']} / {t['pause_min']:.1f} min = "
                        f"**{t['fa_per_min']:.2f}** (≤{t['fa_up95']:.1f}) | {t['hn_fa']} / {t['hn_min']:.1f} min = {t['hn_per_min']:.2f} | "
                        f"{t['near_fa']} | {t['nonhit_per_min']:.2f} | {'%+d' % lr if lr is not None else '–'} (n={t['n_lead_rel']}) | "
                        f"{'%+d' % t['lead_peak_med'] if t['lead_peak_med'] is not None else '–'} | {t['pre_active']} | "
                        f"{s['all'][0]}/{s['all'][1]}, FA {s['fa_per_min']:.2f}/min |")
print("| Modell | Readout | θ f0/f1 | Würfe | Täuschungen | gesamt | Zufall (Mittel, p95) | Pausen-FA (n/Zeit = /min, ≤ 95 %) | Hard-Neg-FA (n/Zeit = /min) | FA nahe Schwung | alle Nicht-Treffer-Alarme /min | Lead vs Release ms (median) | Lead vs Peak ms | y≥θ schon bei Fensterbeginn | 215305 (separat) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))

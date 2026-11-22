"""Find the STILL dodge (still_dodge.py): from a still stand, 30 cm sideways in 0.5-0.6 s, safety first.

Every candidate runs LEFT and RIGHT (mirror) on the same harsh, deployment-like physics
(go2_dodge_sim.harsh_physics: sensor noise, joint-target clamp, friction 0.35-1.1, -1..+3 kg,
motors 80-100 %, 0-30 ms latency) with load measurement. A run fails if it breaks any limit of the
SAFE dodge (validate_dodge.violations: fall, unsettled end, joint margin, foot force, touchdown speed,
trunk acceleration, torque, tilt, < 30 cm) or one of the STILL limits below.

Ranking, safety first:
  1. failures (both sides)
  2. speed requirement met: 30 cm p90 <= T30_P90 s (the 0.5-0.6 s goal, with harsh runs)
  3. load cost: the worst run's highest load as a fraction of its limit (lower = more margin)

  cd sim && python optimize_still.py               # ~15 min on 14 cores -> results/still_optimize.json
  cd sim && python screen_still.py                 # robust candidates on 100 runs per side (the search overfits)
  cd sim && python optimize_still.py --local       # focused search around the screened ones, 16 runs per side
  cd sim && python optimize_still.py --corner --seeds "10;11;12;1" --rand 12
                                                   # same, half of the runs in the harsh corner (weak motors,
                                                   #   +2..3 kg, low friction), safety ranked above speed
  cd sim && python optimize_still.py --validate --corner-rank 0   (or --local-rank / --screened "<name>")
                                                   # 500 harsh runs per side (scaled down until 0 fail)
                                                   #   -> results/still_dodge.json (what the robot runs)
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import go2_dodge_sim as g
import still_dodge as s
import validate_dodge as v

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/still_optimize.json"

SPACE = {
    "kick_vy": (0.8, 2.2),
    "kick_s": (0.15, 0.45),
    "hold_vy": (0.0, 1.0),
    "dodge_s": (0.25, 0.6),
    "brake_vy": (-1.0, 0.0),
    "brake_s": (0.0, 0.3),
    "freq": (3.0, 4.0),
    "stance_width": (0.28, 0.42),
    "footswing": (0.05, 0.12),
    "height": (-0.08, 0.02),
    "phase0": (0.0, 1.0),
}
# STILL limits on top of the SAFE ones
STILL_LIMITS = {
    "y_final_min": 0.33,      # the real dog went ~0.8x the sim distance: 0.33 sim ~ 0.27 real at worst
    "y_final_max": 0.60,      # don't run further than needed
    "y_max_max": 0.65,
    "x_abs_max": 0.15,        # forward/backward drift during and after the dodge
    "yaw_abs_max": 30.0,      # heading change [deg]; not a load: the SAFE right dodge turned up to 29 deg
    "hold_drift_max": 0.005,  # standing still before the trigger [m]
    "end_travel_max": 0.01,   # standing still at the end (last 0.5 s) [m]
}
T30_P90 = 0.60
LOAD_KEYS = {"foot_force_bw": "foot_force_bw_max", "touchdown_p95": "touchdown_p95_max",
             "trunk_acc_g": "trunk_acc_g_max", "torque_frac": "torque_frac_max", "peak_tilt_deg": "peak_tilt_deg_max"}

_POL = None
_PHYS: list[g.Physics] = []
_LIM: dict = {}


def _init(phys, lim):
    global _POL, _PHYS, _LIM
    torch.set_num_threads(1)
    _POL, _PHYS, _LIM = g.WTWPolicy(), phys, lim


def still_violations(m: dict, lim: dict) -> list[str]:
    out = v.violations(m, lim)
    L = STILL_LIMITS
    if m["y_final"] < L["y_final_min"]:
        out.append(f"short {m['y_final']:.2f}")
    if m["y_final"] > L["y_final_max"] or m["y_max"] > L["y_max_max"]:
        out.append(f"far {m['y_max']:.2f}")
    if m["x_abs_max"] > L["x_abs_max"]:
        out.append(f"x_drift {m['x_abs_max']:.2f}")
    if abs(m["yaw_drift_deg"]) > L["yaw_abs_max"]:
        out.append(f"yaw {m['yaw_drift_deg']:.0f}")
    if m["hold_drift_m"] > L["hold_drift_max"]:
        out.append(f"hold_drift {m['hold_drift_m']:.3f}")
    if m["end_travel_m"] > L["end_travel_max"]:
        out.append(f"end_travel {m['end_travel_m']:.3f}")
    return out


def load_cost(m: dict, lim: dict) -> float:
    """Highest load as a fraction of its limit (joint margin: limit / margin)."""
    c = [m[k] / lim[lk] for k, lk in LOAD_KEYS.items() if lk in lim]
    c.append(lim["limit_margin_min"] / max(m["limit_margin"], 1e-3))
    return float(max(c))


def one(d: s.StillDodge, ph: g.Physics, lim: dict) -> dict:
    m = s.metrics(s.run(d, ph, _POL, loads=True))
    m = {k: (float(x) if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) else x)
         for k, x in m.items()}
    m["violations"] = still_violations(m, lim)
    m["load_cost"] = load_cost(m, lim)
    return m


def summarize(ms: list[dict]) -> dict:
    t30 = np.array([9.0 if math.isnan(m["t_30cm"]) else m["t_30cm"] for m in ms])
    bad = [m["violations"] for m in ms if m["violations"]]
    return {
        "n": len(ms), "fails": len(bad), "why": sorted({w.split(" ")[0] for b in bad for w in b}),
        "t30_median": float(np.median(t30)), "t30_p90": float(np.percentile(t30, 90)),
        "t30_p95": float(np.percentile(t30, 95)), "t30_worst": float(t30.max()),
        "y_final_median": float(np.median([m["y_final"] for m in ms])),
        "y_final_min": float(min(m["y_final"] for m in ms)), "y_final_max": float(max(m["y_final"] for m in ms)),
        "load_cost_worst": float(max(m["load_cost"] for m in ms)),
        "load_cost_p90": float(np.percentile([m["load_cost"] for m in ms], 90)),
        "tilt_worst": float(max(m["peak_tilt_deg"] for m in ms)),
        "foot_worst": float(max(m["foot_force_bw"] for m in ms)),
        "acc_worst": float(max(m["trunk_acc_g"] for m in ms)),
        "touch_worst": float(max(m["touchdown_p95"] for m in ms)),
        "torque_worst": float(max(m["torque_frac"] for m in ms)),
        "margin_worst": float(min(m["limit_margin"] for m in ms)),
        "x_worst": float(max(m["x_abs_max"] for m in ms)),
        "yaw_worst": float(max(abs(m["yaw_drift_deg"]) for m in ms)),
        "hold_drift_worst": float(max(m["hold_drift_m"] for m in ms)),
    }


def evaluate(p: dict) -> dict:
    out = {"params": p}
    for side in ("left", "right"):
        d = s.StillDodge.from_params(p, side=side)
        out[side] = summarize([one(d, ph, _LIM) for ph in _PHYS])
    L, R = out["left"], out["right"]
    out["fails"] = L["fails"] + R["fails"]
    out["n"] = L["n"] + R["n"]
    out["t30_p90"] = max(L["t30_p90"], R["t30_p90"])
    out["load_cost"] = max(L["load_cost_worst"], R["load_cost_worst"])
    return out


def score(r: dict) -> tuple:
    return (r["fails"], r["t30_p90"] > T30_P90, round(r["load_cost"], 3), r["t30_p90"])


def sample(rng) -> dict:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()}


def perturb(p: dict, rng, scale: float) -> dict:
    q = {}
    for k, (lo, hi) in SPACE.items():
        x = p[k] + rng.normal(0, scale * (hi - lo))
        q[k] = float(x % 1.0) if k == "phase0" else float(np.clip(x, lo, hi))
    return q


def table(rows: list[dict], k: int = 10) -> None:
    print(f"  {'fail':>6s} {'t30p90':>6s} {'load':>5s} {'yfin L/R':>9s} {'tilt':>4s} {'x':>4s} {'yaw':>3s}  params")
    for r in rows[:k]:
        L, R, p = r["left"], r["right"], r["params"]
        print(f"  {r['fails']:>2d}/{r['n']:<3d} {r['t30_p90']:6.2f} {r['load_cost']:5.2f} "
              f"{L['y_final_median']:.2f}/{R['y_final_median']:.2f} {max(L['tilt_worst'], R['tilt_worst']):4.0f} "
              f"{max(L['x_worst'], R['x_worst']):4.2f} {max(L['yaw_worst'], R['yaw_worst']):3.0f}  "
              + " ".join(f"{k} {x:.2f}" for k, x in p.items())
              + (f"  [{','.join(sorted(set(L['why'] + R['why'])))}]" if r["fails"] else ""), flush=True)


def pool_map(phys, lim, params, workers):
    with Pool(workers, initializer=_init, initargs=(phys, lim)) as pool:
        return pool.map(evaluate, params, chunksize=1)


def local(a) -> None:
    """Focused search around the screened candidates with many more runs per candidate (the broad
    search ranked on 6 runs per side and overfit). Ranked by failures, then 30 cm p90, then load."""
    lim = v.limits()
    rng = np.random.default_rng(a.seed + 7)
    phys = [g.Physics(obs_noise=1.0, target_margin=0.1)] + [g.harsh_physics(rng) for _ in range(a.rand)]
    rows = json.loads((HERE / "results/still_screen.json").read_text())["rows"]
    seeds = [rows[n]["params"] for n in a.seeds.split(";")]
    cand = seeds + [perturb(p, rng, sc) for p in seeds for sc in np.linspace(0.02, 0.08, a.refine_n)]
    print(f"local: {len(cand)} candidates x 2 sides x {len(phys)} harsh physics", flush=True)
    res = sorted(pool_map(phys, lim, cand, a.workers), key=lambda r: (r["fails"], round(r["t30_p90"], 2), r["load_cost"]))
    table(res, 15)
    out = HERE / "results/still_local.json"
    out.write_text(json.dumps({"seeds": a.seeds, "top": res[:30]}, indent=1))
    print(f"-> {out}")


def corner_physics(rng: np.random.Generator) -> g.Physics:
    """The harsh corner where the 500-run validation found torque saturation: weak motors, heavy
    payload, low friction, long latency (still inside harsh_physics' ranges)."""
    return g.Physics(friction=rng.uniform(0.35, 0.5), mass_add=rng.uniform(2.0, 3.0),
                     motor_strength=rng.uniform(0.8, 0.85), latency=rng.uniform(0.01, 0.03),
                     obs_noise=1.0, target_margin=0.1, pre_offset=rng.uniform(-0.3, 0.3),
                     seed=int(rng.integers(1 << 30)))


T30_P90_REQ = 0.62      # corner search: speed requirement (p90 incl. the corner runs)


def corner(a) -> None:
    """Search around still_local.json candidates on half harsh / half corner physics; ranked safety
    first: failures, speed requirement met, then load margin."""
    lim = v.limits()
    rng = np.random.default_rng(a.seed + 13)
    phys = [g.harsh_physics(rng) for _ in range(a.rand)] + [corner_physics(rng) for _ in range(a.rand)]
    top = json.loads((HERE / "results/still_local.json").read_text())["top"]
    seeds = [top[int(i)]["params"] for i in a.seeds.split(";")]
    cand = seeds + [perturb(p, rng, sc) for p in seeds for sc in np.linspace(0.02, 0.08, a.refine_n)]
    print(f"corner: {len(cand)} candidates x 2 sides x {len(phys)} physics ({a.rand} harsh + {a.rand} corner)", flush=True)
    res = sorted(pool_map(phys, lim, cand, a.workers),
                 key=lambda r: (r["fails"], r["t30_p90"] > T30_P90_REQ, round(r["load_cost"], 2), r["t30_p90"]))
    table(res, 15)
    out = HERE / "results/still_corner.json"
    out.write_text(json.dumps({"seeds": a.seeds, "n_harsh": a.rand, "n_corner": a.rand, "top": res[:30]}, indent=1))
    print(f"-> {out}")


def safe_anchor() -> dict:
    p = json.loads((HERE / "results/safe_dodge.json").read_text())["chosen"]["params"]
    return {k: float(p.get(k, 0.0)) for k in SPACE}


def search(a) -> None:
    lim = v.limits()
    rng = np.random.default_rng(a.seed)
    phys = [g.Physics(obs_noise=1.0, target_margin=0.1)] + [g.harsh_physics(rng) for _ in range(a.rand)]
    anchors = [{**safe_anchor(), "phase0": ph} for ph in (0.0, 0.25, 0.5, 0.75)]
    print(f"limits {lim} + {STILL_LIMITS}\nstage 1: {a.samples} candidates x 2 sides x {len(phys)} harsh physics", flush=True)
    rows = pool_map(phys, lim, anchors + [sample(rng) for _ in range(a.samples)], a.workers)
    rows.sort(key=score)
    table(rows)
    print(f"\nstage 2: refine top {a.refine} x {a.refine_n}", flush=True)
    cand = [perturb(r["params"], rng, sc) for r in rows[:a.refine] for sc in np.linspace(0.02, 0.10, a.refine_n)]
    rows = sorted(rows + pool_map(phys, lim, cand, a.workers), key=score)
    table(rows)
    print(f"\nstage 3: top {a.final} on {a.val_n} fresh harsh physics per side", flush=True)
    vrng = np.random.default_rng(a.seed + 100)
    vphys = [g.harsh_physics(vrng) for _ in range(a.val_n)]
    fin = [r for r in rows if r["fails"] == 0][:a.final] or rows[:a.final]
    val = pool_map(vphys, lim, [r["params"] for r in fin], a.workers)
    # neighbours: +-5 % perturbations must also pass on the search set (no knife edge)
    nrng = np.random.default_rng(a.seed + 200)
    nb = pool_map(phys, lim, [perturb(r["params"], nrng, 0.05) for r in fin for _ in range(a.nbr)], a.workers)
    report = []
    for i, (f, vv) in enumerate(zip(fin, val)):
        n = nb[i * a.nbr:(i + 1) * a.nbr]
        report.append({"search": f, "validation": vv, "neighbours_ok": sum(x["fails"] == 0 for x in n),
                       "neighbours_n": len(n), "neighbours_t30_p90_worst": max(x["t30_p90"] for x in n)})
    report.sort(key=lambda r: (r["validation"]["fails"], r["neighbours_n"] - r["neighbours_ok"],
                               r["validation"]["t30_p90"] > T30_P90, round(r["validation"]["load_cost"], 2)))
    print("  validation:")
    table([r["validation"] for r in report], a.final)
    print("  neighbours ok: " + ", ".join(f"{r['neighbours_ok']}/{r['neighbours_n']}" for r in report))
    OUT.write_text(json.dumps({"space": SPACE, "still_limits": STILL_LIMITS, "limits": lim, "t30_p90_goal": T30_P90,
                               "top": rows[:40], "final": report}, indent=1))
    print(f"-> {OUT}")


SCALES = (1.0, 0.97, 0.94, 0.9)


GOAL_REASONS = ("short", "far", "never")      # distance goal, not a load or stability limit


def safety_fails(r: dict) -> int:
    """Runs (both sides) with a fall, unsettled end, load/joint/tilt limit, drift or yaw violation."""
    return sum(any(w.split(" ")[0] not in GOAL_REASONS for w in run["violations"])
               for side in ("left", "right") for run in r[side]["runs"])


def goal_misses(r: dict) -> int:
    return sum(any(w.split(" ")[0] in GOAL_REASONS for w in run["violations"])
               for side in ("left", "right") for run in r[side]["runs"])


def validate(a) -> None:
    """500 harsh runs per side; accepted when NO run breaks a safety limit (fall, unsettled end, load,
    joint margin, torque, tilt, drift, yaw). Runs that only miss the distance window (sim end distance
    0.33-0.60 m, or 30 cm never reached) are counted and reported, not hidden. If a safety limit is
    broken, the sideways commands are scaled down (like the SAFE dodge's scale sweep); --scale fixes
    the scale."""
    lim = v.limits()
    if a.corner_rank is not None:
        p0 = json.loads((HERE / "results/still_corner.json").read_text())["top"][a.corner_rank]["params"]
        a.screened = f"still_corner.json rank {a.corner_rank}"
    elif a.local_rank is not None:
        p0 = json.loads((HERE / "results/still_local.json").read_text())["top"][a.local_rank]["params"]
        a.screened = f"still_local.json rank {a.local_rank}"
    elif a.screened:
        p0 = json.loads((HERE / "results/still_screen.json").read_text())["rows"][a.screened]["params"]
    else:
        p0 = json.loads(OUT.read_text())["final"][a.pick]["search"]["params"]
    print(f"validating {a.screened or f'finalist {a.pick}'}: {p0}")
    rng = np.random.default_rng(a.seed_val)
    phys = [g.harsh_physics(rng) for _ in range(a.runs)]
    sweep = []
    for sc in ((a.scale,) if a.scale else SCALES):
        p = s.StillDodge.from_params(p0).scaled(sc).params()
        p = {k: p[k] for k in SPACE}
        r = pool_map(phys, lim, [p], a.workers)[0] if a.workers <= 1 else _split(phys, lim, p, a.workers)
        r["scale"] = sc
        r["safety_fails"], r["goal_misses"] = safety_fails(r), goal_misses(r)
        sweep.append(r)
        print(f"x{sc:.2f}: safety failures {r['safety_fails']}/{2 * a.runs}, distance misses {r['goal_misses']}/{2 * a.runs} "
              f"(L {r['left']['fails']} R {r['right']['fails']}), t30 p90 {r['t30_p90']:.2f}, load {r['load_cost']:.2f}  "
              f"{r['left']['why'] + r['right']['why']}", flush=True)
        if r["safety_fails"] == 0:
            break
    ok = [r for r in sweep if r["safety_fails"] == 0]
    if not ok:
        print("no scale passed every run")
        return
    best = ok[0]
    gentle = None
    for gs in (0.8, 0.7):          # gentle version for the first tries on the real dog
        pg = {k: s.StillDodge.from_params(best["params"]).scaled(gs).params()[k] for k in SPACE}
        rg = _split(phys[:300], lim, pg, a.workers)
        rg["safety_fails"], rg["goal_misses"] = safety_fails(rg), goal_misses(rg)
        print(f"gentle x{gs}: safety failures {rg['safety_fails']}/{rg['n']}, distance misses {rg['goal_misses']}, "
              f"t30 p90 {rg['t30_p90']:.2f}, y_final L {rg['left']['y_final_min']:.2f}-{rg['left']['y_final_max']:.2f} "
              f"R {rg['right']['y_final_min']:.2f}-{rg['right']['y_final_max']:.2f}, load {rg['load_cost']:.2f}")
        gentle = {"scale": gs, **rg}
        if rg["safety_fails"] == 0:
            break
    s.STILL_PATH.write_text(json.dumps({"chosen": {"params": {**best["params"], "prime": True}, **best},
                                        "gentle": gentle, "sweep": sweep, "limits": lim,
                                        "still_limits": STILL_LIMITS, "runs_per_side": a.runs,
                                        "seed": a.seed_val, "source": a.screened or f"finalist {a.pick}"}, indent=1))
    print(f"-> {s.STILL_PATH}")


def _ev_chunk(args):
    p, idx = args
    out = {}
    for side in ("left", "right"):
        d = s.StillDodge.from_params(p, side=side)
        out[side] = [one(d, _PHYS[i], _LIM) for i in idx]
    return out


def _split(phys, lim, p, workers) -> dict:
    """evaluate() for one candidate, with its runs spread over the pool."""
    chunks = [list(range(i, len(phys), workers)) for i in range(workers)]
    with Pool(workers, initializer=_init, initargs=(phys, lim)) as pool:
        parts = pool.map(_ev_chunk, [(p, c) for c in chunks])
    out = {"params": p}
    for side in ("left", "right"):
        out[side] = summarize([m for part in parts for m in part[side]])
        out[side]["runs"] = [{k: m[k] for k in ("t_30cm", "y_final", "peak_tilt_deg", "load_cost", "violations")}
                             for part in parts for m in part[side]]
    L, R = out["left"], out["right"]
    out.update(fails=L["fails"] + R["fails"], n=L["n"] + R["n"], t30_p90=max(L["t30_p90"], R["t30_p90"]),
               load_cost=max(L["load_cost_worst"], R["load_cost_worst"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--rand", type=int, default=7)
    ap.add_argument("--refine", type=int, default=10)
    ap.add_argument("--refine-n", type=int, default=16)
    ap.add_argument("--final", type=int, default=6)
    ap.add_argument("--val-n", type=int, default=40)
    ap.add_argument("--nbr", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--pick", type=int, default=0, help="--validate: which finalist (0 = best)")
    ap.add_argument("--screened", default=None, help="--validate: a candidate name from results/still_screen.json")
    ap.add_argument("--corner-rank", type=int, default=None, help="--validate: a candidate from results/still_corner.json")
    ap.add_argument("--local-rank", type=int, default=None, help="--validate: a candidate from results/still_local.json")
    ap.add_argument("--scale", type=float, default=None, help="--validate: only this scale of the candidate")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--seed-val", type=int, default=2026)
    ap.add_argument("--corner", action="store_true", help="search around still_local.json ranks --seeds (e.g. 10;1;11) "
                    "on harsh + corner physics")
    ap.add_argument("--local", action="store_true", help="focused search around --seeds (names in still_screen.json)")
    ap.add_argument("--seeds", default="F2 x0.9;F2 x1;F1 (search best) x0.9;SAFE params, phase 0.5 x1", help="; separated")
    a = ap.parse_args()
    if a.corner:
        corner(a)
    elif a.local:
        local(a)
    else:
        validate(a) if a.validate else search(a)


if __name__ == "__main__":
    main()

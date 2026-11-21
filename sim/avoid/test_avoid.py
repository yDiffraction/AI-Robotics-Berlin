"""Scenario suite: SAFE dodge + guard + Nav2 Collision Monitor, in MuJoCo with randomized harsh physics.

Every run uses go2_dodge_sim.harsh_physics (sensor noise, joint clamp, random gait phase at the
trigger, +3 kg, weak motors, 30 ms latency, friction 0.35-1.1) and the simulated L1 lidar.
The robot starts at the origin facing +x; "room" is the free lateral distance from the body side
(0.20 m from the centre line) to the obstacle face.

  # in WSL (ROS 2 Humble), see README:
  python test_avoid.py                       # all families, default counts  (~15 min)
  python test_avoid.py --families forced,popup --n 20 --params my_cm.yaml
Writes ../results/avoid_<tag>.json and prints a pass/fail table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import avoid_sim as A  # noqa: E402
import go2_dodge_sim as g  # noqa: E402

BODY = 0.20          # body half width used for "room"
BAND = 0.06          # rooms this close to NEED may go either way (measurement errs ~4 cm short)


def wall(side: int, room: float, x=0.0, half_len=1.0, height=0.6, thick=0.05):
    yc = side * (BODY + room + thick)
    return A.Obstacle("box", (x, yc, height / 2), (half_len, thick, height / 2))


def box(side, room, x, hx, hy, hz, appear=None):
    return A.Obstacle("box", (x, side * (BODY + room + hy), hz), (hx, hy, hz), appear=appear)


def pole(side, room, x, r=0.025, h=1.0):
    return A.Obstacle("cyl", (x, side * (BODY + room + r), h / 2), (r, h / 2))


def expected_side(room_l: float, room_r: float) -> set:
    """Sides the rule may legitimately choose given true rooms (band around NEED: either)."""
    ok = set()

    def can(r):
        return "yes" if r >= A.NEED + BAND else ("no" if r < A.NEED - BAND else "maybe")
    cl, cr = can(room_l), can(room_r)
    if cl == "yes":
        return {1}
    if cl == "maybe":
        ok.add(1)
    if cr == "yes":
        ok.add(-1)
        return ok
    if cr == "maybe":
        ok.add(-1)
    ok.add(0)
    return ok


def make(family: str, rng: np.random.Generator) -> tuple[A.Scenario, dict]:
    ph = g.harsh_physics(rng)
    if family == "free":     # either side: the floor check may conservatively veto one (~5 %)
        return A.Scenario("free", [], ph), {"sides": {1, -1}}
    if family == "wall_one":
        room = float(rng.uniform(0.05, 1.6))
        return A.Scenario(f"wall left room {room:.2f}", [wall(1, room)], ph), \
            {"sides": expected_side(room, 9.0), "room": room}
    if family == "walls_both":
        rl, rr = rng.uniform(0.05, 1.3, 2)
        return A.Scenario(f"walls L {rl:.2f} R {rr:.2f}", [wall(1, rl), wall(-1, rr)], ph), \
            {"sides": expected_side(rl, rr), "room": (float(rl), float(rr))}
    if family == "forced":
        room = float(rng.uniform(0.15, 0.80))
        return A.Scenario(f"FORCED left, wall room {room:.2f}", [wall(1, room)], ph, side_check=False), \
            {"sides": {1}, "room": room, "monitor_only": True}
    if family == "popup":
        room = float(rng.uniform(0.25, 0.80))
        at = float(rng.uniform(0.0, 0.30))
        x = float(rng.uniform(-0.3, 0.3))
        return A.Scenario(f"popup room {room:.2f} at +{at:.2f}s", [box(1, room, x, 0.15, 0.15, 0.3, appear=at)], ph), \
            {"sides": {1}, "room": room, "at": at}
    if family == "popup_nokeepoff":
        sc, exp = make("popup", rng)
        return replace(sc, name=sc.name + " (no keep-off)", keep_off=False), {**exp, "informational": True}
    if family == "dropout":
        room = float(rng.uniform(0.3, 0.8))
        off = float(rng.uniform(0.0, 0.25))
        return A.Scenario(f"lidar off +{off:.2f}s, FORCED, wall {room:.2f}", [wall(1, room)], ph,
                          lidar_off_at=off, side_check=False), {"sides": {1}, "room": room, "monitor_only": True}
    if family == "dropout_nowatch":
        room = float(rng.uniform(0.3, 0.8))
        off = float(rng.uniform(0.0, 0.25))
        return A.Scenario(f"lidar off +{off:.2f}s, NO lidar watchdog, FORCED, wall {room:.2f}", [wall(1, room)], ph,
                          lidar_off_at=off, side_check=False, lidar_watch=False), \
            {"sides": {1}, "room": room, "monitor_only": True, "informational": True}
    if family == "monitor_down":
        return A.Scenario("monitor not answering", [], ph, monitor="down"), {"sides": {1}, "no_motion": True}
    if family == "pole":
        room = float(rng.uniform(0.1, 1.2))
        x = float(rng.uniform(-0.4, 0.4))
        return A.Scenario(f"pole (5 cm) room {room:.2f} x {x:+.2f}", [pole(1, room, x)], ph), \
            {"sides": expected_side(room, 9.0), "room": room}
    if family == "low_box":
        room = float(rng.uniform(0.1, 1.2))
        return A.Scenario(f"low box (12 cm) room {room:.2f}", [box(1, room, 0.0, 0.15, 0.15, 0.06)], ph), \
            {"sides": expected_side(room, 9.0), "room": room}
    if family == "rear_box":
        room = float(rng.uniform(0.1, 0.8))
        # known lidar blind spot (rearward rays point >= 15 deg down): side choice informational,
        # contact and falls still fail the run
        return A.Scenario(f"box beside hind legs room {room:.2f}", [box(1, room, -0.40, 0.08, 0.08, 0.15)], ph), \
            {"sides": {1, -1, 0}, "room": room, "side_expected": sorted(expected_side(room, 9.0))}
    raise ValueError(family)


def judge(family: str, o: A.Outcome, exp: dict, ref_t30: float | None) -> list[str]:
    bad = []
    if o.fell:
        bad.append("FELL")
    elif not o.stable_end:
        bad.append("unsettled end")
    if o.contact:
        bad.append(f"CONTACT at +{o.contact_t:.2f}s")
    if o.side not in exp["sides"]:
        bad.append(f"side {o.side:+d} not in {sorted(exp['sides'])}")
    if exp.get("no_motion") and abs(o.lateral) > 0.08:
        bad.append(f"moved {o.lateral:.2f} m without monitor")
    if family == "free" and ref_t30 is not None and o.side != 0:
        if o.t30 is None or o.t30 > ref_t30 + 0.04:
            bad.append(f"slower than without monitor ({o.t30} vs {ref_t30:.2f})")
    return bad


FAMILIES = {"free": 25, "wall_one": 30, "walls_both": 30, "forced": 30, "popup": 40, "popup_nokeepoff": 20,
            "dropout": 15, "dropout_nowatch": 10, "monitor_down": 5, "pole": 20, "low_box": 15, "rear_box": 20}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--n", type=int, default=None, help="runs per family (default: per-family counts)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--params", default=str(A.CM_PARAMS))
    ap.add_argument("--tag", default="suite")
    ap.add_argument("--footprint-y", type=float, default=None, help="published footprint half width [m]")
    args = ap.parse_args()
    if args.footprint_y is not None:
        hx = A.FOOTPRINT[0][0]
        A.FOOTPRINT = ((hx, args.footprint_y), (hx, -args.footprint_y), (-hx, -args.footprint_y), (-hx, args.footprint_y))
    torch.set_num_threads(1)
    import show_dodges as sd
    dodge = sd.safe_dodge()
    policy = g.WTWPolicy()
    bench = A.Bench(Path(args.params), HERE.parent / "results" / f"cm_{args.tag}.log")
    rng = np.random.default_rng(args.seed)
    report = {"params": Path(args.params).read_text(), "need": A.NEED, "footprint": A.FOOTPRINT, "families": {}}
    t_start = time.time()
    try:
        for fam in args.families.split(","):
            n = args.n or FAMILIES[fam]
            rows = []
            for i in range(n):
                sc, exp = make(fam, rng)
                ref = None
                if fam == "free":           # same physics without the monitor: speed reference
                    ref = A.run(replace(sc, name="free bypass", monitor="bypass"), bench, policy, dodge).t30
                o = A.run(sc, bench, policy, dodge)
                bad = judge(fam, o, exp, ref)
                rows.append({"scenario": sc.name, "pass": not bad, "why": bad, "exp_sides": sorted(exp["sides"]),
                             "ref_t30": ref, **{k: v for k, v in asdict(o).items() if k != "name"},
                             "informational": exp.get("informational", False)})
                mark = "ok " if not bad else ("info" if exp.get("informational") else "FAIL")
                print(f"  [{mark}] {fam:15s} {sc.name:52s} side {o.side:+d} lat {o.lateral:+.2f} "
                      f"clear {min(o.min_clearance, 9):.3f} t30 {o.t30 if o.t30 is None else round(o.t30, 2)} "
                      f"{'; '.join(bad)}", flush=True)
            passed = sum(r["pass"] for r in rows)
            clear = [r["min_clearance"] for r in rows if math.isfinite(r["min_clearance"])]
            report["families"][fam] = {"runs": n, "passed": passed, "rows": rows,
                                       "min_clearance": min(clear) if clear else None}
            print(f"== {fam}: {passed}/{n} passed" + (f", closest approach {min(clear):.3f} m" if clear else ""),
                  flush=True)
    finally:
        bench.close()
    out = HERE.parent / "results" / f"avoid_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\n{'family':18s} {'passed':>8s}  closest approach")
    for fam, r in report["families"].items():
        info = " (informational)" if fam in ("dropout_nowatch", "popup_nokeepoff") else ""
        mc = f"{r['min_clearance']:.3f} m" if r["min_clearance"] is not None else "-"
        print(f"{fam:18s} {r['passed']:>3d}/{r['runs']:<4d}  {mc}{info}")
    print(f"-> {out}   ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()

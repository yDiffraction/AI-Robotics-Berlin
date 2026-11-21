"""End-to-end tests of supervisor.py over the real Unitree DDS interface (CycloneDDS, private domain 77
on loopback) against fake_go2.FakeGo2. Linux/WSL with the unitree_sdk2py venv:

  python test_supervisor.py            # all scenarios, 3 repeats  (~6 min)
  python test_supervisor.py --only estop_state,child_crash --repeat 1

Each run: the fake robot lies on the floor, the supervisor starts stand_hold.py, which stands the
robot up; once it stands, a fault is injected. Checked: the right trip reason, time from the fault to
the first damping command at the robot, no controller command after the damping started, the
controller process is dead, the robot ends lying down.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

import fake_go2  # noqa: E402
import safety  # noqa: E402

DOMAIN, IFACE = 77, "lo"
PY = sys.executable
STAND_AT = 3.5            # fault injected this long after the controller's first command
L2, B = 1 << safety.BUTTON_BIT["L2"], 1 << safety.BUTTON_BIT["B"]

# name: (child fault, external action, expected reason regex, max latency fault -> first damping [s])
SCENARIOS = {
    "estop_state":     ("none", "keys_state", r"e-stop .*lowstate", 0.03),
    "estop_remote":    ("none", "keys_remote", r"e-stop .*wirelesscontroller", 0.04),
    "child_crash":     ("crash", None, r"controller (exited|silent)", 0.07),
    "child_kill9":     ("none", "kill_child", r"controller (exited|silent)", 0.07),
    "child_hang":      ("hang", None, r"controller silent", 0.07),
    "child_sigstop":   ("none", "stop_child", r"controller silent", 0.07),
    "child_slow":      ("slow", None, r"controller silent", 0.07),
    "child_nan":       ("nan", None, r"bad command: NaN", 0.03),
    "child_wild":      ("wild", None, r"bad command: target outside", 0.03),
    "child_badcrc":    ("badcrc", None, r"controller silent", 0.07),
    "state_lost":      ("none", "state_pause", r"rt/lowstate\) lost", 0.13),
    "tick_stuck":      ("none", "tick_freeze", r"tick stuck", 0.13),
    "remote_lost":     ("none", "remote_pause", r"remote link", 0.55),
    "tilt":            ("none", "push", r"tilt", None),
    "ctrl_c":          ("none", "sigint_sup", r"operator \(Ctrl\+C\)", 0.03),
    "sigterm":         ("none", "sigterm_sup", r"operator \(SIGTERM\)", 0.03),
    "supervisor_kill9": ("none", "kill_sup", None, None),
    "no_fault":        ("none", "other_buttons", None, None),
}


def run_one(robot: fake_go2.FakeGo2, name: str, on_silence: str = "hold") -> dict:
    fault, action, want, max_lat = SCENARIOS[name]
    robot.reset(on_silence).start()
    child = [PY, str(HERE / "stand_hold.py"), "--domain", str(DOMAIN), "--iface", IFACE,
             "--fault", fault, "--fault-at", str(STAND_AT)]
    sup = subprocess.Popen([PY, "-u", str(HERE / "supervisor.py"), "--domain", str(DOMAIN), "--iface", IFACE,
                            "--damp-for", "2.5", "--startup-timeout", "30", "--", *child],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: list[str] = []
    threading.Thread(target=lambda: [out.append(line) for line in sup.stdout], daemon=True).start()
    res = {"scenario": name, "on_silence": on_silence, "problems": []}
    bad = res["problems"]

    def first(kind, after=-1.0):
        return next((t for t, k in list(robot.rx) if k == kind and t > after), None)

    try:
        t_child = None
        while t_child is None and robot.t < 30:
            time.sleep(0.05)
            t_child = first("child")
        if t_child is None:
            bad.append("controller never sent a command")
            return res
        while robot.t < t_child + STAND_AT:
            time.sleep(0.005)
        child_pid = int(re.search(r"pid (\d+)", "".join(out)).group(1))
        t_fault = robot.t
        if action == "keys_state":
            robot.keys_state = L2 | B
        elif action == "keys_remote":
            robot.keys_remote = L2 | B
        elif action == "kill_child":
            os.kill(child_pid, signal.SIGKILL)
        elif action == "stop_child":
            os.kill(child_pid, signal.SIGSTOP)
        elif action == "state_pause":
            robot.state_paused = True
        elif action == "tick_freeze":
            robot.tick_frozen = True
        elif action == "remote_pause":
            robot.remote_paused = True
        elif action == "push":
            robot.push = np.array([0, 0, 0, 80.0, 0, 0])      # roll torque on the trunk for 0.6 s
            time.sleep(0.6)
            robot.push = np.zeros(6)
        elif action == "sigint_sup":
            sup.send_signal(signal.SIGINT)
        elif action == "sigterm_sup":
            sup.send_signal(signal.SIGTERM)
        elif action == "kill_sup":
            sup.kill()
        elif action == "other_buttons":            # L2 alone, B alone, other combos: must not trip
            for keys in (L2, B, L2 | (1 << safety.BUTTON_BIT["A"]), (1 << safety.BUTTON_BIT["R2"]) | B):
                robot.keys_state = robot.keys_remote = keys
                time.sleep(0.4)
            robot.keys_state = robot.keys_remote = 0
        if fault != "none":                          # the child's own fault time, as the robot saw it
            t_fault = t_child + STAND_AT
            if fault in ("nan", "badcrc"):
                t_fault = first("nonfinite" if fault == "nan" else "bad_crc", t_child) or t_fault
        end = robot.t + (6.0 if action == "other_buttons" else 3.5)
        while robot.t < end and sup.poll() is None:
            time.sleep(0.02)
        time.sleep(0.3)
        res["t_fault"] = t_fault
        t_damp = first("damp")
        res["first_damp_latency"] = None if t_damp is None else round(t_damp - t_fault, 4)
        log = robot.log
        z = np.array(log["z"])
        t = np.array(log["t"])
        res["z_before"] = float(z[t < t_fault][-1])
        res["z_end"] = float(z[-1])
        if t_damp is not None:
            after = t > t_damp
            res["acc_peak_after_g"] = float(np.max(np.array(log["acc_g"])[after])) if after.any() else None
        late_child = [tt for tt, k in robot.rx if k == "child" and t_damp is not None and tt > t_damp + 0.005]
        child_alive = False
        try:
            os.kill(child_pid, 0)
            child_alive = True
        except ProcessLookupError:
            pass
        if action == "stop_child" and child_alive:
            os.kill(child_pid, signal.SIGCONT)
        text = "".join(out)
        m = re.search(r"TRIP: (.*?) ->", text)
        res["reason"] = m.group(1) if m else None
        if name == "no_fault":
            if t_damp is not None or res["reason"]:
                bad.append(f"tripped without a fault: {res['reason']}")
            if res["z_end"] < 0.25:
                bad.append(f"robot not standing at the end (z {res['z_end']:.2f})")
            sup.send_signal(signal.SIGINT)
            sup.send_signal(signal.SIGINT)
        elif name == "supervisor_kill9":
            time.sleep(0.3)
            try:
                os.kill(child_pid, 0)
                bad.append("controller still running after the supervisor was killed")
            except ProcessLookupError:
                pass
            last = max(tt for tt, k in robot.rx)
            res["commands_after_kill"] = sum(tt > t_fault + 0.02 for tt, _ in robot.rx)
            res["last_command_after_kill_s"] = round(last - t_fault, 3)
            if last > t_fault + 0.1:
                bad.append(f"commands still arriving {last - t_fault:.2f} s after the kill")
        else:
            if not res["reason"] or not re.search(want, res["reason"]):
                bad.append(f"reason {res['reason']!r} !~ {want!r}")
            if t_damp is None:
                bad.append("no damping command reached the robot")
            elif max_lat is not None and res["first_damp_latency"] > max_lat:
                bad.append(f"damping after {res['first_damp_latency'] * 1000:.0f} ms > {max_lat * 1000:.0f} ms")
            elif res["first_damp_latency"] < 0:
                bad.append("damping before the fault?")
            if late_child:
                bad.append(f"{len(late_child)} controller commands after damping started")
            if child_alive:
                bad.append("controller process still alive after the trip")
            if res["z_end"] > 0.18:
                bad.append(f"robot not lying down at the end (z {res['z_end']:.2f})")
        return res
    finally:
        if sup.poll() is None:
            sup.kill()
        sup.wait()
        robot.stop()
        time.sleep(1.0)                 # let DDS traffic of the dead processes drain
        res["supervisor_output"] = "".join(out)[-1500:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SCENARIOS))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", default="supervisor")
    a = ap.parse_args()
    ChannelFactoryInitialize(DOMAIN, IFACE)
    robot = fake_go2.FakeGo2()
    rows = []
    for rep in range(a.repeat):
        for name in a.only.split(","):
            on_silence = "passive" if rep % 2 else "hold"
            r = run_one(robot, name, on_silence)
            rows.append(r)
            lat = r.get("first_damp_latency")
            print(f"[{'ok ' if not r['problems'] else 'FAIL'}] {name:17s} robot={on_silence:7s} "
                  f"reason={r.get('reason')!s:55.55s} damp after {'-' if lat is None else f'{lat * 1000:5.0f} ms'} "
                  f"z {r.get('z_before', 0):.2f}->{r.get('z_end', 0):.2f}  {'; '.join(r['problems'])}", flush=True)
            if r["problems"]:
                print("    supervisor output tail:\n    " + r["supervisor_output"][-600:].replace("\n", "\n    "))
    fails = [r for r in rows if r["problems"]]
    out = HERE.parent / "results" / f"{a.tag}_test.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\n{len(rows) - len(fails)}/{len(rows)} passed -> {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

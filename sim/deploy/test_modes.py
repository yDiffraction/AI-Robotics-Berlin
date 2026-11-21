"""End to end: go2_modes.py -> supervisor.py -> dodge_controller.py against fake_go2.py, over real
Unitree DDS + RPC on a private domain (77, loopback). Buttons are pressed on the fake remote; one
scenario types keys into a pseudo-terminal instead.

  python test_modes.py            (WSL, ~6 min)   -> ../results/modes_test.json / .log
"""
from __future__ import annotations

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
import dodge_controller as dc  # noqa: E402
import fake_go2 as F  # noqa: E402
import safety  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

DOMAIN = 77
B = safety.BUTTON_BIT
OUT = HERE.parent / "results"
LOG_LINES: list[str] = []


def keys(*names: str) -> int:
    return sum(1 << B[n] for n in names)


def say(s: str) -> None:
    print(s, flush=True)
    LOG_LINES.append(s)


class Session:
    """One go2_modes.py process and its output (with the fake robot's sim time per line)."""

    def __init__(self, robot: F.FakeGo2, keyboard: bool = False, extra: tuple = ()):
        self.robot = robot
        self.master = None
        stdin = subprocess.DEVNULL
        if keyboard:
            self.master, slave = os.openpty()
            stdin = slave
        self.p = subprocess.Popen([sys.executable, "-u", str(HERE / "go2_modes.py"), "--domain", str(DOMAIN),
                                   "--iface", "lo", *extra], stdin=stdin, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, start_new_session=True)
        if keyboard:
            os.close(slave)
        self.lines: list[tuple[float, float, str]] = []      # (wall, sim t, text)
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            line = line.rstrip()
            with self.lock:
                self.lines.append((time.monotonic(), self.robot.t, line))
            LOG_LINES.append(f"      | {line}")

    def wait(self, pattern: str, timeout: float, start: int = 0):
        """First output line matching pattern (from line index `start`): (index, sim t) or None."""
        t_end = time.monotonic() + timeout
        rx = re.compile(pattern)
        while time.monotonic() < t_end:
            with self.lock:
                for i in range(start, len(self.lines)):
                    if rx.search(self.lines[i][2]):
                        return i, self.lines[i][1]
            time.sleep(0.02)
        return None

    def seen(self, pattern: str) -> bool:
        with self.lock:
            return any(re.search(pattern, x[2]) for x in self.lines)

    def press(self, k: int, dur: float = 0.25) -> None:
        self.robot.keys_state = self.robot.keys_remote = k
        time.sleep(dur)
        self.robot.keys_state = self.robot.keys_remote = 0

    def type(self, ch: str) -> None:
        os.write(self.master, ch.encode())

    def close(self) -> None:
        for sig in (signal.SIGINT,):
            if self.p.poll() is None:
                try:
                    os.killpg(self.p.pid, sig)
                except ProcessLookupError:
                    pass
        try:
            self.p.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(self.p.pid, signal.SIGKILL)
        for pat in (f"supervisor.py --domain {DOMAIN}", f"dodge_controller.py --domain {DOMAIN}"):
            subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        if self.master is not None:
            os.close(self.master)


def leftovers() -> list[str]:
    out = subprocess.run(["pgrep", "-af", f"(supervisor|dodge_controller|go2_modes).py --domain {DOMAIN}"],
                         capture_output=True, text=True).stdout
    return [x for x in out.splitlines() if "pgrep" not in x]


class Checks:
    def __init__(self, name: str):
        self.name, self.items = name, []

    def __call__(self, what: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": what, "ok": bool(ok), "detail": detail})
        say(f"   [{'PASS' if ok else 'FAIL'}] {what}" + (f"  ({detail})" if detail else ""))
        return ok

    @property
    def ok(self) -> bool:
        return all(x["ok"] for x in self.items)


def log_window(robot: F.FakeGo2, t0: float, t1: float | None = None) -> dict:
    L = {k: np.array(v) for k, v in robot.log.items()}
    sel = (L["t"] >= t0) & (L["t"] <= (t1 if t1 is not None else 1e9))
    return {k: v[sel] for k, v in L.items()}


def while_sport(robot: F.FakeGo2) -> int:
    return sum(1 for _, kind in robot.rx if kind == "while_sport")


def enter_by_remote(s: Session, c: Checks) -> bool:
    if not c("go2_modes sees sport mode", s.wait(r"SPORT mode 'normal' active", 30) is not None):
        return False
    s.press(keys("L1", "R1"), 1.3)
    ok = c("L1+R1 held 1.3 s -> sport mode released", s.wait(r"sport mode released", 15) is not None)
    return ok and c("controller stands the dog up", s.wait(r"standing\. Ready", 40) is not None)


def dodge_metrics(robot: F.FakeGo2, s: Session, c: Checks, dodge, start_line: int, min_dy: float = 0.30) -> None:
    side = "right" if dodge.cmd.vy < 0 else "left"
    sign = -1.0 if side == "right" else 1.0
    hit = s.wait(r"DODGE:", 3, start_line)
    if not c(f"dodge {side} started", hit is not None and side.upper() in s.lines[hit[0]][2],
             s.lines[hit[0]][2] if hit else ""):
        return
    t_cmd = hit[1]
    done = s.wait(r"dodge done", 20, hit[0])
    if not c("dodge finished, standing again", done is not None):
        return
    time.sleep(0.3)
    W = log_window(robot, t_cmd - 0.1)
    y0 = W["y"][0]
    t_kick = t_cmd + dodge.t0
    dy = sign * (W["y"] - y0)                 # world +y = left (the dog starts facing +x)
    reach = W["t"][(W["t"] >= t_kick) & (dy >= 0.30)]
    t30 = float(reach[0] - t_kick) if len(reach) else float("nan")
    tilt = float(np.abs(np.r_[W["roll"], W["pitch"]]).max())
    end = log_window(robot, done[1] - 0.2, done[1] + 0.2)
    c(f"dodge moved the dog >= {min_dy:.2f} m to the {side.upper()}", dy.max() >= min_dy,
      f"max {dy.max():.2f} m, end {dy[-1]:+.2f} m, x drift {W['x'][-1] - W['x'][0]:+.2f} m")
    if min_dy >= 0.30:
        c("30 cm reached within 0.9 s of the kick (sim worst 0.88 s left, 0.83 s right)", t30 <= 0.9, f"{t30:.2f} s")
    c("tilt stayed below 0.4 rad", tilt < 0.4, f"max {tilt:.2f} rad")
    c("standing at the end (trunk > 0.24 m)", end["z"].min() > 0.24, f"z {end['z'].min():.2f} m")
    s.result = {"t30": t30, "dy_max": float(dy.max()), "tilt": tilt}


def back_by_remote(s: Session, c: Checks, robot: F.FakeGo2, start_line: int) -> None:
    s.press(keys("Y"))
    c("Y -> lying down", s.wait(r"lying down", 5, start_line) is not None)
    c("supervisor: controller finished, dog lying", s.wait(r"supervisor ended: controller finished", 15, start_line) is not None)
    c("sport mode restored", s.wait(r"SPORT mode 'normal' is back", 50, start_line) is not None
      and robot.sport_name == "normal", f"robot mode '{robot.sport_name}'")


def final_checks(s: Session, c: Checks, robot: F.FakeGo2, lying: bool = True) -> None:
    c("no command ever reached the robot while sport mode was active", while_sport(robot) == 0,
      f"{while_sport(robot)} commands")
    if lying:
        z = log_window(robot, robot.t - 0.3)["z"]
        c("dog lying (trunk < 0.16 m)", z.max() < 0.16, f"z {z.max():.2f} m")


# ---- scenarios -------------------------------------------------------------------------------------
def sc_happy(robot: F.FakeGo2, silence: str, button: str = "A", side: str = "left") -> Checks:
    c = Checks(f"remote: enter, {button} = dodge {side}, back ({silence})")
    robot.reset(silence, sport="normal").start()
    s = Session(robot)
    try:
        if enter_by_remote(s, c):
            time.sleep(1.5)
            n = len(s.lines)
            s.press(keys(button))
            dodge_metrics(robot, s, c, dc.load_safe_dodge(side=side), n)
            time.sleep(1.0)
            back_by_remote(s, c, robot, len(s.lines))
            kinds = [k for _, k in robot.rx]
            c("last commands before sport mode were the supervisor's damping", kinds and kinds[-1] == "damp")
        final_checks(s, c, robot)
        os.killpg(s.p.pid, signal.SIGINT)
        c("Ctrl+C in sport mode quits go2_modes", s.wait(r"bye", 5) is not None)
    finally:
        s.close()
        robot.stop()
    return c


def sc_gentle(robot: F.FakeGo2, side: str = "left") -> Checks:
    """Left by remote (A); right by keyboard (o, r, s) in a pseudo-terminal."""
    kb = side == "right"
    c = Checks(f"--gentle: enter, dodge {side} x0.6, back" + (" (keyboard)" if kb else ""))
    robot.reset("hold", sport="normal").start()
    s = Session(robot, keyboard=kb, extra=("--gentle",))
    try:
        if kb:
            ok = c("sport mode seen", s.wait(r"SPORT mode 'normal' active", 30) is not None)
            if ok:
                s.type("o")
                ok = c("o -> standing", s.wait(r"standing\. Ready", 40) is not None)
        else:
            ok = enter_by_remote(s, c)
        if ok:
            c("controller runs the gentle dodge", s.seen(r"SAFE x0\.6 \(gentle\)"))
            time.sleep(1.5)
            n = len(s.lines)
            if kb:
                s.type("r")
            else:
                s.press(keys("A"))
            dodge_metrics(robot, s, c, dc.load_safe_dodge(gentle=True, side=side), n, min_dy=0.20)
            time.sleep(1.0)
            if kb:
                n = len(s.lines)
                s.type("s")
                c("s -> sport mode restored", s.wait(r"SPORT mode 'normal' is back", 50, n) is not None)
            else:
                back_by_remote(s, c, robot, len(s.lines))
        final_checks(s, c, robot)
    finally:
        s.close()
        robot.stop()
    return c


def sc_keyboard(robot: F.FakeGo2) -> Checks:
    c = Checks("keyboard: o, d, s")
    robot.reset("hold", sport="normal").start()
    s = Session(robot, keyboard=True)
    try:
        if c("go2_modes sees sport mode", s.wait(r"SPORT mode 'normal' active", 30) is not None):
            s.type("o")
            if c("o -> standing", s.wait(r"standing\. Ready", 40) is not None):
                time.sleep(1.5)
                n = len(s.lines)
                s.type("d")
                dodge_metrics(robot, s, c, dc.load_safe_dodge(), n)
                n = len(s.lines)
                s.type("s")
                c("s -> sport mode restored", s.wait(r"SPORT mode 'normal' is back", 50, n) is not None)
        final_checks(s, c, robot)
        s.type("q")
        c("q quits", s.wait(r"bye", 5) is not None)
    finally:
        s.close()
        robot.stop()
    return c


def sc_estop_mid_dodge(robot: F.FakeGo2, how: str) -> Checks:
    c = Checks(f"{how} during the kick, then Y")
    robot.reset("passive", sport="normal").start()
    s = Session(robot)
    try:
        if enter_by_remote(s, c):
            time.sleep(1.5)
            n = len(s.lines)
            s.press(keys("A"))
            hit = s.wait(r"DODGE:", 3, n)
            if c("dodge started", hit is not None):
                d = dc.load_safe_dodge()
                time.sleep(max(0.0, d.t0 + 0.25 - (robot.t - hit[1])))      # 0.25 s into the kick
                t_stop = robot.t
                if how == "L2+B":
                    s.press(keys("L2", "B"), 0.3)
                else:
                    os.killpg(s.p.pid, signal.SIGINT)
                trip = s.wait(r"TRIP", 3, n)
                c("supervisor tripped", trip is not None, s.lines[trip[0]][2] if trip else "")
                time.sleep(2.5)
                damp = [t for t, k in robot.rx if k == "damp" and t >= t_stop]
                c("damping reached the robot within 60 ms", bool(damp) and damp[0] - t_stop < 0.06,
                  f"{(damp[0] - t_stop) * 1000:.0f} ms" if damp else "none")
                after = [k for t, k in robot.rx if t > (damp[0] if damp else 0) and k == "child"]
                c("no controller command after the first damping", not after, f"{len(after)}")
                W = log_window(robot, t_stop)
                c("dog down on its belly, not on its side", W["z"][-50:].max() < 0.2 and np.abs(W["roll"][-50:]).max() < 0.6,
                  f"z {W['z'][-1]:.2f} m, roll {W['roll'][-1]:+.2f} rad")
                n = len(s.lines)
                s.press(keys("Y"))
                c("Y releases the damping", s.wait(r"damping stopped", 5, n) is not None)
                c("sport mode restored", s.wait(r"SPORT mode 'normal' is back", 50, n) is not None)
        final_checks(s, c, robot, lying=False)
    finally:
        s.close()
        robot.stop()
    return c


def sc_refused_and_busy(robot: F.FakeGo2) -> Checks:
    c = Checks("A during stand-up is refused; Y during stand-up; SelectMode busy")
    robot.reset("hold", sport="normal").start()
    s = Session(robot)
    try:
        if c("sport mode seen", s.wait(r"SPORT mode 'normal' active", 30) is not None):
            s.press(keys("L1", "R1"), 1.3)
            if c("controller standing up", s.wait(r"\[controller\] standing up", 40) is not None):
                time.sleep(1.2)
                s.press(keys("A"))
                c("A during stand-up -> refused", s.wait(r"dodge refused", 3) is not None)
                c("... and no dodge", not s.seen(r"DODGE:"))
                s.press(keys("Y"))
                c("Y during stand-up -> lies down", s.wait(r"lying down", 3) is not None)
                c("SelectMode busy right after release -> retried", s.wait(r"not ready yet \(code 7002\)", 30) is not None)
                c("sport mode restored", s.wait(r"SPORT mode 'normal' is back", 50) is not None)
        final_checks(s, c, robot)
    finally:
        s.close()
        robot.stop()
    return c


def sc_entry_guards(robot: F.FakeGo2) -> Checks:
    c = Checks("entry needs exactly L1+R1 for 1 s")
    robot.reset("hold", sport="normal").start()
    s = Session(robot)
    try:
        if c("sport mode seen", s.wait(r"SPORT mode 'normal' active", 30) is not None):
            s.press(keys("L1", "R1"), 0.6)
            time.sleep(1.0)
            c("L1+R1 for 0.6 s: nothing", not s.seen("switching"))
            s.press(keys("L1", "R1", "A"), 1.5)
            time.sleep(1.0)
            c("L1+R1+A for 1.5 s: nothing", not s.seen("switching"))
            s.press(keys("A"), 0.3)
            s.press(keys("Y"), 0.3)
            s.press(keys("L2", "B"), 0.3)
            time.sleep(1.0)
            c("A, Y, L2+B in sport mode: nothing", not s.seen("switching|giving") and robot.sport_name == "normal")
            z = log_window(robot, robot.t - 0.3)["z"]
            c("dog still standing in sport mode", z.min() > 0.24, f"z {z.min():.2f}")
        c("nothing sent at all", not robot.rx, f"{len(robot.rx)} commands")
    finally:
        s.close()
        robot.stop()
    return c


def sc_switcher_killed(robot: F.FakeGo2) -> Checks:
    c = Checks("go2_modes killed while our controller stands")
    robot.reset("passive", sport="normal").start()
    s = Session(robot)
    try:
        if enter_by_remote(s, c):
            time.sleep(1.0)
            os.killpg(s.p.pid, signal.SIGKILL)
            t_kill = robot.t
            c("controller lies down by itself (stdin closed)", s.wait(r"lie down requested", 3) is not None)
            c("supervisor finishes cleanly", s.wait(r"controller finished, dog lying", 8) is not None)
            time.sleep(2.5)
            last = max(t for t, _ in robot.rx)
            c("no commands any more (robot gets nothing, lying)", robot.t - last > 1.5, f"last {robot.t - last:.1f} s ago")
            c("supervisor and controller gone", not leftovers(), "; ".join(leftovers()))
            W = log_window(robot, t_kill)
            c("dog lying, never fell (tilt < 0.3)", W["z"][-10:].max() < 0.16 and np.abs(np.r_[W["roll"], W["pitch"]]).max() < 0.3,
              f"z {W['z'][-1]:.2f}")
        final_checks(s, c, robot, lying=False)
    finally:
        s.close()
        robot.stop()
    return c


def main() -> None:
    ChannelFactoryInitialize(DOMAIN, "lo")
    robot = F.FakeGo2()
    results = []
    plan = [lambda: sc_happy(robot, "hold"), lambda: sc_happy(robot, "passive"), lambda: sc_keyboard(robot),
            lambda: sc_estop_mid_dodge(robot, "L2+B"), lambda: sc_estop_mid_dodge(robot, "Ctrl+C"),
            lambda: sc_refused_and_busy(robot), lambda: sc_entry_guards(robot), lambda: sc_switcher_killed(robot),
            lambda: sc_gentle(robot), lambda: sc_happy(robot, "hold", "Right", "right"),
            lambda: sc_happy(robot, "passive", "Left", "left"), lambda: sc_gentle(robot, "right")]
    only = sys.argv[1:]
    for i, run in enumerate(plan):
        if only and str(i) not in only:
            continue
        t0 = time.monotonic()
        say(f"\n== scenario {i}")
        c = run()
        say(f"== {c.name}: {'PASS' if c.ok else 'FAIL'} ({time.monotonic() - t0:.0f} s)")
        results.append({"scenario": c.name, "ok": c.ok, "checks": c.items})
        time.sleep(1.0)
    n_ok = sum(r["ok"] for r in results)
    say(f"\n{n_ok}/{len(results)} scenarios passed")
    OUT.mkdir(exist_ok=True)
    (OUT / "modes_test.json").write_text(json.dumps(results, indent=1))
    (OUT / "modes_test.log").write_text("\n".join(LOG_LINES) + "\n")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()

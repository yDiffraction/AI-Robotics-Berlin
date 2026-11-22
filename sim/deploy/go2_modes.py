"""Switch the Go2 between Unitree's sport mode and our own controller with remote buttons (or keys).

  python go2_modes.py                 # laptop wired to the Go2; finds the 192.168.123.x interface itself

Our code only runs between "enter" and "back"; the rest of the time Unitree's sport mode is in charge
and this program only listens.

  SPORT mode (Unitree in control; walk with the remote as usual)
    hold L1+R1 for 1 s  (or key o)   -> the dog lies down (sport StandDown), sport mode is released,
                                        supervisor.py starts dodge_controller.py, the dog stands up
  OUR mode (dodge_controller.py under supervisor.py)
    D-pad Left  or A  (key l or d)   -> STILL dodge: the dog stands still until the press, then kicks
                                        at once, ~30 cm to the LEFT in ~0.5 s, brakes, stands still
                                        (--trot-start: the previous SAFE dodge, ~2.5 s trotting first)
    D-pad Right       (key r)        -> the same to the RIGHT (mirror image, validated in sim)
    Y        (or key s)              -> the dog lies down, sport mode comes back
    L2+B                             -> E-STOP: controller killed, damping (supervisor.py)
    SPACE or Ctrl+C (laptop)         -> same as L2+B. Repeated presses do nothing more (damping stays on
                                        until s / Y)
  after an E-STOP / trip (dog damped on the floor)
    Y        (or key s)              -> stop damping (after >= 2 s) and give the dog back to sport mode
  after sport mode is back the dog lies on the floor: stand it up with the remote as usual.

Buttons count only when pressed alone (A, Y, D-pad) or exactly together (L1+R1), seen for 60 ms; every
press needs a release first. (B is not used alone: it is half of the L2+B e-stop.) Key q quits (only in sport mode).
"""
from __future__ import annotations

import argparse
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safety  # noqa: E402

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.go2.sport.sport_client import SportClient  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, WirelessController_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

B = safety.BUTTON_BIT
ENTER_KEYS = (1 << B["L1"]) | (1 << B["R1"])
DODGE_LEFT_KEYS = (1 << B["A"], 1 << B["Left"])
DODGE_RIGHT_KEYS = 1 << B["Right"]
BACK_KEYS = 1 << B["Y"]
ENTER_HOLD_S = 1.0
STAND_DOWN_TIMEOUT = 8.0          # sport StandDown must have folded the legs within this
SELECT_TIMEOUT = 45.0             # SelectMode answers "busy" (7002) for ~10 s after ReleaseMode
IGNORE_AFTER_ENTER_S = 1.0
STABLE_SAMPLES = 3                # a button counts once it is seen alone for 3 polls (60 ms)
NAMES = {v: k for k, v in B.items()}


def key_names(k: int) -> str:
    return "+".join(NAMES[b] for b in range(16) if k >> b & 1) or "-"


class Button:
    """Fires once when exactly `keys` (nothing else) has been seen for STABLE_SAMPLES polls in a row.
    (The two button sources update at 500 / 50 Hz, so switching buttons can briefly look like a combo:
    plain edge detection missed presses that way.)"""

    def __init__(self, keys: int):
        self.keys, self.streak = keys, 0

    def update(self, keys: int) -> bool:
        self.streak = self.streak + 1 if keys == self.keys else 0
        return self.streak == STABLE_SAMPLES


def log(s: str) -> None:
    print(f"[modes] {s}", flush=True)


def find_iface() -> str | None:
    """The interface with a 192.168.123.x address (the Go2's network)."""
    out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"\d+:\s+(\S+)\s+inet\s+192\.168\.123\.(\d+)/", line)
        if m:
            return m.group(1)
    return None


class RobotView:
    """What this program reads from the robot: buttons, joints, tilt (rt/lowstate + rt/wirelesscontroller)."""

    def __init__(self):
        self.crc = CRC()
        self.lock = threading.Lock()
        self.keys_state = self.keys_remote = 0
        self.t_state = self.t_remote = 0.0
        self.q = None
        self.tilt = 0.0
        self.s1 = ChannelSubscriber("rt/lowstate", LowState_)
        self.s1.Init(self._on_state, 10)
        self.s2 = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.s2.Init(self._on_remote, 10)

    def _on_state(self, m: LowState_) -> None:
        if m.crc != self.crc.Crc(m):
            return
        with self.lock:
            self.t_state = time.monotonic()
            self.keys_state = safety.keys_from_remote_bytes(m.wireless_remote)
            self.q = [x.q for x in m.motor_state[:12]]
            self.tilt = max(map(abs, safety.tilt_from_quat(m.imu_state.quaternion)))

    def _on_remote(self, m: WirelessController_) -> None:
        with self.lock:
            self.t_remote = time.monotonic()
            self.keys_remote = m.keys

    def keys(self) -> int:
        """Buttons pressed according to either source (only sources heard in the last 0.2 s)."""
        now = time.monotonic()
        with self.lock:
            k = self.keys_state if now - self.t_state < 0.2 else 0
            return k | (self.keys_remote if now - self.t_remote < 0.2 else 0)

    def state_age(self) -> float:
        with self.lock:
            return time.monotonic() - self.t_state if self.t_state else float("inf")

    def lying(self) -> bool:
        with self.lock:
            return self.q is not None and safety.is_lying(self.q)

    def joints(self) -> str:
        with self.lock:
            return "none" if self.q is None else " ".join(f"{x:+.2f}" for x in self.q)


class Keyboard:
    """Single key presses from the terminal (nothing if stdin is not a terminal)."""

    def __init__(self):
        self.old = None
        if sys.stdin.isatty():
            import termios
            import tty
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())          # keys without Enter; Ctrl+C still works

    def read(self) -> str:
        if self.old is None:
            return ""
        out = ""
        while select.select([sys.stdin], [], [], 0)[0]:
            out += os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
        return out.lower()

    def restore(self) -> None:
        if self.old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)


class Modes:
    def __init__(self, a, iface: str | None):
        self.a, self.iface = a, iface
        self.view = RobotView()
        self.sc, self.msc = SportClient(), MotionSwitcherClient()
        for c in (self.sc, self.msc):
            c.SetTimeout(5.0)
            c.Init()
        self.sport_name = a.sport_mode
        self.mode = "sport"
        self.sup: subprocess.Popen | None = None
        self.t_entered = 0.0
        self.n_sigint = 0
        self.estop_sent = False

    # ---- sport side ------------------------------------------------------------------------
    def check_mode(self) -> str | None:
        """Name of the active motion mode ('' = released), None if the robot did not answer."""
        code, res = self.msc.CheckMode()
        if code != 0 or res is None:
            log(f"motion_switcher did not answer (code {code})")
            return None
        return res.get("name", "")

    def enter(self) -> None:
        name = self.check_mode()
        if name is None:
            return
        if name:
            self.sport_name = name
            log(f"switching to OUR mode: sport mode '{name}' lies the dog down ...")
            self.sc.StandDown()
            t0 = time.monotonic()
            while not self.view.lying():
                if time.monotonic() - t0 > STAND_DOWN_TIMEOUT:
                    log(f"ABORT: legs not folded {STAND_DOWN_TIMEOUT:.0f} s after StandDown (joints {self.view.joints()}). "
                        "Sport mode stays in control.")
                    return
                time.sleep(0.05)
            for _ in range(10):
                self.msc.ReleaseMode()
                name = self.check_mode()
                if name == "":
                    break
                self.sc.StandDown()
                time.sleep(1.0)
            if name != "":
                log(f"ABORT: sport mode could not be released (still '{name}'). Sport mode stays in control.")
                return
            log("sport mode released")
        elif not self.view.lying():
            log(f"sport mode is already off but the dog is not lying (joints {self.view.joints()}): not starting")
            return
        cmd_ctl = [sys.executable, "-u", str(HERE / "dodge_controller.py"), "--domain", str(self.a.domain)]
        cmd_sup = [sys.executable, "-u", str(HERE / "supervisor.py"), "--domain", str(self.a.domain),
                   "--startup-timeout", "30"]
        if self.iface:
            cmd_ctl += ["--iface", self.iface]
            cmd_sup += ["--iface", self.iface]
        if self.a.no_remote_check:
            cmd_sup.append("--no-remote-check")
        if self.a.gentle:
            cmd_ctl.append("--gentle")
        if self.a.trot_start:
            cmd_ctl.append("--trot-start")
        # own session: a terminal Ctrl+C reaches only this program, which forwards it deliberately
        self.sup = subprocess.Popen(cmd_sup + ["--"] + cmd_ctl, stdin=subprocess.PIPE, text=True,
                                    start_new_session=True)
        self.mode, self.t_entered = "ours", time.monotonic()
        self.estop_sent = False
        log("OUR mode: dog stands up. D-pad Left / A = dodge left, D-pad Right = dodge right, "
            "Y = back to sport mode, L2+B = E-STOP")

    def hand_back(self) -> None:
        name = self.sport_name
        log(f"giving the dog back to sport mode '{name}' ...")
        t0 = time.monotonic()
        busy_said = False
        while time.monotonic() - t0 < SELECT_TIMEOUT:
            code, _ = self.msc.SelectMode(name)
            now_name = self.check_mode()
            if now_name:
                self.mode = "sport"
                log(f"SPORT mode '{now_name}' is back (dog lying: {self.view.lying()}). Stand it up with the remote as usual. "
                    "Hold L1+R1 1 s (or o) for our mode again.")
                return
            if not busy_said:
                log(f"sport service not ready yet (code {code}), retrying for up to {SELECT_TIMEOUT:.0f} s")
                busy_said = True
            time.sleep(1.0)
        self.mode = "off"
        log("could not restore sport mode. The dog gets NO commands now. Y / s = try again, "
            "or switch the mode in the Unitree app.")

    # ---- our side --------------------------------------------------------------------------
    def tell(self, line: str) -> None:
        try:
            self.sup.stdin.write(line + "\n")
            self.sup.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def signal_sup(self, sig) -> None:
        try:
            os.kill(self.sup.pid, sig)
        except ProcessLookupError:
            pass

    # ---- main loop -------------------------------------------------------------------------
    def on_sigint(self, *_):
        self.n_sigint += 1

    def run(self) -> None:
        kb = Keyboard()
        try:
            self._run(kb)
        finally:
            kb.restore()
            if self.sup is not None and self.sup.poll() is None:
                log("exiting while our mode is active: the controller lies the dog down by itself")

    def _run(self, kb: Keyboard) -> None:
        t0 = time.monotonic()
        while self.view.state_age() > 1.0:
            if time.monotonic() - t0 > 10:
                log("no rt/lowstate from the robot within 10 s. Check: cable, laptop IP 192.168.123.x/24, "
                    f"interface ({self.iface}), WSL mirrored networking + Hyper-V firewall rule (README).")
                return
            time.sleep(0.05)
        name = self.check_mode()
        if name is None:
            return
        if name:
            self.sport_name = name
            log(f"robot connected, SPORT mode '{name}' active. Hold L1+R1 1 s (or o) for our mode, q quits.")
        else:
            self.mode = "off"
            log("robot connected, but sport mode is OFF (released). Y / s = give it back to sport mode, "
                "L1+R1 / o = our mode (dog must be lying), q quits.")
        left_btns = [Button(k) for k in DODGE_LEFT_KEYS]
        right_btn, back_btn = Button(DODGE_RIGHT_KEYS), Button(BACK_KEYS)
        shown = 0
        hold_t = None
        lost_said = False
        handled_sigint = 0
        while True:
            time.sleep(0.02)
            now = time.monotonic()
            typed = kb.read()
            keys = self.view.keys()
            want_left = any([b.update(keys) for b in left_btns])
            want_right, want_back = right_btn.update(keys), back_btn.update(keys)
            if keys != shown and keys:
                log(f"buttons {key_names(keys)}")
            shown = keys
            if self.view.state_age() > 1.0 and not lost_said:
                log("WARNING: no rt/lowstate for 1 s (cable?)")
                lost_said = True
            elif self.view.state_age() < 0.2:
                lost_said = False

            new_sigint = self.n_sigint > handled_sigint
            handled_sigint = self.n_sigint
            if self.mode in ("sport", "off"):
                if new_sigint or "q" in typed:
                    log("bye")
                    return
                # hold_t: None = idle, time = L1+R1 held since, -1 = fired, waiting for all buttons released
                if keys == ENTER_KEYS:
                    if hold_t is None:
                        hold_t = now
                elif keys == 0:
                    hold_t = None
                elif hold_t != -1:
                    hold_t = None                       # another button joined: start over
                fire = hold_t is not None and hold_t != -1 and now - hold_t >= ENTER_HOLD_S
                if fire or "o" in typed:
                    hold_t = -1                         # needs a release before the next trigger
                    self.enter()
                elif self.mode == "off" and (want_back or "s" in typed):
                    self.hand_back()
                continue

            # ---- our mode ----
            code = self.sup.poll()
            if code is not None:
                what = {0: "controller finished, dog lying", 1: "damping stopped after a trip",
                        2: "stopped before our controller took over (nothing sent)"}.get(code, f"exit code {code}")
                log(f"supervisor ended: {what}")
                self.sup.stdin.close()
                self.sup = None
                self.hand_back()
                continue
            if new_sigint or " " in typed:
                if not self.estop_sent:
                    self.estop_sent = True
                    log(f"{'Ctrl+C' if new_sigint else 'SPACE'} -> E-STOP via the supervisor "
                        "(s / Y afterwards gives the dog back to sport mode)")
                    self.signal_sup(signal.SIGINT)
                else:                                  # a 2nd SIGINT would make the supervisor stop damping
                    log("E-STOP already active (damping). s / Y gives the dog back to sport mode.")
            if now - self.t_entered < IGNORE_AFTER_ENTER_S:
                continue
            if want_left or "d" in typed or "l" in typed:
                self.tell("dodge left")
            elif want_right or "r" in typed:
                self.tell("dodge right")
            if want_back or "s" in typed:
                log("back to sport mode requested")
                self.tell("down")
                self.signal_sup(signal.SIGUSR1)    # after a trip: release the damping (>= 2 s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto", help="interface wired to the Go2 (default: the one with 192.168.123.x)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--sport-mode", default="normal", help="mode to restore if it cannot be read from the robot")
    ap.add_argument("--gentle", action="store_true", help="dodge with scaled-down sideways speed (first tries)")
    ap.add_argument("--trot-start", action="store_true",
                    help="previous SAFE dodge (~2 s trotting in place before the kick; drifted on the real dog)")
    ap.add_argument("--no-remote-check", action="store_true", help="passed to supervisor.py (see remote_probe.py)")
    a = ap.parse_args()
    iface = find_iface() if a.iface == "auto" else a.iface
    if a.iface == "auto" and iface is None:
        log("no network interface with a 192.168.123.x address. Wire the Go2 and give the laptop 192.168.123.222/24 "
            "(README), or pass --iface.")
        sys.exit(1)
    log(f"DDS domain {a.domain}, interface {iface}")
    ChannelFactoryInitialize(a.domain, iface)
    m = Modes(a, iface)
    signal.signal(signal.SIGINT, m.on_sigint)
    signal.signal(signal.SIGTERM, m.on_sigint)
    m.run()


if __name__ == "__main__":
    main()

"""Run our low-level Go2 controller under a separate safety process.

  python supervisor.py --iface eth0 -- python my_controller.py --its-args
  (on the robot: domain 0 and the interface wired to the Go2; tests use --domain 77 --iface lo)

What it does
  * starts the controller as a child process in its own session (terminal Ctrl+C reaches only the
    supervisor) that is killed automatically if the supervisor itself dies (PR_SET_PDEATHSIG), so
    a controller never runs unsupervised;
  * listens to rt/lowstate, rt/wirelesscontroller and the controller's rt/lowcmd and runs
    safety.Supervisor (e-stop L2+B, tilt, controller silent/garbage, robot state lost, tick stuck,
    remote silent, Ctrl+C);
  * on a trip: SIGKILLs the controller's process group and publishes DAMPING (kp=0, kd=5) at 500 Hz
    until stopped with a second Ctrl+C (or --damp-for seconds). Only one process ever publishes
    rt/lowcmd at a time.
  * Before the controller has sent its first valid command, the supervisor never publishes anything
    (sport mode may still be active then).
  * Clean finish: the controller exits with code 0 while the dog lies with folded legs -> damping
    for 1 s, then exit 0. Exit code 0 with the dog NOT lying is a trip like any other.
  * SIGUSR1 (go2_modes.py, operator wants sport mode back): after a trip, stop sending damping once
    it has been applied for 2 s, and exit 1. Ignored while the controller runs normally.

What it cannot do: if the cable between this computer and the Go2 is pulled, or this computer
freezes, nothing reaches the robot. What the Go2 does without commands has to be tested once on
the hanging robot (see README).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

signal.signal(signal.SIGUSR1, signal.SIG_IGN)   # until the real handler is set (default action would kill us)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safety  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, WirelessController_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

PERIOD = 0.002
FINISH_DAMP_S = 1.0      # damping after a clean finish (dog already lying)
RELEASE_AFTER_S = 2.0    # SIGUSR1 releases damping only after it has been applied this long


def damping_command() -> LowCmd_:
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0], cmd.head[1] = 0xFE, 0xEF
    cmd.level_flag = 0xFF
    for i in range(20):
        m = cmd.motor_cmd[i]
        m.mode = 0x01
        m.q, m.dq, m.kp, m.tau = 0.0, 0.0, 0.0, 0.0
        m.kd = safety.DAMP_KD if i < 12 else 0.0
    cmd.crc = CRC().Crc(cmd)
    return cmd


def command_arrays(msg: LowCmd_):
    m = msg.motor_cmd[:12]
    return ([x.q for x in m], [x.dq for x in m], [x.kp for x in m], [x.kd for x in m],
            [x.tau for x in m], [x.mode for x in m])


def _die_with_parent() -> None:
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except OSError:
        pass


class SupervisorNode:
    def __init__(self, child_cmd: list[str], limits: safety.Limits, damp_for: float | None, log=lambda s: print(s, flush=True)):
        self.log = log
        self.crc = CRC()
        self.lock = threading.Lock()
        self.sup = safety.Supervisor(limits, time.monotonic())
        self.damp_for = damp_for
        self.damping = damping_command()
        self.sending = False
        self.n_sigint = 0
        self.signal_name = None
        self.release_requested = False
        self.q = None                     # last valid joint positions from rt/lowstate (SDK order)
        self.stats = {"state": 0, "remote": 0, "cmd": 0, "cmd_bad_crc": 0, "state_bad_crc": 0}
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        for topic, typ, cb in (("rt/lowstate", LowState_, self._on_state),
                               ("rt/wirelesscontroller", WirelessController_, self._on_remote),
                               ("rt/lowcmd", LowCmd_, self._on_cmd)):
            s = ChannelSubscriber(topic, typ)
            s.Init(cb, 0)
            setattr(self, "_sub_" + topic.split("/")[1], s)
        self.child = subprocess.Popen(child_cmd, start_new_session=True, preexec_fn=_die_with_parent)
        self.log(f"[supervisor] controller started, pid {self.child.pid}: {' '.join(child_cmd)}")

    # ---- DDS callbacks (Cyclone listener threads) -------------------------------------------
    def _on_state(self, msg: LowState_) -> None:
        ok = msg.crc == self.crc.Crc(msg)
        with self.lock:
            self.stats["state" if ok else "state_bad_crc"] += 1
            if ok:
                self.q = [m.q for m in msg.motor_state[:12]]
            self.sup.on_state(time.monotonic(), ok, msg.tick, msg.imu_state.quaternion, msg.wireless_remote)

    def _on_remote(self, msg: WirelessController_) -> None:
        with self.lock:
            self.stats["remote"] += 1
            self.sup.on_remote(time.monotonic(), msg.keys)

    def _on_cmd(self, msg: LowCmd_) -> None:
        if self.sending:
            return                       # our own damping (only we publish once tripped)
        ok = msg.crc == self.crc.Crc(msg)
        problem = safety.check_command(*command_arrays(msg)) if ok else None
        with self.lock:
            self.stats["cmd" if ok else "cmd_bad_crc"] += 1
            self.sup.on_command(time.monotonic(), ok, problem)

    def on_signal(self, signum, _frame):
        # runs in the main thread between bytecodes: only set flags (taking self.lock here could deadlock)
        if signum == signal.SIGUSR1:
            self.release_requested = True
            return
        self.n_sigint += 1
        self.signal_name = "Ctrl+C" if signum == signal.SIGINT else "SIGTERM"

    # ---- main loop ------------------------------------------------------------------------------
    def _kill_child(self) -> None:
        if self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()

    def run(self) -> int:
        next_t = time.monotonic()
        tripped_at = None
        finished_at = None
        while True:
            now = time.monotonic()
            with self.lock:
                if self.signal_name:
                    self.sup.on_operator_stop(now, self.signal_name)
                code = self.child.poll()
                if code is not None and not self.sup.tripped and finished_at is None:
                    if code == 0 and self.sup.controller_started and self.q is not None and safety.is_lying(self.q):
                        finished_at = now
                        self.sending = True
                        self.log("[supervisor] controller finished, dog lying -> damping for "
                                 f"{FINISH_DAMP_S:.0f} s, then done")
                    else:
                        lying = self.q is not None and safety.is_lying(self.q)
                        self.sup.trip(now, f"controller exited (code {code})" + ("" if lying else ", dog not lying"))
                reason = self.sup.check(now) if finished_at is None else None
                armed = self.sup.controller_started
            if finished_at is not None:
                self.pub.Write(self.damping)
                if now - finished_at >= FINISH_DAMP_S:
                    return 0
                next_t += PERIOD
                time.sleep(max(0.0, next_t - time.monotonic()))
                continue
            if self.release_requested and not self.sending:
                self.release_requested = False          # only meaningful after a trip
            if reason and tripped_at is None:
                tripped_at = now
                self._kill_child()
                if not armed:
                    self.log(f"[supervisor] STOP before the controller took over: {reason}. Nothing sent.")
                    return 2
                self.sending = True
                self.log(f"[supervisor] TRIP: {reason} -> controller killed, sending DAMPING. "
                         f"Ctrl+C to stop sending.")
            if self.sending:
                self.pub.Write(self.damping)
                released = self.release_requested and now - tripped_at >= RELEASE_AFTER_S
                if released or self.n_sigint >= 2 or (self.damp_for is not None and now - tripped_at > self.damp_for):
                    self.log("[supervisor] stopped sending damping"
                             + (" (release requested)" if released else "") + "; the robot now gets no commands.")
                    return 1
            elif self.n_sigint >= 2:
                return 1
            next_t += PERIOD
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None, help="network interface wired to the Go2 (e.g. eth0)")
    ap.add_argument("--no-remote-check", action="store_true",
                    help="do not trip when rt/wirelesscontroller goes silent")
    ap.add_argument("--damp-for", type=float, default=None, help="stop sending damping after this long [s]")
    ap.add_argument("--startup-timeout", type=float, default=60.0)
    ap.add_argument("child", nargs=argparse.REMAINDER, help="-- controller command")
    a = ap.parse_args()
    child = a.child[1:] if a.child[:1] == ["--"] else a.child
    if not child:
        ap.error("give the controller command after --")
    ChannelFactoryInitialize(a.domain, a.iface)
    node = SupervisorNode(child, safety.Limits(remote_check=not a.no_remote_check,
                                               startup_timeout=a.startup_timeout), a.damp_for)
    signal.signal(signal.SIGINT, node.on_signal)
    signal.signal(signal.SIGTERM, node.on_signal)
    signal.signal(signal.SIGUSR1, node.on_signal)
    try:
        code = node.run()
    finally:
        node._kill_child()
        print(f"[supervisor] stats {node.stats}; events {node.sup.events}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()

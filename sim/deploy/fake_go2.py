"""A MuJoCo Go2 that speaks the real Unitree DDS interface, for testing supervisor.py end to end.

Publishes rt/lowstate (500 Hz, tick in ms, IMU, joints, wireless_remote, CRC) and
rt/wirelesscontroller (50 Hz); applies rt/lowcmd (CRC-checked) as joint PD like the motor drivers.
Faults can be injected from the test: remote buttons, stop publishing state / remote, freeze the
tick, push the trunk.

What the real Go2 does when commands stop is unknown; `on_silence` models both guesses
("hold": keep the last command, "passive": zero torque after 0.1 s).

Sport mode (for go2_modes.py): fake 'motion_switcher' (CheckMode / SelectMode / ReleaseMode) and
'sport' (StandDown) RPC services on the real Unitree RPC layer. While a sport mode is active the fake
holds its own pose (PD 60/5) and ignores rt/lowcmd, recording any as 'while_sport' (must never
happen). SelectMode answers 7002 (busy) for SELECT_BUSY_S after ReleaseMode, as reported for the
real Go2.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import go2_dodge_sim as g  # noqa: E402
import safety  # noqa: E402

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import (unitree_go_msg_dds__LowState_,  # noqa: E402
                                        unitree_go_msg_dds__WirelessController_)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, WirelessController_  # noqa: E402
from unitree_sdk2py.rpc.server import Server  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

DT = 0.002
SELECT_BUSY_S = 10.0
SPORT_STAND = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
SPORT_LIE = np.array([-0.2, 1.25, -2.55, 0.2, 1.25, -2.55, -0.2, 1.25, -2.55, 0.2, 1.25, -2.55])


class _FakeSwitcher(Server):
    def __init__(self, robot: "FakeGo2"):
        super().__init__("motion_switcher")
        self.robot = robot

    def Init(self):
        self._SetApiVersion("1.0.0.1")
        self._RegistHandler(1001, lambda p: (0, json.dumps({"form": "0", "name": self.robot.sport_name})), False)
        self._RegistHandler(1002, self._select, False)
        self._RegistHandler(1003, self._release, False)

    def _select(self, p):
        r = self.robot
        if time.monotonic() - r.released_at < SELECT_BUSY_S:
            return 7002, ""
        r.sport_on(json.loads(p)["name"])
        return 0, ""

    def _release(self, _p):
        self.robot.sport_off()
        return 0, ""


class _FakeSport(Server):
    def __init__(self, robot: "FakeGo2"):
        super().__init__("sport")
        self.robot = robot

    def Init(self):
        self._SetApiVersion("1.0.0.1")
        self._RegistHandler(1005, self._stand_down, False)

    def _stand_down(self, _p):
        return (0, "") if self.robot.sport_move(SPORT_LIE, 1.5) else (3103, "")


class FakeGo2:
    """DDS endpoints are created once (Python DDS readers are not reliably released); reset() starts a
    fresh MuJoCo episode on the same endpoints."""

    def __init__(self):
        self.crc = CRC()
        self.state = unitree_go_msg_dds__LowState_()
        self.state.head[0], self.state.head[1] = 0xFE, 0xEF
        self.remote = unitree_go_msg_dds__WirelessController_()
        self.pub_state = ChannelPublisher("rt/lowstate", LowState_)
        self.pub_state.Init()
        self.pub_remote = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.pub_remote.Init()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub.Init(self._on_cmd, 0)
        self.sport_name = ""
        self.released_at = -1e9
        self.servers = [_FakeSwitcher(self), _FakeSport(self)]
        for srv in self.servers:
            srv.Init()
            srv.Start(False)

    # ---- fake sport mode (called from the RPC threads) -------------------------------------------
    def sport_on(self, name: str) -> None:
        with self.lock:
            q = self.d.qpos[self.qadr].copy()
            self.sport_from = self.sport_to = q
            self.sport_t0, self.sport_dur = self.t, 1e-6
            self.sport_name = name

    def sport_off(self) -> None:
        with self.lock:
            if self.sport_name:
                self.released_at = time.monotonic()
            self.sport_name = ""

    def sport_move(self, target, duration: float) -> bool:
        with self.lock:
            if not self.sport_name:
                return False
            self.sport_from = self.sport_target(self.t)
            self.sport_to, self.sport_t0, self.sport_dur = np.asarray(target, float), self.t, duration
            return True

    def sport_target(self, t: float) -> np.ndarray:
        s = min(1.0, (t - self.sport_t0) / self.sport_dur)
        s = 0.5 - 0.5 * math.cos(math.pi * s)
        return self.sport_from + s * (self.sport_to - self.sport_from)

    def reset(self, on_silence: str = "hold", sport: str = "") -> "FakeGo2":
        """New episode. on_silence: what the 'firmware' does without commands.
        sport='': robot lying on the floor, sport mode released; sport='normal': standing in that sport mode."""
        if self.running:
            self.stop()
        m = g.flat_scene_spec().compile()
        m.opt.timestep = DT
        d = mujoco.MjData(m)
        self.m, self.d, self.on_silence = m, d, on_silence
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in safety.SDK_JOINTS])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in safety.SDK_JOINTS])
        self.act = np.array([m.actuator(n.replace("_joint", "")).id for n in safety.SDK_JOINTS])
        self.tau_max = m.actuator_ctrlrange[self.act, 1]
        self.trunk = m.body("base_link").id
        sens = {m.sensor(k).name: (m.sensor_adr[k], m.sensor_dim[k]) for k in range(m.nsensor)}
        self.s_quat, self.s_gyro, self.s_acc = sens["imu_quat"], sens["imu_gyro"], sens["imu_acc"]
        d.qpos[3:7] = [1, 0, 0, 0]
        if sport:
            d.qpos[2] = 0.30
            d.qpos[self.qadr] = SPORT_STAND
        else:
            d.qpos[2] = 0.15
            d.qpos[self.qadr] = [-0.1, 1.3, -2.6, 0.1, 1.3, -2.6, -0.1, 1.3, -2.6, 0.1, 1.3, -2.6]
        mujoco.mj_forward(m, d)
        self.t = 0.0
        with self.lock:
            self.sport_name = sport
            self.released_at = -1e9
            self.sport_from = self.sport_to = d.qpos[self.qadr].copy()
            self.sport_t0, self.sport_dur = 0.0, 1e-6
            self.cmd = None           # (q, dq, kp, kd, tau) arrays of the last valid command
            self.cmd_t = None
            self.rx = []              # (t, kind) per received command: child | damp | bad_crc | nonfinite
        # fault switches (set from the test)
        self.keys_state = 0           # buttons as seen in lowstate.wireless_remote
        self.keys_remote = 0          # buttons as seen on rt/wirelesscontroller
        self.state_paused = self.remote_paused = self.tick_frozen = False
        self.push = np.zeros(6)
        self.t = 0.0
        self.log = {"t": [], "x": [], "y": [], "z": [], "roll": [], "pitch": [], "acc_g": []}
        return self

    def now(self) -> float:
        return self.t

    def _on_cmd(self, msg: LowCmd_) -> None:
        if not self.running:
            return
        t = self.t
        if msg.crc != self.crc.Crc(msg):
            self.rx.append((t, "bad_crc"))
            return
        mc = msg.motor_cmd[:12]
        arr = np.array([[x.q, x.dq, x.kp, x.kd, x.tau] for x in mc]).T
        if not np.isfinite(arr).all():
            self.rx.append((t, "nonfinite"))
            return
        kind = "damp" if (arr[2] == 0).all() and np.allclose(arr[3], safety.DAMP_KD) else "child"
        with self.lock:
            if self.running and self.sport_name:
                self.rx.append((t, "while_sport"))      # the real robot would get two masters
            elif self.running:
                self.cmd, self.cmd_t = arr, t
                self.rx.append((t, kind))

    def start(self) -> "FakeGo2":
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        self.thread.join()

    def _loop(self) -> None:
        m, d = self.m, self.d
        wall0 = time.monotonic()
        k = 0
        tick = 0
        while self.running:
            with self.lock:
                cmd, cmd_t = self.cmd, self.cmd_t
                sport = self.sport_target(self.t) if self.sport_name else None
                if sport is not None:
                    self.cmd = None           # after sport mode, old low-level commands are gone
            q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
            if sport is not None:
                tau = 60.0 * (sport - q) - 5.0 * dq
            elif cmd is None or (self.on_silence == "passive" and self.t - cmd_t > 0.1):
                tau = np.zeros(12)
            else:
                qd, dqd, kp, kd, ff = cmd
                tau = kp * (qd - q) + kd * (dqd - dq) + ff
            d.ctrl[self.act] = np.clip(tau, -self.tau_max, self.tau_max)
            d.xfrc_applied[self.trunk] = self.push
            mujoco.mj_step(m, d)
            self.t += DT
            k += 1
            if not self.tick_frozen:
                tick = int(self.t * 1000)
            quat = d.sensordata[self.s_quat[0]:self.s_quat[0] + 4]
            acc = d.sensordata[self.s_acc[0]:self.s_acc[0] + 3]
            if not self.state_paused:
                s = self.state
                s.imu_state.quaternion = [float(x) for x in quat]
                s.imu_state.gyroscope = [float(x) for x in d.sensordata[self.s_gyro[0]:self.s_gyro[0] + 3]]
                s.imu_state.accelerometer = [float(x) for x in acc]
                roll, pitch = safety.tilt_from_quat(quat)
                s.imu_state.rpy = [roll, pitch, 0.0]
                for i in range(12):
                    ms = s.motor_state[i]
                    ms.q, ms.dq, ms.tau_est = float(q[i]), float(dq[i]), float(d.ctrl[self.act[i]])
                s.tick = tick
                s.wireless_remote = safety.remote_bytes(self.keys_state)
                s.crc = self.crc.Crc(s)
                self.pub_state.Write(s)
            if k % 10 == 0 and not self.remote_paused:
                self.remote.keys = self.keys_remote
                self.pub_remote.Write(self.remote)
            if k % 5 == 0:
                L = self.log
                L["t"].append(self.t)
                L["x"].append(float(d.qpos[0]))
                L["y"].append(float(d.qpos[1]))
                L["z"].append(float(d.qpos[2]))
                roll, pitch = safety.tilt_from_quat(quat)
                L["roll"].append(roll)
                L["pitch"].append(pitch)
                a = d.xmat[self.trunk].reshape(3, 3) @ acc
                L["acc_g"].append(math.sqrt(a[0] ** 2 + a[1] ** 2 + (a[2] - 9.81) ** 2) / 9.81)
            # real-time pacing
            delay = wall0 + self.t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.05:               # fell behind (e.g. CPU hiccup): do not try to catch up
                wall0 = time.monotonic() - self.t

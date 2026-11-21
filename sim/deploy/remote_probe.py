"""Read-only probe for the real Go2: answers the questions the supervisor's checks depend on.
It only subscribes; it never publishes, so it is safe with sport mode on.

  python remote_probe.py --iface auto         # Ctrl+C to stop (Windows: run_go2.ps1 probe)

Prints twice a second:
  lowstate   rate [Hz], tick (must increase), CRC failures, e-stop bits seen, max tilt
  remote     rt/wirelesscontroller rate [Hz], keys, wireless_remote head bytes + keys
  motors     'lost' counter increase since start, max temperature

Do this once:
  0. press L1+R1, A, Y, D-pad Left and Right: each must show under "bytes ... keys" (go2_modes.py reads them there)
  1. press L2+B (sport mode damps the dog, so do it while it lies down)  -> "ESTOP" must show in both
  2. switch the remote OFF for 5 s, then on again -> note what changes:
       rt/wirelesscontroller rate drops to 0     => keep the default remote check
       rate stays, head bytes change             => tell us; the check can use the bytes instead
       nothing changes                           => the remote-off case cannot be detected; run the
                                                    supervisor with --no-remote-check and keep a
                                                    second person at the power button
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safety  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, WirelessController_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=None, help="'auto' = the interface with a 192.168.123.x address")
    ap.add_argument("--domain", type=int, default=0)
    a = ap.parse_args()
    if a.iface == "auto":
        from go2_modes import find_iface
        a.iface = find_iface()
        if a.iface is None:
            sys.exit("no interface with a 192.168.123.x address: wire the Go2 and run setup_laptop.ps1 (README)")
        print(f"interface {a.iface}", flush=True)
    ChannelFactoryInitialize(a.domain, a.iface)
    crc = CRC()
    lock = threading.Lock()
    st = {"n_state": 0, "n_remote": 0, "bad_crc": 0, "tick": None, "head": None, "keys_s": 0, "keys_r": 0,
          "lost0": None, "lost": None, "temp": 0, "tilt": 0.0}

    def on_state(m: LowState_):
        ok = m.crc == crc.Crc(m)
        with lock:
            st["n_state"] += 1
            st["bad_crc"] += not ok
            st["tick"] = m.tick
            st["head"] = bytes(m.wireless_remote[:2]).hex()
            st["keys_s"] = safety.keys_from_remote_bytes(m.wireless_remote)
            lost = [x.lost for x in m.motor_state[:12]]
            st["lost0"] = st["lost0"] or lost
            st["lost"] = lost
            st["temp"] = max(x.temperature for x in m.motor_state[:12])
            st["tilt"] = max(map(abs, safety.tilt_from_quat(m.imu_state.quaternion)))

    def on_remote(m: WirelessController_):
        with lock:
            st["n_remote"] += 1
            st["keys_r"] = m.keys

    s1 = ChannelSubscriber("rt/lowstate", LowState_)
    s1.Init(on_state, 0)
    s2 = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    s2.Init(on_remote, 0)
    last = time.monotonic()
    prev = dict(st)
    names = {v: k for k, v in safety.BUTTON_BIT.items()}
    try:
        while True:
            time.sleep(0.5)
            now = time.monotonic()
            with lock:
                cur = dict(st)
            dt = now - last
            last = now

            def keys(k):
                return "ESTOP" if safety.is_estop(k) else ("+".join(names[b] for b in range(16) if k >> b & 1) or "-")
            lost_up = [b - a for a, b in zip(cur["lost0"] or [], cur["lost"] or [])]
            print(f"lowstate {(cur['n_state'] - prev['n_state']) / dt:6.0f} Hz tick {cur['tick']} crc-bad {cur['bad_crc']} "
                  f"tilt {cur['tilt']:.2f} | remote topic {(cur['n_remote'] - prev['n_remote']) / dt:5.0f} Hz "
                  f"keys {keys(cur['keys_r']):10s} | bytes head {cur['head']} keys {keys(cur['keys_s']):10s} "
                  f"| lost +{max(lost_up, default=0)} temp {cur['temp']} C", flush=True)
            prev = cur
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

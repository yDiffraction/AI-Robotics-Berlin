#!/usr/bin/env python3
"""Board control: which model is loaded, and swapping in a new board folder (backup, copy, restart, md5 check).

  python live_early/hitl/board_ctl.py status
  python live_early/hitl/board_ctl.py load live_early/model_v3          # back to v3
  python live_early/hitl/board_ctl.py load runs/hitl/r1/board           # a fine-tuned model from train.py

--ssh defaults to root@192.168.77.1; with the board cabled straight to the laptop and no IPv4 address on that
adapter, use the IPv6 link-local address, e.g. --ssh "root@fe80::20a:35ff:fe28:d34c%50" (%50 = adapter index).
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWN = {"a0a6f2a9afa1af61fa5d49f6fa33918e": "v2b (live_early/model)", "44f403275d8ea3c87d4e0bf93078cf08": "v3 (live_early/model_v3)"}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def ssh(target, cmd):
    r = subprocess.run(["ssh", *SSH_OPTS, target, cmd], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise SystemExit(f"ssh failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout.strip()


def scp_target(target):
    user, _, host = target.rpartition("@")
    host = f"[{host}]" if ":" in host else host          # IPv6 needs brackets for scp
    return f"{user}@{host}" if user else host


def name_of(md5):
    for p in sorted((HERE / "..").resolve().glob("**/board/weights.txt")) + sorted((HERE / "runs").glob("**/weights.txt")):
        if hashlib.md5(p.read_bytes()).hexdigest() == md5:
            return str(p.parent)
    return KNOWN.get(md5, "unknown")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "load"])
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--ssh", default="root@192.168.77.1")
    a = ap.parse_args()
    if a.cmd == "status":
        md5 = ssh(a.ssh, "md5sum /root/weights.txt | cut -c1-32")
        print(f"board weights md5 {md5} = {KNOWN.get(md5) or name_of(md5)}; spleat-server "
              + ssh(a.ssh, "systemctl is-active spleat-server || true"))
        return
    w = Path(a.folder or "") / "weights.txt"
    if not w.exists():
        sys.exit(f"{w} not found")
    md5 = hashlib.md5(w.read_bytes()).hexdigest()
    n = len(w.read_text().split())
    if n != 78501:
        sys.exit(f"{w}: {n} values, the board firmware expects 78501")
    print(ssh(a.ssh, "cp -n /root/weights.txt /root/weights_backup_$(md5sum /root/weights.txt | cut -c1-32).txt; "
                     "ls /root/weights_backup_*"))
    subprocess.run(["scp", *SSH_OPTS, str(w), f"{scp_target(a.ssh)}:/root/weights.txt"], check=True)
    got = ssh(a.ssh, "systemctl restart spleat-server && sleep 2 && systemctl is-active spleat-server && md5sum /root/weights.txt | cut -c1-32")
    state, board_md5 = got.split()[-2:]
    print(f"spleat-server {state}; board md5 {board_md5}; local {md5}")
    if board_md5 != md5 or state != "active":
        sys.exit("LOAD FAILED: md5 mismatch or server not active")
    print(f"loaded {a.folder}. Run collect.py with --model {a.folder} so the readout uses its theta.")


if __name__ == "__main__":
    main()

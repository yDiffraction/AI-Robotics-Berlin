#!/usr/bin/env python3
"""Collect here, train on the GPU machine at the same time, and decide yourself when the board gets new weights.

  camera ─▶ laptop ─▶ Kria (SPLEAT) ─▶ readout with YOUR theta ─▶ page at http://127.0.0.1:8771 (press SPACE)
     │ every --chunk-s seconds: new chunks + presses + theta ─scp─▶ A100: online_train.py (RL by default)
     │                                                                 │ snapshot every --snapshot-s
     └──── "Update board now" / auto every N min ◀── fetch snapshot ◀──┘ ─▶ scp to the Kria, restart spleat, md5 check

The page has: a theta slider (alarm threshold of the readout, also the theta the RL reward uses on the trainer),
"Update board now" (loads the newest snapshot), an auto-update interval (0 = off, i.e. only when you press the
button), "Back to start model", and the trainer's status. While the board restarts (~3 s) frames are still recorded
with spike = -1.

  python perception/hitl/live_loop.py --host "fe80::20a:35ff:fe28:d34c%50"
  python perception/hitl/live_loop.py --host ... --loss pg,mil --train-args "--bs 16 --lr 1e-4" --snapshot-s 60
  python perception/hitl/live_loop.py --host ... --replay perception/test/x.npy     # replay instead of the camera
"""
import argparse
import io
import json
import subprocess
import sys
import tarfile
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import board_ctl  # noqa: E402
import collect  # noqa: E402

viewer = collect.viewer
from readout_spec import Live as LiveReadout, Spec  # noqa: E402

CODE = ["perception/hitl/train.py", "perception/hitl/online_train.py", "perception/hitl/export_board.py",
        "perception/hitl/readout_spec.py", "perception/snn_model.py", "perception/speedpatch.py",
        "training/early_ctx/labels.json"]
MODEL_FILES = ("model.pt", "model.json", "config.yaml", "readout.json", "weights.txt")


class Gpu:
    """The GPU machine over SSH: code sync, trainer process, uploads, status, snapshot download."""

    def __init__(self, a, run_id):
        self.host, self.dir, self.py = a.gpu, a.gpu_dir, a.gpu_python
        self.opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15"]
        if a.gpu_key:
            self.opts = ["-i", str(Path(a.gpu_key).expanduser())] + self.opts
        self.root = f"{self.dir}/live/{run_id}"        # relative to the remote home
        self.proc = None

    def ssh(self, cmd, data=None, timeout=60):
        r = subprocess.run(["ssh", *self.opts, self.host, cmd], input=data, capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"ssh {self.host}: {r.stderr.decode(errors='replace').strip()[-300:]}")
        return r.stdout.decode(errors="replace")

    def scp(self, srcs, dst, cwd=None, timeout=120):
        """Local paths relative to cwd: Windows scp would read 'C:' as a host name."""
        r = subprocess.run(["scp", "-q", *self.opts, *srcs, dst], capture_output=True, timeout=timeout, cwd=cwd)
        if r.returncode:
            raise RuntimeError(f"scp: {r.stderr.decode(errors='replace').strip()[-300:]}")

    def sync_code(self, model_dir: Path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for f in CODE:
                tar.add(REPO / f, arcname=f)
            for f in MODEL_FILES:
                tar.add(model_dir / f, arcname=(model_dir / f).relative_to(REPO).as_posix())
        self.ssh(f"mkdir -p {self.dir} && tar xf - -C {self.dir}", data=buf.getvalue(), timeout=180)

    def start_trainer(self, args):
        """Detached from this SSH connection (nohup + setsid), so a network drop does not kill the training.
        It stops on stop_trainer(), or by itself when no upload has arrived for --idle-exit-s (laptop gone)."""
        rid = self.root.rsplit("/", 1)[1]
        self.ssh(f"mkdir -p {self.root}/sessions && cd {self.dir} && (nohup setsid {self.py} -W ignore "
                 f"perception/hitl/online_train.py --root live/{rid} {args} > live/{rid}/train.log 2>&1 < /dev/null & "
                 f"echo $! > live/{rid}/trainer.pid)", timeout=60)
        self.proc = True

    def stop_trainer(self):
        if self.proc is None:
            return
        try:
            self.ssh(f"kill $(cat {self.root}/trainer.pid) 2>/dev/null; true", timeout=30)
        except Exception as e:
            print(f"could not stop the trainer ({e}); it exits by itself after --idle-exit-s without uploads", flush=True)

    def status(self):
        out = self.ssh(f"cat {self.root}/status.json 2>/dev/null; echo; echo ===; cat {self.root}/latest.json 2>/dev/null; echo; "
                       f"echo ===; tail -n 3 {self.root}/train.log 2>/dev/null", timeout=30)
        st, lt, log = (out.split("===") + ["", "", ""])[:3]
        js = lambda s: json.loads(s) if s.strip().startswith("{") else None  # noqa: E731
        return js(st), js(lt), log.strip()


class Live:
    def __init__(self, a):
        self.a = a
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.lock = threading.Lock()                 # one board update at a time
        self.stop = threading.Event()
        self.board_busy = threading.Event()
        self.gpu = None if a.no_gpu else Gpu(a, self.run_id)
        self.local_runs = HERE / "runs" / f"live_{self.run_id}"
        self.latest = None
        self.last_update = time.time()
        self.uploaded = set()

    # ------------------------------------------------------------------ board
    def push(self, folder: Path, label, step=None):
        """Copy a board folder's weights.txt onto the board, restart spleat-server, verify the md5."""
        a, st = self.a, self.state
        w = folder / "weights.txt"
        md5 = hashlib.md5(w.read_bytes()).hexdigest()
        if len(w.read_text().split()) != 78501:
            raise RuntimeError(f"{w}: wrong number of values")
        self.board_busy.set()
        try:
            subprocess.run(["scp", "-q", *board_ctl.SSH_OPTS, "weights.txt", f"{board_ctl.scp_target(a.ssh)}:/root/weights.txt"],
                           check=True, capture_output=True, timeout=60, cwd=folder)
            got = board_ctl.ssh(a.ssh, "systemctl restart spleat-server && sleep 2 && systemctl is-active spleat-server "
                                       "&& md5sum /root/weights.txt | cut -c1-32")
        finally:
            self.board_busy.clear()
        state, board_md5 = got.split()[-2:]
        if board_md5 != md5 or state != "active":
            raise RuntimeError(f"board md5 {board_md5} / {state} after load (expected {md5})")
        with st.lock:
            if self.a.eval:
                self.close_score(st.k)
            st.extra["board"] = {"label": label, "step": step, "md5": md5, "t": time.time()}
        self.rec.event(st.k, f"board_model:{label}:{md5}")
        self.last_update = time.time()

    def update_now(self, why):
        if not self.lock.acquire(blocking=False):
            return "an update is already running"
        try:
            lt = self.latest
            if lt is None:
                raise RuntimeError("no snapshot from the trainer yet")
            d = self.local_runs / lt["dir"]
            d.mkdir(parents=True, exist_ok=True)
            self.set_msg(f"{why}: downloading snapshot {lt['step']} ...")
            self.gpu.scp([f"{self.gpu.host}:{self.gpu.root}/snaps/{lt['dir']}/{f}" for f in
                          ("weights.txt", "readout.json", "model.pt", "export_log.json")], ".", cwd=d)
            self.set_msg(f"{why}: loading step {lt['step']} onto the board ...")
            self.push(d, f"step {lt['step']}", lt["step"])
            self.set_msg(f"{why}: board runs step {lt['step']} ({time.strftime('%H:%M:%S')})")
            return "ok"
        except Exception as e:
            self.set_msg(f"{why}: FAILED: {e}")
            return str(e)
        finally:
            self.lock.release()

    def revert(self):
        with self.lock:
            try:
                d = Path(self.a.model)
                self.set_msg(f"loading {d.name} ...")
                self.push(d, d.name, 0)
                self.set_msg(f"board runs {d.name} ({time.strftime('%H:%M:%S')})")
            except Exception as e:
                self.set_msg(f"back to the start model FAILED: {e}")

    # ------------------------------------------------------------------ eval mode
    def find_models(self, start_dir: Path):
        """The start model and every snapshot downloaded to runs/live_*/<step>/ (the ones that went onto the board)."""
        out = [{"label": f"start: {start_dir.name}", "dir": str(start_dir), "step": 0, "run": "",
                "md5": hashlib.md5((start_dir / "weights.txt").read_bytes()).hexdigest()}]
        for w in sorted((HERE / "runs").glob("live_*/*/weights.txt")):
            d = w.parent
            run = d.parent.name.replace("live_", "")
            out.append({"label": f"{run} step {int(d.name)}", "dir": str(d), "step": int(d.name), "run": run,
                        "md5": hashlib.md5(w.read_bytes()).hexdigest()})
        return out

    def load_model(self, m):
        if not self.lock.acquire(blocking=False):
            self.set_msg(f"{m['label']} not loaded: another load is still running")
            return
        try:
            self.set_msg(f"loading {m['label']} ...")
            self.push(Path(m["dir"]), m["label"], m["step"])
            self.set_msg(f"board runs {m['label']} ({time.strftime('%H:%M:%S')}); its score counts from now")
        except Exception as e:
            self.set_msg(f"loading {m['label']} FAILED: {e}")
        finally:
            self.lock.release()

    def close_score(self, k):
        """Settle the score of the model leaving the board and add it to its row (state lock held)."""
        sc = self.score
        sc.update(10 ** 9)
        label = self.state.extra["board"]["label"]
        t = self.stats.setdefault(label, {"frames": 0, "presses": 0, "hits": 0, "misses": 0, "fas": 0, "lat": []})
        t["frames"] += k - self.score_k0
        t["presses"] += len(sc.all)
        t["hits"] += sc.hits
        t["misses"] += sc.misses
        t["fas"] += sc.fas
        t["lat"] += sc.lat
        self.score, self.score_k0 = collect.Score(60, 30), k

    def model_table(self, k):
        """Every model with its accumulated score; the model on the board includes its running score."""
        cur = self.state.extra["board"]["label"]
        rows = []
        for i, m in enumerate(self.models):
            t = dict(self.stats.get(m["label"], {"frames": 0, "presses": 0, "hits": 0, "misses": 0, "fas": 0, "lat": []}))
            if m["label"] == cur:
                sc = self.score
                t = {"frames": t["frames"] + k - self.score_k0, "presses": t["presses"] + len(sc.all),
                     "hits": t["hits"] + sc.hits, "misses": t["misses"] + sc.misses, "fas": t["fas"] + sc.fas,
                     "lat": t["lat"] + sc.lat}
            mins = t["frames"] / 6000
            n = t["hits"] + t["misses"]
            rows.append({"i": i, "label": m["label"], "md5": m["md5"][:8], "on_board": m["label"] == cur,
                         "min": round(mins, 1), "presses": t["presses"], "hits": t["hits"], "misses": t["misses"],
                         "fas": t["fas"], "fa_per_min": round(t["fas"] / mins, 2) if mins > 0.05 else None,
                         "recall": round(t["hits"] / n, 2) if n else None,
                         "lat_ms": float(np.median(t["lat"])) if t["lat"] else None})
        return rows

    def camera(self, stop):
        """viewer.camera_frames, restarted (camera + SSH tunnel) whenever it drops or goes quiet, instead of
        ending the run. The frame counter, readout and recording just carry on."""
        a = self.a
        while not stop.is_set():
            try:
                for f in viewer.camera_frames(a, stop):
                    yield f
                why = "camera stream ended"
            except (OSError, ConnectionError, SystemExit) as e:   # socket timeout, reset, failed start
                why = f"camera dropped ({type(e).__name__}: {e})"[:160]
            if stop.is_set():
                return
            self.set_msg(f"{why}; reconnecting ({time.strftime('%H:%M:%S')})")
            time.sleep(1.0)

    def set_msg(self, m):
        print(m, flush=True)
        with self.state.lock:
            self.state.extra["msg"] = m

    # ------------------------------------------------------------------ background threads
    def uploader(self):
        while not self.stop.wait(self.a.chunk_s / 2):
            self.upload_once()

    def upload_once(self):
        if self.gpu is None:
            return
        d = self.rec.out
        new = [p for p in sorted(d.glob("chunk_*.npz")) if p.name not in self.uploaded]
        files = [p.name for p in new] + [f for f in ("events.jsonl", "control.json", "meta.json") if (d / f).exists()]
        try:
            self.gpu.scp(files, f"{self.gpu.host}:{self.gpu.root}/sessions/{d.name}/", cwd=d)
            self.uploaded.update(p.name for p in new)
            with self.state.lock:
                self.state.extra["upload"] = {"chunks": len(self.uploaded), "t": time.time(), "err": None}
        except Exception as e:
            with self.state.lock:
                self.state.extra["upload"] = {"chunks": len(self.uploaded), "t": time.time(), "err": str(e)[-200:]}

    def poller(self):
        while not self.stop.wait(self.a.poll_s):
            try:
                st, lt, log = self.gpu.status()
            except Exception as e:
                with self.state.lock:
                    self.state.extra["trainer"] = {"state": f"unreachable: {e}"[:200]}
                continue
            self.latest = lt
            with self.state.lock:
                self.state.extra["trainer"] = st or {"state": "starting", "log": log[-300:]}
                self.state.extra["latest"] = lt
                board_step = (self.state.extra.get("board") or {}).get("step") or 0
            auto = self.state.extra.get("auto_min", 0)
            if (auto > 0 and lt and lt["step"] > board_step and time.time() - self.last_update >= auto * 60):
                threading.Thread(target=self.update_now, args=(f"auto ({auto:g} min)",), daemon=True).start()

    # ------------------------------------------------------------------ page controls
    def write_control(self):
        (self.rec.out / "control.json").write_text(json.dumps({"theta": self.readout.theta, "t": time.time()}))

    def on_post(self, path):
        u = urlparse(path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        st = self.state
        if u.path == "/theta":
            lo, hi = self.spec.theta_range
            th = min(hi, max(lo, float(q["v"])))
            self.readout.theta = th
            with st.lock:
                st.extra["theta"] = th
            self.rec.event(st.k, f"theta:{th}")
            self.write_control()
            return 200, {"theta": th}
        if u.path == "/watch":
            on = q.get("on", "0") == "1"
            with st.lock:
                st.extra["watching"] = on
            self.rec.event(st.k, f"watch:{int(on)}")
            return 200, {"watching": on}
        if u.path == "/auto":
            m = max(0.0, float(q.get("min", 0)))
            with st.lock:
                st.extra["auto_min"] = m
            self.last_update = time.time() if m > 0 else self.last_update
            return 200, {"auto_min": m}
        if u.path == "/update":
            if self.gpu is None:
                return 400, {"err": "no GPU trainer (--no-gpu)"}
            threading.Thread(target=self.update_now, args=("manual",), daemon=True).start()
            return 202, {"ok": True}
        if u.path == "/load":
            m = self.models[int(q["i"])]
            threading.Thread(target=self.load_model, args=(m,), daemon=True).start()
            return 202, {"ok": True, "label": m["label"]}
        if u.path == "/revert":
            threading.Thread(target=self.revert, daemon=True).start()
            return 202, {"ok": True}
        return 404, {"err": "unknown"}

    # ------------------------------------------------------------------ main loop
    def run(self):
        a = self.a
        model_dir = Path(a.model)
        self.spec = spec = Spec(model_dir)
        theta = a.theta if a.theta is not None else spec.theta
        self.readout = LiveReadout(spec, theta)
        self.rec = collect.Recorder(HERE / "sessions" / f"{self.run_id}_live", chunk=int(a.chunk_s * 100))
        (self.rec.out / "meta.json").write_text(json.dumps({"live_loop": True, "run_id": self.run_id, "model": str(model_dir.resolve()),
                                                            "theta": theta, "args": vars(a), "started": time.time()}, indent=1))
        self.state = st = collect.State({"theta": theta, "primary": "board", "model": model_dir.resolve().name,
                                         "source": a.replay or "camera", "out": str(self.rec.out), "live": True,
                                         "theta_range": spec.theta_range, "y_range": [min(spec.y_neg, 0.0), spec.y_pos],
                                         "rule": spec.kind, "start_model": model_dir.name})
        st.extra = {"theta": theta, "auto_min": a.auto_min, "msg": "", "board": {"label": model_dir.name, "step": 0},
                    "trainer": {"state": "off (--no-gpu)" if self.gpu is None else "starting"}, "latest": None,
                    "upload": None, "gpu": a.gpu if self.gpu else None, "watching": a.watching}
        st.on_post = self.on_post
        self.write_control()

        md5_board = board_ctl.ssh(a.ssh, "cp -n /root/weights.txt /root/weights_backup_$(md5sum /root/weights.txt | cut -c1-32).txt; "
                                         "md5sum /root/weights.txt | cut -c1-32")
        md5_local = hashlib.md5((model_dir / "weights.txt").read_bytes()).hexdigest()
        self.models = self.find_models(model_dir)
        if a.eval:
            cur = next((m for m in self.models if m["md5"] == md5_board), None)
            st.extra["board"] = {"label": cur["label"] if cur else f"unknown ({md5_board[:8]})", "md5": md5_board,
                                 "step": cur["step"] if cur else None}
            st.extra["eval"] = True
        else:
            if md5_board != md5_local:
                print(f"WARNING: board runs {md5_board}, --model {model_dir} is {md5_local}; loading {model_dir} first", flush=True)
                self.push(model_dir, model_dir.name, 0)
            st.extra["board"]["md5"] = md5_local

        if self.gpu:
            print(f"syncing code to {a.gpu}:{a.gpu_dir} and starting the trainer ({a.loss})", flush=True)
            self.gpu.sync_code(model_dir)
            self.gpu.ssh(f"mkdir -p {self.gpu.root}/sessions/{self.rec.out.name}")
            rel = model_dir.relative_to(REPO).as_posix()
            self.gpu.start_trainer(f"--model-dir {rel} --loss {a.loss} --mil-theta {theta} --snapshot-s {a.snapshot_s} {a.train_args}")
            threading.Thread(target=self.uploader, daemon=True).start()
            threading.Thread(target=self.poller, daemon=True).start()

        srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), collect.make_handler(st))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"window: http://127.0.0.1:{a.http_port}/   (session {self.rec.out})", flush=True)

        stop = self.stop
        if a.replay:
            a.fast, a.loop = True, a.loop
            frames = collect.paced(collect.session_frames(Path(a.replay)) if Path(a.replay).is_dir()
                                   else viewer.replay_frames(a, stop), a.speed, stop)
        else:
            if ":" in a.host:
                viewer.start_camera = collect.start_camera_tunnel
            frames = self.camera(stop)

        board, t_try = None, 0.0
        self.score, self.score_k0 = collect.Score(60, 30), 0
        self.stats = {}                                    # label -> accumulated eval numbers
        recent, rtts = deque(maxlen=3), []
        t_fps, n_fps, t_start = time.perf_counter(), 0, time.time()
        try:
            for k, f in enumerate(frames):
                if a.max_frames and k >= a.max_frames:
                    break
                with st.lock:
                    evs, st.pending = st.pending, []
                for ev in evs:
                    if ev == "down" and not st.held:
                        st.held = True
                        self.score.press(k)
                        self.rec.event(k, "down")
                    elif ev == "up":
                        st.held = False
                        self.rec.event(k, "up")
                    elif ev == "undo" and self.score.undo(k):
                        self.rec.event(k, "undo")
                out = np.full(spec.n_used, -1, np.int8)     # -1 = no board output this frame
                if not self.board_busy.is_set():
                    if board is None and time.time() - t_try > 0.5:
                        t_try = time.time()
                        try:
                            board = viewer.Board(a.host, a.port)
                        except OSError:
                            board = None
                    if board is not None:
                        try:
                            t0 = time.perf_counter()
                            ob, _ = board.infer(f)
                            rtts.append((time.perf_counter() - t0) * 1e3)
                            out = (ob[:spec.n_used] > 0).astype(np.int8)
                        except (OSError, ConnectionError):
                            board = None              # spleat restarting: reconnect on a later frame
                elif board is not None:
                    try:
                        board.sock.close()
                    except OSError:
                        pass
                    board = None
                y, al = self.readout.step(k, np.maximum(out, 0))
                s = int(out[spec.pos_channel])
                if al:
                    self.score.alarm(k)
                self.score.update(k)
                self.rec.add(f, out, -1, y, int(al), int(st.held), time.time(), int(st.extra["watching"]))
                n_fps += 1
                now = time.perf_counter()
                with st.lock:
                    st.rows.append([round(y, 3), s, int(al), int(st.held), int("down" in evs)])
                    st.seq += 1
                    st.k = k
                    st.score = self.score.as_dict()
                    st.n_press = len(self.score.all)
                    recent.append(f)
                    if k % 4 == 0:
                        acc = np.zeros((2, collect.GRID, collect.GRID), bool)
                        for g in recent:
                            acc |= g > 0
                        st.img = acc[1].astype(np.uint8) * 2 | acc[0].astype(np.uint8)
                    if now - t_fps >= 1.0 and a.eval:
                        st.extra["models"] = self.model_table(k)
                    if now - t_fps >= 1.0:
                        st.fps = n_fps / (now - t_fps)
                        st.rtt = round(float(np.median(rtts[-100:])), 2) if rtts else -1.0
                        st.minutes = (time.time() - t_start) / 60
                        t_fps, n_fps = now, 0
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            st.done = True
            self.rec.close()
            if self.gpu:
                self.upload_once()
                self.gpu.stop_trainer()
            print(json.dumps({"session": str(self.rec.out), "frames": st.k + 1, "presses": st.n_press,
                              "board": st.extra.get("board"), "latest_snapshot": self.latest,
                              "gpu_run": f"{a.gpu}:{self.gpu.root}" if self.gpu else None}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.77.1", help="board (SPLEAT); IPv6 link-local works, e.g. fe80::...%%50")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--ssh", default=None, help="board SSH target (default root@<host>)")
    ap.add_argument("--model", default=str(HERE.parent / "model_d_neg11"),
                    help="board folder to start from (inside the repo; loaded if the board differs); perception/model_v3 for v3")
    ap.add_argument("--theta", type=float, default=None, help="initial theta (default: the model's readout.json)")
    ap.add_argument("--watching", action="store_true", help="start in 'just watching' mode (not training data)")
    ap.add_argument("--auto-min", type=float, default=0, help="auto-update the board every N minutes (0 = only by button)")
    ap.add_argument("--gpu", default="team-1-user@38.128.233.249")
    ap.add_argument("--gpu-key", default="~/.ssh/hackathon_a100")
    ap.add_argument("--gpu-dir", default="hitl", help="folder on the GPU machine, relative to its home")
    ap.add_argument("--gpu-python", default="~/snn_pretrain/.venv/bin/python")
    ap.add_argument("--no-gpu", action="store_true", help="collect only, no trainer")
    ap.add_argument("--eval", action="store_true",
                    help="final evaluation: no trainer; pick the start model or any downloaded snapshot (runs/live_*) "
                         "in the window, each keeps its own hit / miss / false-alarm score")
    ap.add_argument("--loss", default="pg", help="trainer objective: pg (RL), or e.g. pg,mil / sup,mil")
    ap.add_argument("--snapshot-s", type=float, default=120, help="trainer exports a snapshot this often")
    ap.add_argument("--train-args", default="--bs 16", help="extra online_train.py / train.py options")
    ap.add_argument("--chunk-s", type=float, default=10, help="upload granularity (seconds of frames per chunk)")
    ap.add_argument("--poll-s", type=float, default=10)
    ap.add_argument("--listen-port", type=int, default=3334)
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--replay", help="recording / session folder instead of the camera")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--http-port", type=int, default=8771)
    a = ap.parse_args()
    a.ssh = a.ssh or f"root@{a.host}"
    if a.eval:
        a.no_gpu = True
    Live(a).run()


if __name__ == "__main__":
    main()

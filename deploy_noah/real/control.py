"""Keyboard control + safety gate between Noah's throw_reflex and the Go2 (run by run.sh, not by itself).

throw_reflex is remapped to publish on /noah/api/sport/request. This gate forwards those requests to the dog
(--out, /api/sport/request) ONLY while ARMED. It starts DISARMED: blocked requests are printed, not sent.

  a        ARM: forward the reflex's commands to the dog
  Space    STOP: disarm + StopMove (the dog stands where it is)
  x        DAMP: disarm + Damp, the same as the remote's L2+B: the motors go limp, the dog drops to the floor
  u        stand up: StandUp, then BalanceStand after 2 s (only while disarmed; also clears the
           locked stand Noah's duck leaves behind)
  d        lie down: StandDown (only while disarmed)
  t / f    fake THROW / FEINT score burst (only with --scores keys, i.e. no Kria): 0.6 s wind-up + 0.4 s
           throw / feint, as in Noah's Gazebo thrower. two_stage: arm, then back off (t) or walk back (f)
  h        status
  q, Ctrl+C  StopMove, then quit (the launch is stopped by run.sh)

--scores keys: no Kria. This gate publishes /durance/gestures itself at 30 Hz (idle; t/f send 6 frames of throw /
feint). The stream also keeps throw_reflex's stale watchdog fed; if this gate dies, the stream stops and
throw_reflex sends its own StopMove (which can then only reach the dog through this gate, so: StopMove on exit).
"""
import argparse
import json
import select
import sys
import termios
import time
import tty

import rclpy
from msgs.msg import Clearance, Gestures
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from unitree_api.msg import Request, Response

DAMP, BALANCE_STAND, STOP_MOVE, STAND_UP, STAND_DOWN, MOVE = 1001, 1002, 1003, 1004, 1005, 1008
NAMES = {1001: "Damp", 1002: "BalanceStand", 1003: "StopMove", 1004: "StandUp", 1005: "StandDown",
         1006: "RecoveryStand", 1008: "Move"}
CLEAR_NAMES = {Clearance.CLEAR: "CLEAR", Clearance.OBSTACLE: "OBSTACLE", Clearance.UNKNOWN: "UNKNOWN"}
ARM_QUIET = 3.0         # arming refused until the reflex has been silent this long (no half-finished dodge)
SCORE_HZ = 30.0
# fake gestures = the score shape of Noah's Gazebo thrower (throw_scene.py): 0.6 s wind-up that looks the same for
# throw and feint, then 0.4 s of throw or feint. Long enough for gesture_trigger (arm, then dash / disarm).
IDLE_SCORES, WINDUP_SCORES = (0.05, 0.05, 0.9), (0.35, 0.35, 0.3)
GESTURE = {"t": [WINDUP_SCORES] * 18 + [(0.85, 0.1, 0.05)] * 12,
           "f": [WINDUP_SCORES] * 18 + [(0.1, 0.85, 0.05)] * 12}


def describe(api_id, parameter):
    name = NAMES.get(api_id, str(api_id))
    if api_id == MOVE and parameter:
        try:
            v = json.loads(parameter)
            return f"{name} vx={v.get('x', 0):+.2f} vy={v.get('y', 0):+.2f} wz={v.get('z', 0):+.2f}"
        except ValueError:
            pass
    return name


class Control:
    def __init__(self, args):
        self.args = args
        self.node = rclpy.create_node("noah_control")
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.out = self.node.create_publisher(Request, args.out, reliable)
        self.node.create_subscription(Request, "/noah/api/sport/request", self.on_reflex, reliable)
        self.node.create_subscription(Response, "/api/sport/response", self.on_response, reliable)
        self.node.create_subscription(String, "/reflex/event", self.on_event, reliable)
        self.node.create_subscription(Clearance, "/durance/clearance", self.on_clearance, reliable)
        self.node.create_subscription(Gestures, "/durance/gestures", self.on_gestures, reliable)
        self.armed = False
        self.req_id = 1_000_000           # own ids, apart from throw_reflex's 1, 2, 3, ...
        self.sent_at = {}                 # api_id -> time this gate last sent it (to show the dog's answers)
        self.last_reflex = 0.0
        self.last_print = None            # two_stage_reflex re-sends Move every 0.1 s: print changes only
        self.clearance = None
        self.clearance_count = 0
        self.score_count = 0
        self.t_status = time.monotonic()
        self.pending = []                 # (due, api_id) delayed own commands
        self.burst = []                   # queued fake score frames
        if args.scores == "keys":
            self.gest_pub = self.node.create_publisher(Gestures, "/durance/gestures", reliable)
            self.node.create_timer(1.0 / SCORE_HZ, self.publish_scores)

    # --- output ------------------------------------------------------------------------------------------------
    def log(self, text):
        print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)

    def send(self, api_id, parameter=None, why=""):
        self.req_id += 1
        req = Request()
        req.header.identity.id = self.req_id
        req.header.identity.api_id = api_id
        req.parameter = json.dumps(parameter) if parameter is not None else ""
        self.out.publish(req)
        self.sent_at[api_id] = time.monotonic()
        self.log(f"SENT {describe(api_id, req.parameter)}{'  (' + why + ')' if why else ''}")

    # --- inputs ------------------------------------------------------------------------------------------------
    def on_reflex(self, req):
        self.last_reflex = time.monotonic()
        what = describe(req.header.identity.api_id, req.parameter)
        if self.armed:
            self.out.publish(req)
        line = f"reflex -> dog: {what}" if self.armed else f"reflex BLOCKED (disarmed): {what}"
        if line != self.last_print:
            self.log(line)
            self.last_print = line

    def on_response(self, res):
        api_id = res.header.identity.api_id
        if self.armed or time.monotonic() - self.sent_at.get(api_id, -1e9) < 3.0:
            code = res.header.status.code
            self.log(f"dog answered {NAMES.get(api_id, api_id)}: {'ok' if code == 0 else 'ERROR code ' + str(code)}")

    def on_event(self, msg):
        try:
            e = json.loads(msg.data)
        except ValueError:
            return
        kind = e.get("event")
        if kind == "dodge":
            self.log(f"reflex: THROW p={e.get('p_throw', 0):.2f} -> sidestep {'left' if e.get('side', 1) > 0 else 'right'}")
        elif kind == "duck":
            self.log(f"reflex: THROW p={e.get('p_throw', 0):.2f}, space not clear -> duck")
        elif kind == "feint_ignored":
            self.log(f"reflex: feint p={e.get('p_feint', 0):.2f} -> ignored")
        elif kind == "stale_stop":
            self.log("reflex: score stream stale -> StopMove")
        elif kind == "ignored":
            self.log(f"reflex: {e.get('trigger')} ignored (state {e.get('state')})")
        elif kind in ("armed", "dash", "hold", "return", "idle"):      # two_stage_reflex
            text = {"armed": "wind-up -> ARMED, leaning in", "dash": "THROW -> backing off",
                    "hold": "backed off, waiting", "return": "walking back to the start",
                    "idle": "at the start"}[kind]
            reason = f" ({e['reason']})" if "reason" in e else ""
            self.log(f"reflex: {text}{reason}   [{e.get('distance', 0) * 100:+.0f} cm]")

    def on_clearance(self, msg):
        self.clearance_count += 1
        if msg.state != self.clearance:
            self.log(f"LiDAR clearance: {CLEAR_NAMES.get(msg.state, msg.state)}"
                     f"{'  (free: lean-in / sidestep allowed)' if msg.state == Clearance.CLEAR else '  (not free: no lean-in; sidestep reflex ducks)'}")
            self.clearance = msg.state

    def on_gestures(self, _msg):
        self.score_count += 1

    def publish_scores(self):
        msg = Gestures()
        msg.scores[0], msg.scores[1], msg.scores[2] = self.burst.pop(0) if self.burst else IDLE_SCORES
        self.gest_pub.publish(msg)

    # --- keys --------------------------------------------------------------------------------------------------
    def key(self, k):
        if k == " ":
            self.armed = False
            self.pending.clear()
            self.send(STOP_MOVE, why="STOP, disarmed")
        elif k == "x":
            self.armed = False
            self.pending.clear()
            self.send(DAMP, why="DAMP, disarmed: motors limp")
        elif k == "a":
            quiet = time.monotonic() - self.last_reflex
            if self.armed:
                self.log("already ARMED")
            elif quiet < ARM_QUIET:
                self.log(f"not armed: the reflex sent a command {quiet:.1f} s ago, try again in a moment")
            else:
                self.armed = True
                self.log("ARMED: the reflex now drives the dog.  Space = STOP, x = DAMP")
        elif k in "ud":
            if self.armed:
                self.log("disarm first (Space)")
            elif k == "u":
                self.send(STAND_UP, why="stand up")
                self.pending.append((time.monotonic() + 2.0, BALANCE_STAND))
            else:
                self.send(STAND_DOWN, why="lie down")
        elif k in "tf":
            if self.args.scores != "keys":
                self.log("t/f only without the Kria (scores:=keys)")
            else:
                self.burst = list(GESTURE[k])
                self.log(f"fake {'THROW' if k == 't' else 'FEINT'}: 0.6 s wind-up, then 0.4 s {'throw' if k == 't' else 'feint'}")
        elif k == "h":
            self.status()

    def status(self):
        dt = max(time.monotonic() - self.t_status, 1e-3)
        self.log(f"{'ARMED' if self.armed else 'disarmed'} | out {self.args.out} | scores {self.args.scores} "
                 f"{self.score_count / dt:.0f} Hz | clearance {CLEAR_NAMES.get(self.clearance, 'none yet')} "
                 f"{self.clearance_count / dt:.0f} Hz")
        if self.clearance_count == 0:
            self.log("  no clearance yet: obstacle_detection has no LiDAR height map (dog on? see README)")
        if self.score_count == 0:
            self.log("  no scores: Kria not streaming? (without Kria: scores:=keys)")
        self.score_count = self.clearance_count = 0
        self.t_status = time.monotonic()

    def tick(self):
        now = time.monotonic()
        due = [p for p in self.pending if p[0] <= now]
        self.pending = [p for p in self.pending if p[0] > now]
        for _, api_id in due:
            if not self.armed:
                self.send(api_id, why="stand up, part 2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/api/sport/request")
    ap.add_argument("--scores", choices=["kria", "keys"], default="kria")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl+C -> KeyboardInterrupt, ROS still up for StopMove
    c = Control(args)
    print(f"\n=== Noah's throw reflex: {args.label}  (commands -> {args.out}, scores from {args.scores}) ===\n"
          "  a ARM   Space STOP   x DAMP (= L2+B)   u stand up   d lie down   h status   q quit"
          + ("\n  t fake throw   f fake feint" if args.scores == "keys" else "") +
          "\nDISARMED. The reflex runs, but nothing reaches the dog until you press a.\n", flush=True)

    tty_in = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if tty_in else None
    if tty_in:
        tty.setcbreak(sys.stdin.fileno())          # single keys without Enter; Ctrl+C still raises
    t_first_status = time.monotonic() + 3.0
    try:
        while True:
            rclpy.spin_once(c.node, timeout_sec=0.005)
            c.tick()
            if t_first_status and time.monotonic() > t_first_status:
                c.status()
                t_first_status = None
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "":                       # stdin closed
                    raise KeyboardInterrupt
                if ch == "q":
                    raise KeyboardInterrupt
                c.key(ch.lower() if ch != " " else ch)
    except KeyboardInterrupt:
        pass
    finally:
        c.armed = False
        try:
            c.send(STOP_MOVE, why="quit")
            t_end = time.monotonic() + 0.3         # let the StopMove go out before the participant disappears
            while time.monotonic() < t_end:
                rclpy.spin_once(c.node, timeout_sec=0.02)
        finally:
            if tty_in:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            c.node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()

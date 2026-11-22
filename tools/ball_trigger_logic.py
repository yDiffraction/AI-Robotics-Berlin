#!/usr/bin/env python3
"""Ball-dodge trigger: SNN output spikes per 10 ms frame -> "arm" / "dash" / "disarm" events.

Pure Python 3 (stdlib only, no ROS, no numpy). The same file runs in the ROS node
(sim/ros_ws/src/go2_sim_tools/go2_sim_tools/ball_trigger.py imports an identical copy that sits next to it;
tests/test_ball_trigger_logic.py checks that both copies match), in the offline check
runs/ball/deploy/simulate_trigger.py, and stand-alone against the board (prints events, no ROS):

  python3 tools/ball_trigger_logic.py --host 192.168.77.1 --port 3333 [--swing-channel 2]

Input per frame: one value per SNN output channel (0/1 spikes). Ball models, padded to 11 outputs:
ch0 throw (ball in flight), ch1 no throw, ch2 swing (wind-up; only models trained with --swing), ch3..10 silent.

Rule (every parameter is documented in TriggerParams):
  evidence    d = sum over the last `window` frames of (throw - no_throw); population heads average the
              channels of a class first.
  dash        d >= theta, at least `min_fill` valid frames since the last (re)start, and armed (with
              require_arm off, "arm" is sent right before "dash": two_stage_reflex only dashes when armed).
  arm         swing rate (swing spikes per frame over `arm_window` frames) >= theta_arm, edge triggered.
  disarm      no dash within arm_timeout after the arm, or any reset while armed.
  refractory  after a dash nothing fires for `refractory` s; the next dash also needs d < theta in between.
  reset       window cleared and disarm if armed, on: no frame for more than `watchdog` s (tick() checks
              between frames), frame-id jump of more than `max_missing`, frame id or time going back, an
              invalid frame (wrong length, NaN, value other than 0/1 = not raw spikes, spike on an unused
              channel = another model loaded, server error) and on (re)connects. Never dash on stale or
              partial data: after a reset `min_fill` fresh frames are needed again.

Why no minimum-activity guard: d is a difference of spike counts, not a ratio. The old ratio
s_throw / total jumps to 1.0 on one spike in a quiet window; here one spike gives d = 1, and d >= theta
needs theta more throw than no-throw spikes within `window` frames whatever the overall activity, so a
quiet scene keeps d near 0 and there is nothing to normalise.
"""

import argparse
import math
import socket
import struct
import sys
import time
from collections import deque

ARM, DASH, DISARM = "arm", "dash", "disarm"
EVENTS = (ARM, DASH, DISARM)
EPS = 1e-9


def _finite(v):
    return isinstance(v, float) and not math.isnan(v) and not math.isinf(v)


class TriggerParams(object):
    """Parameters of BallTrigger. Times in seconds, windows in frames (one frame = 10 ms at 100 Hz).

    name              default  meaning
    window            10       W: frames summed for the throw evidence d
    theta             3.0      dash when d >= theta (d = sum of throw - no_throw spikes over W frames);
                               must be <= window
    throw_channels    (0,)     output channels voting "throw" (several = population head, averaged)
    nothrow_channels  (1,)     output channels voting "no throw"
    swing_channel     None     wind-up channel (2 for models trained with --swing); None = no arm stage
    arm_window        10       frames for the swing rate
    theta_arm         0.5      arm when the swing rate (spikes per frame over arm_window) >= theta_arm;
                               edge triggered: the rate has to drop below theta_arm before the next arm
    require_arm       None     dash only while armed; None = True if a swing channel is set, else False
    arm_before_dash   True     a dash while not armed sends "arm" just before "dash" (two_stage_reflex
                               ignores a dash that does not follow an arm)
    arm_timeout       1.5      disarm when no dash follows an arm within this time
    refractory        2.0      after a dash nothing fires for this long
    watchdog          0.2      no frame for longer than this: reset and disarm (tick() between frames)
    max_missing       3        frame ids may skip at most this many frames, more = reset; -1 ignores ids
    min_fill          None     valid frames needed after a (re)start before arm or dash may fire;
                               None = max(window, arm_window)
    n_channels        None     required outputs per frame (11 on the board); None = any number that
                               covers the used channels
    strict_binary     True     every value must be exactly 0 or 1 (raw spikes; catches gesture_node
                               running with window > 1)
    foreign_guard     True     a spike on a channel that is not used makes the frame invalid (silent
                               padding channels firing = another model is loaded)
    """

    DEFAULTS = (
        ("window", 10), ("theta", 3.0), ("throw_channels", (0,)), ("nothrow_channels", (1,)),
        ("swing_channel", None), ("arm_window", 10), ("theta_arm", 0.5), ("require_arm", None),
        ("arm_before_dash", True), ("arm_timeout", 1.5), ("refractory", 2.0), ("watchdog", 0.2),
        ("max_missing", 3), ("min_fill", None), ("n_channels", None), ("strict_binary", True),
        ("foreign_guard", True),
    )

    def __init__(self, **kw):
        names = [k for k, _ in self.DEFAULTS]
        unknown = sorted(set(kw) - set(names))
        if unknown:
            raise ValueError("unknown trigger parameter(s): %s" % ", ".join(unknown))
        for k, v in self.DEFAULTS:
            setattr(self, k, kw.get(k, v))

    def replace(self, **kw):
        d = self.as_dict()
        d.update(kw)
        return TriggerParams(**d)

    def as_dict(self):
        return dict((k, getattr(self, k)) for k, _ in self.DEFAULTS)

    def validate(self):
        """Raise ValueError for settings that cannot work (a trigger that can never fire is a bug too)."""
        def need(cond, msg):
            if not cond:
                raise ValueError(msg)

        def is_int(v):
            return isinstance(v, int) and not isinstance(v, bool)

        need(is_int(self.window) and self.window >= 1, "window must be an integer >= 1")
        need(is_int(self.arm_window) and self.arm_window >= 1, "arm_window must be an integer >= 1")
        need(0 < float(self.theta) <= self.window, "theta must be in (0, window]: d can never exceed window")
        need(0 < float(self.theta_arm) <= 1, "theta_arm must be in (0, 1]")
        need(float(self.arm_timeout) > 0, "arm_timeout must be > 0")
        need(float(self.refractory) >= 0, "refractory must be >= 0")
        need(float(self.watchdog) > 0, "watchdog must be > 0")
        need(is_int(self.max_missing) and self.max_missing >= -1, "max_missing must be an integer >= -1")
        chans = list(self.throw_channels) + list(self.nothrow_channels)
        need(len(self.throw_channels) > 0 and len(self.nothrow_channels) > 0,
             "throw_channels and nothrow_channels must not be empty")
        if self.swing_channel is not None:
            chans.append(self.swing_channel)
        need(all(is_int(c) and c >= 0 for c in chans), "channels must be integers >= 0")
        need(len(set(chans)) == len(chans), "throw, no-throw and swing channels must not overlap")
        need(not (self.require_arm and self.swing_channel is None),
             "require_arm needs a swing channel (it would never dash)")
        need(self.min_fill is None or (is_int(self.min_fill) and self.min_fill >= 1),
             "min_fill must be None or an integer >= 1")
        need(self.n_channels is None or (is_int(self.n_channels) and self.n_channels > max(chans)),
             "n_channels must be None or larger than every used channel")
        return self


class BallTrigger(object):
    """Turns per-frame output spikes into "arm" / "dash" / "disarm" events (see the module docstring).

    step(spikes, t, frame_id=None) once per frame, tick(t) between frames (the watchdog), reset(why) on
    reconnects, bad_frame(t, why) for results that could not be decoded. Each returns the events in order.
    Not thread safe: call it from one thread or under a lock.
    """

    def __init__(self, params=None, **overrides):
        if params is None:
            params = TriggerParams(**overrides)
        elif overrides:
            params = params.replace(**overrides)
        self.p = params.validate()
        p = self.p
        self.throw_ch = tuple(p.throw_channels)
        self.nothrow_ch = tuple(p.nothrow_channels)
        self.swing_ch = p.swing_channel
        self.used = set(self.throw_ch) | set(self.nothrow_ch)
        if self.swing_ch is not None:
            self.used.add(self.swing_ch)
        self.need = max(self.used) + 1
        self.require_arm = (self.swing_ch is not None) if p.require_arm is None else bool(p.require_arm)
        if p.min_fill is not None:
            self.min_fill = p.min_fill
        else:
            self.min_fill = max(p.window, p.arm_window if self.swing_ch is not None else 1)
        self.counts = {"frames": 0, "invalid": 0, "dup_ids": 0, "resets": 0, ARM: 0, DASH: 0, DISARM: 0}
        self.reset_reasons = {}
        self.last_reason = "start"
        self._ev = deque(maxlen=p.window)
        self._sw = deque(maxlen=p.arm_window)
        self._last_t = None
        self._last_id = None
        self._stale = False                 # tick() already reset for the current silence
        self.armed = False
        self._arm_t = 0.0
        self._refr_until = float("-inf")
        self._dash_ready = True             # d was below theta since the last dash
        self._arm_ready = True              # the swing rate was below theta_arm since the last arm
        self.fill = 0                       # valid frames since the last (re)start
        self.evidence = 0.0
        self.swing_rate = 0.0
        self.last_values = (0.0, 0.0, 0.0)  # throw, no_throw, swing of the last valid frame

    # ------------------------------------------------------------------ inputs
    def step(self, spikes, t, frame_id=None):
        """Feed one frame. spikes: one value per output channel; t: arrival time in s (monotonic);
        frame_id: the SPLEAT frame id if known. Returns the events of this frame, e.g. ["arm", "dash"]."""
        events = []
        p = self.p
        try:
            t = float(t)
        except (TypeError, ValueError):
            t = float("nan")
        if not _finite(t):
            self.counts["frames"] += 1
            self.counts["invalid"] += 1
            self._reset("invalid frame: bad time stamp", events, "invalid")
            return events
        if self._last_t is not None:
            dt = t - self._last_t
            if dt < 0:
                self._reset("time went back by %d ms" % round(-dt * 1000), events, "time")
            elif dt > p.watchdog and not self._stale:
                self._reset("gap of %d ms between frames" % round(dt * 1000), events, "gap")
        if frame_id is not None and p.max_missing >= 0:
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                frame_id = None
        if frame_id is not None and p.max_missing >= 0:
            if self._last_id is not None:
                jump = frame_id - self._last_id
                if jump < 0:
                    self._reset("frame id went back (%d -> %d)" % (self._last_id, frame_id), events, "ids")
                elif jump == 0:
                    self.counts["dup_ids"] += 1
                elif jump - 1 > p.max_missing:
                    self._reset("%d frames missing" % (jump - 1), events, "ids")
            self._last_id = frame_id
        self._last_t = t
        self._stale = False
        self.counts["frames"] += 1

        parsed = self._parse(spikes)
        if isinstance(parsed, str):
            self.counts["invalid"] += 1
            self._reset("invalid frame: " + parsed, events, "invalid")
            return events
        throw_v, no_v, swing_v = parsed
        self.last_values = parsed
        self._ev.append(throw_v - no_v)
        self.evidence = sum(self._ev)
        if self.swing_ch is not None:
            self._sw.append(swing_v)
            self.swing_rate = sum(self._sw) / float(p.arm_window)
        self.fill += 1

        if self.armed and t - self._arm_t > p.arm_timeout:
            self._disarm("arm timeout", events)
        if self.evidence < p.theta - EPS:
            self._dash_ready = True
        if self.swing_ch is not None and self.swing_rate < p.theta_arm - EPS:
            self._arm_ready = True
        if t < self._refr_until or self.fill < self.min_fill:
            return events
        if (self.swing_ch is not None and not self.armed and self._arm_ready
                and self.swing_rate >= p.theta_arm - EPS):
            self._arm(t, "swing rate %.2f" % self.swing_rate, events)
            self._arm_ready = False
        if self._dash_ready and self.evidence >= p.theta - EPS and (self.armed or not self.require_arm):
            if not self.armed and p.arm_before_dash:
                self._arm(t, "right before the dash", events)
            events.append(DASH)
            self.counts[DASH] += 1
            self.armed = False
            self._dash_ready = False
            self._refr_until = t + p.refractory
            self.last_reason = "dash at d=%g" % self.evidence
        return events

    def tick(self, t):
        """Call between frames (e.g. every 20 ms): resets and disarms when frames stop coming, and
        enforces the arm timeout without frames. Returns the events."""
        events = []
        if self._last_t is None:
            return events
        if not self._stale and t - self._last_t > self.p.watchdog:
            self._reset("no frame for %d ms" % round((t - self._last_t) * 1000), events, "watchdog")
            self._stale = True
        elif self.armed and t - self._arm_t > self.p.arm_timeout:
            self._disarm("arm timeout", events)
        return events

    def bad_frame(self, t, why):
        """A result arrived but could not be used (undecodable, server error): reset like any invalid frame."""
        events = []
        self.counts["frames"] += 1
        self.counts["invalid"] += 1
        try:
            t = float(t)
            if _finite(t):
                self._last_t = t
                self._stale = False
        except (TypeError, ValueError):
            pass
        self._reset("invalid frame: " + str(why), events, "invalid")
        return events

    def reset(self, why="external"):
        """Forget everything about the stream (e.g. after a reconnect); disarms if armed."""
        events = []
        self._reset(why, events, "external")
        self._last_t = None
        self._last_id = None
        self._stale = False
        return events

    # ------------------------------------------------------------------ state
    @property
    def ready(self):
        return self.fill >= self.min_fill

    def refractory_left(self, t):
        return max(0.0, self._refr_until - t)

    def state(self, t=None):
        s = {"evidence": self.evidence, "swing_rate": self.swing_rate, "armed": self.armed,
             "fill": self.fill, "ready": self.ready, "counts": dict(self.counts),
             "reset_reasons": dict(self.reset_reasons), "last_reason": self.last_reason}
        if t is not None:
            s["refractory_left"] = round(self.refractory_left(t), 3)
        return s

    # ------------------------------------------------------------------ internals
    def _parse(self, spikes):
        """-> (throw, no_throw, swing) of one frame, or a string saying why the frame is invalid."""
        p = self.p
        try:
            vals = [float(v) for v in spikes]
        except (TypeError, ValueError):
            return "not a list of numbers"
        n = len(vals)
        if p.n_channels is not None and n != p.n_channels:
            return "%d outputs, expected %d" % (n, p.n_channels)
        if n < self.need:
            return "%d outputs, the trigger reads channel %d" % (n, self.need - 1)
        for i, v in enumerate(vals):
            if not 0.0 <= v <= 1.0:  # also catches NaN
                return "output %d is %r, outside [0, 1]" % (i, v)
            if p.strict_binary and v != 0.0 and v != 1.0:
                return "output %d is %g, not a spike (gesture_node needs window 1)" % (i, v)
            if p.foreign_guard and v != 0.0 and i not in self.used:
                return "spike on unused output %d (another model loaded?)" % i
        throw_v = sum(vals[c] for c in self.throw_ch) / float(len(self.throw_ch))
        no_v = sum(vals[c] for c in self.nothrow_ch) / float(len(self.nothrow_ch))
        swing_v = vals[self.swing_ch] if self.swing_ch is not None else 0.0
        return throw_v, no_v, swing_v

    def _arm(self, t, why, events):
        events.append(ARM)
        self.counts[ARM] += 1
        self.armed = True
        self._arm_t = t
        self.last_reason = "arm: " + why

    def _disarm(self, why, events):
        events.append(DISARM)
        self.counts[DISARM] += 1
        self.armed = False
        self.last_reason = "disarm: " + why

    def _reset(self, why, events, key):
        if self.armed:
            self._disarm(why, events)
        self._ev.clear()
        self._sw.clear()
        self.fill = 0
        self.evidence = 0.0
        self.swing_rate = 0.0
        self._dash_ready = True
        self._arm_ready = True
        self.counts["resets"] += 1
        self.reset_reasons[key] = self.reset_reasons.get(key, 0) + 1
        self.last_reason = why


# ----------------------------------------------------------------------------- SPLEAT result stream
MAX_RESULT = 1 << 16  # results are ~70 bytes; a larger length prefix means this is not a SPLEAT stream


def _cbor_head(buf, i):
    if i >= len(buf):
        raise ValueError("truncated CBOR")
    ib = buf[i]
    major, info = ib >> 5, ib & 31
    i += 1
    if info < 24:
        return major, info, info, i
    if info > 27:
        raise ValueError("indefinite or reserved CBOR length (info %d)" % info)
    n = 1 << (info - 24)
    if i + n > len(buf):
        raise ValueError("truncated CBOR")
    return major, info, int.from_bytes(bytes(buf[i:i + n]), "big"), i + n


def cbor_decode(buf, i=0, depth=0):
    """Minimal CBOR decoder (definite lengths only) -> (value, end). Enough for the SPLEAT maps."""
    if depth > 16:
        raise ValueError("CBOR nested too deep")
    major, info, arg, i = _cbor_head(buf, i)
    if major == 0:
        return arg, i
    if major == 1:
        return -1 - arg, i
    if major in (2, 3):
        end = i + arg
        if end > len(buf):
            raise ValueError("truncated CBOR string")
        data = bytes(buf[i:end])
        return (data if major == 2 else data.decode("utf-8")), end
    if major in (4, 5):
        items = []
        for _ in range(arg * (2 if major == 5 else 1)):
            v, i = cbor_decode(buf, i, depth + 1)
            items.append(v)
        if major == 4:
            return items, i
        try:
            return dict(zip(items[0::2], items[1::2])), i
        except TypeError:
            raise ValueError("unhashable CBOR map key")
    if major == 6:
        return cbor_decode(buf, i, depth + 1)  # tag: keep the tagged value
    if info == 25:
        return struct.unpack(">e", bytes(buf[i - 2:i]))[0], i
    if info == 26:
        return struct.unpack(">f", bytes(buf[i - 4:i]))[0], i
    if info == 27:
        return struct.unpack(">d", bytes(buf[i - 8:i]))[0], i
    return {20: False, 21: True, 22: None, 23: None}.get(arg, arg), i


def decode_result(body):
    """One SPLEAT result (CBOR map) -> (frame_id or None, [float per output], latency or None, error text).
    Raises ValueError for anything that is not a result map with a float32 payload."""
    msg, end = cbor_decode(body)
    if end != len(body):
        raise ValueError("%d trailing bytes" % (len(body) - end))
    if not isinstance(msg, dict):
        raise ValueError("result is not a CBOR map")
    payload = msg.get("payload")
    if not isinstance(payload, bytes) or not payload or len(payload) % 4:
        raise ValueError("payload missing or not float32")
    values = list(struct.unpack("<%df" % (len(payload) // 4), payload))
    fid = msg.get("id")
    if not isinstance(fid, int) or isinstance(fid, bool):
        fid = None
    err = msg.get("error")
    err = "" if err is None or err is False or err == "" else str(err)
    return fid, values, msg.get("latency"), err


class SpleatStream(object):
    """Blocking TCP reader for the SPLEAT result stream (the server sends every result to every client)."""

    def __init__(self, host, port, connect_timeout=1.0, read_timeout=0.1, max_frame=MAX_RESULT):
        self.host, self.port = host, int(port)
        self.connect_timeout, self.read_timeout, self.max_frame = connect_timeout, read_timeout, max_frame
        self.sock = None
        self.buf = bytearray()

    @property
    def connected(self):
        return self.sock is not None

    def connect(self):
        """One connection attempt; True when connected."""
        self.close()
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except (OSError, ValueError):
            return False
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.read_timeout)
        self.sock, self.buf = s, bytearray()
        self._quickack()
        return True

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _quickack(self):
        # Linux: acknowledge at once. A server with Nagle on (no TCP_NODELAY) otherwise holds every
        # small result until our delayed ACK, and results arrive in bursts. Not sticky: re-armed per recv.
        opt = getattr(socket, "TCP_QUICKACK", None)
        if opt is not None and self.sock is not None:
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, opt, 1)
            except OSError:
                pass

    def read(self):
        """Wait up to read_timeout for data; return the complete results (CBOR bodies) received so far.
        Raises ConnectionError when the connection is gone or the stream is not SPLEAT framing."""
        if self.sock is None:
            raise ConnectionError("not connected")
        try:
            chunk = self.sock.recv(65536)
        except socket.timeout:
            return []
        except OSError as e:
            raise ConnectionError(str(e))
        if not chunk:
            raise ConnectionError("server closed the connection")
        self._quickack()
        self.buf += chunk
        frames = []
        while len(self.buf) >= 4:
            n = struct.unpack_from(">I", self.buf, 0)[0]
            if n > self.max_frame:
                raise ConnectionError("length prefix %d: not a SPLEAT result stream" % n)
            if len(self.buf) < 4 + n:
                break
            frames.append(bytes(self.buf[4:4 + n]))
            del self.buf[:4 + n]
        return frames


# ----------------------------------------------------------------------------- stand-alone monitor
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the ball trigger on a live SPLEAT result stream and "
                                             "print its events (no ROS, no robot).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--theta", type=float, default=3.0)
    ap.add_argument("--swing-channel", type=int, default=-1, help="-1 = model without swing output")
    ap.add_argument("--require-arm", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--n-channels", type=int, default=11, help="0 = any")
    ap.add_argument("--no-ids", action="store_true", help="ignore frame ids (gap check by time only)")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between status lines")
    a = ap.parse_args(argv)
    trig = BallTrigger(window=a.window, theta=a.theta,
                       swing_channel=None if a.swing_channel < 0 else a.swing_channel,
                       require_arm={"auto": None, "yes": True, "no": False}[a.require_arm],
                       n_channels=a.n_channels or None, max_missing=-1 if a.no_ids else 3)
    stream = SpleatStream(a.host, a.port)
    t0 = time.monotonic()
    next_status, frames_at_status = t0 + a.every, 0

    def show(events, t, fid):
        for e in events:
            print("%8.3f s  %-6s d=%-5g swing=%.2f frame=%s  (%s)" % (
                t - t0, e.upper(), trig.evidence, trig.swing_rate, fid, trig.last_reason), flush=True)

    try:
        while True:
            now = time.monotonic()
            if not stream.connected:
                if not stream.connect():
                    print("no connection to %s:%d, retrying" % (a.host, a.port), flush=True)
                    time.sleep(1.0)
                    continue
                print("connected to %s:%d" % (a.host, a.port), flush=True)
                show(trig.reset("connected"), now, None)
            try:
                bodies = stream.read()
            except ConnectionError as e:
                print("connection lost: %s" % e, flush=True)
                stream.close()
                show(trig.reset("connection lost"), time.monotonic(), None)
                continue
            for body in bodies:
                t = time.monotonic()
                try:
                    fid, values, _lat, err = decode_result(body)
                except ValueError as e:
                    show(trig.bad_frame(t, "undecodable result: %s" % e), t, None)
                    continue
                show(trig.bad_frame(t, "server error: " + err) if err else trig.step(values, t, fid), t, fid)
            t = time.monotonic()
            show(trig.tick(t), t, None)
            if t >= next_status:
                n = trig.counts["frames"]
                print("%8.3f s  %.1f frames/s  d=%g armed=%s  counts %s  resets %s  last: %s" % (
                    t - t0, (n - frames_at_status) / a.every, trig.evidence, trig.armed, trig.counts,
                    trig.reset_reasons, trig.last_reason), flush=True)
                next_status, frames_at_status = t + a.every, n
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Frame-to-event simulator modelled on a GenX320-style DVS pixel.

Per pixel: log intensity -> first-order photoreceptor low-pass (bandwidth grows with
brightness) -> ON/OFF contrast thresholds with fixed-pattern mismatch -> refractory
period -> events. Between two input frames the log intensity is linearly interpolated
in adaptive sub-steps so fast motion yields many, correctly ordered events (as in v2e).
Background activity (leak + shot noise) is added at the end.

All parameters are drawn per clip by `SensorParams.sample(rng)` so a dataset spans a
range of sensors/settings; calibrate the ranges with tools/calibrate_sim.py against
real recordings.
"""

from dataclasses import asdict, dataclass

import numpy as np

SENSOR = 320
DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])


@dataclass
class SensorParams:
    c_on: float = 0.2            # ON contrast threshold (log units)
    c_off: float = 0.2           # OFF contrast threshold
    sigma: float = 0.03          # threshold mismatch (std, log units)
    refractory_us: float = 300.0
    cutoff_hz: float = 300.0     # photoreceptor bandwidth at full brightness
    leak_hz: float = 0.1         # per-pixel ON leak events
    shot_hz: float = 0.5         # per-pixel noise events (both polarities)
    max_per_step: int = 4        # cap of events per pixel per sub-step

    @staticmethod
    def sample(rng, scale=1.0):
        c = rng.uniform(0.12, 0.3) * scale
        return SensorParams(
            c_on=c * rng.uniform(0.85, 1.15), c_off=c * rng.uniform(0.85, 1.15),
            sigma=rng.uniform(0.01, 0.05), refractory_us=rng.uniform(100, 1000),
            cutoff_hz=rng.uniform(80, 600), leak_hz=rng.uniform(0.001, 0.01),
            shot_hz=rng.uniform(0.002, 0.03))   # the Kria camera runs with a noise filter

    def as_dict(self):
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def simulate(frames, times_us, params, rng, max_substeps=12):
    """frames: float array [N, H, W] in [0, 1]; times_us: [N] timestamps (microseconds).

    Returns a structured event array (DTYPE, big-endian like the .bin files), sorted by time.
    """
    frames = np.asarray(frames, np.float32)
    n, h, w = frames.shape
    eps = 0.02
    log_in = np.log(frames + eps)
    c_on = np.clip(params.c_on + rng.normal(0, params.sigma, (h, w)), 0.03, None).astype(np.float32)
    c_off = np.clip(params.c_off + rng.normal(0, params.sigma, (h, w)), 0.03, None).astype(np.float32)
    lp = log_in[0].copy()                     # photoreceptor output
    ref = lp.copy()                           # memorised level at the last event
    last = np.full((h, w), -1e12)             # last event time per pixel
    xs, ys, ps, ts = [], [], [], []

    for i in range(1, n):
        t0, t1 = float(times_us[i - 1]), float(times_us[i])
        d = np.abs(log_in[i] - log_in[i - 1]).max()
        k = int(np.clip(np.ceil(d / (0.5 * min(params.c_on, params.c_off))), 1, max_substeps))
        for j in range(1, k + 1):
            u = j / k
            target = log_in[i - 1] * (1 - u) + log_in[i] * u
            t = t0 + (t1 - t0) * u
            dt = (t1 - t0) / k * 1e-6
            # brightness-dependent bandwidth: dark pixels respond slower
            bright = np.exp(target) - eps
            fc = params.cutoff_hz * np.clip(bright, 0.05, 1.0)
            alpha = 1.0 - np.exp(-2 * np.pi * fc * dt)
            lp += alpha * (target - lp)
            diff = lp - ref
            ready = (t - last) >= params.refractory_us
            n_on = np.where(ready, np.floor(np.maximum(diff, 0) / c_on), 0).astype(np.int32)
            n_off = np.where(ready, np.floor(np.maximum(-diff, 0) / c_off), 0).astype(np.int32)
            for cnt, pol, c, sign in ((n_on, 1, c_on, 1.0), (n_off, 0, c_off, -1.0)):
                yy, xx = np.nonzero(cnt)
                if len(yy) == 0:
                    continue
                m = np.minimum(cnt[yy, xx], params.max_per_step)
                ref[yy, xx] += sign * cnt[yy, xx] * c[yy, xx]   # all crossings consume the change
                xr, yr = np.repeat(xx, m), np.repeat(yy, m)
                xs.append(xr); ys.append(yr)
                ps.append(np.full(len(xr), pol, np.uint8))
                ts.append(t - rng.random(len(xr)) * (t1 - t0) / k)
                last[yy, xx] = t

    duration = float(times_us[-1] - times_us[0])
    t_start = float(times_us[0])
    for rate, pol in ((params.leak_hz, 1), (params.shot_hz, None)):
        cnt = rng.poisson(rate * h * w * duration / 1e6)
        xs.append(rng.integers(0, w, cnt)); ys.append(rng.integers(0, h, cnt))
        ps.append(np.full(cnt, pol, np.uint8) if pol is not None else rng.integers(0, 2, cnt).astype(np.uint8))
        ts.append(t_start + rng.uniform(0, duration, cnt))

    ev = np.empty(sum(len(a) for a in xs), DTYPE)
    if len(ev):
        ev["x"], ev["y"] = np.concatenate(xs), np.concatenate(ys)
        ev["polarity"] = np.concatenate(ps)
        ev["timestamp"] = (np.concatenate(ts) - t_start).astype(np.int64)
    return ev[np.argsort(ev["timestamp"], kind="stable")]


def simulate_batch(frames, times_us, params, rng, device="cuda", max_substeps=12):
    """GPU version of `simulate` for a batch of clips with the same frame times.

    frames: uint8 [B, N, H, W] (0..255) or float [B, N, H, W] in [0, 1]; times_us: [N];
    params: list of B SensorParams. Same pixel model as `simulate`; the number of sub-steps per
    frame is the maximum over the batch (finer steps for some clips only add accuracy).
    Returns a list of B event arrays (DTYPE), timestamps relative to times_us[0].
    """
    import torch
    dev = torch.device(device)
    b, n, h, w = frames.shape
    eps = 0.02
    f = torch.as_tensor(np.asarray(frames), device=dev)
    scale = 1 / 255 if f.dtype == torch.uint8 else 1.0
    log_of = lambda i: torch.log(f[:, i].float() * scale + eps)
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2**62)))
    col = lambda name: torch.tensor([getattr(p, name) for p in params], device=dev, dtype=torch.float32).view(b, 1, 1)
    sigma = col("sigma")
    c_on = (col("c_on") + sigma * torch.randn(b, h, w, device=dev, generator=gen)).clamp(min=0.03)
    c_off = (col("c_off") + sigma * torch.randn(b, h, w, device=dev, generator=gen)).clamp(min=0.03)
    cut, refr = col("cutoff_hz"), col("refractory_us")
    maxp = col("max_per_step")
    cmin = torch.minimum(col("c_on"), col("c_off")).view(b)
    times = [float(t) for t in times_us]
    prev = log_of(0)
    lp, ref = prev.clone(), prev.clone()
    last = torch.full((b, h, w), -1e9, device=dev)
    chunks = []
    for i in range(1, n):
        cur = log_of(i)
        t0, t1 = times[i - 1], times[i]
        d = (cur - prev).abs().amax(dim=(1, 2))
        k = int(torch.clamp(torch.ceil(d / (0.5 * cmin)), 1, max_substeps).max())
        counts = torch.zeros((k, 2, b, h, w), dtype=torch.int16, device=dev)
        dt = (t1 - t0) / k * 1e-6
        for j in range(1, k + 1):
            u = j / k
            target = prev * (1 - u) + cur * u
            t = t0 + (t1 - t0) * u
            bright = torch.exp(target) - eps
            alpha = 1.0 - torch.exp(-2 * np.pi * cut * bright.clamp(0.05, 1.0) * dt)
            lp += alpha * (target - lp)
            diff = lp - ref
            ready = (t - last) >= refr
            n_on = torch.where(ready, torch.floor(diff.clamp(min=0) / c_on), 0.0)
            n_off = torch.where(ready, torch.floor((-diff).clamp(min=0) / c_off), 0.0)
            ref += n_on * c_on - n_off * c_off
            fired = (n_on + n_off) > 0
            last = torch.where(fired, torch.full_like(last, t), last)
            counts[j - 1, 0] = torch.minimum(n_off, maxp).to(torch.int16)
            counts[j - 1, 1] = torch.minimum(n_on, maxp).to(torch.int16)
        idx = counts.nonzero()                                 # [M, 5]: substep, pol, clip, y, x
        if len(idx):
            m = counts[tuple(idx.T)].long()
            idx = idx.repeat_interleave(m, dim=0)
            t_ev = t0 + (t1 - t0) * (idx[:, 0].float() + 1 - torch.rand(len(idx), device=dev, generator=gen)) / k
            chunks.append(torch.cat([idx[:, [2, 4, 3, 1]].float(), (t_ev - times[0])[:, None]], 1))
        prev = cur
    allev = torch.cat(chunks).cpu().numpy().astype(np.float64) if chunks else np.zeros((0, 5))
    duration = times[-1] - times[0]
    out = []
    for c in range(b):
        e = allev[allev[:, 0] == c]
        p = params[c]
        xs, ys, ps, ts = [e[:, 1]], [e[:, 2]], [e[:, 3]], [e[:, 4]]
        for rate, pol in ((p.leak_hz, 1), (p.shot_hz, None)):
            cnt = rng.poisson(rate * h * w * duration / 1e6)
            xs.append(rng.integers(0, w, cnt)); ys.append(rng.integers(0, h, cnt))
            ps.append(np.full(cnt, pol) if pol is not None else rng.integers(0, 2, cnt))
            ts.append(rng.uniform(0, duration, cnt))
        ev = np.empty(sum(len(a) for a in xs), DTYPE)
        ev["x"], ev["y"] = np.concatenate(xs), np.concatenate(ys)
        ev["polarity"] = np.concatenate(ps)
        ev["timestamp"] = np.concatenate(ts).astype(np.int64)
        out.append(ev[np.argsort(ev["timestamp"], kind="stable")])
    return out

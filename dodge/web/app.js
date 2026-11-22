/* Throw Reflex web UI (dodge/web/app.js), served by dodge/snn_bridge.py.
 *
 * Everything comes from the bridge's HTTP API:
 *   GET  /api/state            snapshot {model, camera, fpga, ros: {available, auto}, robot, color, counters, ...}
 *   GET  /stream               server-sent events, each data line is JSON:
 *        frame      <= 25 Hz  {t, w: 160, h: 160, on, off, rate}   on/off: base64 bitmaps, 1 bit/pixel, MSB first
 *        snn        <= 25 Hz  {frames: [{k, s: [ch0, ch1, ch2], y}], theta, latency_ms, fps}
 *        detection            {t, k, y, theta, acted}               acted = /reflex/dash was published
 *        status     1 Hz      the /api/state snapshot (also on every change)
 *        robot                a /dodge/event of go2_dodge.py, forwarded
 *        color                {seq}: a new JPEG at /color.jpg
 *   POST /api/auto {enabled}   connect the SNN to ROS (auto-dodge); returns the snapshot
 *   POST /api/theta {theta}    readout threshold
 *   POST /api/color {enabled}  poll the Go2 color camera
 *   GET  /color.jpg            latest color image (404 if none)
 *
 * ?demo=1 replaces the backend with a simulation at the end of this file (a person throwing in the event image,
 * spikes that rise ~300 ms before the release, the readout crossing θ, the robot's dodge phases, status changes),
 * so the page can be reviewed without the robot. Keys in demo mode: T throw, F feint, X dodge, Y auto-dodge,
 * C camera, P FPGA. Plain browser JavaScript, no libraries, no external URLs (the robot network is offline).
 */
'use strict';

(() => {
  // ============================================================================================ constants
  const query = new URLSearchParams(location.search);
  const DEMO = query.has('demo') && query.get('demo') !== '0';

  const W = 160, H = 160, NPIX = W * H, NBYTES = NPIX >> 3;   // event image: sensor / 2
  const FRAME_HZ = 100;                                        // SNN frames are 10 ms bins
  const SPAN = 600;                                            // frames on screen (6 s)
  const RING = 2048;                                           // frames kept (power of two)
  const OVERLAY_MS = 1600;                                     // THROW DETECTED on screen
  const TRAIL_MS = 70;                                         // event image fade time constant
  const STALE_MS = 3500;                                       // stream silent this long -> reconnecting
  const LOG_MAX = 6;
  const HOLD_S = 1.0, RETURN_SPEED = 0.25;                     // go2_dodge defaults, only for the progress bars

  const C = {                                                  // canvas colors, see the tokens in style.css
    paper: [251, 251, 253], on: [10, 132, 255], off: [255, 159, 10],
    line: '#0071e3', area0: 'rgba(0, 113, 227, 0.16)', area1: 'rgba(0, 113, 227, 0)',
    hot: '#ff3b30', hotArea0: 'rgba(255, 59, 48, 0.26)', hotArea1: 'rgba(255, 59, 48, 0.06)',
    thetaInk: '#d70015', thetaBg: 'rgba(255, 59, 48, 0.1)',
    grid: 'rgba(0, 0, 0, 0.07)', grid2: 'rgba(0, 0, 0, 0.035)', base: 'rgba(0, 0, 0, 0.12)', label: '#86868b',
    spike: '#1d1d1f', spike2: '#aeaeb2', track: 'rgba(120, 120, 128, 0.07)', det: 'rgba(255, 59, 48, 0.3)',
  };
  const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", "Inter", system-ui, sans-serif';

  // ============================================================================================ helpers
  const $ = (id) => document.getElementById(id);
  const now = () => performance.now();
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const num = (v, fallback) => (isNum(v) ? v : fallback);

  function setText(node, text) {                               // touch the DOM only on change
    if (node.textContent !== text) node.textContent = text;
  }
  function setData(node, key, value) {
    const v = String(value);
    if (node.dataset[key] !== v) node.dataset[key] = v;
  }
  function fmtRate(r) {
    if (!isNum(r)) return '—';
    if (r >= 1e6) return (r / 1e6).toFixed(r >= 1e7 ? 0 : 1) + 'M';
    if (r >= 1e3) return Math.round(r / 1e3) + 'k';
    return String(Math.round(r));
  }
  function fmtVal(v, mode = app.mode) {                         // readout y and theta
    if (!isNum(v)) return '—';
    if (mode === 'diff') return Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtClock(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  }
  function fmtAgo(ms) {
    const s = ms / 1000;
    if (s < 1) return 'just now';
    if (s < 60) return Math.floor(s) + ' s ago';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    return Math.floor(s / 3600) + ' h ago';
  }
  function fmtMeters(m) { return isNum(m) ? m.toFixed(2) + ' m' : '—'; }
  function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  }

  // ============================================================================================ DOM
  const el = {
    layout: $('layout'), banner: $('conn-banner'), bannerText: $('conn-text'),
    demoBadge: $('demo-badge'), footKeys: $('foot-keys'), toast: $('toast'),
    pillCam: $('pill-camera'), pillCamV: $('pill-camera-v'),
    pillFpga: $('pill-fpga'), pillFpgaV: $('pill-fpga-v'),
    pillRobot: $('pill-robot'), pillRobotV: $('pill-robot-v'),
    auto: $('auto'), autoSwitch: $('auto-switch'), autoState: $('auto-state'), autoHintText: $('auto-hint-text'),
    stage: $('stage'), canvas: $('events'), camRate: $('cam-rate'), camSub: $('cam-sub'),
    stageBadge: $('stage-badge'), stageBadgeText: $('stage-badge-text'),
    stageEmpty: $('stage-empty'), stageEmptyTitle: $('stage-empty-title'), stageEmptySub: $('stage-empty-sub'),
    throwBox: $('throw'), throwTitle: $('throw-title'), throwSubText: $('throw-sub-text'),
    colorToggle: $('color-toggle'), colorCard: $('card-color'), colorImg: $('color-img'),
    colorEmpty: $('color-empty'), colorEmptyTitle: $('color-empty-title'), colorEmptySub: $('color-empty-sub'),
    colorMeta: $('color-meta'), colorAge: $('color-age'),
    snnSub: $('snn-sub'), snnTag: $('snn-tag'), statY: $('stat-y'), snnY: $('snn-y'), snnTheta: $('snn-theta'),
    snnLast: $('snn-last'), snnPerf: $('snn-perf'), rasterLabel: $('raster-label'),
    raster: $('raster'), chart: $('chart'),
    snnEmpty: $('snn-empty'), snnEmptyTitle: $('snn-empty-title'), snnEmptySub: $('snn-empty-sub'),
    theta: $('theta'), thetaOut: $('theta-out'),
    rstate: $('rstate'), robotState: $('robot-state'), robotSub: $('robot-sub'),
    phases: $('phases'), phaseItems: Array.from(document.querySelectorAll('#phases .phase')),
    kpiDist: $('kpi-dist'), kpiDet: $('kpi-det'), kpiDodges: $('kpi-dodges'),
    log: $('log'), logEmpty: $('log-empty'),
  };

  // ============================================================================================ app state
  const app = {
    status: null,                 // last /api/state snapshot
    connected: false,             // stream open and not silent
    everConnected: false,
    lastMsg: now(),
    mode: 'ema',                  // readout: ema | diff
    channel: 0,
    auto: false, autoKnown: false, autoPending: null,          // {value, until}
    theta: null, thetaDragging: false, thetaHoldUntil: 0,
    y: null, latency: null, fps: null, camRate: null,
    lastDetAt: 0, pendingDetRow: null,                         // detection waiting for its /dodge/event dash
    colorOn: false, colorPendingUntil: 0,
  };

  // ============================================================================================ event image
  /* 160 x 160 canvas scaled up with crisp pixels. Every frame marks its ON / OFF pixels at full strength; the
   * strengths fade with TRAIL_MS at display rate, so motion leaves a short, gentle trail. A 64 x 64 lookup table
   * holds the composite colour (OFF over the paper, ON over that). */
  const eventView = (() => {
    const ctx = el.canvas.getContext('2d', { alpha: false });
    const img = ctx.createImageData(W, H);
    const px = new Uint32Array(img.data.buffer);
    const gOn = new Float32Array(NPIX), gOff = new Float32Array(NPIX);
    const LV = 64, lut = new Uint32Array(LV * LV);
    const ease = (x) => (x < 0.03 ? 0 : Math.pow(x, 0.8));
    for (let i = 0; i < LV; i++) {
      for (let j = 0; j < LV; j++) {
        const a = ease(i / (LV - 1)), b = ease(j / (LV - 1));
        const ch = [0, 1, 2].map((c) => {
          const under = C.paper[c] * (1 - b) + C.off[c] * b;
          return Math.round(under * (1 - a) + C.on[c] * a);
        });
        lut[i * LV + j] = (255 << 24) | (ch[2] << 16) | (ch[1] << 8) | ch[0];   // RGBA, little endian
      }
    }
    let peak = 0, blank = false;

    function mark(b64, g) {
      if (!b64) return;
      let s;
      try { s = atob(b64); } catch (e) { return; }
      const n = Math.min(s.length, NBYTES);
      for (let i = 0; i < n; i++) {
        const v = s.charCodeAt(i);
        if (!v) continue;
        const o = i << 3;
        if (v & 128) g[o] = 1;
        if (v & 64) g[o + 1] = 1;
        if (v & 32) g[o + 2] = 1;
        if (v & 16) g[o + 3] = 1;
        if (v & 8) g[o + 4] = 1;
        if (v & 4) g[o + 5] = 1;
        if (v & 2) g[o + 6] = 1;
        if (v & 1) g[o + 7] = 1;
      }
    }

    return {
      push(frame) {
        if (frame.w && frame.w !== W) return;                // contract: 160 x 160
        mark(frame.on, gOn);
        mark(frame.off, gOff);
        peak = 1;
      },
      draw(dt) {
        if (peak < 0.004) {
          if (!blank) { px.fill(lut[0]); ctx.putImageData(img, 0, 0); blank = true; }
          return;
        }
        const d = Math.exp(-dt / TRAIL_MS), L = LV - 1;
        for (let p = 0; p < NPIX; p++) {
          const a = gOn[p], b = gOff[p];
          if (a === 0 && b === 0) { px[p] = lut[0]; continue; }
          px[p] = lut[((a * L + 0.5) | 0) * LV + ((b * L + 0.5) | 0)];
          gOn[p] = a * d < 0.01 ? 0 : a * d;
          gOff[p] = b * d < 0.01 ? 0 : b * d;
        }
        peak *= d;
        blank = false;
        ctx.putImageData(img, 0, 0);
      },
    };
  })();

  // ============================================================================================ SNN plot
  /* Spike raster (readout channel; for the diff readout also the channel it is compared with) and the readout
   * y(t) over the last 6 s, with θ as a dashed line, the part above θ in red and the detections as dots. The
   * x axis runs on a display clock that advances smoothly between the 25 Hz batches. */
  function hiDpiCanvas(canvas) {
    const c = { canvas, ctx: canvas.getContext('2d'), w: 0, h: 0, dpr: 1 };
    const fit = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      c.w = r.width; c.h = r.height; c.dpr = dpr;
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    new ResizeObserver(fit).observe(canvas);
    window.addEventListener('resize', fit);
    fit();
    return c;
  }

  const plot = (() => {
    const kArr = new Float64Array(RING).fill(-1);
    const yArr = new Float32Array(RING);
    const sArr = [new Float32Array(RING), new Float32Array(RING), new Float32Array(RING)];
    const dets = [];                                        // {k, y, acted}
    let kHead = -1, kDisp = -1, rate = FRAME_HZ, lastPushAt = 0, lastPushK = -1;
    let diffHi = 4;                                         // eased upper bound for the diff readout
    const chart = hiDpiCanvas(el.chart), raster = hiDpiCanvas(el.raster);
    const R = 58;                                           // right margin: y labels and the θ tag
    const hair = (c) => 1 / c.dpr;
    const snap = (c, v) => (Math.round(v * c.dpr) + 0.5) / c.dpr;

    function reset() {
      kArr.fill(-1); dets.length = 0; kHead = -1; kDisp = -1; lastPushK = -1;
    }

    function push(frames) {
      if (!Array.isArray(frames) || !frames.length) return;
      for (const f of frames) {
        if (!f || !isNum(f.k)) continue;
        if (kHead >= 0 && f.k < kHead - 200) reset();      // the bridge restarted
        const i = f.k & (RING - 1);
        kArr[i] = f.k;
        yArr[i] = isNum(f.y) ? f.y : NaN;
        const s = Array.isArray(f.s) ? f.s : [];
        for (let c = 0; c < 3; c++) sArr[c][i] = isNum(s[c]) ? s[c] : 0;
        if (f.k > kHead) kHead = f.k;
        if (isNum(f.y)) app.y = f.y;
      }
      const t = now();
      if (lastPushK >= 0 && t > lastPushAt && kHead > lastPushK) {
        const r = ((kHead - lastPushK) * 1000) / (t - lastPushAt);
        rate += (clamp(r, 5, 400) - rate) * 0.15;          // measured frames/s, for the display clock
      }
      lastPushAt = t; lastPushK = kHead;
      if (kDisp < 0 || kHead - kDisp > 120) kDisp = kHead - 4;
    }

    function advance(dt) {
      if (kHead < 0) return;
      const target = kHead - Math.max(3, rate * 0.045);    // about one batch behind the newest frame
      kDisp += (dt * rate) / 1000;
      kDisp += (target - kDisp) * Math.min(1, dt / 300);
      if (kDisp > kHead) kDisp = kHead;
    }

    function addDetection(d) {
      const k = isNum(d.k) ? d.k : kHead;
      if (k < 0) return;
      dets.push({ k, y: isNum(d.y) ? d.y : app.theta, acted: !!d.acted });
      while (dets.length > 40) dets.shift();
    }

    function yRange() {
      if (app.mode !== 'diff') return [0, 1];
      let lo = 0, hi = 1;
      const k0 = Math.floor(kDisp - SPAN);
      for (let k = Math.max(0, k0); k <= kHead; k++) {
        const i = k & (RING - 1);
        if (kArr[i] !== k) continue;
        const y = yArr[i];
        if (y < lo) lo = y;
        if (y > hi) hi = y;
      }
      const target = Math.max(hi, num(app.theta, 1) * 1.3, 2);
      diffHi += (target - diffHi) * 0.08;
      return [Math.min(0, Math.floor(lo)), Math.max(diffHi, 1)];
    }

    function ticks(lo, hi) {
      if (app.mode !== 'diff') return [0, 0.5, 1];
      const span = hi - lo, raw = span / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= 5) || 10 * mag;
      const out = [];
      for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
      return out;
    }

    function drawRaster() {
      const c = raster, g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      if (w < 20) return;
      const pw = w - R, k1 = kDisp, k0 = k1 - SPAN;
      const X = (k) => ((k - k0) / SPAN) * pw;
      const chans = app.mode === 'diff' ? [app.channel, app.channel === 0 ? 1 : 0] : [app.channel];
      const gap = 3, rowH = (h - gap * (chans.length - 1)) / chans.length;
      g.font = `500 11px ${FONT}`;
      g.textBaseline = 'middle';
      chans.forEach((ch, r) => {
        const top = r * (rowH + gap);
        g.fillStyle = C.track;
        roundRect(g, 0, top, pw, rowH, Math.min(5, rowH / 2));
        g.fill();
        let count = 0;
        if (ch <= 2 && kHead >= 0) {
          const arr = sArr[ch];
          g.fillStyle = r === 0 ? C.spike : C.spike2;
          const kEnd = Math.floor(k1);
          for (let k = Math.max(Math.ceil(k0), kHead - RING + 1); k <= kEnd; k++) {
            const i = k & (RING - 1);
            if (kArr[i] !== k) continue;
            const v = arr[i];
            if (v > 0) {
              g.globalAlpha = Math.min(1, 0.55 + 0.45 * v);
              g.fillRect(X(k) - 0.6, top + 2.5, 1.2, rowH - 5);
              if (k > kEnd - FRAME_HZ) count++;
            }
          }
          g.globalAlpha = 1;
        }
        g.fillStyle = C.label;
        g.textAlign = 'left';
        g.fillText(kHead >= 0 ? `${count}/s` : '', pw + 8, top + rowH / 2 + 0.5);
      });
      g.fillStyle = C.det;
      for (const d of dets) {
        if (d.k < k0 || d.k > k1) continue;
        g.fillRect(snap(c, X(d.k)) - 0.5, 0, 1, h);
      }
    }

    function drawChart() {
      const c = chart, g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      if (w < 40) return;
      const T = 12, B = 24, pw = w - R, ph = h - T - B;
      const k1 = kDisp, k0 = k1 - SPAN;
      const [lo, hi] = yRange();
      const X = (k) => ((k - k0) / SPAN) * pw;
      const Y = (v) => T + (1 - (clamp(v, lo, hi) - lo) / (hi - lo)) * ph;
      const theta = app.theta, yT = isNum(theta) ? Y(theta) : null;

      // grid, y labels, time axis
      g.lineWidth = hair(c);
      g.font = `500 11px ${FONT}`;
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      for (const v of ticks(lo, hi)) {
        const y = snap(c, Y(v));
        g.strokeStyle = v === 0 ? C.base : C.grid;
        g.beginPath(); g.moveTo(0, y); g.lineTo(pw, y); g.stroke();
        if (yT === null || Math.abs(y - yT) > 12) {
          g.fillStyle = C.label;
          g.fillText(app.mode === 'diff' ? String(v) : v.toFixed(1), pw + 8, y);
        }
      }
      g.textBaseline = 'alphabetic';
      for (const s of [6, 4, 2, 0]) {
        const x = pw * (1 - s / 6);
        if (s > 0 && s < 6) {
          g.strokeStyle = C.grid2;
          g.beginPath(); g.moveTo(snap(c, x), T); g.lineTo(snap(c, x), T + ph); g.stroke();
        }
        g.fillStyle = C.label;
        g.textAlign = s === 6 ? 'left' : s === 0 ? 'right' : 'center';
        g.fillText(s ? `−${s} s` : 'now', x, h - 6);
      }

      // the readout as runs of consecutive frames
      const runs = [];
      if (kHead >= 0) {
        let run = null;
        const kEnd = Math.floor(k1);
        for (let k = Math.max(Math.floor(k0), kHead - RING + 1); k <= kEnd; k++) {
          const i = k & (RING - 1), y = yArr[i];
          if (kArr[i] !== k || !Number.isFinite(y)) { run = null; continue; }
          if (!run) { run = []; runs.push(run); }
          run.push(X(k), Y(y));
        }
      }
      g.save();
      g.beginPath(); g.rect(0, 0, pw, h); g.clip();
      const base = Y(clamp(0, lo, hi));
      const trace = (r) => { g.moveTo(r[0], r[1]); for (let j = 2; j < r.length; j += 2) g.lineTo(r[j], r[j + 1]); };
      const area = g.createLinearGradient(0, T, 0, T + ph);
      area.addColorStop(0, C.area0); area.addColorStop(1, C.area1);
      g.lineJoin = 'round'; g.lineCap = 'round';
      for (const r of runs) {
        if (r.length < 4) continue;
        g.beginPath(); g.moveTo(r[0], base); for (let j = 0; j < r.length; j += 2) g.lineTo(r[j], r[j + 1]);
        g.lineTo(r[r.length - 2], base); g.closePath();
        g.fillStyle = area; g.fill();
        g.beginPath(); trace(r); g.strokeStyle = C.line; g.lineWidth = 1.75; g.stroke();
      }
      if (yT !== null) {                                    // above θ: red line, red wash down to θ
        g.save();
        g.beginPath(); g.rect(0, 0, pw, yT); g.clip();
        const hot = g.createLinearGradient(0, T, 0, yT);
        hot.addColorStop(0, C.hotArea0); hot.addColorStop(1, C.hotArea1);
        for (const r of runs) {
          if (r.length < 4) continue;
          g.beginPath(); g.moveTo(r[0], yT); for (let j = 0; j < r.length; j += 2) g.lineTo(r[j], r[j + 1]);
          g.lineTo(r[r.length - 2], yT); g.closePath();
          g.fillStyle = hot; g.fill();
          g.beginPath(); trace(r); g.strokeStyle = C.hot; g.lineWidth = 2; g.stroke();
        }
        g.restore();
      }
      for (const d of dets) {                               // detections
        if (d.k < k0 - 2 || d.k > k1) continue;
        const x = X(d.k);
        g.fillStyle = C.det;
        g.fillRect(snap(c, x) - 0.5, T, 1, ph);
      }
      g.restore();

      if (yT !== null) {                                    // θ line and its tag
        g.setLineDash([4, 4]);
        g.strokeStyle = C.hot; g.lineWidth = 1.25;
        g.beginPath(); g.moveTo(0, snap(c, yT)); g.lineTo(pw, snap(c, yT)); g.stroke();
        g.setLineDash([]);
        const txt = `θ ${fmtVal(theta)}`;
        g.font = `600 11px ${FONT}`;
        const tw = Math.ceil(g.measureText(txt).width) + 12;
        g.fillStyle = C.thetaBg;
        roundRect(g, pw + 4, yT - 9, Math.min(tw, R - 4), 18, 6); g.fill();
        g.fillStyle = C.thetaInk; g.textAlign = 'left'; g.textBaseline = 'middle';
        g.fillText(txt, pw + 10, yT + 0.5);
      }
      for (const d of dets) {                               // detection dots on top
        if (d.k < k0 || d.k > k1) continue;
        const x = X(d.k), y = Y(num(d.y, theta));
        g.beginPath(); g.arc(x, y, 6, 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill();
        g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2);
        if (d.acted) { g.fillStyle = C.hot; g.fill(); }
        else { g.fillStyle = '#fff'; g.fill(); g.lineWidth = 1.5; g.strokeStyle = C.hot; g.stroke(); }
      }
      const last = runs[runs.length - 1];                   // live head
      if (last && last.length >= 2 && kHead - k1 < 30) {
        const x = last[last.length - 2], y = last[last.length - 1];
        const over = yT !== null && y < yT;
        g.beginPath(); g.arc(x, y, 7, 0, Math.PI * 2);
        g.fillStyle = over ? 'rgba(255, 59, 48, 0.16)' : 'rgba(0, 113, 227, 0.14)'; g.fill();
        g.beginPath(); g.arc(x, y, 3.5, 0, Math.PI * 2);
        g.fillStyle = over ? C.hot : C.line; g.fill();
        g.lineWidth = 1.5; g.strokeStyle = '#fff'; g.stroke();
      }
    }

    return {
      push, advance, addDetection, reset,
      draw() { drawRaster(); drawChart(); },
      hasData: () => kHead >= 0,
      setRows(n) { el.raster.style.height = n > 1 ? '34px' : ''; },
    };
  })();

  // ============================================================================================ status
  function applyStatus(s) {
    if (!s || typeof s !== 'object') return;
    app.status = s;
    const m = s.model || {};
    const mode = m.readout === 'diff' ? 'diff' : 'ema';
    if (mode !== app.mode) { app.mode = mode; configureTheta(); plot.setRows(mode === 'diff' ? 2 : 1); }
    app.channel = isNum(m.channel) ? m.channel : 0;

    // auto-dodge: keep an optimistic local change until the bridge confirms it
    const auto = !!(s.ros && s.ros.auto);
    const pend = app.autoPending && now() < app.autoPending.until ? app.autoPending : null;
    if (pend) {
      if (auto === pend.value) app.auto = auto;
    } else if (auto !== app.auto) {
      app.auto = auto;
      if (app.autoKnown) logToggle(auto, 'Y toggle');
    }
    app.autoKnown = true;
    renderAuto();
    renderRobot();

    if (isNum(m.theta) && !app.thetaDragging && now() > app.thetaHoldUntil) setTheta(m.theta);
    if (s.fpga) {
      if (isNum(s.fpga.latency_ms)) app.latency = s.fpga.latency_ms;
      if (isNum(s.fpga.fps)) app.fps = s.fpga.fps;
    }
    if (s.counters) {
      setText(el.kpiDet, String(num(s.counters.detections, 0)));
      setText(el.kpiDodges, String(num(s.counters.dodges, 0)));
    }
    if (s.robot) robotFromStatus(s.robot);
    if (s.color && now() > app.colorPendingUntil) {
      if (!!s.color.enabled !== app.colorOn) setColorVisible(!!s.color.enabled, false);
      if (app.colorOn && isNum(s.color.seq) && s.color.seq > color.seq && s.color.available) loadColor(s.color.seq);
    }
    renderModel();
    renderPills();
    renderEmpties();
  }

  function renderModel() {
    const m = (app.status && app.status.model) || {};
    const name = m.name || 'SNN';
    const readout = app.mode === 'diff' ? 'Σ(ch0 − ch1) readout' : 'EMA readout';
    setText(el.snnSub, `${name} · ${readout}`);
    const fpga = (app.status && app.status.fpga) || {};
    const mock = !!m.mock || fpga.state === 'mock';
    el.snnTag.hidden = !(mock || fpga.state === 'offline');
    setText(el.snnTag, mock ? 'Mock SNN' : 'Offline');
    setData(el.snnTag, 'tone', mock ? 'warn' : 'bad');
    const ch = app.channel;
    const label = app.mode === 'diff' ? `ch ${ch} − ch ${ch === 0 ? 1 : 0}` : `ch ${ch}`;
    if (el.rasterLabel.lastChild.textContent !== label) el.rasterLabel.lastChild.textContent = label;
  }

  function setPill(pill, valueEl, tone, value, live) {
    setData(pill, 'tone', tone);
    setData(pill, 'live', !!live);
    setText(valueEl, value);
  }

  function renderPills() {
    const s = app.status;
    if (!app.connected || !s) {
      setPill(el.pillCam, el.pillCamV, 'idle', '—');
      setPill(el.pillFpga, el.pillFpgaV, 'idle', '—');
      setPill(el.pillRobot, el.pillRobotV, 'idle', '—');
      el.stageBadge.hidden = true;
      return;
    }
    const cam = s.camera || {};
    const rate = num(app.camRate, num(cam.rate, null));
    const camTone = { streaming: 'ok', replay: 'info', waiting: 'warn', offline: 'bad' }[cam.state] || 'idle';
    const camText = { streaming: `${fmtRate(rate)} ev/s`, replay: `Replay · ${fmtRate(rate)} ev/s`,
                      waiting: 'Waiting', offline: 'Offline' }[cam.state] || '—';
    setPill(el.pillCam, el.pillCamV, camTone, camText, cam.state === 'streaming' || cam.state === 'replay');
    el.stageBadge.hidden = !(cam.state === 'streaming' || cam.state === 'replay');
    setData(el.stageBadge, 'tone', cam.state === 'replay' ? 'info' : 'ok');
    setText(el.stageBadgeText, cam.state === 'replay' ? 'Replay' : 'Live');

    const fpga = s.fpga || {};
    const mock = fpga.state === 'mock' || (s.model && s.model.mock);
    if (mock) setPill(el.pillFpga, el.pillFpgaV, 'warn', 'Mock', true);
    else if (fpga.state === 'connected') {
      const lat = isNum(app.latency) ? `${app.latency.toFixed(1)} ms` : '—';
      const fps = isNum(app.fps) ? ` · ${Math.round(app.fps)} fps` : '';
      setPill(el.pillFpga, el.pillFpgaV, 'ok', lat + fps, true);
    } else setPill(el.pillFpga, el.pillFpgaV, fpga.state === 'offline' ? 'bad' : 'idle',
                   fpga.state === 'offline' ? 'Offline' : '—');

    const r = robotView();
    setPill(el.pillRobot, el.pillRobotV, r.tone, r.title, r.moving);
  }

  function renderEmpties() {
    const s = app.status || {};
    const cam = s.camera || {};
    let show = true, tone = '', title = '', sub = '';
    if (!app.connected) {
      tone = app.everConnected ? 'warn' : '';
      title = app.everConnected ? 'Connection lost' : 'Connecting…';
      sub = app.everConnected ? 'The page will catch up as soon as the bridge answers again.'
                              : 'Waiting for snn_bridge.py on the robot.';
    } else if (cam.state === 'waiting') {
      tone = 'warn'; title = 'Waiting for the event camera'; sub = 'despot_debug_camera has not connected yet.';
    } else if (cam.state === 'offline') {
      tone = 'bad'; title = 'Event camera offline'; sub = 'No events from the camera. Is despot_debug_camera running?';
    } else show = false;
    el.stageEmpty.hidden = !show;
    if (show) {
      setData(el.stageEmpty, 'tone', tone || 'idle');
      setText(el.stageEmptyTitle, title);
      setText(el.stageEmptySub, sub);
    }

    const fpga = s.fpga || {};
    let snnShow = false, sTone = 'idle', sTitle = '', sSub = '';
    if (app.connected && fpga.state === 'offline') {
      snnShow = true; sTone = 'bad'; sTitle = 'FPGA offline'; sSub = 'Reconnecting to the SPLEAT server…';
    } else if (app.connected && !plot.hasData()) {
      snnShow = true; sTitle = 'Waiting for SNN output'; sSub = 'The first frames arrive once the camera streams.';
    }
    el.snnEmpty.hidden = !snnShow;
    if (snnShow) {
      setData(el.snnEmpty, 'tone', sTone);
      setText(el.snnEmptyTitle, sTitle);
      setText(el.snnEmptySub, sSub);
    }
  }

  // ============================================================================================ auto-dodge switch
  function renderAuto() {
    const s = app.status;
    const rosOk = !s || !s.ros || s.ros.available !== false;
    const on = app.auto;
    setData(el.auto, 'on', on);
    setData(el.auto, 'warn', on && !rosOk);
    el.autoSwitch.setAttribute('aria-checked', String(on));
    setText(el.autoState, on ? (rosOk ? 'Connected' : 'On · no ROS') : 'Disconnected');
    setText(el.autoHintText, rosOk ? 'on the remote toggles' : 'ROS is not running');
    el.autoSwitch.title = on ? 'SNN detections make the robot dodge. Click to disconnect.'
                             : 'Detections are only shown. Click to let the SNN trigger the dodge.';
  }

  el.autoSwitch.addEventListener('click', async () => {
    const next = !app.auto;
    app.auto = next;
    app.autoPending = { value: next, until: now() + 4000 };
    el.autoSwitch.setAttribute('aria-busy', 'true');
    renderAuto();
    renderRobot();
    try {
      const s = await api('api/auto', { enabled: next });
      app.autoPending = { value: next, until: now() + 1500 };   // ignore snapshots sent before the change
      logToggle(next, 'Web');
      if (s && s.ros) applyStatus(s);
    } catch (e) {
      app.autoPending = null;
      app.auto = !next;
      renderAuto();
      renderRobot();
      toast('Auto-dodge was not changed: the bridge did not answer.');
    } finally {
      el.autoSwitch.removeAttribute('aria-busy');
    }
  });

  // ============================================================================================ threshold
  function configureTheta() {
    const diff = app.mode === 'diff';
    el.theta.min = diff ? '1' : '0.05';
    el.theta.max = diff ? '20' : '0.99';
    el.theta.step = diff ? '1' : '0.01';
    if (isNum(app.theta)) setTheta(app.theta);
  }

  function setTheta(v) {
    app.theta = v;
    if (Number(el.theta.value) !== v) el.theta.value = String(v);
    const min = Number(el.theta.min), max = Number(el.theta.max);
    el.theta.style.setProperty('--pct', `${(clamp((v - min) / (max - min), 0, 1) * 100).toFixed(2)}%`);
    el.theta.setAttribute('aria-valuetext', `θ ${fmtVal(v)}`);
    setText(el.thetaOut, fmtVal(v));
    setText(el.snnTheta, fmtVal(v));
  }

  let thetaTimer = 0, thetaSent = null;
  function postTheta(v) {
    if (v === thetaSent) return;
    thetaSent = v;
    api('api/theta', { theta: v }).then((s) => { if (s && s.model) applyStatus(s); })
      .catch(() => toast('θ was not changed: the bridge did not answer.'));
  }
  el.theta.addEventListener('input', () => {
    app.thetaDragging = true;
    const v = Number(el.theta.value);
    setTheta(v);
    clearTimeout(thetaTimer);
    thetaTimer = setTimeout(() => postTheta(v), 180);
  });
  el.theta.addEventListener('change', () => {
    app.thetaDragging = false;
    app.thetaHoldUntil = now() + 1500;
    clearTimeout(thetaTimer);
    postTheta(Number(el.theta.value));
  });

  // ============================================================================================ THROW DETECTED
  let throwTimer = 0;
  function showThrow(acted) {
    const box = el.throwBox;
    box.hidden = true;
    box.classList.toggle('throw--passive', !acted);
    setText(el.throwTitle, acted ? 'THROW DETECTED' : 'Throw detected');
    setText(el.throwSubText, acted ? 'Dodging' : 'Auto-dodge off');
    void box.offsetWidth;                                     // restart the CSS animations
    box.hidden = false;
    clearTimeout(throwTimer);
    throwTimer = setTimeout(() => { box.hidden = true; }, OVERLAY_MS);
  }

  function onDetection(d) {
    const acted = !!d.acted;
    plot.addDetection(d);
    showThrow(acted);
    app.lastDetAt = now();
    const row = addLog({
      source: 'SNN', cls: 'snn', action: 'Throw detected',
      detail: acted ? 'dodge sent' : 'auto-dodge off',
    });
    app.pendingDetRow = acted ? { row, until: now() + 1500 } : null;
  }

  // ============================================================================================ robot
  const robot = { state: 'unknown', speed: null, out: null, toGo: null, lastEventAt: 0, dodge: null, lastDist: null };
  const ROBOT_TEXT = {
    idle: ['Idle', 'ok'], dash: ['Dodging', 'bad'], hold: ['Holding', 'warn'], return: ['Returning', 'info'],
    unknown: ['No data', 'idle'], noros: ['No ROS', 'idle'],
  };

  function robotView() {
    const rosOff = app.status && app.status.ros && app.status.ros.available === false;
    const st = rosOff ? 'noros' : robot.state;
    const [title, tone] = ROBOT_TEXT[st] || ROBOT_TEXT.unknown;
    return { state: st, title, tone, moving: st === 'dash' || st === 'return' };
  }

  function robotFromStatus(r) {
    if (r.last && robot.lastDist === null && isNum(r.last.out_m)) {
      robot.lastDist = r.last.out_m;
      renderKpis();
    }
    if (now() - robot.lastEventAt < 2000) return;             // the event stream is fresher
    const st = ['idle', 'dash', 'hold', 'return'].includes(r.state) ? r.state : 'unknown';
    if (st !== robot.state) { robot.state = st; renderRobot(); }
  }

  function srcLabel(src) {
    const s = String(src || '');
    const m = s.match(/button\s+([a-z0-9]+)/i);
    if (m) return { label: `${m[1].toUpperCase()} button`, cls: m[1].toLowerCase() === 'x' ? 'x' : '' };
    if (s.includes('/reflex/dash')) return { label: 'Reflex', cls: '' };
    if (s.includes('/dodge/dash')) return { label: 'Topic', cls: '' };
    return { label: s || 'Robot', cls: '' };
  }

  function onRobot(e) {
    if (!e || typeof e !== 'object') return;
    const t = now();
    robot.lastEventAt = t;
    const d = robot.dodge;
    const close = (phase) => { if (d && d.took[phase] === undefined && d.phase === phase) d.took[phase] = (t - d.t0) / 1000; };
    switch (e.event) {
      case 'dash': {
        robot.state = 'dash';
        robot.speed = num(e.speed, null);
        robot.out = null; robot.toGo = null;
        const src = srcLabel(e.source);
        let row = null;
        const pend = app.pendingDetRow;
        if (pend && t < pend.until && !src.cls) {           // the SNN's /reflex/dash: same row as the detection
          row = pend.row; row.detail = 'dodging'; updateLog();
        } else {
          row = addLog({ source: src.label, cls: src.cls, action: src.cls === 'x' ? 'Manual dodge' : 'Dodge',
                         detail: isNum(e.speed) ? `${e.speed.toFixed(1)} m/s` : '' });
        }
        app.pendingDetRow = null;
        robot.dodge = { phase: 'dash', t0: t, took: {}, row,
                        plan: { dash: num(e.duration, 0.5), hold: HOLD_S, return: null } };
        break;
      }
      case 'hold':
        if (!d) robot.dodge = { phase: 'hold', t0: t, took: {}, plan: { dash: 0.5, hold: HOLD_S, return: null } };
        else { close('dash'); d.phase = 'hold'; d.t0 = t; }
        robot.state = 'hold';
        robot.out = num(e.out_m, num(e.away_m, null));
        setRow(robot.dodge, robot.out !== null ? `${robot.out.toFixed(2)} m out` : 'holding');
        break;
      case 'return':
        if (!d) robot.dodge = { phase: 'return', t0: t, took: {}, plan: { dash: 0.5, hold: HOLD_S, return: null } };
        else { close('hold'); d.phase = 'return'; d.t0 = t; }
        robot.state = 'return';
        robot.toGo = num(e.to_go_m, robot.out);
        robot.dodge.plan.return = isNum(robot.toGo) ? Math.max(0.4, robot.toGo / RETURN_SPEED) : 2;
        setRow(robot.dodge, isNum(robot.toGo) ? `${robot.toGo.toFixed(2)} m back…` : 'returning');
        break;
      case 'idle':
        if (d && d.phase !== 'done') {
          close(d.phase);
          d.phase = 'done';
          const out = num(e.out_m, robot.out);
          if (isNum(out)) { robot.lastDist = out; robot.out = out; }
          setRow(d, isNum(out) ? `${out.toFixed(2)} m · back` : (e.reason || 'done'));
        }
        robot.state = 'idle';
        break;
      case 'stop':
        robot.state = 'idle';
        addLog({ source: 'Stop', cls: '', action: 'Stopped', detail: e.reason || '' });
        break;
      case 'ignored': {
        const pend = app.pendingDetRow;
        if (pend && t < pend.until) { pend.row.detail = 'robot busy'; updateLog(); app.pendingDetRow = null; }
        else {
          const src = srcLabel(e.source);
          addLog({ source: src.label, cls: src.cls, action: 'Ignored', detail: e.state ? `busy (${e.state})` : 'busy' });
        }
        break;
      }
      default:
        break;
    }
    renderRobot();
    renderKpis();
    renderPills();
  }

  function setRow(dodge, detail) {
    if (dodge && dodge.row) { dodge.row.detail = detail; updateLog(); }
  }

  function renderRobot() {
    const v = robotView();
    setData(el.rstate, 'state', v.state === 'noros' ? 'unknown' : v.state);
    setData(el.rstate, 'moving', v.moving);
    setText(el.robotState, v.title);
    let sub;
    switch (v.state) {
      case 'idle': sub = app.auto ? 'Ready · the SNN can trigger a dodge' : 'Ready · X on the remote dodges'; break;
      case 'dash': sub = isNum(robot.speed) ? `Moving back at ${robot.speed.toFixed(1)} m/s` : 'Moving back'; break;
      case 'hold': sub = isNum(robot.out) ? `${robot.out.toFixed(2)} m out · standing` : 'Standing'; break;
      case 'return': sub = isNum(robot.toGo) ? `Walking ${robot.toGo.toFixed(2)} m back to the start` : 'Walking back'; break;
      case 'noros': sub = 'The bridge runs without ROS'; break;
      default: sub = 'No event from go2_dodge yet';
    }
    setText(el.robotSub, sub);
  }

  function renderKpis() {
    const d = robot.lastDist;
    const html = isNum(d) ? `${d.toFixed(2)}<small>m</small>` : '—';
    if (el.kpiDist.innerHTML !== html) el.kpiDist.innerHTML = html;
  }

  let phaseCache = '';
  function tickPhases(t) {
    const d = robot.dodge;
    let key = '';
    const parts = el.phaseItems.map((li) => {
      const p = li.dataset.phase;
      let frac = 0, label = '—', active = false;
      if (d) {
        if (d.took[p] !== undefined) { frac = 1; label = `${d.took[p].toFixed(1)} s`; }
        else if (d.phase === p) {
          const s = (t - d.t0) / 1000;
          frac = d.plan[p] ? Math.min(0.96, s / d.plan[p]) : 0.5;
          label = `${s.toFixed(1)} s`;
          active = true;
        }
      }
      key += `${(frac * 100).toFixed(1)}|${label}|${active};`;
      return { li, frac, label, active };
    });
    const done = !!d && d.phase === 'done';
    key += done;
    if (key === phaseCache) return;
    phaseCache = key;
    for (const p of parts) {
      p.li.querySelector('.phase__bar i').style.width = `${(p.frac * 100).toFixed(1)}%`;
      setText(p.li.querySelector('.phase__time'), p.label);
      setData(p.li, 'active', p.active);
    }
    setData(el.phases, 'done', done);
  }

  // ============================================================================================ activity log
  const logRows = [];
  let logSeq = 0;
  function addLog(row) {
    row.id = ++logSeq;
    row.at = new Date();
    logRows.unshift(row);
    if (logRows.length > LOG_MAX) logRows.length = LOG_MAX;
    renderLog(row.id);
    return row;
  }
  function updateLog() { renderLog(-1); }
  function renderLog(newId) {
    const items = logRows.map((r) => {
      const li = document.createElement('li');
      li.className = 'log__item' + (r.id === newId ? ' is-new' : '');
      const time = document.createElement('span');
      time.className = 'log__time'; time.textContent = fmtClock(r.at);
      const src = document.createElement('span');
      src.className = 'src' + (r.cls ? ` src--${r.cls}` : ''); src.textContent = r.source;
      const act = document.createElement('span');
      act.className = 'log__action'; act.textContent = r.action;
      const det = document.createElement('span');
      det.className = 'log__detail'; det.textContent = r.detail || '';
      li.append(time, src, act, det);
      return li;
    });
    el.log.replaceChildren(...items);
    el.logEmpty.hidden = logRows.length > 0;
  }
  function logToggle(on, source) {
    addLog({ source, cls: source === 'Web' ? 'web' : 'y', action: on ? 'Auto-dodge on' : 'Auto-dodge off',
             detail: on ? 'SNN → robot' : 'watch only' });
  }

  // ============================================================================================ color camera
  const color = { seq: -1, wanted: -1, loading: false, url: null, lastAt: 0, shownAt: 0 };

  function setColorVisible(on, animate = true) {
    app.colorOn = on;
    el.colorToggle.checked = on;
    el.colorCard.hidden = !on;
    el.layout.dataset.color = on ? 'on' : 'off';
    if (on) {
      color.shownAt = now();
      if (animate) {
        el.colorCard.classList.remove('color-enter');
        void el.colorCard.offsetWidth;
        el.colorCard.classList.add('color-enter');
      }
    }
    renderColor();
  }

  function renderColor() {
    if (!app.colorOn) return;
    const st = (app.status && app.status.color) || {};
    const fresh = color.lastAt && now() - color.lastAt < 3000;
    const waited = now() - color.shownAt > 4000;
    if (fresh) { el.colorEmpty.hidden = true; el.colorImg.hidden = false; }
    else if (!color.lastAt || waited) {
      el.colorEmpty.hidden = false;
      const unavailable = waited && !st.available;
      setData(el.colorEmpty, 'tone', unavailable ? 'warn' : 'idle');
      setText(el.colorEmptyTitle, unavailable ? 'Color camera not available'
                                 : color.lastAt ? 'Color image paused' : 'Connecting to the Go2 camera…');
      setText(el.colorEmptySub, unavailable ? 'The Go2 videohub does not answer. The event camera is unaffected.'
                               : 'Asking the videohub for a frame.');
    }
    if (color.lastAt) setText(el.colorAge, fmtAgo(now() - color.lastAt));
    const im = el.colorImg;
    setText(el.colorMeta, im.naturalWidth ? `JPEG · ${im.naturalWidth} × ${im.naturalHeight}` : 'Go2 videohub');
  }

  async function loadColor(seq) {
    if (color.loading) { color.wanted = Math.max(color.wanted, seq); return; }
    color.loading = true;
    let url = null;
    try {
      if (DEMO) url = demo.colorURL();
      else {
        const r = await fetch(`color.jpg?seq=${seq}`, { cache: 'no-store' });
        if (!r.ok) throw new Error(String(r.status));
        url = URL.createObjectURL(await r.blob());
      }
      const im = new Image();
      im.src = url;
      await im.decode();
      el.colorImg.src = url;
      if (color.url && color.url.startsWith('blob:')) URL.revokeObjectURL(color.url);
      color.url = url;
      color.seq = seq;
      color.lastAt = now();
      renderColor();
    } catch (e) {
      if (url && url.startsWith('blob:') && url !== color.url) URL.revokeObjectURL(url);
    } finally {
      color.loading = false;
      if (app.colorOn && color.wanted > color.seq) { const w = color.wanted; color.wanted = -1; loadColor(w); }
    }
  }

  el.colorToggle.addEventListener('change', async () => {
    const on = el.colorToggle.checked;
    app.colorPendingUntil = now() + 4000;
    setColorVisible(on);
    try {
      const s = await api('api/color', { enabled: on });
      app.colorPendingUntil = now() + 1200;
      if (s && s.color) applyStatus(s);
    } catch (e) {
      app.colorPendingUntil = 0;
      setColorVisible(!on, false);
      toast('The color camera could not be switched: the bridge did not answer.');
    }
  });

  // ============================================================================================ stream handlers
  const handlers = {
    frame(d) {
      eventView.push(d);
      if (isNum(d.rate)) { app.camRate = d.rate; setText(el.camRate, fmtRate(d.rate)); }
    },
    snn(d) {
      plot.push(d.frames);
      if (isNum(d.theta) && !app.thetaDragging && now() > app.thetaHoldUntil && d.theta !== app.theta) setTheta(d.theta);
      if (isNum(d.latency_ms)) app.latency = d.latency_ms;
      if (isNum(d.fps)) app.fps = d.fps;
    },
    detection: onDetection,
    status: applyStatus,
    robot: onRobot,
    color(d) { if (app.colorOn && isNum(d.seq)) loadColor(d.seq); },
  };

  function dispatch(type, data) {
    markAlive();
    const fn = handlers[type];
    if (fn && data && typeof data === 'object') {
      try { fn(data); } catch (e) { console.error(`[${type}]`, e); }
    }
  }

  // ============================================================================================ connection
  async function api(path, body) {
    if (DEMO) return demo.handle(path, body);
    const r = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const type = r.headers.get('content-type') || '';
    return type.includes('json') ? r.json() : null;
  }

  async function fetchState() {
    try {
      const r = await fetch('api/state', { cache: 'no-store' });
      if (r.ok) applyStatus(await r.json());
    } catch (e) { /* the stream watchdog reports it */ }
  }

  let source = null, reopenTimer = 0, backoff = 1000;
  function openStream() {
    clearTimeout(reopenTimer);
    if (source) source.close();
    source = new EventSource('stream');
    source.onopen = () => { backoff = 1000; fetchState(); };
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) scheduleReopen();   // e.g. an HTTP error: no auto retry
    };
    for (const type of Object.keys(handlers)) {
      source.addEventListener(type, (ev) => {
        let data;
        try { data = JSON.parse(ev.data); } catch (e) { return; }
        dispatch(type, data);
      });
    }
  }
  function scheduleReopen() {
    clearTimeout(reopenTimer);
    reopenTimer = setTimeout(openStream, backoff);
    backoff = Math.min(backoff * 2, 10000);
  }

  function markAlive() {
    app.lastMsg = now();
    if (!app.connected) setConnected(true);
  }

  function setConnected(on) {
    app.connected = on;
    if (on) app.everConnected = true;
    el.banner.hidden = on;
    setText(el.bannerText, app.everConnected ? 'Reconnecting to the robot…' : 'Connecting to the robot…');
    el.layout.dataset.stale = String(!on && app.everConnected);
    renderPills();
    renderEmpties();
  }

  function watchdog() {
    const quiet = now() - app.lastMsg;
    if (app.connected && quiet > STALE_MS) setConnected(false);
    if (!app.connected && el.banner.hidden && quiet > 2500) setConnected(false);   // first connect: show the banner
    if (DEMO || !source) return;
    if (quiet > 8000 && source.readyState === EventSource.OPEN) scheduleReopen();      // open but silent
  }

  // ============================================================================================ misc UI
  let toastTimer = 0;
  function toast(text) {
    el.toast.textContent = text;
    el.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.hidden = true; }, 3200);
  }

  function slowTick() {                                        // 4 Hz: relative times and live values
    setText(el.snnLast, app.lastDetAt ? fmtAgo(now() - app.lastDetAt) : '—');
    const perf = isNum(app.latency) ? `${app.latency.toFixed(1)} ms` + (isNum(app.fps) ? ` · ${Math.round(app.fps)} fps` : '') : '—';
    setText(el.snnPerf, app.status && app.status.fpga && app.status.fpga.state === 'offline' ? 'offline' : perf);
    setText(el.snnY, fmtVal(app.y));
    setData(el.statY, 'over', isNum(app.y) && isNum(app.theta) && app.y >= app.theta);
    renderPills();
    renderEmpties();
    renderColor();
  }

  let lastFrameT = now();
  function loop(t) {
    const dt = clamp(t - lastFrameT, 0, 100);
    lastFrameT = t;
    eventView.draw(dt);
    plot.advance(dt);
    plot.draw();
    tickPhases(t);
    if (isNum(app.y)) {
      setText(el.snnY, fmtVal(app.y));
      setData(el.statY, 'over', isNum(app.theta) && app.y >= app.theta);
    }
    requestAnimationFrame(loop);
  }

  // ============================================================================================ demo
  /* Simulation of the whole backend for ?demo=1. A person stands in front of the robot and throws every 7-10 s
   * (every fourth time a feint): the event image is computed like a real event camera (thresholded brightness
   * changes between 10 ms renders of the scene, plus sensor noise and a few hot pixels), the SNN spikes rise
   * about 300 ms before the release, the EMA readout crosses θ, and with auto-dodge on the robot dashes back
   * (the scene shrinks and shakes in the event image), holds and walks back. The remote's Y and X, a short
   * FPGA dropout and the color camera are simulated too. */
  function createDemo() {
    const sim = {
      t: 0, k: 0, y: 0, theta: 0.85, refrUntil: -1, auto: false, rosOk: true,
      cam: 'streaming', fpga: 'connected', rate: 0, latency: 4.1,
      robot: 'idle', last: null, away: 0, awayFn: null, det: 0, dodges: 0,
      color: false, colorAvail: false, colorSeq: 0, nextColor: 0,
      throws: [], nextThrow: 2.6, throwCount: 0,
      queue: [], nextStatus: 0, nextX: 21, nextY: 7, yPhase: 0, nextOutage: 58,
    };
    const scene = document.createElement('canvas');
    scene.width = W; scene.height = H;
    const sg = scene.getContext('2d', { willReadFrequently: true });
    const photo = document.createElement('canvas');
    photo.width = 640; photo.height = 360;
    const pg = photo.getContext('2d');
    let prevLum = null;
    const hot = [[23, 141], [131, 37], [97, 152]];

    const at = (dt, fn) => sim.queue.push({ t: sim.t + dt, fn });
    const rand = (a, b) => a + Math.random() * (b - a);

    // ----- the thrower: joints in event-image coordinates
    const HAND = [[0, -5, 38], [0.3, -22, -4], [0.55, -9, -30], [0.64, 5, -23], [0.7, 15, -7], [0.86, 25, 27],
                  [1.2, 3, 37], [1.6, -5, 38]];
    const ELBOW = [[0, -4, 20], [0.3, -16, 7], [0.55, -14, -10], [0.64, -5, -13], [0.7, 4, -3], [0.86, 12, 14],
                   [1.2, 0, 19], [1.6, -4, 20]];
    const LEAN = [[0, 0, 0], [0.5, -4, 0], [0.72, 5, 0], [1.0, 3, 0], [1.6, 0, 0]];
    function spline(u, F) {
      const n = F.length;
      if (u <= F[0][0]) return [F[0][1], F[0][2]];
      if (u >= F[n - 1][0]) return [F[n - 1][1], F[n - 1][2]];
      let i = 1;
      while (F[i][0] < u) i++;
      const p0 = F[Math.max(0, i - 2)], p1 = F[i - 1], p2 = F[i], p3 = F[Math.min(n - 1, i + 1)];
      const f = (u - p1[0]) / (p2[0] - p1[0]), f2 = f * f, f3 = f2 * f;
      const cr = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * f + (2 * a - 5 * b + 4 * c - d) * f2 + (-a + 3 * b - 3 * c + d) * f3);
      return [cr(p0[1], p1[1], p2[1], p3[1]), cr(p0[2], p1[2], p2[2], p3[2])];
    }
    function pose(t) {
      const th = sim.throws.find((x) => t >= x.ts && t < x.ts + 1.6);
      const u = th ? t - th.ts : -1;
      const sway = 0.9 * Math.sin(t * 1.7) + 0.6 * Math.sin(t * 0.63);
      const lean = u >= 0 ? spline(u, LEAN)[0] : 0;
      const cx = 80 + sway + lean * 0.6;
      const S = [cx - 15, 64];
      const idle = 1.5 * Math.sin(t * 2.1);
      const e = u >= 0 ? spline(u, ELBOW) : [-4 + idle * 0.3, 20];
      const h = u >= 0 ? spline(u, HAND) : [-5 + idle, 38];
      const p = {
        cx, lean, head: [cx + lean * 0.5 + 0.4 * Math.sin(t * 0.9), 44],
        shoulderR: S, elbowR: [S[0] + e[0], S[1] + e[1]], handR: [S[0] + h[0], S[1] + h[1]],
        shoulderL: [cx + 15, 64], elbowL: [cx + 19, 84], handL: [cx + 20 - idle * 0.4, 102],
        ball: null,
      };
      const kind = th ? th.kind : 'throw';
      if (!th || kind === 'feint' || u < 0.7) {
        p.ball = { x: p.handR[0], y: p.handR[1] - 2, r: 4 };
      } else {
        const tau = u - 0.7, T = 0.42;
        if (tau < 0.38) {
          const f = tau / T, x0 = S[0] + 15, y0 = S[1] - 9;
          p.ball = { x: x0 + (84 - x0) * f * 0.5, y: y0 + 46 * f * f, r: Math.min(70, 4 / (1 - f)) };
        }
      }
      return p;
    }

    // ----- scene renders: grey levels for the event camera, colors for the Go2 camera
    function drawPerson(g, p, pal) {
      const limb = (a, b, w, c) => { g.strokeStyle = c; g.lineWidth = w; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); };
      g.lineCap = 'round'; g.lineJoin = 'round';
      const cx = p.cx;
      limb([cx - 6, 104], [cx - 9, 150], 9.5, pal.legs);
      limb([cx + 6, 104], [cx + 9, 150], 9.5, pal.legs);
      if (pal.shoes) { limb([cx - 10, 150], [cx - 13, 152], 6, pal.shoes); limb([cx + 10, 150], [cx + 13, 152], 6, pal.shoes); }
      g.fillStyle = pal.torso;
      g.beginPath();
      g.moveTo(cx - 16, 60); g.lineTo(cx + 16, 60); g.lineTo(cx + 12, 106); g.lineTo(cx - 12, 106); g.closePath();
      g.fill();
      limb(p.shoulderL, p.elbowL, 7.5, pal.arms); limb(p.elbowL, p.handL, 6.5, pal.arms);
      limb([cx, 52], [cx, 60], 7, pal.skin);
      g.fillStyle = pal.skin;
      g.beginPath(); g.arc(p.head[0], p.head[1], 9, 0, Math.PI * 2); g.fill();
      if (pal.hair) {
        g.fillStyle = pal.hair;
        g.beginPath(); g.arc(p.head[0], p.head[1] - 2, 9.2, Math.PI * 1.02, Math.PI * 1.98); g.fill();
      }
      limb(p.shoulderR, p.elbowR, 7.5, pal.arms); limb(p.elbowR, p.handR, 6.5, pal.arms);
      if (pal.hands) {
        g.fillStyle = pal.hands;
        for (const hnd of [p.handR, p.handL]) { g.beginPath(); g.arc(hnd[0], hnd[1], 3.4, 0, Math.PI * 2); g.fill(); }
      }
      if (p.ball) {
        g.fillStyle = pal.ball;
        g.beginPath(); g.arc(p.ball.x, p.ball.y, p.ball.r, 0, Math.PI * 2); g.fill();
        if (pal.ballHi) {
          g.fillStyle = pal.ballHi;
          g.beginPath(); g.arc(p.ball.x - p.ball.r * 0.3, p.ball.y - p.ball.r * 0.35, p.ball.r * 0.35, 0, Math.PI * 2); g.fill();
        }
      }
    }
    const grey = (v) => { const c = Math.round(v * 255); return `rgb(${c},${c},${c})`; };
    const GREY = { legs: grey(0.2), torso: grey(0.16), arms: grey(0.17), skin: grey(0.3), ball: grey(0.96) };
    function egoTransform(g, scale, offX, offY) {      // the robot backing off: the scene shrinks around (80, 90)
      const s = 3.5 / (3.5 + sim.away), j = sim.robot === 'dash' ? 1.2 * scale : 0;
      g.setTransform(scale * s, 0, 0, scale * s, scale * 80 * (1 - s) + offX + rand(-j, j),
                     scale * 90 * (1 - s) + offY + rand(-j, j));
    }
    function renderLum(t) {
      const g = sg;
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.fillStyle = grey(0.62); g.fillRect(0, 0, W, H);
      egoTransform(g, 1, 0, 0);
      g.fillStyle = grey(0.5); g.fillRect(-120, 122, 400, 160);
      g.fillStyle = grey(0.88); g.fillRect(112, 18, 36, 46);
      g.strokeStyle = grey(0.36); g.lineWidth = 4; g.strokeRect(14, 22, 34, 102);
      g.fillStyle = grey(0.42); g.fillRect(124, 98, 28, 26);
      drawPerson(g, pose(t), GREY);
      const d = g.getImageData(0, 0, W, H).data, lum = new Uint8Array(NPIX);
      for (let i = 0; i < NPIX; i++) lum[i] = d[i * 4];
      return lum;
    }
    function renderPhoto() {
      const g = pg, w = photo.width, h = photo.height, t = sim.t;
      g.setTransform(1, 0, 0, 1, 0, 0);
      const wall = g.createLinearGradient(0, 0, 0, h);
      wall.addColorStop(0, '#efebe5'); wall.addColorStop(0.7, '#e2dbd0');
      g.fillStyle = wall; g.fillRect(0, 0, w, h);
      egoTransform(g, 2, 160, -20);
      const floor = g.createLinearGradient(0, 122, 0, 200);
      floor.addColorStop(0, '#cbbfae'); floor.addColorStop(1, '#b3a591');
      g.fillStyle = floor; g.fillRect(-200, 122, 560, 120);
      g.fillStyle = '#f7f5f1'; g.fillRect(-200, 120, 560, 2.5);
      const sky = g.createLinearGradient(0, 18, 0, 64);
      sky.addColorStop(0, '#cfe4f7'); sky.addColorStop(1, '#f1f6fb');
      g.fillStyle = '#fbfaf8'; g.fillRect(108, 14, 44, 54);
      g.fillStyle = sky; g.fillRect(112, 18, 36, 46);
      g.fillStyle = '#fbfaf8'; g.fillRect(129, 18, 2, 46);
      g.fillStyle = '#d9c7ab'; g.fillRect(16, 24, 30, 98);
      g.strokeStyle = '#bba585'; g.lineWidth = 3; g.strokeRect(14.5, 22.5, 33, 101);
      g.fillStyle = '#9c8466'; g.beginPath(); g.arc(41, 76, 1.6, 0, Math.PI * 2); g.fill();
      g.fillStyle = '#c9a675'; g.fillRect(124, 98, 28, 26);
      g.fillStyle = '#b8935f'; g.fillRect(124, 98, 28, 4);
      g.fillStyle = 'rgba(0,0,0,0.08)';
      g.beginPath(); g.ellipse(80, 152, 26, 4, 0, 0, Math.PI * 2); g.fill();
      drawPerson(g, pose(t), { legs: '#2d3340', torso: '#3d5a80', arms: '#3d5a80', skin: '#e3b494', hair: '#3a2a20',
                               hands: '#e3b494', shoes: '#f2f2f2', ball: '#ff7a45', ballHi: 'rgba(255,255,255,0.45)' });
      g.setTransform(1, 0, 0, 1, 0, 0);
      const vig = g.createRadialGradient(w / 2, h / 2, h * 0.35, w / 2, h / 2, w * 0.62);
      vig.addColorStop(0, 'rgba(0,0,0,0)'); vig.addColorStop(1, 'rgba(0,0,0,0.16)');
      g.fillStyle = vig; g.fillRect(0, 0, w, h);
    }

    // ----- one 40 ms camera frame from four 10 ms renders
    function cameraFrame() {
      const on = new Uint8Array(NBYTES), off = new Uint8Array(NBYTES);
      let count = 0;
      const set = (bits, i) => { const b = 128 >> (i & 7), o = i >> 3; if (!(bits[o] & b)) { bits[o] |= b; count++; } };
      for (let s = 3; s >= 0; s--) {
        const lum = renderLum(sim.t - s * 0.01);
        if (prevLum) {
          for (let i = 0; i < NPIX; i++) {
            const dl = lum[i] - prevLum[i];
            if (dl > 16 && Math.random() < 0.9) set(on, i);
            else if (dl < -16 && Math.random() < 0.9) set(off, i);
          }
        }
        prevLum = lum;
      }
      for (let n = 0; n < 70; n++) set(Math.random() < 0.5 ? on : off, (Math.random() * NPIX) | 0);   // noise
      for (const [x, y] of hot) if (Math.random() < 0.7) set(on, y * W + x);
      const b64 = (bytes) => { let s = ''; for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]); return btoa(s); };
      sim.rate += (count * 260 + rand(-4000, 4000) - sim.rate) * 0.35;
      return { t: sim.t, w: W, h: H, on: b64(on), off: b64(off), rate: Math.max(0, Math.round(sim.rate)) };
    }

    // ----- SNN: spikes on channel 0 rise before the release; EMA readout as on the robot
    function spikeProb(t) {
      let p = 0.012;
      for (const th of sim.throws) {
        const u = t - th.ts;
        if (u > -0.3 && u < 1.2) p += 0.94 * Math.exp(-0.5 * Math.pow((u - 0.44) / 0.12, 2));
      }
      return Math.min(0.97, p);
    }
    function snnBatch() {
      const frames = [];
      for (let i = 0; i < 4; i++) {
        const tf = sim.t - (3 - i) * 0.01;
        const s0 = Math.random() < spikeProb(tf) ? 1 : 0;
        const prev = sim.y;
        sim.y = 0.85 * sim.y + 0.15 * s0;
        sim.k++;
        frames.push({ k: sim.k, s: [s0, 0, 0], y: +sim.y.toFixed(4) });
        if (prev < sim.theta && sim.y >= sim.theta && tf >= sim.refrUntil) {
          sim.refrUntil = tf + 1.0;
          const acted = sim.auto && sim.rosOk, kDet = sim.k, yDet = sim.y, theta = sim.theta;
          sim.det++;
          at(0.005, () => {
            dispatch('detection', { t: tf, k: kDet, y: +yDet.toFixed(4), theta, acted });
            if (acted) at(0.03, () => dash('/reflex/dash'));
            emitStatus();
          });
        }
      }
      sim.latency += (rand(3.7, 4.6) - sim.latency) * 0.2;
      return { frames, theta: sim.theta, latency_ms: +sim.latency.toFixed(2), fps: 100 };
    }

    // ----- go2_dodge
    function robotEvent(e) {
      e.wall_time = Date.now() / 1000;
      sim.last = e;
      dispatch('robot', e);
    }
    function dash(source) {
      if (sim.robot !== 'idle') { robotEvent({ event: 'ignored', source, state: sim.robot }); return; }
      const out = rand(0.49, 0.57), back = out / 0.25 + rand(0.1, 0.4);
      const t0 = sim.t;
      sim.dodges++;
      sim.robot = 'dash';
      sim.awayFn = (t) => {
        const u = t - t0;
        if (u < 0.5) return out * (1 - Math.pow(1 - u / 0.5, 2));
        if (u < 1.5) return out;
        if (u < 1.5 + back) return out * (1 - (u - 1.5) / back);
        return 0;
      };
      robotEvent({ event: 'dash', source, speed: 1.1, duration: 0.5, odometry: true, away_m: 0 });
      at(0.5, () => { sim.robot = 'hold'; robotEvent({ event: 'hold', out_m: +out.toFixed(3), away_m: +out.toFixed(3) }); });
      at(1.5, () => { sim.robot = 'return'; robotEvent({ event: 'return', to_go_m: +out.toFixed(3), odometry: true, away_m: +out.toFixed(3) }); });
      at(1.5 + back, () => {
        sim.robot = 'idle'; sim.awayFn = null;
        robotEvent({ event: 'idle', reason: 'back at the start', out_m: +out.toFixed(3), away_m: 0.012 });
        emitStatus();
      });
      emitStatus();
    }

    // ----- status
    function snapshot() {
      return {
        model: { name: 'v3 early throw-motion', readout: 'ema', theta: sim.theta, channel: 0, mock: false },
        camera: { state: sim.cam, rate: Math.round(sim.rate) },
        fpga: { state: sim.fpga, latency_ms: sim.fpga === 'connected' ? +sim.latency.toFixed(2) : null,
                fps: sim.fpga === 'connected' ? 100 : 0 },
        ros: { available: sim.rosOk, auto: sim.auto },
        robot: { state: sim.robot, last: sim.last },
        color: { enabled: sim.color, available: sim.colorAvail, seq: sim.colorSeq },
        counters: { detections: sim.det, dodges: sim.dodges },
        server_time: Date.now() / 1000,
      };
    }
    function emitStatus() { dispatch('status', snapshot()); sim.nextStatus = sim.t + 1; }

    function throwAt(t, kind) {
      sim.throws.push({ ts: t, kind });
      sim.throws = sim.throws.filter((x) => x.ts > t - 3);
    }

    // ----- 25 Hz tick
    function tick() {
      sim.t += 0.04;
      const t = sim.t;
      sim.queue.sort((a, b) => a.t - b.t);
      while (sim.queue.length && sim.queue[0].t <= t) sim.queue.shift().fn();

      if (t >= sim.nextThrow) {                               // scripted thrower
        sim.throwCount++;
        throwAt(t, sim.throwCount % 4 === 0 ? 'feint' : 'throw');
        sim.nextThrow = t + rand(7, 10);
      }
      if (t >= sim.nextY) {                                   // somebody presses Y on the remote
        sim.auto = !sim.auto;
        emitStatus();
        sim.yPhase++;
        sim.nextY = t + (sim.auto ? rand(40, 55) : rand(12, 16));
      }
      if (t >= sim.nextX) {                                   // ... and X now and then
        const busy = sim.throws.some((x) => t > x.ts - 1 && t < x.ts + 2.5);
        if (sim.robot === 'idle' && !busy) { dash('button X (lowstate)'); sim.nextX = t + rand(28, 40); }
        else sim.nextX = t + 1.5;
      }
      if (t >= sim.nextOutage && sim.fpga === 'connected') {   // a short FPGA dropout
        sim.fpga = 'offline'; sim.y = 0; emitStatus();
        at(2.6, () => { sim.fpga = 'connected'; emitStatus(); });
        sim.nextOutage = t + rand(80, 110);
      }
      sim.away = sim.awayFn ? sim.awayFn(t) : 0;

      if (sim.cam === 'streaming') dispatch('frame', cameraFrame());
      else { prevLum = null; sim.rate = 0; }
      if (sim.fpga === 'connected' && sim.cam === 'streaming') dispatch('snn', snnBatch());
      else sim.y = 0;
      if (sim.color && sim.colorAvail && t >= sim.nextColor) {
        sim.colorSeq++;
        sim.nextColor = t + 0.2;
        dispatch('color', { seq: sim.colorSeq });
      }
      if (t >= sim.nextStatus) emitStatus();
    }

    function onKey(e) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const k = e.key.toLowerCase();
      if (k === 't' || k === 'f') { throwAt(sim.t + 0.05, k === 't' ? 'throw' : 'feint'); sim.nextThrow = sim.t + rand(7, 10); }
      else if (k === 'x') dash('button X (lowstate)');
      else if (k === 'y') { sim.auto = !sim.auto; emitStatus(); }
      else if (k === 'c') { sim.cam = sim.cam === 'streaming' ? 'waiting' : 'streaming'; emitStatus(); }
      else if (k === 'p') { sim.fpga = sim.fpga === 'connected' ? 'offline' : 'connected'; sim.y = 0; emitStatus(); }
      else return;
      e.preventDefault();
    }

    return {
      start() {
        el.demoBadge.hidden = false;
        el.demoBadge.title = 'Synthetic data, no robot. Keys: T throw, F feint, X dodge, Y auto-dodge, C camera, P FPGA';
        el.footKeys.hidden = false;
        document.addEventListener('keydown', onKey);
        emitStatus();
        setInterval(tick, 40);
      },
      handle(path, body) {
        body = body || {};
        if (path.endsWith('api/auto')) sim.auto = !!body.enabled;
        else if (path.endsWith('api/theta') && isNum(body.theta)) sim.theta = clamp(body.theta, 0.05, 0.99);
        else if (path.endsWith('api/color')) {
          sim.color = !!body.enabled;
          sim.colorAvail = false;
          if (sim.color) at(1.1, () => { if (sim.color) { sim.colorAvail = true; emitStatus(); } });
        }
        return new Promise((resolve) => setTimeout(() => resolve(snapshot()), 90));
      },
      colorURL() { renderPhoto(); return photo.toDataURL('image/jpeg', 0.82); },
    };
  }
  const demo = DEMO ? createDemo() : null;

  // ============================================================================================ boot
  configureTheta();
  renderAuto();
  renderRobot();
  renderLog(-1);
  renderPills();
  renderEmpties();
  setInterval(watchdog, 500);
  setInterval(slowTick, 250);
  requestAnimationFrame(loop);
  if (DEMO) demo.start();
  else { fetchState(); openStream(); }
})();

#!/usr/bin/env python3
"""Offline analysis of live sessions recorded with live_view.py --save: run every candidate model
on every recording (float network, neuron state carried across the whole stream, exactly like the
live loop) and store per-frame scores, alarms and motion features.

Per recording, writes <out>/<name>.npz with
  frames (n), t0_us
  ev        events per 10 ms frame
  active    active 4x4 blocks per frame (of 6400, either polarity)
  spread    fraction of the 10x10 coarse grid (32x32 px cells) with >= 3 active blocks:
            ~1 when the whole image moves (camera shake / ego-motion), small for one person
  cx, cy    centroid of the active blocks (0..80)
  spikes_<model>                 float-model output spikes [n, 2]
  score_<model>, alarm_<model>   readout score and alarm frames (bool) per model
  ..._q35sim                     E_moredd_q35 RiverOps Q3.5/Q4.12 PTQ simulation,
                                kept separate from float and board outputs

Also writes summary.json and summary.md with alarm totals, compact alarm clusters,
and the float-vs-Q3.5 simulation comparison for E_moredd_q35.

  .venv/bin/python throwdet/analyze_live.py ~/throwdet_live/*.bin --out ~/throwdet_live/analysis
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from live_readout import ThrowReadout  # noqa: E402
from live_view import EVENT_DTYPE, make_stateful  # noqa: E402
from model import ThrowSNN  # noqa: E402

MODELS = ["D_e10_fold3", "E_moredd_q35", "A_neg", "B_neg_shiftx", "C_all", "E2_moredd_fold3"]
ROOT = Path(__file__).resolve().parent


def load_frames(path, chunk_ev=5_000_000):
    """Binary [F, 2, 80, 80] frames (uint8) + per-frame features, without holding all events as int64."""
    raw = np.memmap(path, np.uint8, "r")
    n = struct.unpack("!I", bytes(raw[:4]))[0] // EVENT_DTYPE.itemsize
    ev = np.memmap(path, EVENT_DTYPE, "r", offset=4, shape=(n,))
    t0 = int(ev["t"][0])
    F = int((int(ev["t"][-1]) - t0) // 10_000) + 1
    frames = np.zeros((F, 2, 80, 80), np.uint8)
    counts = np.zeros(F, np.int64)
    for a in range(0, n, chunk_ev):
        e = ev[a:a + chunk_ev]
        f = (e["t"].astype(np.int64) - t0) // 10_000
        k = (f >= 0) & (f < F) & (e["x"] < 320) & (e["y"] < 320) & (e["p"] < 2)
        f = f[k]
        frames[f, e["p"][k], e["y"][k] // 4, e["x"][k] // 4] = 1
        counts += np.bincount(f, minlength=F)
    return frames, counts, t0


def features(frames):
    act = frames.max(1)                                   # [F, 80, 80]
    active = act.reshape(len(act), -1).sum(1)
    coarse = act.reshape(len(act), 10, 8, 10, 8).sum((2, 4))
    spread = (coarse >= 3).reshape(len(act), -1).mean(1)
    ys, xs = np.arange(80)[None, :, None], np.arange(80)[None, None, :]
    w = np.maximum(active, 1)
    cx = (act * xs).sum((1, 2)) / w
    cy = (act * ys).sum((1, 2)) / w
    return active, spread, cx, cy


def load_model(name, dev):
    d = ROOT / "models" / name
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    net = ThrowSNN(2, arch=ck["arch"], tau=ck["args"]["tau"])
    net.load_state_dict(ck["model"])
    net.eval().to(dev)
    make_stateful(net, torch)
    return net


def run_float_model(name, frames, dev, chunk=500):
    net = load_model(name, dev)
    out = []
    with torch.no_grad():
        for a in range(0, len(frames), chunk):
            x = torch.from_numpy(frames[a:a + chunk])[None].to(dev).float()
            out.append(net(x)[0].flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32, copy=False)


def run_e_q35_sim(frames, dev, chunk=500):
    """Run E's exported Q3.5 weights and Q4.12 potentials in RiverOps PTQ simulation."""
    from riverops.nn import FixedPointFormat, QuantConfig, prepare_ptq

    d = ROOT / "models" / "E_moredd_q35"
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    model = ThrowSNN(2, arch=ck["arch"], tau=ck["args"]["tau"])
    model.load_state_dict(ck["model"])
    model.eval()
    net = model.export_module().eval()
    qcfg = QuantConfig(
        weight_format=FixedPointFormat(total_bits=8, frac_bits=5, signed=True),
        potential_format=FixedPointFormat(total_bits=16, frac_bits=12, signed=True),
        enable_weight_quant=True,
        enable_potential_quant=True,
        qat_mode=False,
    )
    net = prepare_ptq(net, qcfg, fuse=True).eval().to(dev)
    make_stateful(net, torch)
    out = []
    with torch.no_grad():
        for a in range(0, len(frames), chunk):
            x = torch.from_numpy(frames[a:a + chunk])[None].to(dev).float()
            out.append(net(x)[0].flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32, copy=False)


def scores_and_alarms(spikes, readout_path):
    ro = ThrowReadout(str(readout_path))
    score = np.zeros(len(spikes), np.float32)
    alarm = np.zeros(len(spikes), bool)
    for i, (s0, s1) in enumerate(spikes):
        alarm[i] = ro.update(s0, s1)
        score[i] = ro.score
    return score, alarm


def alarm_clusters(alarm, gap_s=2.5):
    """Group nearby alarms into compact time ranges; times are relative to recording start."""
    ix = np.flatnonzero(alarm)
    if not len(ix):
        return []
    groups = [[int(ix[0])]]
    max_gap = int(round(gap_s * 100))
    for frame in ix[1:]:
        frame = int(frame)
        if frame - groups[-1][-1] <= max_gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def fmt_clusters(alarm, limit=6):
    groups = alarm_clusters(alarm)
    parts = []
    for g in groups[:limit]:
        start, end = g[0] / 100, g[-1] / 100
        if len(g) == 1:
            parts.append(f"{start:.2f}s")
        else:
            parts.append(f"{start:.2f}-{end:.2f}s x{len(g)}")
    if len(groups) > limit:
        parts.append(f"+{len(groups) - limit} clusters")
    return "; ".join(parts) if parts else "none"


def count_close_matches(a, b, tolerance_frames=30):
    """Greedy one-to-one alarm matching within 300 ms."""
    left, right = np.flatnonzero(a), np.flatnonzero(b)
    i = j = matches = 0
    while i < len(left) and j < len(right):
        delta = int(left[i]) - int(right[j])
        if abs(delta) <= tolerance_frames:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return int(matches)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_summaries(out_dir, records, model_names, fixed_name):
    fingerprint_names = list(dict.fromkeys([*model_names, "E_moredd", "E_moredd_q35"]))
    fingerprints = {}
    for name in fingerprint_names:
        d = ROOT / "models" / name
        fingerprints[name] = {
            file_name: sha256_file(d / file_name)
            for file_name in ("model.pt", "weights.txt", "readout.json")
        }
    parent_readout = json.loads((ROOT / "models/E_moredd/readout.json").read_text())
    q35_readout = json.loads((ROOT / "models/E_moredd_q35/readout.json").read_text())
    live_keys = ("classes", "frame_ms", "input", "score", "alpha", "threshold", "hold_frames", "refractory_ms")
    e_variant = {
        "live_variant_selected": "E_moredd_q35",
        "live_variant_basis": "repository deployment notes designate E_moredd_q35 for Q3.5 firmware",
        "historical_board_model_verifiable_from_recording": False,
        "float_checkpoint_identical_sha256": fingerprints["E_moredd"]["model.pt"] == fingerprints["E_moredd_q35"]["model.pt"],
        "exported_weights_identical_sha256": fingerprints["E_moredd"]["weights.txt"] == fingerprints["E_moredd_q35"]["weights.txt"],
        "live_readout_settings_identical": all(parent_readout.get(k) == q35_readout.get(k) for k in live_keys),
        "q35_format": q35_readout.get("ptq_check", {}).get("weight_format"),
        "potential_format": q35_readout.get("ptq_check", {}).get("potential_format"),
        "export_sample_decision_agreement": q35_readout.get("ptq_check", {}).get("decision_agreement"),
        "artifact_sha256": fingerprints,
    }
    total_alarm_counts = {}
    for r in records:
        for name, count in r["alarm_counts"].items():
            total_alarm_counts[name] = total_alarm_counts.get(name, 0) + int(count)
    payload = {
        "inference": "float PyTorch model.pt with recurrent neuron potentials carried across 10 ms frames",
        "fixed_point_check": {
            "model": fixed_name,
            "engine": "RiverOps PTQ simulation, Q3.5 weights and Q4.12 potentials",
            "is_fpga_capture": False,
            "note": "Raw event recordings contain no paired FPGA output spikes; this is an offline fixed-point simulation, not measured board parity.",
        },
        "e_variant_provenance": e_variant,
        "alarm_cluster_gap_seconds": 2.5,
        "total_duration_s": float(sum(r["duration_s"] for r in records)),
        "total_event_count": int(sum(r["event_count"] for r in records)),
        "total_alarm_counts": total_alarm_counts,
        "records": records,
        "models": list(model_names),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Offline live-recording comparison",
        "",
        f"Float inference: PyTorch `model.pt`, state carried continuously across 10 ms frames. Models: {', '.join(model_names)}.",
        f"Fixed-point check: `{fixed_name}` in RiverOps PTQ simulation (Q3.5 weights, Q4.12 potentials). This is not an FPGA capture; no paired board outputs are present in the raw recordings.",
        f"E variant: repository deployment notes designate `E_moredd_q35` for the Q3.5 firmware. The event files do not encode the active board model, so the exact historical board load cannot be verified from these captures. `E_moredd_q35` and parent `E_moredd` have identical float-checkpoint SHA256: `{e_variant['float_checkpoint_identical_sha256']}`; the exported integer weights differ (same SHA256: `{e_variant['exported_weights_identical_sha256']}`). The q35 export declares `{e_variant['q35_format']}` weights / `{e_variant['potential_format']}` potentials; live readout settings match parent: `{e_variant['live_readout_settings_identical']}`. The stored `{e_variant['export_sample_decision_agreement']:.3f}` decision agreement is the export's separate 160-sample check, not a result from these recordings.",
        "Alarm clusters group alarms separated by at most 2.5 s. Alarm times are relative to each recording start.",
        "",
        "## Per-recording counts",
        "",
        "| recording | duration | events | " + " | ".join(model_names) + f" | {fixed_name} Q3.5 sim |",
        "|---|---:|---:|" + "---:|" * (len(model_names) + 1),
    ]
    for r in records:
        row = [f"`{r['recording']}`", f"{r['duration_s']:.1f}s", f"{r['event_count']:,}"]
        row.extend(str(r["alarm_counts"][m]) for m in model_names)
        row.append(str(r["alarm_counts"][fixed_name + "_q35sim"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Totals across recordings", "", f"{payload['total_duration_s']:.1f} s across {len(records)} recordings; {payload['total_event_count']:,} events.", "", "| model | alarms |", "|---|---:|"]
    for name, count in total_alarm_counts.items():
        lines.append(f"| `{name}` | {count} |")

    lines += ["", "## Compact alarm clusters", "", "Times in seconds; `xN` is N alarms in one cluster.", ""]
    for r in records:
        lines.append(f"### {r['recording']}")
        lines.append("")
        for m in model_names:
            lines.append(f"- `{m}` ({r['alarm_counts'][m]}): {r['alarm_clusters'][m]}")
        qname = fixed_name + "_q35sim"
        lines.append(f"- `{qname}` ({r['alarm_counts'][qname]}): {r['alarm_clusters'][qname]}")
        lines.append("")

    lines += ["## E float vs Q3.5 simulation", "", "| recording | exact spike-element agreement | identical two-neuron frames | float alarms | Q3.5 sim alarms | alarms matched within 0.3 s |", "|---|---:|---:|---:|---:|---:|"]
    for r in records:
        q = r["e_q35_parity"]
        lines.append(f"| `{r['recording']}` | {q['spike_element_agreement']:.3f} | {q['identical_spike_frames']}/{r['frames']} | {q['float_alarms']} | {q['q35sim_alarms']} | {q['alarms_matched_within_0_3s']} |")
    parity_records = [(r, r["e_q35_parity"]) for r in records if r["e_q35_parity"] is not None]
    parity_frames = sum(r["frames"] for r, _ in parity_records)
    if parity_records and parity_frames:
        exact_spike_agreement = sum(q["spike_element_agreement"] * r["frames"] for r, q in parity_records) / parity_frames
        identical_frames = sum(q["identical_spike_frames"] for _, q in parity_records)
        float_alarms = sum(q["float_alarms"] for _, q in parity_records)
        q35_alarms = sum(q["q35sim_alarms"] for _, q in parity_records)
        close_matches = sum(q["alarms_matched_within_0_3s"] for _, q in parity_records)
        lines.append(f"| **all recordings** | **{exact_spike_agreement:.3f}** | **{identical_frames}/{parity_frames}** | **{float_alarms}** | **{q35_alarms}** | **{close_matches}** |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--skip-e-q35-sim", action="store_true", help="skip the separate RiverOps fixed-point simulation of E")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_names = [m for m in args.models.split(",") if m]
    fixed_name = "E_moredd_q35"
    do_fixed = not args.skip_e_q35_sim and fixed_name in model_names
    records = []
    for path in args.recordings:
        frames, counts, t0 = load_frames(path)
        active, spread, cx, cy = features(frames)
        res = dict(frames=len(frames), t0_us=t0, ev=counts, active=active, spread=spread, cx=cx, cy=cy,
                   recording=Path(path).name, inference_backend="float_pytorch")
        alarms_by_model = {}
        for m in model_names:
            spikes = run_float_model(m, frames, dev)
            res[f"spikes_{m}"] = spikes
            res[f"score_{m}"], res[f"alarm_{m}"] = scores_and_alarms(spikes, ROOT / "models" / m / "readout.json")
            alarms_by_model[m] = res[f"alarm_{m}"]
        if do_fixed:
            fixed_spikes = run_e_q35_sim(frames, dev)
            qname = fixed_name + "_q35sim"
            res[f"spikes_{qname}"] = fixed_spikes
            res[f"score_{qname}"], res[f"alarm_{qname}"] = scores_and_alarms(
                fixed_spikes, ROOT / "models" / fixed_name / "readout.json")
            alarms_by_model[qname] = res[f"alarm_{qname}"]
            float_spikes = res[f"spikes_{fixed_name}"]
            float_alarm = res[f"alarm_{fixed_name}"]
            qalarm = res[f"alarm_{qname}"]
            res["e_q35_spike_element_agreement"] = np.asarray(np.mean(float_spikes == fixed_spikes), np.float64)
            res["e_q35_identical_spike_frames"] = np.asarray(np.sum(np.all(float_spikes == fixed_spikes, axis=1)), np.int64)
            res["e_q35_alarm_matches_0_3s"] = np.asarray(count_close_matches(float_alarm, qalarm), np.int64)
        else:
            qname = None
        counts_by_model = {m: int(alarms_by_model[m].sum()) for m in model_names}
        clusters_by_model = {m: fmt_clusters(alarms_by_model[m]) for m in model_names}
        if qname:
            counts_by_model[qname] = int(alarms_by_model[qname].sum())
            clusters_by_model[qname] = fmt_clusters(alarms_by_model[qname])
            parity = {
                "spike_element_agreement": float(res["e_q35_spike_element_agreement"]),
                "identical_spike_frames": int(res["e_q35_identical_spike_frames"]),
                "float_alarms": counts_by_model[fixed_name],
                "q35sim_alarms": counts_by_model[qname],
                "alarms_matched_within_0_3s": int(res["e_q35_alarm_matches_0_3s"]),
            }
        else:
            parity = None
        records.append({
            "recording": Path(path).name,
            "frames": int(len(frames)),
            "duration_s": float(len(frames) / 100),
            "t0_us": int(t0),
            "event_count": int(counts.sum()),
            "alarm_counts": counts_by_model,
            "alarm_clusters": clusters_by_model,
            "e_q35_parity": parity,
        })
        res["model_names"] = np.asarray(model_names)
        np.savez_compressed(out / (Path(path).stem + ".npz"), **res)
        print(f"{Path(path).name}: {len(frames) / 100:.0f} s, alarms " +
              ", ".join(f"{m.split('_')[0]} {counts_by_model[m]}" for m in model_names) +
              (f", E-Q3.5sim {counts_by_model[qname]}" if qname else ""), flush=True)
        del frames
    save_summaries(out, records, model_names, fixed_name if do_fixed else "")


if __name__ == "__main__":
    main()

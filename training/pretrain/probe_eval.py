"""Few-shot transfer evaluation of pretrained checkpoints (more seeds, two modes).

  head : frozen backbone, new head            (representation quality)
  full : whole net fine-tuned at low LR       (how the throw/fake model will actually be trained)

Each seed draws K recordings/class of the probe classes from the train users, trains on random
windows, evaluates on all recordings of those classes from the held-out users.

usage: python probe_eval.py ../runs/sweep_mid/best.pt [more.pt ...] --seeds 8 --shots 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import speedpatch  # noqa: F401
from model import ThrowSNN, arch_from_config, rate_readout
from train2 import EventWindows, augment, evaluate


def run_probe(ck, tr, te, pc, mode, seed, args, dev, log=None):
    """Returns (window accuracy on held-out users, training-loss curve)."""
    shots, steps, T, bs = args.shots, args.steps, args.T, args.bs
    cfg = ck["config"]
    model = ThrowSNN(cfg["num_classes"], arch=arch_from_config(cfg)).to(dev)
    if mode != "scratch":  # scratch = same shape, random init: control for "does pretraining help?"
        model.load_state_dict(ck["model"])
    model.reset_head(len(pc))
    y_index = torch.full((11,), -1, dtype=torch.long)
    y_index[torch.tensor(pc)] = torch.arange(len(pc))
    label_map = torch.arange(len(pc), device=dev)  # probe classes 7,8,9 are mirror-symmetric
    rng = np.random.default_rng(args.seed_offset + seed)
    torch.manual_seed(args.seed_offset + seed)
    pick = np.concatenate([rng.choice(np.flatnonzero(tr.labels == c), shots, replace=False) for c in pc])
    if mode == "head":
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        groups = [{"params": model.head.parameters(), "lr": args.head_lr}]
    elif mode == "full":
        groups = [{"params": model.backbone.parameters(), "lr": args.backbone_lr},
                  {"params": model.head.parameters(), "lr": args.head_lr}]
    else:
        groups = [{"params": model.parameters(), "lr": args.scratch_lr}]
    opt = torch.optim.Adam(groups)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    model.head.train() if mode == "head" else model.train()
    curve, run = [], []
    for step in range(steps):
        idx = rng.choice(pick, bs)
        items = [(i, int(rng.integers(0, max(tr.n_frames[i] - T, 0) + 1))) for i in idx]
        x = tr.batch(items, T).to(dev).float()
        y = y_index[torch.from_numpy(tr.labels[idx])].to(dev)
        if args.aug:
            x, y = augment(x, y, label_map, args)
        if mode == "head":
            with torch.no_grad():
                feats = model.backbone(x)
            r = rate_readout(model.head(feats).flatten(2))
        else:
            r = rate_readout(model(x))
        loss = F.mse_loss(r, F.one_hot(y, len(pc)).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run.append((loss.item(), (r.argmax(1) == y).float().mean().item()))
        if (step + 1) % (steps // 10) == 0:
            l, a = np.mean(run, 0)
            curve.append({"step": step + 1, "loss": round(float(l), 4), "train_acc": round(float(a), 3)})
            run = []
            if log:
                print(f"  {log} seed{seed} {mode} {curve[-1]}", flush=True)
    return evaluate(model, te, T, y_index, dev)["win_acc"], curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--probe-classes", default="7,8,9")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--modes", default="head,full")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed-offset", type=int, default=1000, help="100 = same clips as train2 quick probe")
    ap.add_argument("--head-lr", type=float, default=3e-3)
    ap.add_argument("--backbone-lr", type=float, default=5e-4)
    ap.add_argument("--scratch-lr", type=float, default=2e-3)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--flip", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=8)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()
    pc = [int(c) for c in args.probe_classes.split(",")]
    dev = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    data = Path(args.data)
    tr, te = EventWindows(data / "dvsg_train.npz", pc), EventWindows(data / "dvsg_test.npz", pc)
    for path in args.ckpts:
        ck = torch.load(path, map_location=dev)
        res = {"ckpt": path, "arch": ck["config"].get("arch_name", "?"), "shots": args.shots}
        for mode in args.modes.split(","):
            outs = [run_probe(ck, tr, te, pc, mode, s, args, dev, log=res["arch"]) for s in range(args.seeds)]
            accs = [o[0] for o in outs]
            res[mode] = {"mean": round(float(np.mean(accs)), 4),
                         "sem": round(float(np.std(accs, ddof=1) / np.sqrt(len(accs))), 4),
                         "seeds": [round(a, 3) for a in accs],
                         "final_train_loss": [o[1][-1]["loss"] for o in outs],
                         "final_train_acc": [o[1][-1]["train_acc"] for o in outs]}
            print(f"PROBE_MODE {res['arch']} {mode} {json.dumps(res[mode])}", flush=True)
        print("PROBE " + json.dumps(res), flush=True)
        res["protocol"] = {k: v for k, v in vars(args).items() if k != "ckpts"}
        Path(path).with_name(f"probe_{args.shots}shot{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

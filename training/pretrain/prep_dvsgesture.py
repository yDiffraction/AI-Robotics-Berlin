"""Convert DVS Gesture (Tonic/Zenodo npy release) to the board's frame format.

Board format (Durance deployment docs (riversdk.dev, not included here), despot): 320x320 GenX320 downsampled x4 -> 80x80,
frames every 10 ms, 2 polarity channels.  DVS128 (128x128) is rescaled to 80x80.

Each recording is stored as sorted unique uint32 codes
    code = ((frame * 2 + p) * 80 + y) * 80 + x
i.e. the set of active (frame, polarity, pixel) cells = binary frames, sparse.

usage: python -I prep_dvsgesture.py <extracted_dir> <out.npz>
  <extracted_dir> contains ibmGestureTrain/ or ibmGestureTest/ user folders with <label>.npy
"""
import sys
from pathlib import Path

import numpy as np

HW, SRC_HW, FRAME_MS = 80, 128, 10.0
CELLS = 2 * HW * HW


def encode(ev: np.ndarray) -> np.ndarray:
    x = (ev[:, 0].astype(np.int64) * HW) // SRC_HW
    y = (ev[:, 1].astype(np.int64) * HW) // SRC_HW
    p = ev[:, 2].astype(np.int64)
    f = (ev[:, 3] // FRAME_MS).astype(np.int64)
    codes = ((f * 2 + p) * HW + y) * HW + x
    assert codes.max() < 2**32
    return np.unique(codes).astype(np.uint32)


def main(src: str, out: str) -> None:
    files = sorted(Path(src).rglob("*.npy"))
    all_codes, offsets, labels, users, n_frames = [], [0], [], [], []
    for i, fp in enumerate(files):
        ev = np.load(fp, allow_pickle=False)
        c = encode(ev)
        all_codes.append(c)
        offsets.append(offsets[-1] + len(c))
        labels.append(int(fp.stem))
        users.append(int(fp.parent.name.split("_")[0].replace("user", "")))
        n_frames.append(int(c.max() // CELLS) + 1)
        if i % 100 == 0:
            print(f"{i}/{len(files)} {fp.parent.name}/{fp.name}: {len(ev)} ev -> {len(c)} cells, "
                  f"{n_frames[-1]} frames", flush=True)
    np.savez(out, codes=np.concatenate(all_codes), offsets=np.array(offsets, np.int64),
             labels=np.array(labels, np.int64), users=np.array(users, np.int64),
             n_frames=np.array(n_frames, np.int64))
    print(f"saved {out}: {len(files)} recordings, {offsets[-1]} active cells, "
          f"classes {np.bincount(labels).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

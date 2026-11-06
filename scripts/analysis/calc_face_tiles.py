"""
Per-frame face feature at the 120 tile centers, from Tarun's SAM3 face masks (annotations/pipeline/).
Each frame: union of included face masks (human + Muppet + Pixar bird; other animal faces excluded), Gaussian blur
(sigma=46 display px, zero outside the frame), sampled at the tile centers. No masks for a clip -> all zeros.

Input:  data/local_cache/face_masks/<clip>/{frames.csv, masks.npz} (copied from annotations/output/current on the server)
Output: data/results/face/<clip>_face_tiles_frames.csv (video, frame_time_s, tile_000..tile_119)

Usage: python scripts/analysis/calc_face_tiles.py [--clips frank_play pixar_birds ...]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tile_pooling import CENTERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_DIR = PROJECT_ROOT / "data" / "local_cache" / "face_masks"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "face"
STIM_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
CLIPS = ["frank_complex", "frank_objects", "frank_play", "pixar_birds", "sesameindia_1", "sesameindia_2",
         "sesameus_1", "sesameus_2", "slow_animals", "slow_hands", "slow_inanimate", "slow_people"]
EXCLUDED = {"penguin face", "horse face", "turtle face", "bird face", "dog face", "sheep face", "chicken faces",
            "goat face", "turkey face", "cow face", "llama face"}  # non-Pixar animal faces (user, 2026-11-05)
VIDEO_W, VIDEO_H = 1280, 720
TO_VIDEO = VIDEO_W / 1920  # display px -> video px
SIGMA_PX = 46.0 * TO_VIDEO
TILE_COLS = [f"tile_{i:03d}" for i in range(120)]


def kernels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Separable Gaussian evaluated at the tile centers: blur-then-sample = Ky @ mask @ Kx.T on the unique rows/cols.
    xy = CENTERS[["center_x", "center_y"]].to_numpy() * TO_VIDEO
    ux, ix = np.unique(xy[:, 0], return_inverse=True)
    uy, iy = np.unique(xy[:, 1], return_inverse=True)
    norm = 1 / (np.sqrt(2 * np.pi) * SIGMA_PX)  # 1D kernel sums to 1 over an unbounded image (zero padding)
    kx = norm * np.exp(-(np.arange(VIDEO_W)[None, :] - ux[:, None]) ** 2 / (2 * SIGMA_PX ** 2))
    ky = norm * np.exp(-(np.arange(VIDEO_H)[None, :] - uy[:, None]) ** 2 / (2 * SIGMA_PX ** 2))
    return kx, ky, ix, iy


def face_keys(clip: str, npz) -> dict:
    # frame -> mask keys (r{prompt_idx}_{prompt}|{frame}) of included face prompts
    d = pd.read_csv(MASK_DIR / clip / "frames.csv", dtype={"prompt_idx": str})
    ok = set(d.loc[(d.category == "face") & ~d.prompt.isin(EXCLUDED), "prompt_idx"])
    out = {}
    for k in npz.files:
        name, fr = k.rsplit("|", 1)
        if name[1:].split("_", 1)[0] in ok:
            out.setdefault(int(fr), []).append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kx, ky, ix, iy = kernels()
    for clip in args.clips:
        cap = cv2.VideoCapture(str(STIM_DIR / f"{clip}_stripped.mp4"))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vals = np.zeros((n, 120))
        npz_path = MASK_DIR / clip / "masks.npz"
        n_face = 0
        if npz_path.exists():  # frank_objects has no SAM3 output (no faces)
            npz = np.load(npz_path)
            for fr, keys in face_keys(clip, npz).items():
                m = np.zeros((VIDEO_H, VIDEO_W), bool)
                for k in keys:
                    m |= npz[k]
                vals[fr] = (ky @ m.astype(float) @ kx.T)[iy, ix]
                n_face += 1
        df = pd.DataFrame(vals, columns=TILE_COLS)
        df.insert(0, "frame_time_s", np.arange(n) / fps)
        df.insert(0, "video", clip)
        df.to_csv(OUT_DIR / f"{clip}_face_tiles_frames.csv", index=False)
        print(f"{clip}: {n} frames, {n_face} with faces, max {vals.max():.3f}")


if __name__ == "__main__":
    main()

"""
Compare two ways to turn SAM3 face masks into smoothed face maps, on 20 assorted frames:
(A) each 240px-diameter tile circle gets 1 if any face pixel falls inside it, else 0; that value is placed at the
    tile center as a Gaussian bump (sigma from --sigma, display px), bumps summed, divided by the all-ones peak;
(B) the binary pixel mask itself, Gaussian-smoothed with the same sigma.
Faces = human + Muppet + Pixar bird faces; other animal faces excluded.

Inputs: data/local_cache/face_masks/<clip>/{frames.csv, masks.npz} (copied from annotations/output/current on the server)
Output: data/local_cache/face_masks/smoothing_comparison/sigma<N>/ (frame PNGs + maps.csv + tiles.csv + frames.csv),
plotted by plot_face_map_smoothing.R

Usage: python scripts/decision_making/compare_face_map_smoothing.py [--sigma 46]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
from tile_pooling import CENTERS  # noqa: E402

MASK_DIR = PROJECT_ROOT / "data" / "local_cache" / "face_masks"
STIM_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
EXCLUDED = {"penguin face", "horse face", "turtle face", "bird face", "dog face", "sheep face", "chicken faces",
            "goat face", "turkey face", "cow face", "llama face"}  # non-Pixar animal faces (user, 2026-11-05)
VIDEO_W, VIDEO_H = 1280, 720
TO_VIDEO = VIDEO_W / 1920  # display px -> video px
STEP = 4  # map output resolution (video px per cell)
TILE_XY = CENTERS[["center_x", "center_y"]].to_numpy() * TO_VIDEO
TILE_RADIUS = 120.0 * TO_VIDEO  # 240px-diameter circles on the 1920 canvas

# (clip, label kind, rule[, n frames=2]), each frame from a different shot; rules use per-frame stats of included faces.
PICKS = [
    ("frank_play", "human, one big face", lambda f: (f.n == 1) & (f.max_area > 0.06)),
    ("frank_complex", "human, one mid-size face", lambda f: (f.n == 1) & (f.max_area < 0.03)),
    ("frank_complex", "human, a few faces", lambda f: (f.n >= 2) & (f.n <= 3), 1),
    ("frank_complex", "human, baby", lambda f: f.has_baby),
    ("sesameindia_1", "human, many small faces", lambda f: (f.n >= 6) & (f.max_area < 0.003)),
    ("slow_people", "human, tiny faces", lambda f: f.n >= 2),
    ("sesameus_1", "Muppet, large", lambda f: (f.n_muppet >= 1) & (f.max_area > 0.04)),
    ("sesameus_2", "Muppet + humans", lambda f: (f.n_muppet >= 1) & (f.n >= 4)),
    ("sesameindia_2", "Muppet + human, mixed sizes", lambda f: (f.n_muppet >= 1) & (f.n >= 2) & (f.n <= 4)),
    ("pixar_birds", "bird, one big face", lambda f: (f.n == 1) & (f.max_area > 0.05)),
    ("pixar_birds", "birds, many small faces", lambda f: (f.n >= 6) & (f.max_area < 0.01)),
]


def frame_stats(clip: str) -> pd.DataFrame:
    d = pd.read_csv(MASK_DIR / clip / "frames.csv")
    d = d[(d.category == "face") & ~d.prompt.isin(EXCLUDED)]
    d = d.assign(muppet=d.prompt.str.lower().str.contains("muppet"), baby=d.prompt.eq("baby face"))
    return d.groupby("frame_idx_global").agg(n=("obj_id", "size"), max_area=("mask_area_frac", "max"),
                                             n_muppet=("muppet", "sum"), has_baby=("baby", "any"),
                                             shot=("shot_idx", "first"))


def pick_frames() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for clip, kind, rule, *n in PICKS:
        f = frame_stats(clip)
        f = f[rule(f)]
        shots = rng.choice(f.shot.unique(), size=min(n[0] if n else 2, f.shot.nunique()), replace=False)
        for s in shots:  # two different shots, frame nearest the middle of the qualifying run
            g = f[f.shot == s].index.to_numpy()
            rows.append(dict(clip=clip, kind=kind, frame=int(g[len(g) // 2])))
    return pd.DataFrame(rows)


def union_mask(npz, keys_by_frame: dict, frame: int) -> np.ndarray:
    m = np.zeros((VIDEO_H, VIDEO_W), bool)
    for k in keys_by_frame.get(frame, []):
        m |= npz[k]
    return m


def included_keys(npz) -> dict:
    # key format r{prompt_idx}_{prompt}|{frame}; only face prompts that aren't excluded animals
    out = {}
    for k in npz.files:
        name, fr = k.rsplit("|", 1)
        prompt = name.split("_", 1)[1]
        if "face" in prompt.lower() and prompt not in EXCLUDED:
            out.setdefault(int(fr), []).append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=46.0, help="display px (46 = gaze/features)")
    args = ap.parse_args()
    SIGMA_PX = args.sigma * TO_VIDEO  # video px
    OUT_DIR = MASK_DIR / "smoothing_comparison" / f"sigma{args.sigma:g}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = pick_frames()
    ys, xs = np.mgrid[STEP // 2:VIDEO_H:STEP, STEP // 2:VIDEO_W:STEP]
    d2 = (xs.ravel()[:, None] - TILE_XY[None, :, 0]) ** 2 + (ys.ravel()[:, None] - TILE_XY[None, :, 1]) ** 2
    w = np.exp(-d2 / (2 * SIGMA_PX ** 2))
    w /= (w @ np.ones(120)).max()  # 1 = every tile around a point contains a face (bump sum peaks ~3.7 at sigma=92)
    maps, tiles = [], []
    for clip, g in picks.groupby("clip", sort=False):
        npz = np.load(MASK_DIR / clip / "masks.npz")
        keys = included_keys(npz)
        cap = cv2.VideoCapture(str(STIM_DIR / f"{clip}_stripped.mp4"))
        for p in g.itertuples():
            fid = f"{clip}_{p.frame}"
            cap.set(cv2.CAP_PROP_POS_FRAMES, p.frame)
            ok, img = cap.read()
            assert ok, fid
            cv2.imwrite(str(OUT_DIR / f"{fid}.png"), cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
            m = union_mask(npz, keys, p.frame)
            ti = np.clip(np.round(TILE_XY).astype(int), 0, [VIDEO_W - 1, VIDEO_H - 1])
            dist = distance_transform_edt(~m)  # px to the nearest face pixel
            a_tiles = (dist[ti[:, 1], ti[:, 0]] <= TILE_RADIUS).astype(float) if m.any() else np.zeros(120)
            b_full = gaussian_filter(m.astype(float), SIGMA_PX, mode="constant")
            maps.append(pd.DataFrame(dict(frame_id=fid, x=xs.ravel(), y=ys.ravel(),
                                          mask=m[ys.ravel(), xs.ravel()].astype(int),
                                          a=w @ a_tiles, b=b_full[ys.ravel(), xs.ravel()])))
            tiles.append(pd.DataFrame(dict(frame_id=fid, tile=np.arange(120), x=TILE_XY[:, 0], y=TILE_XY[:, 1],
                                           a_center=a_tiles, b_at_center=b_full[ti[:, 1], ti[:, 0]])))
    picks["frame_id"] = picks["clip"] + "_" + picks["frame"].astype(str)
    picks.to_csv(OUT_DIR / "frames.csv", index=False)
    pd.concat(maps).to_csv(OUT_DIR / "maps.csv", index=False)
    pd.concat(tiles).to_csv(OUT_DIR / "tiles.csv", index=False)
    print(picks.to_string())


if __name__ == "__main__":
    main()

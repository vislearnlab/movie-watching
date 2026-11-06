"""
Spatial resolution of the four IG features on single frames: 10 frames per block, each feature's 120 tile values as
the IG models see them. Motion/saliency/meaning: per-frame patch values, point-smoothed at the tile centers (sigma=46,
load_features); meaning = average of fine and coarse (sigma=92), each min-max scaled within the frame.
Face (current): blurred mask sampled at centers (calc_face_tiles.py). Face (box option): mean of the binary mask over
each 240px square, then point-smoothed like motion/saliency.

Output: data/local_cache/feature_resolution/ (frame PNGs, frames.csv, tiles.csv), plotted by plot_feature_resolution.R
Usage:  python scripts/decision_making/compare_feature_resolution.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis" / "information_gain"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
import config as C  # noqa: E402
from calc_face_tiles import MASK_DIR, VIDEO_H, VIDEO_W, face_keys  # noqa: E402
from load_features import TILE_XY, frame_feature, load_meaning_raw, meaning_frames, point_smoothing_matrix  # noqa: E402
from tile_pooling import tile_box_means  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "local_cache" / "feature_resolution"
BLOCK_CLIPS = {"frank": ["frank_complex", "frank_objects", "frank_play"], "pixar": ["pixar_birds"],
               "sesame": ["sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2"],
               "slow": ["slow_animals", "slow_hands", "slow_inanimate", "slow_people"]}
N_PER_BLOCK = 10


def nearest(frames: tuple, t: float) -> np.ndarray:
    times, vals = frames
    return vals[np.abs(times - t).argmin()]


def mm(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w_fine = point_smoothing_matrix(TILE_XY, C.SIGMA_FEATURE_FINE)
    picks, tiles = [], []
    for block, clips in BLOCK_CLIPS.items():
        # spread N_PER_BLOCK frames over the block's clips, evenly in time, skipping the first and last second
        per_clip = np.array_split(np.arange(N_PER_BLOCK), len(clips))
        for clip, idx in zip(clips, per_clip):
            times_m, fine, coarse, c_xy = meaning_frames({clip: load_meaning_raw(clip)})[clip]
            w_coarse = point_smoothing_matrix(c_xy, C.SIGMA_FEATURE_COARSE)
            sal, mot, face = (frame_feature(clip, f) for f in ("saliency", "motion", "face"))
            npz = np.load(MASK_DIR / clip / "masks.npz") if (MASK_DIR / clip / "masks.npz").exists() else None
            keys = face_keys(clip, npz) if npz is not None else {}
            cap = cv2.VideoCapture(str(PROJECT_ROOT / "stimuli" / "main_blocks" / f"{clip}_stripped.mp4"))
            fps = cap.get(cv2.CAP_PROP_FPS)
            for t_target in np.linspace(1, C.DURATIONS[clip] - 1, len(idx) + 2)[1:-1]:
                k = np.abs(times_m - t_target).argmin()
                t = times_m[k]  # a meaning frame, so all four features exist at t
                fr = int(round(t * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, img = cap.read()
                assert ok
                fid = f"{clip}_{fr}"
                cv2.imwrite(str(OUT_DIR / f"{fid}.png"), cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA))
                m = np.zeros((VIDEO_H, VIDEO_W), bool)
                for key in keys.get(fr, []):
                    m |= npz[key]
                vals = dict(motion=w_fine @ nearest(mot, t), saliency=w_fine @ nearest(sal, t),
                            meaning=(mm(w_fine @ fine[k]) + mm(w_coarse @ coarse[k])) / 2,
                            face_current=nearest(face, t), face_box=w_fine @ tile_box_means(m.astype(float)))
                for f, v in vals.items():
                    tiles.append(pd.DataFrame(dict(frame_id=fid, feature=f, tile=np.arange(120),
                                                   x=TILE_XY[:, 0], y=TILE_XY[:, 1], value=v)))
                picks.append(dict(block=block, clip=clip, frame=fr, time_s=round(t, 3), frame_id=fid))
    pd.DataFrame(picks).to_csv(OUT_DIR / "frames.csv", index=False)
    pd.concat(tiles).to_csv(OUT_DIR / "tiles.csv", index=False)
    print(pd.DataFrame(picks).groupby("block").size())


if __name__ == "__main__":
    main()

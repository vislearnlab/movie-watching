"""
Per-segment 120-tile feature vectors. Patch values (240px box means for motion/saliency, DeepMeaning
ratings for meaning, face-box coverage for face, from ~2 Hz review frames) are averaged over the segment's frames, plotted at their patch centers,
Gaussian-smoothed (normalized weighted average; fine sigma=46, coarse meaning sigma=92), and sampled
at the 120 tile centers. Cached as a long table: smoothed unscaled columns (saliency_raw, motion_raw,
face_raw, meaning_fine, meaning_coarse) plus the scaled saliency, motion, meaning, face used by the models.

Usage: python scripts/analysis/information_gain/load_features.py
"""

import sys

import numpy as np
import pandas as pd

import config as C
from segments import segment_table

sys.path.insert(0, str(C.PROJECT_ROOT / "scripts" / "analysis"))
from tile_pooling import CENTERS  # noqa: E402

TILE_COLS = [f"tile_{i:03d}" for i in range(120)]
TILE_XY = CENTERS[["center_x", "center_y"]].to_numpy()  # display px


def point_smoothing_matrix(points_xy: np.ndarray, sigma: float) -> np.ndarray:
    """(120, n_points): value at each tile center = Gaussian-weighted average of values plotted at points_xy."""
    d2 = ((TILE_XY[:, None, :] - points_xy[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma ** 2))
    return w / w.sum(axis=1, keepdims=True)


def frame_feature(clip: str, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = C.FEATURE_TILE_DIR / name / f"{clip}_{name}_tiles_frames.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.frame_time_s.to_numpy(), df[TILE_COLS].to_numpy()


def segment_mean(frames: tuple | None, s: float, e: float, needs_prev: bool = False) -> np.ndarray:
    # Mean of frames inside [s, e); motion (needs_prev) also drops flow frames whose previous frame is before s (cross-cut).
    if frames is None:
        return None
    t, m = frames
    sel = (t >= s) & (t < e)
    if needs_prev:
        sel &= (t - np.median(np.diff(t))) >= s - 1e-9
    return m[sel].mean(0) if sel.any() else np.full(m.shape[1], np.nan)


def load_meaning_raw(clip: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    fine = C.MEANING_DIR / f"deepmeaning_{clip}_fine{C.MEANING_FINE_PATCH}" / f"{clip}.csv"
    coarse = C.MEANING_DIR / f"deepmeaning_{clip}_coarse{C.MEANING_COARSE_PATCH}" / f"{clip}.csv"
    if not (fine.exists() and coarse.exists()):
        return None
    return pd.read_csv(fine), pd.read_csv(coarse)


def meaning_frames(raw: dict) -> dict:
    """clip -> (frame_times, fine (n_frames, 120), coarse (n_frames, 28), coarse centers in display px), raw ratings."""
    out = {}
    for clip, (fine, coarse) in raw.items():
        frames = sorted(set(fine.frame_index) & set(coarse.frame_index))
        f = fine[fine.frame_index.isin(frames)].sort_values(["frame_index", "patch_index"])
        c = coarse[coarse.frame_index.isin(frames)].sort_values(["frame_index", "patch_index"])
        n_c = c.patch_index.nunique()
        assert f.patch_index.nunique() == 120 and len(f) == 120 * len(frames) and len(c) == n_c * len(frames)
        to_disp = 1 / C.CANVAS_TO_SOURCE
        assert np.allclose(f.iloc[:120][["patch_center_x", "patch_center_y"]].to_numpy() * to_disp, TILE_XY)
        c_xy = c.iloc[:n_c][["patch_center_x", "patch_center_y"]].to_numpy() * to_disp
        times = f.groupby("frame_index").frame_time_sec.first().to_numpy()
        out[clip] = (times, f.predicted_meaning.to_numpy().reshape(len(frames), 120),
                     c.predicted_meaning.to_numpy().reshape(len(frames), n_c), c_xy)
    return out


def build(segment_set: str) -> pd.DataFrame:
    segs = segment_table(segment_set)
    raw_meaning = {c: r for c in C.CLIPS if (r := load_meaning_raw(c)) is not None}
    meaning = meaning_frames(raw_meaning)
    w_fine = point_smoothing_matrix(TILE_XY, C.SIGMA_FEATURE_FINE)
    nan120 = np.full(120, np.nan)
    rows = []
    for clip in C.CLIPS:
        sal, mot, face = frame_feature(clip, "saliency"), frame_feature(clip, "motion"), frame_feature(clip, "face")
        m = meaning.get(clip)
        w_coarse = point_smoothing_matrix(m[3], C.SIGMA_FEATURE_COARSE) if m else None
        for seg in segs[segs.video == clip].itertuples():
            s, e = seg.seg_start_s, seg.seg_end_s
            # Segment-mean patch values -> plotted at centers, smoothed, sampled (linear, so order doesn't matter).
            sv, mv, fv = segment_mean(sal, s, e), segment_mean(mot, s, e, needs_prev=True), segment_mean(face, s, e)
            vec = dict(saliency=w_fine @ sv if sv is not None else nan120,
                       motion=w_fine @ mv if mv is not None else nan120,
                       face=w_fine @ fv if fv is not None else nan120,
                       meaning_fine=w_fine @ segment_mean((m[0], m[1]), s, e) if m else nan120,
                       meaning_coarse=w_coarse @ segment_mean((m[0], m[2]), s, e) if m else nan120)
            rows.append(pd.DataFrame(dict(segment_set=segment_set, video=clip, seg_idx=seg.seg_idx,
                                          tile=np.arange(120), **vec)))
    return scale_features(pd.concat(rows, ignore_index=True).rename(columns={"saliency": "saliency_raw", "motion": "motion_raw", "face": "face_raw"}))


def zscore_by_segment(df: pd.DataFrame, col: str) -> pd.Series:
    # Exactly flat segments get 0 everywhere (feature says nothing -> model falls back to the floor).
    g = df.groupby(["video", "seg_idx"])[col]
    sd = g.transform("std")
    return ((df[col] - g.transform("mean")) / sd.where(sd > 1e-12)).fillna(0).where(df[col].notna())


def minmax(x: pd.Series) -> pd.Series:
    return (x - x.min()) / (x.max() - x.min())


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    if C.FEATURE_SCALING == "segment_z":
        # Same operation for every feature; meaning's two scales are each z-scored before averaging.
        df["saliency"] = zscore_by_segment(df, "saliency_raw")
        df["motion"] = zscore_by_segment(df, "motion_raw")
        df["face"] = zscore_by_segment(df, "face_raw")
        df["meaning_avg"] = (zscore_by_segment(df, "meaning_fine") + zscore_by_segment(df, "meaning_coarse")) / 2
        df["meaning"] = zscore_by_segment(df, "meaning_avg")
    else:  # "corpus" (Kümmerer et al. 2015): min-max rescale to [0, 1] jointly across all segments, not per segment
        df["meaning_avg"] = (minmax(df.meaning_fine) + minmax(df.meaning_coarse)) / 2
        df["saliency"], df["motion"], df["meaning"] = minmax(df.saliency_raw), minmax(df.motion_raw), minmax(df.meaning_avg)
        df["face"] = minmax(df.face_raw)
    return df.drop(columns=["meaning_avg"], errors="ignore")


def main():
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for segment_set in C.SEGMENT_SETS:
        df = build(segment_set)
        out = C.CACHE_DIR / f"features_{segment_set}.csv"
        df.to_csv(out, index=False)
        n_seg = df.groupby(["video", "seg_idx"]).ngroups
        avail = df.groupby("video")[C.FEATURES].apply(lambda d: d.notna().all()).astype(int)
        print(f"{segment_set}: {n_seg} segments -> {out}\nfeature available per clip:\n{avail}")


if __name__ == "__main__":
    main()

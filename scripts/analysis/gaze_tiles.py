"""
Shared gaze loading and closed-form tile values, used by every tile-based gaze analysis (ISC tile-peak,
entropy, within/between, information gain, montages). Moved out of calc_concordance_group.py (now archived).

Tile values are a duration-weighted Gaussian evaluated directly at tile centers -- identical to blurring the
fixation map and sampling it (figures/checks/closed_form_vs_raster_blur_verification.png).
Reads fixation data from a LOCAL COPY only -- never the live server mount.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DISPLAY_W, DISPLAY_H = 1920, 1080
MIN_MOVIES = 2  # participant-level exclusion: fewer usable movies than this -> dropped (METHODS_DECISIONS 2026-11-05)


def eligible_pids(fixation_dir: Path, param_code: str, min_movies: int = MIN_MOVIES) -> set[str]:
    """Participants with fixation files for >= min_movies movies; I2MC writes none for excluded blocks/movies."""
    counts = pd.Series([p.name.split("_")[0] for p in fixation_dir.glob(f"*_{param_code}.csv")], dtype=str).value_counts()
    return set(counts[counts >= min_movies].index)


def load_video_fixations(fixation_dir: Path, video: str, param_code: str) -> pd.DataFrame:
    """Eligible participants' fixation rows for one video, from a LOCAL COPY only."""
    fix_paths = sorted(fixation_dir.glob(f"*_{video}_{param_code}.csv"))
    assert fix_paths, f"No fixation CSVs found in {fixation_dir} for video={video!r}, code={param_code!r}."

    dfs = [pd.read_csv(p) for p in fix_paths]
    fix_df = pd.concat(dfs, ignore_index=True)
    fix_df = fix_df[fix_df["pid"].isin(eligible_pids(fixation_dir, param_code))].copy()
    print(f"Loaded {len(fix_df)} fixations from {fix_df['pid'].nunique()} participants for {video}.")
    return fix_df


def build_participant_arrays(fix_df: pd.DataFrame) -> dict[str, tuple[np.ndarray, ...]]:
    """{pid: (startT, endT, img_x, img_y)} as numpy arrays; y flipped from Tobii bottom-left to image top-left origin."""
    arrays = {}
    for pid, g in fix_df.groupby("pid"):
        arrays[pid] = (
            g["startT"].to_numpy(),
            g["endT"].to_numpy(),
            g["xpos"].to_numpy(),
            DISPLAY_H - g["ypos"].to_numpy(),
        )
    return arrays


def closed_form_tile_values(startT: np.ndarray, endT: np.ndarray, img_x: np.ndarray, img_y: np.ndarray,
                             window_start_ms: float, window_end_ms: float, sigma_px: float,
                             centers_x: np.ndarray, centers_y: np.ndarray) -> np.ndarray:
    """Duration-weighted Gaussian density at each tile center for one participant's fixations in one window."""
    overlap_ms = np.minimum(endT, window_end_ms) - np.maximum(startT, window_start_ms)
    mask = overlap_ms > 0
    if not mask.any():
        return np.zeros(len(centers_x))

    x, y, w = img_x[mask], img_y[mask], overlap_ms[mask]
    dist_sq = (x[:, None] - centers_x[None, :]) ** 2 + (y[:, None] - centers_y[None, :]) ** 2
    gauss = np.exp(-dist_sq / (2 * sigma_px ** 2)) / (2 * np.pi * sigma_px ** 2)
    return (w[:, None] * gauss).sum(axis=0)

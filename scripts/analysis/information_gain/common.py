"""Shared helpers: tile columns, gaze loading, log-likelihood, and the participant-equal averaging rule."""

import numpy as np
import pandas as pd

import config as C

TILE_COLS = [f"tile_{i:03d}" for i in range(120)]
UNIFORM = np.full(120, 1 / 120)


def excluded_trials_path(segment_set: str):
    return C.RESULTS_DIR / f"excluded_trials_{segment_set}.csv"


def load_gaze(segment_set: str, apply_exclusions: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(C.CACHE_DIR / f"gaze_{segment_set}.parquet")
    df = df[df.n_fix >= C.MIN_FIXATIONS]
    if apply_exclusions and C.EXCLUDE_NEGATIVE_CEILING:
        path = excluded_trials_path(segment_set)
        assert path.exists(), f"{path} missing: run exclusions.py first"
        ex = pd.read_csv(path)
        df = df[~df.set_index(["video", "participant"]).index.isin(ex.set_index(["video", "participant"]).index)]
    return df.reset_index(drop=True)


def log_lik(g: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Average log2-likelihood per unit looking time: sum_t g_t log2 q_t, row-wise."""
    assert np.all(q > 0) and np.allclose(q.sum(axis=-1), 1)
    return (g * np.log2(q)).sum(axis=-1)


def group_average(df: pd.DataFrame, col: str) -> float:
    # Within participant across segments, then across participants (equal weight per cell, then per participant).
    return df.groupby("participant")[col].mean().mean()


def clip_sums(gaze: pd.DataFrame) -> tuple[dict, dict]:
    """Per-clip sum of normalized gaze vectors and cell count, for fast leave-clips-out pooling."""
    sums = {c: g[TILE_COLS].to_numpy().sum(0) for c, g in gaze.groupby("video")}
    counts = {c: len(g) for c, g in gaze.groupby("video")}
    return sums, counts


def pooled_excluding(sums: dict, counts: dict, exclude: set) -> np.ndarray:
    keep = [c for c in sums if c not in exclude]
    v = sum(sums[c] for c in keep) / sum(counts[c] for c in keep)
    return v / v.sum()

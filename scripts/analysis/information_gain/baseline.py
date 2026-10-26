"""
Floor (baseline): where people look regardless of content. For each held-out clip c,
floor_c = (1 - eps_c) * pooled gaze of all other clips + eps_c * uniform, with eps_c chosen by
inner leave-one-clip-out on the other clips only (clip c never informs its own floor).

Output: floor_<segment_set>.parquet (video, eps, tile_000..tile_119) (cache) + floor_<set>.csv (results, for R)

Usage: python scripts/analysis/information_gain/baseline.py
"""

import numpy as np
import pandas as pd

import config as C
from common import TILE_COLS, UNIFORM, clip_sums, group_average, load_gaze, log_lik, pooled_excluding

EPS_GRID = np.concatenate([[0.0], np.logspace(-6, -0.3, 40)])


def choose_eps(gaze: pd.DataFrame, sums: dict, counts: dict, train: list) -> float:
    """eps maximizing group-average held-out log-likelihood, inner leave-one-clip-out within train."""
    sub = gaze[gaze.video.isin(train)]
    g = sub[TILE_COLS].to_numpy()
    # Each training cell scored against the pooled floor of the other training clips.
    pooled = np.stack([pooled_excluding(sums, counts, set(sums) - set(train) | {v}) for v in sub.video])
    best_eps, best_ll = None, -np.inf
    for eps in EPS_GRID:
        q = (1 - eps) * pooled + eps * UNIFORM
        if np.any(q <= 0):
            continue
        ll = group_average(sub.assign(ll=log_lik(g, q)), "ll")
        if ll > best_ll:
            best_eps, best_ll = eps, ll
    return best_eps


def build(segment_set: str, gaze: pd.DataFrame | None = None) -> pd.DataFrame:
    gaze = load_gaze(segment_set) if gaze is None else gaze
    sums, counts = clip_sums(gaze)
    rows = []
    for clip in C.CLIPS:
        train = [c for c in C.CLIPS if c != clip]
        eps = choose_eps(gaze, sums, counts, train)
        floor = (1 - eps) * pooled_excluding(sums, counts, {clip}) + eps * UNIFORM
        assert np.all(floor > 0) and np.isclose(floor.sum(), 1)
        rows.append(dict(segment_set=segment_set, video=clip, eps=eps, **dict(zip(TILE_COLS, floor))))
    return pd.DataFrame(rows)


def main():
    for segment_set in C.SEGMENT_SETS:
        df = build(segment_set)
        out = C.CACHE_DIR / f"floor_{segment_set}.parquet"
        df.to_parquet(out, index=False)
        print(f"{segment_set}: eps per held-out clip {df.eps.min():.1e}-{df.eps.max():.1e} -> {out}")
        df.to_csv(C.RESULTS_DIR / f"floor_{segment_set}.csv", index=False)  # for plot_information_gain_checks.R


if __name__ == "__main__":
    main()

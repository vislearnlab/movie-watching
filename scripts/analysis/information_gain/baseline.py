"""
Floor (baseline): where people look regardless of content. For each held-out clip c,
floor_c = (1 - eps_c) * pooled gaze of all other clips + eps_c * uniform, with eps_c chosen by
inner leave-one-clip-out on the other clips only (clip c never informs its own floor).

Output: floor_<segment_set>.parquet (video, eps, tile_000..tile_119) + figures/checks/information_gain_floor_<set>.png

Usage: python scripts/analysis/information_gain/baseline.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def plot(df: pd.DataFrame, segment_set: str):
    fig, axes = plt.subplots(3, 4, figsize=(14, 6.4))
    vmax = df[TILE_COLS].to_numpy().max()
    for ax, row in zip(axes.flat, df.itertuples()):
        im = ax.imshow(np.array([getattr(row, t) for t in TILE_COLS]).reshape(8, 15), cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"held out: {row.video}\neps={row.eps:.1e}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, label="floor probability per tile")
    fig.suptitle(f"Floor per held-out clip (pooled gaze from the other 11 clips), 15x8 tiles, {segment_set}")
    out = C.CHECK_FIG_DIR / f"information_gain_floor_{segment_set}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    for segment_set in C.SEGMENT_SETS:
        df = build(segment_set)
        out = C.CACHE_DIR / f"floor_{segment_set}.parquet"
        df.to_parquet(out, index=False)
        print(f"{segment_set}: eps per held-out clip {df.eps.min():.1e}-{df.eps.max():.1e} -> {out}")
        plot(df, segment_set)


if __name__ == "__main__":
    main()

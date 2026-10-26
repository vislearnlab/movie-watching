"""
Per-quadrant montages of 12 examples each from the unchopped segments (slow included as its real ~15s segments).
Quadrant labels come from all_clips_by_segment.csv (pooled 40/60 split, as in plot_data.quadrants). Examples are
ranked by how far into the corner they sit (smaller of the two percentile distances past the thresholds) and
picked with a per-clip cap that is raised only if needed to reach 12. Selective instead ranks by lowest within
and highlights the lowest-within participant with >= 3 fixations in the segment. Panels reuse montage_best_examples.py's
drawing: frame at the segment midpoint, group fixations in blue, the most quadrant-typical participant highlighted.

Output: figures/within_between/montages_unchopped/<quadrant>_12_examples.png

Usage: python scripts/analysis/montage_quadrant_examples.py [--alt coordinated | --two-target]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis" / "information_gain"))

from calc_within_between_cosine import WINDOW_S, within_between_for_windows  # noqa: E402
from montage_best_examples import (  # noqa: E402
    ALL_CLIPS, QUADRANT_MARKER_COLORS, QUADRANTS, STIMULI_DIR, build_clip_vectors, draw_example_panel,
    get_video_resolution, quadrant_score,
)
from plot_data import QUADRANT_Q, WB_PATH, quadrants  # noqa: E402

N_EXAMPLES, N_COLS = 12, 4
MIN_FIX_SELECTIVE = 3
HIGHLIGHT = QUADRANT_MARKER_COLORS | {"concentrated": "#C62828"}  # red reads better than green on the blue group dots
OUT_DIR = PROJECT_ROOT / "figures" / "within_between" / "montages_unchopped"


def n_fixations(arr: tuple, s: float, e: float) -> int:
    startT, endT, _, _ = arr
    return int(((np.minimum(endT, e * 1000) - np.maximum(startT, s * 1000)) > 0).sum())


def corner_scores(w: pd.DataFrame) -> pd.Series:
    # Percentile distance past each axis threshold, toward the quadrant's corner; score = the smaller of the two.
    pw = w.within_person_cosine_dist.rank(pct=True)
    pb = w.between_person_cosine_dist.rank(pct=True)
    lo, hi = QUADRANT_Q, 1 - QUADRANT_Q
    d_w = np.where(w.quadrant.isin(["coordinated", "scattered"]), pw - hi, lo - pw)  # NaN quadrants scored, then dropped
    d_b = np.where(w.quadrant.isin(["selective", "scattered"]), pb - hi, lo - pb)
    return pd.Series(np.minimum(d_w, d_b), index=w.index)


def pick_diverse(c: pd.DataFrame, n: int) -> pd.DataFrame:
    c = c.sort_values("corner", ascending=False)
    for cap in range(1, n + 1):
        picked = c.groupby("video").head(cap)
        if len(picked) >= n:
            return picked.head(n) if cap == 1 else c[c.index.isin(picked.index)].head(n)
    return c


def kmeans2(X: np.ndarray, w: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    ax = int(np.ptp(X[:, 1]) > np.ptp(X[:, 0]))
    c = X[[np.argmin(X[:, ax]), np.argmax(X[:, ax])]].astype(float)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
        for k in (0, 1):
            if (lab == k).any():
                c[k] = np.average(X[lab == k], axis=0, weights=w[lab == k])
    return lab, c


def two_target_score(arrays: dict, s: float, e: float) -> float:
    # Group fixations split into 2 duration-weighted clusters: separation/spread (capped at 4) x balance x share of
    # people who visit both clusters.
    X, W, P = [], [], []
    for pid, (st, en, x, y) in arrays.items():
        ov = np.minimum(en, e * 1000) - np.maximum(st, s * 1000)
        m = ov > 0
        X.append(np.c_[x[m], y[m]]); W.append(ov[m]); P += [pid] * int(m.sum())
    X, W, P = np.vstack(X), np.concatenate(W), np.array(P)
    if len(X) < 10:
        return np.nan
    lab, c = kmeans2(X, W)
    spread = np.sqrt(np.average(((X - c[lab]) ** 2).sum(1), weights=W))
    share = np.array([W[lab == k].sum() for k in (0, 1)]) / W.sum()
    both = np.mean([len(set(lab[P == q])) == 2 for q in np.unique(P) if (P == q).sum() >= 2])
    return min(np.linalg.norm(c[0] - c[1]) / spread, 4) / 4 * share.min() / share.max() * both


def alternate(c: pd.DataFrame, used: pd.DataFrame, n: int) -> pd.DataFrame:
    # Unused segments first; fill from used ones, preferring clips the unused set lacks.
    rest = c[~c.index.isin(used.index)].sort_values("corner", ascending=False)
    fill = used[~used.video.isin(rest.video)].sort_values("corner", ascending=False)
    fill = pd.concat([fill, used[~used.index.isin(fill.index)].sort_values("corner", ascending=False)])
    return pd.concat([rest, fill]).head(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two-target", action="store_true",
                   help="coordinated half of a median split, ranked by two-target score (coordinated_two_target_12_examples.png)")
    p.add_argument("--alt", choices=QUADRANTS, help="render only this quadrant with segments not in its main montage (<quad>_12_examples_alt.png)")
    a = p.parse_args()
    w = pd.read_csv(WB_PATH).dropna(subset=["within_person_cosine_dist"]).merge(quadrants("cuts"), on=["video", "seg_idx"])
    w["corner"] = corner_scores(w)  # percentiles over all valid segments, matching the threshold quantiles
    w = w.dropna(subset=["quadrant"])
    sel = w.quadrant == "selective"
    w.loc[sel, "corner"] = -w.loc[sel, "within_person_cosine_dist"]  # selective: favor low within over corner depth
    picks = {q: pick_diverse(w[w.quadrant == q], N_EXAMPLES) for q in QUADRANTS}
    if a.alt:
        picks = {a.alt: alternate(w[w.quadrant == a.alt], picks[a.alt], N_EXAMPLES)}
    if a.two_target:
        all_w = pd.read_csv(WB_PATH).dropna(subset=["within_person_cosine_dist"])
        half = all_w[(all_w.within_person_cosine_dist >= all_w.within_person_cosine_dist.median())
                     & (all_w.between_person_cosine_dist <= all_w.between_person_cosine_dist.median())].copy()
        arr = {v: build_clip_vectors(v)[1] for v in half.video.unique()}
        half["corner"] = [two_target_score(arr[r.video], r.seg_start_s, r.seg_end_s) for r in half.itertuples()]
        picks = {"coordinated": pick_diverse(half.dropna(subset=["corner"]), N_EXAMPLES)}

    needed = sorted({v for p in picks.values() for v in p.video})
    clip_data = {v: build_clip_vectors(v) for v in needed}
    video_res = {v: get_video_resolution(STIMULI_DIR / f"{v}.mp4") for v in ALL_CLIPS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for quad, p in picks.items():
        n_rows = int(np.ceil(len(p) / N_COLS))
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(5 * N_COLS, 3.6 * n_rows), squeeze=False)
        n_cand = len(half) if a.two_target else len(w[w.quadrant == quad])
        print(f"\n{quad}: {n_cand} candidates, {p.video.nunique()} clips in the 12")
        for k, ax in enumerate(axes.flat):
            if k >= len(p):
                ax.axis("off")
                continue
            r = p.iloc[k]
            pids, arrays, all_vecs, n_win, _ = clip_data[r.video]
            w_idx = [i for i in range(n_win) if i * WINDOW_S >= r.seg_start_s and (i + 1) * WINDOW_S <= r.seg_end_s]
            res = within_between_for_windows(pids, all_vecs[:, w_idx, :])
            if quad == "selective":  # lowest own within among people with >= MIN_FIX_SELECTIVE fixations in the segment
                ok = [pid for pid in res if n_fixations(arrays[pid], r.seg_start_s, r.seg_end_s) >= MIN_FIX_SELECTIVE]
                best = min(ok or res, key=lambda pid: res[pid][0])
            else:
                best = max(res, key=lambda pid: quadrant_score(quad, *res[pid]))
            ex = dict(video=r.video, seg_idx=int(r.seg_idx), sub_idx=None, sub_start_s=r.seg_start_s,
                      sub_end_s=r.seg_end_s, arrays=arrays, best_pid=best)
            draw_example_panel(ax, ex, video_res, HIGHLIGHT[quad])
            ax.set_title(f"within {r.within_person_cosine_dist:.2f}, between {r.between_person_cosine_dist:.2f}", fontsize=22)
            print(f"  {r.video} seg{int(r.seg_idx)} within={r.within_person_cosine_dist:.3f} "
                  f"between={r.between_person_cosine_dist:.3f} score={r.corner:.3f}")
        fig.tight_layout()
        tag = "_two_target" if a.two_target else ""
        out = OUT_DIR / f"{quad}{tag}_12_examples{'_alt' if a.alt else ''}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("Saved", out)


if __name__ == "__main__":
    main()

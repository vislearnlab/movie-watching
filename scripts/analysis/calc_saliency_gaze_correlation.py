"""
Per-segment Pearson correlation between the group gaze tile-vector and the saliency tile-
vector (FineGrained), averaged by quadrant. See METHODS_DECISIONS.md for method notes.

Usage:
    python scripts/analysis/calc_saliency_gaze_correlation.py --q 0.4
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from calc_within_between_cosine import MIN_WINDOWS, WINDOW_S  # noqa: E402
from montage_best_examples import ALL_CLIPS, CHOPPED_CLIPS, build_clip_vectors  # noqa: E402
from plot_within_between_timecourse import classify_quadrant, pooled_quantile_split  # noqa: E402

CANONICAL_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
SLOW_CHOPPED_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "slow_chopped_by_subsegment.csv"
SALIENCY_DIR = PROJECT_ROOT / "data" / "results" / "saliency"
OUT_FIG_DIR = PROJECT_ROOT / "figures" / "within_between"

QUAD_ORDER = ["concentrated", "coordinated", "selective", "scattered", "unclassified"]
QUAD_COLORS = {"concentrated": "#C8E6C9", "coordinated": "#D1C4E9",
               "selective": "#FFF9C4", "scattered": "#FFCDD2", "unclassified": "#E0E0E0"}


def group_mean_vector(pids, seg_vecs):
    own_means = []
    for i in range(len(pids)):
        own_by_window = seg_vecs[i]
        valid_w = np.where(own_by_window.sum(axis=1) > 0)[0]
        if len(valid_w) < MIN_WINDOWS:
            continue
        own_means.append(own_by_window[valid_w].mean(axis=0))
    if len(own_means) < 3:
        return None
    return np.mean(own_means, axis=0)


def segment_list_for(video: str, slow_chopped: bool) -> pd.DataFrame:
    if video not in CHOPPED_CLIPS or not slow_chopped:
        df = pd.read_csv(CANONICAL_PATH)
        return df[df["video"] == video].copy()
    df = pd.read_csv(SLOW_CHOPPED_PATH)
    return df[df["video"] == video].copy()


def run_analysis(slow_chopped: bool, thresholds: tuple) -> tuple:
    rows = []
    gaze_vecs, sal_vecs = [], []
    for video in ALL_CLIPS:
        print(f"Processing {video} (slow_chopped={slow_chopped})...")
        pids, arrays, all_vecs, n_windows_total, _ = build_clip_vectors(video)
        sal_df = pd.read_csv(SALIENCY_DIR / f"{video}_saliency_tiles.csv")
        tile_cols = [c for c in sal_df.columns if c.startswith("tile_")]

        for _, seg in segment_list_for(video, slow_chopped).iterrows():
            s, e = seg["seg_start_s"], seg["seg_end_s"]
            w_idx = [w for w in range(n_windows_total) if w * WINDOW_S >= s and (w + 1) * WINDOW_S <= e]
            if len(w_idx) < MIN_WINDOWS:
                continue

            gaze_vec = group_mean_vector(pids, all_vecs[:, w_idx, :])
            if gaze_vec is None:
                continue

            sal_rows = sal_df[sal_df["window_idx"].isin(w_idx)]
            if sal_rows.empty:
                continue
            sal_vec = sal_rows[tile_cols].mean(axis=0).to_numpy()

            if gaze_vec.std() == 0 or sal_vec.std() == 0:
                continue
            r, _ = pearsonr(gaze_vec, sal_vec)

            if pd.isna(seg["within_person_cosine_dist"]):
                continue
            quad = classify_quadrant(seg["within_person_cosine_dist"], seg["between_person_cosine_dist"],
                                      *thresholds)
            rows.append(dict(video=video, seg_idx=seg["seg_idx"], quad=quad or "unclassified", r=r))
            gaze_vecs.append(gaze_vec)
            sal_vecs.append(sal_vec)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    gaze_arr, sal_arr = np.array(gaze_vecs), np.array(sal_vecs)
    n = len(gaze_arr)
    null_means = []
    for _ in range(500):
        shuffled = rng.permutation(n)
        rs = [pearsonr(gaze_arr[i], sal_arr[shuffled[i]])[0] for i in range(n)]
        null_means.append(np.mean(rs))
    null_floor = np.mean(null_means)
    return df, null_floor


def draw_panel(ax, df: pd.DataFrame, null_floor: float, title: str, quad_order: list):
    summary = df.groupby("quad")["r"].agg(["mean", "sem", "count"]).reindex(quad_order)
    ax.bar(quad_order, summary["mean"], yerr=summary["sem"],
           color=[QUAD_COLORS[q] for q in quad_order], edgecolor="black", capsize=4)
    for i, q in enumerate(quad_order):
        ax.text(i, 0.01, f"n={int(summary['count'][q])}", ha="center", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(null_floor, color="red", linestyle="--", linewidth=1.2,
               label=f"null = {null_floor:.3f}")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("Pearson r (saliency vector vs. group gaze vector)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=0.4, help="low/high percentile split (0.5 = median)")
    args = parser.parse_args()
    q_tag = f"q{int(args.q*100)}"
    quad_order = [x for x in QUAD_ORDER if x != "unclassified" or args.q < 0.5]

    canonical = pd.read_csv(CANONICAL_PATH)
    thresholds = pooled_quantile_split(canonical, args.q)

    df_chopped, null_chopped = run_analysis(slow_chopped=True, thresholds=thresholds)
    out_csv = SALIENCY_DIR / f"saliency_gaze_correlation_by_segment_{q_tag}.csv"
    df_chopped.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df_chopped)} segments)")

    df_unchopped, null_unchopped = run_analysis(slow_chopped=False, thresholds=thresholds)
    out_csv_unchopped = SALIENCY_DIR / f"saliency_gaze_correlation_by_segment_{q_tag}_slowunchopped.csv"
    df_unchopped.to_csv(out_csv_unchopped, index=False)
    print(f"Saved {out_csv_unchopped} ({len(df_unchopped)} segments)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    draw_panel(axes[0], df_chopped, null_chopped, f"Slow chopped (n={len(df_chopped)})", quad_order)
    draw_panel(axes[1], df_unchopped, null_unchopped, f"Slow unchopped (n={len(df_unchopped)})", quad_order)
    pct_label = f"{int(args.q*100)}th/{int((1-args.q)*100)}th percentile split"
    fig.suptitle(f"Saliency-gaze correlation by quadrant ({pct_label})")
    fig.tight_layout()
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = OUT_FIG_DIR / f"saliency_gaze_correlation_by_quadrant_{q_tag}.png"
    fig.savefig(out_fig, dpi=150)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()

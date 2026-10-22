"""
Per-segment partial correlation of gaze with motion (controlling for saliency) and gaze with
saliency (controlling for motion), averaged by quadrant at a 50/50 (median) split.

Usage:
    python scripts/analysis/calc_motion_saliency_partial_correlation.py
"""

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

from calc_within_between_cosine import BLOCKS, MIN_WINDOWS, WINDOW_S  # noqa: E402
from calc_motion_gaze_correlation import group_mean_vector, segment_list_for  # noqa: E402
from montage_best_examples import ALL_CLIPS, build_clip_vectors  # noqa: E402
from plot_within_between_timecourse import classify_quadrant, pooled_quantile_split  # noqa: E402

CANONICAL_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
MOTION_DIR = PROJECT_ROOT / "data" / "results" / "motion"
SALIENCY_DIR = PROJECT_ROOT / "data" / "results" / "saliency"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "motion"
OUT_FIG = PROJECT_ROOT / "figures" / "within_between" / "motion_saliency_partial_correlation_byblock_q50.png"

Q = 0.5
BLOCK_ORDER = ["frank", "pixar", "sesame", "slow"]
BLOCK_COLORS = {"frank": "#BBDEFB", "pixar": "#FFE0B2", "sesame": "#FFCDD2", "slow": "#E1BEE7"}


def partial_r(r_ab: float, r_ac: float, r_bc: float) -> float:
    denom = np.sqrt((1 - r_ac**2) * (1 - r_bc**2))
    return np.nan if denom == 0 else (r_ab - r_ac * r_bc) / denom


def run_analysis(slow_chopped: bool, thresholds: tuple) -> tuple:
    rows = []
    gaze_vecs, motion_vecs, sal_vecs = [], [], []
    for video in ALL_CLIPS:
        print(f"Processing {video} (slow_chopped={slow_chopped})...")
        pids, arrays, all_vecs, n_windows_total, _ = build_clip_vectors(video)
        motion_df = pd.read_csv(MOTION_DIR / f"{video}_motion_tiles.csv")
        sal_df = pd.read_csv(SALIENCY_DIR / f"{video}_saliency_tiles.csv")
        motion_cols = [c for c in motion_df.columns if c.startswith("tile_")]
        sal_cols = [c for c in sal_df.columns if c.startswith("tile_")]

        for _, seg in segment_list_for(video, slow_chopped).iterrows():
            s, e = seg["seg_start_s"], seg["seg_end_s"]
            w_idx = [w for w in range(n_windows_total) if w * WINDOW_S >= s and (w + 1) * WINDOW_S <= e]
            if len(w_idx) < MIN_WINDOWS:
                continue

            gaze_vec = group_mean_vector(pids, all_vecs[:, w_idx, :])
            if gaze_vec is None:
                continue

            motion_rows = motion_df[motion_df["window_idx"].isin(w_idx)]
            sal_rows = sal_df[sal_df["window_idx"].isin(w_idx)]
            if motion_rows.empty or sal_rows.empty:
                continue
            motion_vec = motion_rows[motion_cols].mean(axis=0).to_numpy()
            sal_vec = sal_rows[sal_cols].mean(axis=0).to_numpy()

            if gaze_vec.std() == 0 or motion_vec.std() == 0 or sal_vec.std() == 0:
                continue
            r_gm, _ = pearsonr(gaze_vec, motion_vec)
            r_gs, _ = pearsonr(gaze_vec, sal_vec)
            r_ms, _ = pearsonr(motion_vec, sal_vec)
            pr_motion = partial_r(r_gm, r_gs, r_ms)
            pr_saliency = partial_r(r_gs, r_gm, r_ms)
            if np.isnan(pr_motion) or np.isnan(pr_saliency):
                continue

            if pd.isna(seg["within_person_cosine_dist"]):
                continue
            quad = classify_quadrant(seg["within_person_cosine_dist"], seg["between_person_cosine_dist"],
                                      *thresholds)
            rows.append(dict(video=video, block=BLOCKS[video], seg_idx=seg["seg_idx"], quad=quad,
                              pr_motion=pr_motion, pr_saliency=pr_saliency))
            gaze_vecs.append(gaze_vec)
            motion_vecs.append(motion_vec)
            sal_vecs.append(sal_vec)

    df = pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    gaze_arr = np.array(gaze_vecs)
    motion_arr, sal_arr = np.array(motion_vecs), np.array(sal_vecs)
    n = len(gaze_arr)
    null_motion, null_saliency = [], []
    for _ in range(500):
        shuffled = rng.permutation(n)
        pm, ps = [], []
        for i in range(n):
            r_gm, _ = pearsonr(gaze_arr[i], motion_arr[shuffled[i]])
            r_gs, _ = pearsonr(gaze_arr[i], sal_arr[shuffled[i]])
            r_ms, _ = pearsonr(motion_arr[shuffled[i]], sal_arr[shuffled[i]])
            pm.append(partial_r(r_gm, r_gs, r_ms))
            ps.append(partial_r(r_gs, r_gm, r_ms))
        null_motion.append(np.nanmean(pm))
        null_saliency.append(np.nanmean(ps))
    return df, np.mean(null_motion), np.mean(null_saliency)


def draw_panel(ax, df: pd.DataFrame, null_motion: float, null_saliency: float, title: str):
    rng = np.random.default_rng(0)
    summary = df.groupby("block")[["pr_motion", "pr_saliency"]].agg(["mean", "count"]).reindex(BLOCK_ORDER)
    x = np.arange(len(BLOCK_ORDER))
    w = 0.35
    ax.bar(x - w/2, summary["pr_motion"]["mean"],
           width=w, color=[BLOCK_COLORS[b] for b in BLOCK_ORDER], edgecolor="black",
           hatch="//", label="motion | saliency", zorder=1)
    ax.bar(x + w/2, summary["pr_saliency"]["mean"],
           width=w, color=[BLOCK_COLORS[b] for b in BLOCK_ORDER], edgecolor="black",
           label="saliency | motion", zorder=1)

    for i, b in enumerate(BLOCK_ORDER):
        sub = df[df["block"] == b]
        jitter_m = rng.uniform(-w * 0.35, w * 0.35, size=len(sub))
        jitter_s = rng.uniform(-w * 0.35, w * 0.35, size=len(sub))
        ax.scatter(i - w/2 + jitter_m, sub["pr_motion"], s=10, color="black", alpha=0.3, zorder=2)
        ax.scatter(i + w/2 + jitter_s, sub["pr_saliency"], s=10, color="black", alpha=0.3, zorder=2)
        ax.text(i - w/2, -0.02, f"n={int(summary['pr_motion']['count'][b])}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(null_motion, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label=f"null (motion|saliency) = {null_motion:.3f}")
    ax.axhline(null_saliency, color="darkred", linestyle=":", linewidth=1, alpha=0.7,
               label=f"null (saliency|motion) = {null_saliency:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(BLOCK_ORDER, rotation=20)
    ax.set_ylabel("Partial Pearson r with group gaze vector")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    canonical = pd.read_csv(CANONICAL_PATH)
    thresholds = pooled_quantile_split(canonical, Q)

    df_chopped, null_m_chopped, null_s_chopped = run_analysis(slow_chopped=True, thresholds=thresholds)
    out_csv = OUT_DIR / "motion_saliency_partial_correlation_q50.csv"
    df_chopped.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df_chopped)} segments)")

    df_unchopped, null_m_unchopped, null_s_unchopped = run_analysis(slow_chopped=False, thresholds=thresholds)
    out_csv_unchopped = OUT_DIR / "motion_saliency_partial_correlation_q50_slowunchopped.csv"
    df_unchopped.to_csv(out_csv_unchopped, index=False)
    print(f"Saved {out_csv_unchopped} ({len(df_unchopped)} segments)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    draw_panel(axes[0], df_chopped, null_m_chopped, null_s_chopped, f"Slow chopped (n={len(df_chopped)})")
    draw_panel(axes[1], df_unchopped, null_m_unchopped, null_s_unchopped, f"Slow unchopped (n={len(df_unchopped)})")
    fig.suptitle("Motion vs. saliency, partial correlation with gaze by block")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()

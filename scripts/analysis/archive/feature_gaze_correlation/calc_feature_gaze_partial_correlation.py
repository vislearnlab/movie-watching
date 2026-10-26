"""
Per-segment partial correlation of gaze with each of N content features (motion, saliency,
center bias, ...), controlling for all the other active features at once. Generalizes
calc_motion_saliency_partial_correlation.py's pairwise formula to any number of features via
residualized regression. See METHODS_DECISIONS.md for method notes.

Usage:
    python scripts/analysis/calc_feature_gaze_partial_correlation.py --features motion,center --groupby clip
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

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # archived: scripts/analysis/archive/feature_gaze_correlation
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from calc_within_between_cosine import BLOCKS, MIN_WINDOWS, WINDOW_S  # noqa: E402
from calc_motion_gaze_correlation import group_mean_vector, segment_list_for  # noqa: E402
from montage_best_examples import ALL_CLIPS, build_clip_vectors  # noqa: E402
from plot_within_between_timecourse import classify_quadrant, pooled_quantile_split  # noqa: E402
from plot_patch_grid import compute_patch_grid  # noqa: E402

CANONICAL_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
MOTION_DIR = PROJECT_ROOT / "data" / "results" / "motion"
SALIENCY_DIR = PROJECT_ROOT / "data" / "results" / "saliency"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "motion"
OUT_FIG_DIR = PROJECT_ROOT / "figures" / "within_between"

Q = 0.5
BLOCK_ORDER = ["frank", "pixar", "sesame", "slow"]
BLOCK_COLORS = {"frank": "#BBDEFB", "pixar": "#FFE0B2", "sesame": "#FFCDD2", "slow": "#E1BEE7"}
CLIP_ORDER = ["frank_complex", "frank_objects", "frank_play", "pixar_birds",
              "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2",
              "slow_animals", "slow_hands", "slow_inanimate", "slow_people"]
CLIP_COLORS = {
    "frank_complex": "#1976D2", "frank_objects": "#90CAF9", "frank_play": "#FB8C00",
    "pixar_birds": "#FFB74D",
    "sesameindia_1": "#2E7D32", "sesameindia_2": "#81C784",
    "sesameus_1": "#C62828", "sesameus_2": "#EF9A9A",
    "slow_animals": "#7E57C2", "slow_hands": "#D1C4E9",
    "slow_inanimate": "#6D4C41", "slow_people": "#BCAAA4",
}
QUAD_ORDER = ["concentrated", "coordinated", "selective", "scattered"]
QUAD_COLORS = {"concentrated": "#C8E6C9", "coordinated": "#D1C4E9",
               "selective": "#FFF9C4", "scattered": "#FFCDD2"}
HATCHES = ["//", "", "xx", "..", "++", "oo"]  # one per feature, cycles if >6

# Center bias: fixed anisotropic Gaussian on the 120-tile grid, approved 2026-10-22
# (sigma_x=384px, sigma_y=216px -- 1920x1080 display, 20% tighter than the first draft).
_DISPLAY_W, _DISPLAY_H = 1920, 1080
_centers, _ = compute_patch_grid(240, 15, _DISPLAY_W, _DISPLAY_H)  # 2026-10-22: moved from 180px to 240px (50% overlap)
_cx0, _cy0 = _DISPLAY_W / 2, _DISPLAY_H / 2
_sigma_x, _sigma_y = _DISPLAY_W / 4 * 0.8, _DISPLAY_H / 4 * 0.8
_dx, _dy = _centers["center_x"] - _cx0, _centers["center_y"] - _cy0
CENTER_BIAS_VEC = np.exp(-(_dx**2 / (2 * _sigma_x**2) + _dy**2 / (2 * _sigma_y**2))).to_numpy()


FEATURES = {
    "motion": dict(label="motion", load=lambda video: pd.read_csv(MOTION_DIR / f"{video}_motion_tiles.csv")),
    "saliency": dict(label="saliency", load=lambda video: pd.read_csv(SALIENCY_DIR / f"{video}_saliency_tiles.csv")),
    "center": dict(label="center", load=lambda video: None),  # constant vector, no per-video file
}


def feature_vector(name: str, video: str, w_idx: list, cache: dict) -> np.ndarray:
    if name == "center":
        return CENTER_BIAS_VEC
    key = (name, video)
    if key not in cache:
        df = FEATURES[name]["load"](video)
        cache[key] = (df, [c for c in df.columns if c.startswith("tile_")])
    df, tile_cols = cache[key]
    rows = df[df["window_idx"].isin(w_idx)]
    if rows.empty:
        return None
    return rows[tile_cols].mean(axis=0).to_numpy()


def partial_corr(y: np.ndarray, x: np.ndarray, others: list) -> float:
    """Partial correlation of y with x, controlling for all vectors in others.
    others=[] reduces to plain Pearson r; generalizes to any number of controls."""
    if not others:
        r, _ = pearsonr(y, x)
        return r
    Z = np.column_stack(others + [np.ones(len(y))])
    beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid_y = y - Z @ beta_y
    beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
    resid_x = x - Z @ beta_x
    if resid_y.std() == 0 or resid_x.std() == 0:
        return np.nan
    r, _ = pearsonr(resid_y, resid_x)
    return r


def run_analysis(features: list, slow_chopped: bool, thresholds: tuple) -> tuple:
    rows = []
    gaze_vecs = []
    feat_vecs = {f: [] for f in features}
    cache = {}

    for video in ALL_CLIPS:
        print(f"Processing {video} (slow_chopped={slow_chopped})...")
        pids, arrays, all_vecs, n_windows_total, _ = build_clip_vectors(video)

        for _, seg in segment_list_for(video, slow_chopped).iterrows():
            s, e = seg["seg_start_s"], seg["seg_end_s"]
            w_idx = [w for w in range(n_windows_total) if w * WINDOW_S >= s and (w + 1) * WINDOW_S <= e]
            if len(w_idx) < MIN_WINDOWS:
                continue

            gaze_vec = group_mean_vector(pids, all_vecs[:, w_idx, :])
            if gaze_vec is None or gaze_vec.std() == 0:
                continue

            this_seg_feats = {}
            ok = True
            for f in features:
                v = feature_vector(f, video, w_idx, cache)
                if v is None or v.std() == 0:
                    ok = False
                    break
                this_seg_feats[f] = v
            if not ok:
                continue

            if pd.isna(seg["within_person_cosine_dist"]):
                continue
            quad = classify_quadrant(seg["within_person_cosine_dist"], seg["between_person_cosine_dist"],
                                      *thresholds)

            row = dict(video=video, block=BLOCKS[video], seg_idx=seg["seg_idx"], quad=quad)
            for f in features:
                others = [this_seg_feats[g] for g in features if g != f]
                row[f"pr_{f}"] = partial_corr(gaze_vec, this_seg_feats[f], others)
                feat_vecs[f].append(this_seg_feats[f])
            rows.append(row)
            gaze_vecs.append(gaze_vec)

    df = pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    gaze_arr = np.array(gaze_vecs)
    feat_arrs = {f: np.array(v) for f, v in feat_vecs.items()}
    n = len(gaze_arr)
    null_means = {f: [] for f in features}
    for _ in range(500):
        shuffled = rng.permutation(n)
        per_feat = {f: [] for f in features}
        for i in range(n):
            shuffled_feats = {f: feat_arrs[f][shuffled[i]] for f in features}
            for f in features:
                others = [shuffled_feats[g] for g in features if g != f]
                per_feat[f].append(partial_corr(gaze_arr[i], shuffled_feats[f], others))
        for f in features:
            null_means[f].append(np.nanmean(per_feat[f]))
    nulls = {f: float(np.mean(null_means[f])) for f in features}
    return df, nulls


def draw_panel(ax, df: pd.DataFrame, features: list, nulls: dict, groupby: str, title: str,
               line_labels: dict = None):
    if groupby == "clip":
        order, colors = [c for c in CLIP_ORDER if c in df["video"].unique()], CLIP_COLORS
        key = "video"
    elif groupby == "block":
        order, colors = BLOCK_ORDER, BLOCK_COLORS
        key = "block"
    else:
        order, colors = [q for q in QUAD_ORDER if q in df["quad"].unique()], QUAD_COLORS
        key = "quad"

    rng = np.random.default_rng(0)
    n_feat = len(features)
    group_w = 0.8
    bar_w = group_w / n_feat

    for gi, g in enumerate(order):
        sub = df[df[key] == g]
        gx = gi
        for fi, f in enumerate(features):
            bx = gx - group_w / 2 + bar_w * (fi + 0.5)
            color = colors[g]
            mean_val = sub[f"pr_{f}"].mean()
            ax.bar(bx, mean_val, width=bar_w * 0.92, color=color, edgecolor="black",
                   hatch=HATCHES[fi % len(HATCHES)], zorder=1)
            jitter = rng.uniform(-bar_w * 0.35, bar_w * 0.35, size=len(sub))
            ax.scatter(bx + jitter, sub[f"pr_{f}"], s=9, color="black", alpha=0.3, zorder=2)
        ax.text(gx, -0.025, f"n={len(sub)}", ha="center", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8)
    for fi, f in enumerate(features):
        label = line_labels[f] if line_labels else f"null ({FEATURES[f]['label']}|rest)"
        ax.axhline(nulls[f], color="red", linestyle=["--", ":", "-."][fi % 3], linewidth=1, alpha=0.7,
                   label=f"{label} = {nulls[f]:.3f}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45 if groupby == "clip" else 20, ha="right" if groupby == "clip" else "center")
    ax.set_ylabel("Partial Pearson r with group gaze vector")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def run_benchmark_analysis(primary: str, benchmark: str, slow_chopped: bool, thresholds: tuple) -> tuple:
    """Raw (non-partial) correlation of gaze with `primary`, per segment, plus two reference
    lines: primary's genuine shuffled-pairing null, and benchmark's raw mean correlation
    (not shuffled -- meaningless for a feature with no between-segment variability, e.g.
    center bias, so it's reported as a plain benchmark level instead of a null)."""
    rows = []
    gaze_vecs, primary_vecs = [], []
    cache = {}

    for video in ALL_CLIPS:
        print(f"Processing {video} (slow_chopped={slow_chopped})...")
        pids, arrays, all_vecs, n_windows_total, _ = build_clip_vectors(video)

        for _, seg in segment_list_for(video, slow_chopped).iterrows():
            s, e = seg["seg_start_s"], seg["seg_end_s"]
            w_idx = [w for w in range(n_windows_total) if w * WINDOW_S >= s and (w + 1) * WINDOW_S <= e]
            if len(w_idx) < MIN_WINDOWS:
                continue
            gaze_vec = group_mean_vector(pids, all_vecs[:, w_idx, :])
            if gaze_vec is None or gaze_vec.std() == 0:
                continue
            p_vec = feature_vector(primary, video, w_idx, cache)
            b_vec = feature_vector(benchmark, video, w_idx, cache)
            if p_vec is None or p_vec.std() == 0 or b_vec is None or b_vec.std() == 0:
                continue
            if pd.isna(seg["within_person_cosine_dist"]):
                continue
            quad = classify_quadrant(seg["within_person_cosine_dist"], seg["between_person_cosine_dist"],
                                      *thresholds)
            r_primary, _ = pearsonr(gaze_vec, p_vec)
            r_benchmark, _ = pearsonr(gaze_vec, b_vec)
            rows.append(dict(video=video, block=BLOCKS[video], seg_idx=seg["seg_idx"], quad=quad,
                              r_primary=r_primary, r_benchmark=r_benchmark))
            gaze_vecs.append(gaze_vec)
            primary_vecs.append(p_vec)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    gaze_arr, primary_arr = np.array(gaze_vecs), np.array(primary_vecs)
    n = len(gaze_arr)
    null_means = []
    for _ in range(500):
        shuffled = rng.permutation(n)
        rs = [pearsonr(gaze_arr[i], primary_arr[shuffled[i]])[0] for i in range(n)]
        null_means.append(np.mean(rs))
    primary_null = float(np.mean(null_means))
    benchmark_level = float(df["r_benchmark"].mean())
    return df, primary_null, benchmark_level


def draw_benchmark_panel(ax, df: pd.DataFrame, primary: str, benchmark: str,
                          primary_null: float, benchmark_level: float, groupby: str, title: str):
    if groupby == "clip":
        order, colors = [c for c in CLIP_ORDER if c in df["video"].unique()], CLIP_COLORS
        key = "video"
    elif groupby == "block":
        order, colors = BLOCK_ORDER, BLOCK_COLORS
        key = "block"
    else:
        order, colors = [q for q in QUAD_ORDER if q in df["quad"].unique()], QUAD_COLORS
        key = "quad"

    rng = np.random.default_rng(0)
    for gi, g in enumerate(order):
        sub = df[df[key] == g]
        mean_val = sub["r_primary"].mean()
        ax.bar(gi, mean_val, width=0.6, color=colors[g], edgecolor="black", hatch="//", zorder=1)
        jitter = rng.uniform(-0.2, 0.2, size=len(sub))
        ax.scatter(gi + jitter, sub["r_primary"], s=10, color="black", alpha=0.3, zorder=2)
        ax.text(gi, -0.03, f"n={len(sub)}", ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(primary_null, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"{FEATURES[primary]['label']} null (shuffled) = {primary_null:.3f}")
    ax.axhline(benchmark_level, color="teal", linestyle=":", linewidth=1.5, alpha=0.8,
               label=f"{FEATURES[benchmark]['label']} (real, unshuffled) = {benchmark_level:.3f}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45 if groupby == "clip" else 20, ha="right" if groupby == "clip" else "center")
    ax.set_ylabel(f"Pearson r ({FEATURES[primary]['label']} vs. group gaze vector)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="comma-separated, e.g. motion,center or motion,saliency,center")
    parser.add_argument("--primary", help="benchmark mode: the feature whose raw correlation to plot")
    parser.add_argument("--benchmark", help="benchmark mode: feature used as a reference line, not partialed out")
    parser.add_argument("--groupby", choices=["clip", "block", "quad"], default="clip")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--raw-reference-lines", action="store_true",
                         help="plot partial-correlation bars (both features) but use raw-correlation "
                              "reference lines (first feature: shuffled null; second: real unshuffled "
                              "mean) instead of the degenerate partial-correlation null. Needs exactly 2 --features.")
    args = parser.parse_args()

    canonical = pd.read_csv(CANONICAL_PATH)
    thresholds = pooled_quantile_split(canonical, args.q)

    if args.primary or args.benchmark:
        assert args.primary and args.benchmark, "benchmark mode needs both --primary and --benchmark"
        primary, benchmark = args.primary, args.benchmark
        for f in (primary, benchmark):
            assert f in FEATURES, f"Unknown feature {f!r}, choose from {list(FEATURES)}"
        tag = f"{primary}_vs_{benchmark}"

        df_chopped, null_c, bench_c = run_benchmark_analysis(primary, benchmark, True, thresholds)
        out_csv = OUT_DIR / f"feature_gaze_benchmark_{tag}_q{int(args.q*100)}.csv"
        df_chopped.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} ({len(df_chopped)} segments)")

        df_unchopped, null_u, bench_u = run_benchmark_analysis(primary, benchmark, False, thresholds)
        out_csv_u = OUT_DIR / f"feature_gaze_benchmark_{tag}_q{int(args.q*100)}_slowunchopped.csv"
        df_unchopped.to_csv(out_csv_u, index=False)
        print(f"Saved {out_csv_u} ({len(df_unchopped)} segments)")

        fig, axes = plt.subplots(1, 2, figsize=(20, 7) if args.groupby == "clip" else (16, 6.5), sharey=True)
        draw_benchmark_panel(axes[0], df_chopped, primary, benchmark, null_c, bench_c,
                              args.groupby, f"Slow chopped (n={len(df_chopped)})")
        draw_benchmark_panel(axes[1], df_unchopped, primary, benchmark, null_u, bench_u,
                              args.groupby, f"Slow unchopped (n={len(df_unchopped)})")
        fig.suptitle(f"{FEATURES[primary]['label']} vs. gaze, benchmarked against "
                     f"{FEATURES[benchmark]['label']}, by {args.groupby}")
        fig.tight_layout()
        OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_fig = OUT_FIG_DIR / f"feature_benchmark_{tag}_by{args.groupby}_q{int(args.q*100)}.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
        return

    features = args.features.split(",")
    for f in features:
        assert f in FEATURES, f"Unknown feature {f!r}, choose from {list(FEATURES)}"

    df_chopped, nulls_chopped = run_analysis(features, slow_chopped=True, thresholds=thresholds)
    tag = "_".join(features)
    out_csv = OUT_DIR / f"feature_gaze_partial_correlation_{tag}_q{int(args.q*100)}.csv"
    df_chopped.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df_chopped)} segments)")

    df_unchopped, nulls_unchopped = run_analysis(features, slow_chopped=False, thresholds=thresholds)
    out_csv_u = OUT_DIR / f"feature_gaze_partial_correlation_{tag}_q{int(args.q*100)}_slowunchopped.csv"
    df_unchopped.to_csv(out_csv_u, index=False)
    print(f"Saved {out_csv_u} ({len(df_unchopped)} segments)")

    line_labels, fig_tag = None, tag
    if args.raw_reference_lines:
        assert len(features) == 2, "--raw-reference-lines needs exactly 2 --features"
        primary, benchmark = features
        _, raw_null_c, raw_bench_c = run_benchmark_analysis(primary, benchmark, True, thresholds)
        _, raw_null_u, raw_bench_u = run_benchmark_analysis(primary, benchmark, False, thresholds)
        nulls_chopped = {primary: raw_null_c, benchmark: raw_bench_c}
        nulls_unchopped = {primary: raw_null_u, benchmark: raw_bench_u}
        line_labels = {primary: f"{FEATURES[primary]['label']} null (shuffled)",
                       benchmark: f"{FEATURES[benchmark]['label']} (real, unshuffled)"}
        fig_tag = f"{tag}_rawlines"

    fig, axes = plt.subplots(1, 2, figsize=(20, 7) if args.groupby == "clip" else (16, 6.5), sharey=True)
    draw_panel(axes[0], df_chopped, features, nulls_chopped, args.groupby, f"Slow chopped (n={len(df_chopped)})",
               line_labels=line_labels)
    draw_panel(axes[1], df_unchopped, features, nulls_unchopped, args.groupby, f"Slow unchopped (n={len(df_unchopped)})",
               line_labels=line_labels)
    fig.suptitle(f"{' vs. '.join(FEATURES[f]['label'] for f in features)}, partial correlation with gaze by {args.groupby}")
    fig.tight_layout()
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = OUT_FIG_DIR / f"feature_partial_correlation_{fig_tag}_by{args.groupby}_q{int(args.q*100)}.png"
    fig.savefig(out_fig, dpi=150)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()

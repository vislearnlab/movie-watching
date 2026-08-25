"""
Plot face-gaze analysis results across all 4 annotated clips.

Reads CSVs produced by run_all_clips.py and generates:

  1. face_proportions.png       — face-looking proportion per clip (adult/infant)
                                   with random baseline overlay
  2. face_vs_metadata.png       — face-looking vs # faces & vs face area per clip
  3. summary_stats.png          — summary statistics table / bar chart
  4. isc_timeseries_{clip}.png  — ISC over time per clip (aa / ii / ai) + baselines
  5. isc_timeseries_avg.png     — ISC over time averaged across clips + baselines

Usage:
    python analysis/plot_results.py \\
        --results_dir analysis/results \\
        --output_dir analysis/results/plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

CLIPS = ["frank_complex", "frank_objects", "frank_play", "sesameus_1"]

CLIP_LABELS = {
    "frank_complex": "Frank\nComplex",
    "frank_objects": "Frank\nObjects",
    "frank_play": "Frank\nPlay",
    "sesameus_1": "Sesame\nUS 1",
}

COMPARISON_LABELS = {
    "adult_adult": "Adult–Adult",
    "infant_infant": "Infant–Infant",
    "adult_infant": "Adult–Infant",
}

# Colors
GROUP_COLORS = {"adult": "#2196F3", "infant": "#FF9800"}
COMPARISON_COLORS = {
    "adult_adult": "#1976D2",
    "infant_infant": "#E65100",
    "adult_infant": "#388E3C",
}
RANDOM_ALPHA = 0.35
BASELINE_LINESTYLE = "--"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> dict:
    def _read(name):
        path = os.path.join(results_dir, name)
        if os.path.isfile(path):
            return pd.read_csv(path)
        print(f"WARNING: {path} not found — some plots may be skipped.")
        return None

    return dict(
        props=_read("face_proportions.csv"),
        baseline=_read("random_prop_baseline.csv"),
        isc_summary=_read("isc_summary.csv"),
        isc_ts=_read("isc_timeseries.csv"),
        meta=_read("clip_face_metadata.csv"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Face-looking proportions per clip, by group, with random baseline
# ─────────────────────────────────────────────────────────────────────────────

def plot_face_proportions(props_df: pd.DataFrame, baseline_df: pd.DataFrame, out_path: str):
    groups = ["adult", "infant"]
    n_clips = len(CLIPS)
    fig, axes = plt.subplots(1, n_clips, figsize=(3.5 * n_clips, 4.5), sharey=True)
    fig.suptitle("Face-Looking Proportion per Clip", fontsize=13, fontweight="bold", y=1.01)

    for ax, clip in zip(axes, CLIPS):
        ax.set_title(CLIP_LABELS[clip])
        ax.set_xlabel("Group")
        if ax == axes[0]:
            ax.set_ylabel("Proportion looking at face")

        x_positions = np.arange(len(groups))
        for xi, grp in enumerate(groups):
            color = GROUP_COLORS[grp]

            # Individual participant dots
            sub = props_df[(props_df["clip"] == clip) & (props_df["group"] == grp)]
            vals = sub["prop_looking_at_face"].dropna().values

            if len(vals) == 0:
                continue

            # Jitter
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals, color=color, alpha=0.45, s=20, zorder=3)

            # Mean ± SEM bar
            mean = vals.mean()
            sem = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.bar(xi, mean, width=0.55, color=color, alpha=0.7, zorder=2)
            ax.errorbar(xi, mean, yerr=sem, fmt="none", color="black", capsize=4, linewidth=1.5, zorder=4)

            # Random baseline (dashed line across bar width)
            base_sub = baseline_df[(baseline_df["clip"] == clip) & (baseline_df["group"] == grp)]
            base_vals = base_sub["baseline_prop"].dropna().values
            if len(base_vals) > 0:
                base_mean = base_vals.mean()
                ax.hlines(base_mean, xi - 0.28, xi + 0.28, colors="black",
                          linestyles=BASELINE_LINESTYLE, linewidth=1.5, zorder=5, label="_nolegend_")

        ax.set_xticks(x_positions)
        ax.set_xticklabels([g.capitalize() for g in groups])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Legend
    handles = [
        mpatches.Patch(color=GROUP_COLORS["adult"], alpha=0.7, label="Adult (mean ± SEM)"),
        mpatches.Patch(color=GROUP_COLORS["infant"], alpha=0.7, label="Infant (mean ± SEM)"),
        plt.Line2D([0], [0], color="black", linestyle=BASELINE_LINESTYLE, linewidth=1.5, label="Random baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07),
               frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Face-looking vs face metadata scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_face_vs_metadata(props_df: pd.DataFrame, meta_df: pd.DataFrame, out_path: str):
    # Clip-level mean proportions per group
    clip_means = (
        props_df.groupby(["clip", "group"])["prop_looking_at_face"]
        .mean()
        .reset_index()
        .rename(columns={"prop_looking_at_face": "mean_prop"})
    )
    # Overall mean across all participants
    overall = (
        props_df.groupby("clip")["prop_looking_at_face"]
        .mean()
        .reset_index()
        .rename(columns={"prop_looking_at_face": "mean_prop"})
    )
    overall["group"] = "overall"
    clip_means = pd.concat([clip_means, overall], ignore_index=True)

    # Merge with metadata
    merged = clip_means.merge(meta_df, on="clip")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Face-Looking vs Face Properties (per clip)", fontsize=12, fontweight="bold")

    x_vars = [
        ("mean_n_faces_per_frame", "Mean # faces per frame"),
        ("mean_face_area_frac_per_frame", "Mean face area fraction"),
    ]

    for ax, (x_col, x_label) in zip(axes, x_vars):
        for grp, color in [("adult", GROUP_COLORS["adult"]),
                            ("infant", GROUP_COLORS["infant"]),
                            ("overall", "#6D4C41")]:
            sub = merged[merged["group"] == grp]
            if sub.empty:
                continue
            ax.scatter(sub[x_col], sub["mean_prop"], color=color, s=80,
                       label=grp.capitalize(), zorder=3)
            # Clip labels
            for _, row in sub.iterrows():
                ax.annotate(
                    CLIP_LABELS[row["clip"]].replace("\n", " "),
                    (row[x_col], row["mean_prop"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7, color=color,
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean proportion looking at face")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, max(merged["mean_prop"].max() * 1.2, 0.1))
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Summary stats
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_stats(props_df: pd.DataFrame, isc_summary_df: pd.DataFrame, meta_df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Summary Statistics", fontsize=13, fontweight="bold")

    # ── Panel A: face-looking proportion by clip and group ────────────────────
    ax = axes[0]
    ax.set_title("A. Face-Looking Proportion by Clip")

    groups = ["adult", "infant"]
    x = np.arange(len(CLIPS))
    width = 0.35

    for gi, grp in enumerate(groups):
        means, sems = [], []
        for clip in CLIPS:
            sub = props_df[(props_df["clip"] == clip) & (props_df["group"] == grp)]
            vals = sub["prop_looking_at_face"].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        offset = (gi - 0.5) * width
        ax.bar(x + offset, means, width, label=grp.capitalize(),
               color=GROUP_COLORS[grp], alpha=0.8)
        ax.errorbar(x + offset, means, yerr=sems, fmt="none",
                    color="black", capsize=3, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([CLIP_LABELS[c] for c in CLIPS])
    ax.set_ylabel("Mean proportion (± SEM)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)

    # ── Panel B: overall ISC by comparison type ───────────────────────────────
    ax = axes[1]
    ax.set_title("B. Overall ISC by Comparison Type (mean ± SD)")

    if isc_summary_df is not None and not isc_summary_df.empty:
        actual = isc_summary_df[isc_summary_df["isc_type"] == "actual"]
        random = isc_summary_df[isc_summary_df["isc_type"] == "random"]
        comparisons = ["adult_adult", "infant_infant", "adult_infant"]
        xi = np.arange(len(comparisons))

        for gi, (isc_type_df, label, ls, alpha) in enumerate([
            (actual, "Actual (within-clip)", "-", 0.8),
            (random, "Random (cross-clip)", BASELINE_LINESTYLE, 0.5),
        ]):
            means_by_comp = []
            sds_by_comp = []
            for comp in comparisons:
                sub = isc_type_df[isc_type_df["comparison"] == comp]
                # Average across clips
                valid = sub["mean_r"].dropna()
                means_by_comp.append(valid.mean() if len(valid) > 0 else np.nan)
                sds_by_comp.append(valid.std() if len(valid) > 1 else 0)

            colors = [COMPARISON_COLORS[c] for c in comparisons]
            offset = (gi - 0.5) * 0.35
            for xi_val, (m, s, c) in enumerate(zip(means_by_comp, sds_by_comp, colors)):
                if np.isnan(m):
                    continue
                bar_alpha = alpha
                ax.bar(xi_val + offset, m, 0.35, color=c, alpha=bar_alpha,
                       label=f"{COMPARISON_LABELS[comparisons[xi_val]]} ({label})" if gi == 0 else "_nolegend_",
                       hatch="///" if gi == 1 else "")
                ax.errorbar(xi_val + offset, m, yerr=s, fmt="none",
                            color="black", capsize=3, linewidth=1.2)

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(np.arange(len(comparisons)))
        ax.set_xticklabels([COMPARISON_LABELS[c] for c in comparisons])
        ax.set_ylabel("Mean Pearson r (avg over clips ± SD)")
        ax.set_ylim(-0.15, 0.35)

        # Custom legend for actual vs random
        legend_handles = [
            mpatches.Patch(facecolor="gray", alpha=0.8, label="Actual (within-clip)"),
            mpatches.Patch(facecolor="gray", alpha=0.5, hatch="///", label="Random (cross-clip)"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ISC time series helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_isc_timeseries_on_ax(ax, isc_ts_df, clip=None, comparison="adult_adult"):
    """
    Plot actual ISC + random baseline over time for one comparison type on `ax`.
    If clip is None, average over all clips.
    """
    color = COMPARISON_COLORS[comparison]
    label = COMPARISON_LABELS[comparison]

    for isc_type, ls, alpha_fill, alpha_line in [
        ("actual", "-", 0.15, 0.9),
        ("random", BASELINE_LINESTYLE, 0.08, 0.6),
    ]:
        if clip is not None:
            sub = isc_ts_df[
                (isc_ts_df["clip"] == clip)
                & (isc_ts_df["comparison"] == comparison)
                & (isc_ts_df["isc_type"] == isc_type)
            ].sort_values("window_start_ms")
        else:
            # Average across clips by window_idx
            sub = (
                isc_ts_df[
                    (isc_ts_df["comparison"] == comparison)
                    & (isc_ts_df["isc_type"] == isc_type)
                ]
                .groupby("window_idx")
                .agg(
                    mean_r=("mean_r", "mean"),
                    sd_r=("sd_r", "mean"),
                    window_start_ms=("window_start_ms", "mean"),
                )
                .reset_index()
                .sort_values("window_start_ms")
            )

        if sub.empty:
            continue

        t = sub["window_start_ms"].values / 1000  # seconds
        r = sub["mean_r"].values
        sd = sub["sd_r"].values

        suffix = "" if isc_type == "actual" else " (random)"
        ax.plot(t, r, color=color, linestyle=ls, linewidth=1.5,
                alpha=alpha_line, label=f"{label}{suffix}")
        ax.fill_between(t, r - sd, r + sd, color=color, alpha=alpha_fill)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: ISC over time per clip
# ─────────────────────────────────────────────────────────────────────────────

def plot_isc_timeseries_per_clip(isc_ts_df: pd.DataFrame, out_dir: str):
    comparisons = ["adult_adult", "infant_infant", "adult_infant"]

    for clip in CLIPS:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"ISC over Time — {CLIP_LABELS[clip].replace(chr(10), ' ')}",
                     fontsize=12, fontweight="bold")

        for ax, comp in zip(axes, comparisons):
            _plot_isc_timeseries_on_ax(ax, isc_ts_df, clip=clip, comparison=comp)
            ax.set_ylabel("Mean Pearson r")
            ax.set_title(COMPARISON_LABELS[comp])
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            ax.set_ylim(-0.4, 0.6)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"isc_timeseries_{clip}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: ISC over time averaged across clips
# ─────────────────────────────────────────────────────────────────────────────

def plot_isc_timeseries_avg(isc_ts_df: pd.DataFrame, out_path: str):
    comparisons = ["adult_adult", "infant_infant", "adult_infant"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("ISC over Time — Averaged Across Clips", fontsize=12, fontweight="bold")

    for ax, comp in zip(axes, comparisons):
        _plot_isc_timeseries_on_ax(ax, isc_ts_df, clip=None, comparison=comp)
        ax.set_ylabel("Mean Pearson r")
        ax.set_title(COMPARISON_LABELS[comp])
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_ylim(-0.4, 0.6)

    axes[-1].set_xlabel("Time from clip start (s)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: All clips ISC over time — one panel per comparison, all clips overlaid
# ─────────────────────────────────────────────────────────────────────────────

def plot_isc_all_clips_overlaid(isc_ts_df: pd.DataFrame, out_path: str):
    """One row per comparison type, each clip as a separate line."""
    comparisons = ["adult_adult", "infant_infant", "adult_infant"]
    clip_colors = ["#1976D2", "#388E3C", "#E65100", "#7B1FA2"]

    fig, axes = plt.subplots(len(comparisons), 1, figsize=(12, 9), sharex=False)
    fig.suptitle("ISC over Time per Clip (actual solid, random dashed)", fontsize=12, fontweight="bold")

    for ax, comp in zip(axes, comparisons):
        ax.set_title(COMPARISON_LABELS[comp])
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

        for clip, color in zip(CLIPS, clip_colors):
            for isc_type, ls, alpha in [("actual", "-", 0.9), ("random", BASELINE_LINESTYLE, 0.5)]:
                sub = isc_ts_df[
                    (isc_ts_df["clip"] == clip)
                    & (isc_ts_df["comparison"] == comp)
                    & (isc_ts_df["isc_type"] == isc_type)
                ].sort_values("window_start_ms")

                if sub.empty:
                    continue
                t = sub["window_start_ms"].values / 1000
                r = sub["mean_r"].values
                lbl = f"{CLIP_LABELS[clip].replace(chr(10), ' ')}" if isc_type == "actual" else "_nolegend_"
                ax.plot(t, r, color=color, linestyle=ls, linewidth=1.4, alpha=alpha, label=lbl)

        ax.set_ylabel("Mean Pearson r")
        ax.set_ylim(-0.5, 0.7)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot face-gaze analysis results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="analysis/results")
    parser.add_argument("--output_dir", default="analysis/results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_results(args.results_dir)

    # ── Plot 1: face proportions ──────────────────────────────────────────────
    if data["props"] is not None and data["baseline"] is not None:
        print("Plotting face proportions...")
        plot_face_proportions(
            data["props"], data["baseline"],
            os.path.join(args.output_dir, "face_proportions.png"),
        )

    # ── Plot 2: face vs metadata ──────────────────────────────────────────────
    if data["props"] is not None and data["meta"] is not None:
        print("Plotting face vs metadata...")
        plot_face_vs_metadata(
            data["props"], data["meta"],
            os.path.join(args.output_dir, "face_vs_metadata.png"),
        )

    # ── Plot 3: summary stats ─────────────────────────────────────────────────
    if data["props"] is not None:
        print("Plotting summary stats...")
        plot_summary_stats(
            data["props"], data["isc_summary"], data["meta"],
            os.path.join(args.output_dir, "summary_stats.png"),
        )

    # ── Plots 4 & 5: ISC over time ────────────────────────────────────────────
    if data["isc_ts"] is not None:
        print("Plotting ISC timeseries per clip...")
        plot_isc_timeseries_per_clip(data["isc_ts"], args.output_dir)

        print("Plotting ISC timeseries averaged...")
        plot_isc_timeseries_avg(
            data["isc_ts"],
            os.path.join(args.output_dir, "isc_timeseries_avg.png"),
        )

        print("Plotting ISC all clips overlaid...")
        plot_isc_all_clips_overlaid(
            data["isc_ts"],
            os.path.join(args.output_dir, "isc_timeseries_all_clips.png"),
        )

    print("\nAll plots saved to", args.output_dir)


if __name__ == "__main__":
    main()

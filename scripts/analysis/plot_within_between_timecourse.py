"""
Plot within-person and between-person cosine distance over time, one clip at a time or
one multi-panel figure per block (one panel per clip in that block).

Reads data/results/within_between_cosine/all_clips_by_segment.csv (output of
calc_within_between_cosine.py) and plots both metrics against segment start time.

Usage:
    python scripts/analysis/plot_within_between_timecourse.py --clip pixar_birds
    python scripts/analysis/plot_within_between_timecourse.py --block frank
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = "data/results/within_between_cosine/all_clips_by_segment.csv"
OUT_DIR = "figures/within_between/timecourse"

WITHIN_COLOR = "#1976D2"
BETWEEN_COLOR = "#E65100"

# Quadrant background shading (pastel, distinct from the within/between line colors above).
# Tertile split (top/bottom third on BOTH axes), computed pooled across all clips' valid
# segments -- a sharper version of the four-quadrant framework in METHODS_DECISIONS.md
# (2026-10-19 entry, which used a median split). Segments in the middle third on either axis
# are left unclassified (unshaded) rather than forced into a quadrant.
QUADRANT_COLORS = {
    "concentrated": "#C8E6C9",  # low within, low between
    "coordinated": "#D1C4E9",   # high within, low between
    "selective": "#FFF9C4",     # low within, high between
    "scattered": "#FFCDD2",     # high within, high between
}
UNCLASSIFIED_COLOR = "#E0E0E0"  # middle band on at least one axis

MARKER_SIZE_SCALE = 12  # points^2 per second of segment duration
TERTILE = 1 / 3


def classify_quadrant(within: float, between: float, w_low: float, w_high: float,
                       b_low: float, b_high: float) -> str | None:
    low_within, high_within = within <= w_low, within >= w_high
    low_between, high_between = between <= b_low, between >= b_high
    if low_within and low_between:
        return "concentrated"
    if high_within and low_between:
        return "coordinated"
    if low_within and high_between:
        return "selective"
    if high_within and high_between:
        return "scattered"
    return None  # middle third on at least one axis -- unclassified


def draw_clip(ax, df: pd.DataFrame, clip: str, thresholds: tuple, shade_unclassified: bool = False):
    w_low, w_high, b_low, b_high = thresholds
    sub = df[df["video"] == clip].sort_values("seg_start_s")

    # Background shading: one band per segment, colored by which quadrant that segment falls in.
    # Unclassified (middle band on at least one axis) segments are shaded grey if
    # shade_unclassified, otherwise left unshaded.
    for _, row in sub.iterrows():
        if pd.isna(row["within_person_cosine_dist"]) or pd.isna(row["between_person_cosine_dist"]):
            continue
        quad = classify_quadrant(row["within_person_cosine_dist"], row["between_person_cosine_dist"],
                                  w_low, w_high, b_low, b_high)
        if quad is None:
            if not shade_unclassified:
                continue
            color = UNCLASSIFIED_COLOR
        else:
            color = QUADRANT_COLORS[quad]
        ax.axvspan(row["seg_start_s"], row["seg_end_s"], color=color, alpha=0.6, lw=0)

    # Lines (NaN segments create natural gaps)
    ax.plot(sub["seg_start_s"], sub["within_person_cosine_dist"], "-",
            color=WITHIN_COLOR, label="within-person", linewidth=1.5, zorder=3)
    ax.plot(sub["seg_start_s"], sub["between_person_cosine_dist"], "-",
            color=BETWEEN_COLOR, label="between-person", linewidth=1.5, zorder=3)

    # Markers, sized by segment duration
    ax.scatter(sub["seg_start_s"], sub["within_person_cosine_dist"],
               s=sub["seg_dur_s"] * MARKER_SIZE_SCALE, color=WITHIN_COLOR, zorder=4)
    ax.scatter(sub["seg_start_s"], sub["between_person_cosine_dist"],
               s=sub["seg_dur_s"] * MARKER_SIZE_SCALE, color=BETWEEN_COLOR, zorder=4)

    ax.set_xlabel("Time (s, segment start)")
    ax.set_title(clip)
    ax.spines[["top", "right"]].set_visible(False)


def pooled_quantile_split(df: pd.DataFrame, q: float = TERTILE) -> tuple:
    valid = df.dropna(subset=["within_person_cosine_dist", "between_person_cosine_dist"])
    w_low, w_high = valid["within_person_cosine_dist"].quantile([q, 1 - q])
    b_low, b_high = valid["between_person_cosine_dist"].quantile([q, 1 - q])
    return w_low, w_high, b_low, b_high


def shared_legend(fig, ax, show_unclassified: bool = False):
    line_handles, line_labels = ax.get_legend_handles_labels()
    quad_handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in QUADRANT_COLORS.values()]
    labels = list(QUADRANT_COLORS.keys())
    if show_unclassified:
        quad_handles.append(plt.Rectangle((0, 0), 1, 1, color=UNCLASSIFIED_COLOR, alpha=0.6))
        labels.append("unclassified")
    fig.legend(line_handles + quad_handles, line_labels + labels,
               frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.02))


def plot_clip(df: pd.DataFrame, clip: str, out_dir: str, q: float = TERTILE,
              shade_unclassified: bool = False):
    thresholds = pooled_quantile_split(df, q)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    draw_clip(ax, df, clip, thresholds, shade_unclassified=shade_unclassified)
    ax.set_ylabel("Cosine distance")
    shared_legend(fig, ax, show_unclassified=shade_unclassified)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = f"{out_dir}/{clip}_within_between_timecourse.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_block(df: pd.DataFrame, block: str, out_dir: str, med_df: pd.DataFrame = None,
               title_suffix: str = "", out_suffix: str = "", q: float = TERTILE,
               shade_unclassified: bool = False):
    """med_df: dataframe to compute quadrant threshold from (defaults to df itself).
    Pass the canonical all_clips_by_segment.csv here when plotting an alternate segmentation
    (e.g. artificially chopped), so quadrant boundaries stay comparable across versions."""
    thresholds = pooled_quantile_split(med_df if med_df is not None else df, q)

    clips = df.loc[df["block"] == block, "video"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(clips), 1, figsize=(14, 3.5 * len(clips)), sharex=True, sharey=True)
    if len(clips) == 1:
        axes = [axes]

    for ax, clip in zip(axes, clips):
        draw_clip(ax, df, clip, thresholds, shade_unclassified=shade_unclassified)
        ax.set_ylabel("Cosine distance")
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time (s, segment start)")

    fig.suptitle(f"{block} block{title_suffix}", y=1.0)
    shared_legend(fig, axes[0], show_unclassified=shade_unclassified)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = f"{out_dir}/{block}_within_between_timecourse{out_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clip")
    group.add_argument("--block")
    args = parser.parse_args()

    df = pd.read_csv(RESULTS_PATH)
    if args.clip:
        plot_clip(df, args.clip, OUT_DIR)
    else:
        plot_block(df, args.block, OUT_DIR)


if __name__ == "__main__":
    main()

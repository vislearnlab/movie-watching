"""
Scatter plot of between-person vs. within-person cosine distance, one point per segment,
with the four-quadrant regions shaded and labeled. Point size = segment duration.

Quadrant boundaries use a configurable low/high percentile split (pooled across all 12
clips' valid segments, same convention as plot_within_between_timecourse.py and
plot_quadrant_distribution.py). Segments in the middle band on either axis fall in an
unshaded/grey cross between the four quadrants rather than being forced into one.

Usage:
    python scripts/analysis/plot_quadrant_scatter.py --clip pixar_birds --q 0.4
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from plot_within_between_timecourse import (
    QUADRANT_COLORS, UNCLASSIFIED_COLOR, classify_quadrant, pooled_quantile_split,
)

RESULTS_PATH = "data/results/within_between_cosine/all_clips_by_segment.csv"
OUT_DIR = "figures/within_between"

POINT_COLOR = "#5C6BC0"
MARKER_SIZE_SCALE = 12  # points^2 per second of segment duration, matches the timecourse script
UNCLASSIFIED_POINT_ALPHA = 0.15
CLASSIFIED_POINT_ALPHA = 0.85


def split_classified(sub: pd.DataFrame, thresholds: tuple) -> tuple:
    """Split a dataframe into (classified, unclassified) rows based on quadrant membership."""
    w_low, w_high, b_low, b_high = thresholds
    quad = sub.apply(lambda r: classify_quadrant(r["within_person_cosine_dist"],
                                                   r["between_person_cosine_dist"],
                                                   w_low, w_high, b_low, b_high), axis=1)
    return sub[quad.notna()], sub[quad.isna()]

BLOCK_ORDER = ["frank", "pixar", "sesame", "slow"]
BLOCK_CLIP_COLORS = {
    "frank": {"frank_complex": "#1976D2", "frank_objects": "#90CAF9", "frank_play": "#FB8C00"},
    "pixar": {"pixar_birds": "#FFB74D"},
    "sesame": {"sesameindia_1": "#2E7D32", "sesameindia_2": "#81C784",
               "sesameus_1": "#C62828", "sesameus_2": "#EF9A9A"},
    "slow": {"slow_animals": "#7E57C2", "slow_hands": "#D1C4E9",
             "slow_inanimate": "#6D4C41", "slow_people": "#BCAAA4"},
}


def draw_quadrant_grid(ax, thresholds: tuple, xlim: tuple, ylim: tuple):
    w_low, w_high, b_low, b_high = thresholds
    col_edges = [xlim[0], w_low, w_high, xlim[1]]
    row_edges = [ylim[0], b_low, b_high, ylim[1]]

    grid_colors = [
        [QUADRANT_COLORS["concentrated"], UNCLASSIFIED_COLOR, QUADRANT_COLORS["coordinated"]],
        [UNCLASSIFIED_COLOR, UNCLASSIFIED_COLOR, UNCLASSIFIED_COLOR],
        [QUADRANT_COLORS["selective"], UNCLASSIFIED_COLOR, QUADRANT_COLORS["scattered"]],
    ]

    for row in range(3):
        for col in range(3):
            x0, x1 = col_edges[col], col_edges[col + 1]
            y0, y1 = row_edges[row], row_edges[row + 1]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    color=grid_colors[row][col], alpha=0.6, lw=0, zorder=0))


def plot_clip_scatter(df: pd.DataFrame, clip: str, thresholds: tuple, out_dir: str, q: float):
    sub = df[(df["video"] == clip)].dropna(
        subset=["within_person_cosine_dist", "between_person_cosine_dist"])

    fig, ax = plt.subplots(figsize=(8, 7))

    pad_x = (sub["within_person_cosine_dist"].max() - sub["within_person_cosine_dist"].min()) * 0.08
    pad_y = (sub["between_person_cosine_dist"].max() - sub["between_person_cosine_dist"].min()) * 0.08
    xlim = (sub["within_person_cosine_dist"].min() - pad_x, sub["within_person_cosine_dist"].max() + pad_x)
    ylim = (sub["between_person_cosine_dist"].min() - pad_y, sub["between_person_cosine_dist"].max() + pad_y)

    draw_quadrant_grid(ax, thresholds, xlim, ylim)
    classified, unclassified = split_classified(sub, thresholds)
    ax.scatter(unclassified["within_person_cosine_dist"], unclassified["between_person_cosine_dist"],
               s=unclassified["seg_dur_s"] * MARKER_SIZE_SCALE, color=POINT_COLOR,
               alpha=UNCLASSIFIED_POINT_ALPHA, edgecolor="none", zorder=2)
    ax.scatter(classified["within_person_cosine_dist"], classified["between_person_cosine_dist"],
               s=classified["seg_dur_s"] * MARKER_SIZE_SCALE, color=POINT_COLOR,
               alpha=CLASSIFIED_POINT_ALPHA, edgecolor="white", linewidth=0.5, zorder=3)

    w_low, w_high, b_low, b_high = thresholds
    label_kwargs = dict(fontsize=11, style="italic", color="#555555")
    ax.text(xlim[0] + pad_x * 0.3, b_high + pad_y * 0.3, "SELECTIVE", **label_kwargs)
    ax.text(w_high + pad_x * 0.3, b_high + pad_y * 0.3, "SCATTERED", **label_kwargs)
    ax.text(xlim[0] + pad_x * 0.3, ylim[0] + pad_y * 0.3, "CONCENTRATED", **label_kwargs)
    ax.text(w_high + pad_x * 0.3, ylim[0] + pad_y * 0.3, "COORDINATED", **label_kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("within-person cosine distance (own windows vs. own mean)")
    ax.set_ylabel("between-person cosine distance (own mean vs. leave-one-out group mean)")
    pct_label = f"{int(q * 100)}th/{int((1 - q) * 100)}th percentile split"
    ax.set_title(f"{clip}: within vs. between, per segment (n={len(sub)}, size = duration)\n{pct_label}",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = f"{out_dir}/{clip}_within_vs_between_scatter_q{int(q*100)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def draw_block_panel(ax, sub: pd.DataFrame, block: str, thresholds: tuple,
                      xlim: tuple, ylim: tuple, pad_x: float, pad_y: float):
    w_low, w_high, b_low, b_high = thresholds
    label_kwargs = dict(fontsize=10, style="italic", color="#555555")

    draw_quadrant_grid(ax, thresholds, xlim, ylim)

    for clip, color in BLOCK_CLIP_COLORS[block].items():
        clip_sub = sub[sub["video"] == clip]
        classified, unclassified = split_classified(clip_sub, thresholds)
        ax.scatter(unclassified["within_person_cosine_dist"], unclassified["between_person_cosine_dist"],
                   s=unclassified["seg_dur_s"] * MARKER_SIZE_SCALE, color=color,
                   alpha=UNCLASSIFIED_POINT_ALPHA, edgecolor="none", zorder=2)
        ax.scatter(classified["within_person_cosine_dist"], classified["between_person_cosine_dist"],
                   s=classified["seg_dur_s"] * MARKER_SIZE_SCALE, color=color,
                   alpha=CLASSIFIED_POINT_ALPHA, edgecolor="white", linewidth=0.5, zorder=3, label=clip)

    ax.text(xlim[0] + pad_x, b_high + pad_y, "SELECTIVE", **label_kwargs)
    ax.text(w_high + pad_x, b_high + pad_y, "SCATTERED", **label_kwargs)
    ax.text(xlim[0] + pad_x, ylim[0] + pad_y, "CONCENTRATED", **label_kwargs)
    ax.text(w_high + pad_x, ylim[0] + pad_y, "COORDINATED", **label_kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("within-person cosine distance")
    ax.set_ylabel("between-person cosine distance")
    ax.set_title(f"{block} (n={len(sub)})")
    ax.legend(frameon=False, loc="upper left", fontsize=9, markerscale=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def shared_xlim_ylim(valid: pd.DataFrame) -> tuple:
    pad_x = (valid["within_person_cosine_dist"].max() - valid["within_person_cosine_dist"].min()) * 0.05
    pad_y = (valid["between_person_cosine_dist"].max() - valid["between_person_cosine_dist"].min()) * 0.05
    xlim = (valid["within_person_cosine_dist"].min() - pad_x, valid["within_person_cosine_dist"].max() + pad_x)
    ylim = (valid["between_person_cosine_dist"].min() - pad_y, valid["between_person_cosine_dist"].max() + pad_y)
    return xlim, ylim, pad_x, pad_y


def plot_byblock_grid(df: pd.DataFrame, thresholds: tuple, out_dir: str, q: float):
    valid = df.dropna(subset=["within_person_cosine_dist", "between_person_cosine_dist"])
    xlim, ylim, pad_x, pad_y = shared_xlim_ylim(valid)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, block in zip(axes.flat, BLOCK_ORDER):
        sub = valid[valid["block"] == block]
        draw_block_panel(ax, sub, block, thresholds, xlim, ylim, pad_x, pad_y)

    pct_label = f"{int(q * 100)}th/{int((1 - q) * 100)}th percentile split"
    fig.suptitle(f"within vs. between, split by block -- color = clip, size = segment duration "
                 f"({pct_label}, unclassified in grey)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = f"{out_dir}/all_clips_within_vs_between_byblock_q{int(q*100)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_single_block(df_block: pd.DataFrame, block: str, thresholds: tuple, xlim_ref: pd.DataFrame,
                       out_dir: str, q: float, title_suffix: str = "", out_suffix: str = ""):
    """df_block: data to plot (e.g. an artificially chopped version). xlim_ref: dataframe to
    derive shared axis limits from (pass the canonical unchopped data for comparability)."""
    sub = df_block.dropna(subset=["within_person_cosine_dist", "between_person_cosine_dist"])
    xlim, ylim, pad_x, pad_y = shared_xlim_ylim(
        xlim_ref.dropna(subset=["within_person_cosine_dist", "between_person_cosine_dist"]))

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_block_panel(ax, sub, block, thresholds, xlim, ylim, pad_x, pad_y)

    pct_label = f"{int(q * 100)}th/{int((1 - q) * 100)}th percentile split"
    ax.set_title(f"{block} (n={len(sub)}){title_suffix}\n"
                 f"color = clip, size = segment duration ({pct_label}, unclassified in grey)",
                 fontsize=10)
    fig.tight_layout()
    out_path = f"{out_dir}/{block}_within_vs_between_scatter_q{int(q*100)}{out_suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clip")
    group.add_argument("--byblock", action="store_true")
    parser.add_argument("--q", type=float, default=0.4, help="low/high percentile split (0.5 = median)")
    args = parser.parse_args()

    df = pd.read_csv(RESULTS_PATH)
    thresholds = pooled_quantile_split(df, args.q)
    if args.clip:
        plot_clip_scatter(df, args.clip, thresholds, OUT_DIR, args.q)
    else:
        plot_byblock_grid(df, thresholds, OUT_DIR, args.q)


if __name__ == "__main__":
    main()

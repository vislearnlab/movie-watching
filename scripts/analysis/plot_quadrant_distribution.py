"""
Grouped bar chart of the four-quadrant distribution (concentrated/coordinated/selective/
scattered) by block, median-split vs. top/bottom-third ("strict") side by side.

Reads data/results/within_between_cosine/all_clips_by_segment.csv and classifies every
valid segment using both threshold schemes from plot_within_between_timecourse.py, pooled
across all 12 clips, then aggregates to % of segments per block.

Usage:
    python scripts/analysis/plot_quadrant_distribution.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_within_between_timecourse import QUADRANT_COLORS, classify_quadrant

RESULTS_PATH = "data/results/within_between_cosine/all_clips_by_segment.csv"
OUT_PATH = "figures/within_between/quadrant_distribution_by_block.png"

BLOCK_ORDER = ["frank", "pixar", "sesame", "slow"]
UNCLASSIFIED_COLOR = "#E0E0E0"
SPLIT_QUANTILE = 0.4  # low/high cut at the 40th/60th percentile (pass 1/3 for tertiles)


def pooled_quantile_split(df: pd.DataFrame, q: float) -> tuple:
    valid = df.dropna(subset=["within_person_cosine_dist", "between_person_cosine_dist"])
    w_low, w_high = valid["within_person_cosine_dist"].quantile([q, 1 - q])
    b_low, b_high = valid["between_person_cosine_dist"].quantile([q, 1 - q])
    return w_low, w_high, b_low, b_high


def block_quadrant_pcts(valid: pd.DataFrame, thresholds: tuple, categories: list) -> pd.DataFrame:
    w_low, w_high, b_low, b_high = thresholds
    quad = valid.apply(lambda r: classify_quadrant(r["within_person_cosine_dist"],
                                                     r["between_person_cosine_dist"],
                                                     w_low, w_high, b_low, b_high), axis=1)
    quad = quad.fillna("unclassified")
    ct = pd.crosstab(valid["block"], quad, normalize="index") * 100
    for cat in categories:
        if cat not in ct.columns:
            ct[cat] = 0.0
    return ct.loc[BLOCK_ORDER, categories]


def draw_panel(ax, pcts: pd.DataFrame, categories: list, title: str):
    n_blocks = len(pcts.index)
    n_cats = len(categories)
    bar_w = 0.8 / n_cats
    x = np.arange(n_blocks)

    for i, cat in enumerate(categories):
        color = QUADRANT_COLORS.get(cat, UNCLASSIFIED_COLOR)
        offset = (i - (n_cats - 1) / 2) * bar_w
        ax.bar(x + offset, pcts[cat].values, width=bar_w, color=color,
               edgecolor="white", linewidth=0.5, label=cat)

    ax.set_xticks(x)
    ax.set_xticklabels(pcts.index)
    ax.set_ylabel("% of segments")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(70, pcts.values.max() * 1.15))


def main():
    df = pd.read_csv(RESULTS_PATH)
    valid = df.dropna(subset=["within_person_cosine_dist", "between_person_cosine_dist"]).copy()

    med_within = valid["within_person_cosine_dist"].median()
    med_between = valid["between_person_cosine_dist"].median()
    median_thresholds = (med_within, med_within, med_between, med_between)
    median_categories = ["concentrated", "coordinated", "selective", "scattered"]
    median_pcts = block_quadrant_pcts(valid, median_thresholds, median_categories)

    split_thresholds = pooled_quantile_split(df, SPLIT_QUANTILE)
    split_categories = ["concentrated", "coordinated", "selective", "scattered", "unclassified"]
    split_pcts = block_quadrant_pcts(valid, split_thresholds, split_categories)

    pct_label = f"{int(SPLIT_QUANTILE * 100)}th/{int((1 - SPLIT_QUANTILE) * 100)}th percentile"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    draw_panel(axes[0], median_pcts, median_categories, "Median split")
    draw_panel(axes[1], split_pcts, split_categories, pct_label)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Quadrant distribution by block", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

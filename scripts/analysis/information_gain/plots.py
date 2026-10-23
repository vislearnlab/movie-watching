"""
Figures: headline partition (4 bars + stacked bar vs. ceiling), supplementary model bars, fitted
nonlinearities (check), per-segment ceiling vs. entropy, per-segment ceiling vs. proportion explained.

Default (unchopped, 20 knots, floor) is the only configuration plotted routinely; pass other args as needed.

Usage: python scripts/analysis/information_gain/plots.py [--segment-set cuts] [--variant knots_floor]
"""

import argparse
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from feature_models import KNOTS, VARIANTS
from common import group_average
from partition import partition
from scoring import MODELS

COLORS = {"saliency": "#2a78d6", "motion": "#eb6834", "meaning": "#1baf7a", "shared": "#8a8984"}
INK, INK2 = "#0b0b0b", "#52514e"
BLOCK_COLORS = {"frank": "#eda100", "pixar": "#e87ba4", "sesame": "#008300", "slow": "#6250d6"}  # palette slots 4-7, distinct from features
WB_PATH = C.PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
WB_SLOW_CHOPPED_PATH = WB_PATH.parent / "slow_chopped_by_subsegment.csv"
QUADRANT_Q = 0.4  # plot_quadrant_scatter.py default
QUADRANT_TITLE_COLORS = {"coordinated": INK, "concentrated": INK, "selective": INK, "scattered": INK}  # source pastels unreadable as text
ENTROPY_PATH = C.PROJECT_ROOT / "data" / "results" / "entropy_by_segment" / "entropy_by_segment.csv"


def quadrants(segment_set: str = "cuts") -> pd.DataFrame:
    # Same classification as plot_quadrant_scatter.py (pooled quantile split, middle band unclassified).
    # Thresholds always from the unchopped segments; slow_chopped swaps in slow sub-segments (as montage_slow_chopped_selective.py).
    w = pd.read_csv(WB_PATH)
    wl, wh = w.within_person_cosine_dist.quantile([QUADRANT_Q, 1 - QUADRANT_Q])
    bl, bh = w.between_person_cosine_dist.quantile([QUADRANT_Q, 1 - QUADRANT_Q])
    if segment_set == "slow_chopped":
        w = pd.concat([w[w.block != "slow"], pd.read_csv(WB_SLOW_CHOPPED_PATH)], ignore_index=True)
    lo_w, hi_w = w.within_person_cosine_dist <= wl, w.within_person_cosine_dist >= wh
    lo_b, hi_b = w.between_person_cosine_dist <= bl, w.between_person_cosine_dist >= bh
    w["quadrant"] = np.select([lo_w & lo_b, hi_w & lo_b, lo_w & hi_b, hi_w & hi_b],
                              ["concentrated", "coordinated", "selective", "scattered"], None)
    return w[["video", "seg_idx", "quadrant"]]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.yaxis.label.set_color(INK2)


def headline(s: pd.Series, tag: str, caption: str, points: pd.DataFrame | None = None):
    # y = proportion of the ceiling (ratio of group averages). With points: violins of the per-unit
    # values, group value marked as a black line; without: plain bars.
    c = s.ceiling
    cols = ["shared", "unique_motion", "unique_saliency", "unique_meaning"]
    bars = [("shared", s.shared / c, "shared"), ("unique\nmotion", s.unique_motion / c, "motion"),
            ("unique\nsaliency", s.unique_saliency / c, "saliency"), ("unique\nmeaning", s.unique_meaning / c, "meaning")]
    total = s.ig_SMX / c
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.8), gridspec_kw=dict(width_ratios=[4, 1.4]))
    lo, hi = min(0.0, *(v for _, v, _ in bars)), max(total, *(v for _, v, _ in bars))
    if points is None:
        for i, (lab, v, key) in enumerate(bars):
            ax.bar(i, v, width=0.62, color=COLORS[key])
    else:
        rng = np.random.default_rng(0)
        for i, (col, (lab, v, key)) in enumerate(zip(cols, bars)):
            y = points[col].to_numpy() / c
            vp = ax.violinplot(y, positions=[i], widths=0.7, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(COLORS[key]); body.set_edgecolor(COLORS[key]); body.set_alpha(0.55)
            ax.scatter(i + rng.uniform(-0.06, 0.06, len(y)), y, s=5, color=INK, alpha=0.3, linewidths=0, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [v, v], color=INK, lw=2, zorder=4)
            lo, hi = min(lo, y.min()), max(hi, y.max())
    ax.set_xticks(range(4), [f"{b[0]}\n{b[1]:+.1%}" for b in bars])
    # Stacked: positive pieces upward from 0, negative pieces downward, total marked.
    pos, neg = 0.0, 0.0
    for lab, v, key in bars:
        base = pos if v >= 0 else neg
        ax2.bar(0, v, bottom=base, width=0.5, color=COLORS[key], edgecolor="white", linewidth=1.5)
        pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
    ax2.plot([-0.3, 0.3], [total] * 2, color=INK, lw=1.5)
    ax2.text(0.32, total, f"all three\n{total:.1%}", va="center", fontsize=8, color=INK)
    ax2.set_xticks([0], ["stacked"])
    ax2.set_xlim(-0.6, 1.0)
    hi = max(hi, pos)
    pad = 0.05 * (hi - min(lo, neg))
    for a in (ax, ax2):
        a.axhline(0, color=INK, lw=0.8)
        a.set_ylim(min(lo, neg) - pad, hi + pad)
        if hi + pad >= 1:
            a.axhline(1, color=INK2, ls="--", lw=1)
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        style(a)
    ax.set_ylabel(f"information gain, proportion of ceiling\n(100% = ceiling = {c:.2f} bits)")
    fig.suptitle(f"Overall IG\n{tag}", fontsize=11)
    fig.tight_layout()
    fig.text(0.01, -0.06, caption, fontsize=7.5, color=INK2, wrap=True)
    return fig


def headline_by_group(df: pd.DataFrame, by: list, tag: str, caption: str, group: str, colors: dict, ncols: int | None = None):
    # One panel per group (block or quadrant); y = proportion of that group's ceiling (ratio of group averages).
    cols = ["shared", "unique_motion", "unique_saliency", "unique_meaning"]
    labs = [("shared", "shared"), ("unique\nmotion", "motion"), ("unique\nsaliency", "saliency"), ("unique\nmeaning", "meaning")]
    blocks = [b for b in colors if b in set(df[group])]
    ncols = ncols or len(blocks)
    nrows = -(-len(blocks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 4.6 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for ax, b in zip(axes, blocks):
        d = df[df[group] == b].dropna(subset=[f"ig_{m}" for m in MODELS])
        ig = {m: group_average(d, f"ig_{m}") for m in MODELS}
        c, part = group_average(d, "ig_ceiling"), partition(ig)
        pts = unit_partition(d, by)
        for i, (col, (lab, key)) in enumerate(zip(cols, labs)):
            y = pts[col].to_numpy() / c
            vp = ax.violinplot(y, positions=[i], widths=0.7, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(COLORS[key]); body.set_edgecolor(COLORS[key]); body.set_alpha(0.55)
            ax.scatter(i + rng.uniform(-0.06, 0.06, len(y)), y, s=5, color=INK, alpha=0.3, linewidths=0, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [part[col] / c] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(4), [f"{lab}\n{part[col] / c:+.1%}" for col, (lab, _) in zip(cols, labs)], fontsize=8)
        ax.set_title(f"{b}: {len(pts)} {'segments' if 'seg_idx' in by else 'participants'}\n"
                     f"ceiling {c:.2f} bits, all three {ig['SMX'] / c:.1%}", fontsize=9, color=colors[b])
        ax.axhline(0, color=INK, lw=0.8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        style(ax)
    for ax in axes[::ncols]:
        ax.set_ylabel(f"information gain, proportion of {group}'s ceiling")
    fig.suptitle(f"Overall IG by {group}\n{tag}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.03 * (nrows > 1)))
    fig.text(0.01, -0.09 / nrows, caption, fontsize=7.5, color=INK2, wrap=True)
    return fig


def supplementary(s: pd.Series, tag: str):
    models = [("saliency", "ig_S", "saliency"), ("motion", "ig_M", "motion"), ("meaning", "ig_X", "meaning"),
              ("all three", "ig_SMX", "shared")]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for i, (lab, col, key) in enumerate(models):
        ax.bar(i, s[col], width=0.62, color=COLORS[key])
        ax.text(i, s[col] + 0.01, f"{s[col]:.3f}\n({s[col] / s.ceiling:.0%})", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(s.ceiling, color=INK2, ls="--", lw=1)
    ax.text(3.4, s.ceiling, f"ceiling {s.ceiling:.2f}", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(4), [m[0] for m in models])
    ax.set_ylabel("information gain over floor (bits)")
    ax.set_title(f"Single-feature and all-three models (% of ceiling), {tag}", fontsize=10)
    style(ax)
    fig.tight_layout()
    return fig


def curves_fig(curves: pd.DataFrame, tag: str):
    kcols = [c for c in curves.columns if c.startswith("k")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, feat in zip(axes, C.FEATURES):
        for r in curves[curves.feature == feat].itertuples():
            ax.plot(KNOTS, [getattr(r, k) for k in kcols], color=COLORS[feat], alpha=0.5, lw=1.2)
        ax.set_yscale("log")
        ax.set_title(f"{feat} (one line per held-out clip)", fontsize=10)
        ax.set_xlabel("feature value (rescaled 0-1 across segments)")
        style(ax)
    axes[0].set_ylabel("f(x) relative to f(0), log scale")
    fig.suptitle(f"Fitted nonlinearities, {tag}", fontsize=11)
    fig.tight_layout()
    return fig


def unit_partition(df: pd.DataFrame, by: list) -> pd.DataFrame:
    """Partition per unit (segment or participant) from mean IGs within the unit (bits)."""
    cols = [f"ig_{m}" for m in MODELS]
    seg = df.dropna(subset=cols).groupby(by)[cols + ["ig_ceiling"]].mean()
    parts = seg.apply(lambda r: pd.Series(partition({m: r[f"ig_{m}"] for m in MODELS})), axis=1)
    return seg.join(parts).reset_index()


def per_segment(df: pd.DataFrame) -> pd.DataFrame:
    seg = df.dropna(subset=["ig_SMX"]).groupby(["video", "seg_idx"]).agg(
        ceiling=("ig_ceiling", "mean"), ig_SMX=("ig_SMX", "mean"), n=("participant", "size")).reset_index()
    seg["prop_SMX"] = seg.ig_SMX / seg.ceiling
    seg["block"] = seg.video.map(C.BLOCKS)
    return seg


def scatter(seg, x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for b, g in seg.groupby("block"):
        ax.scatter(g[x], g[y], s=18, alpha=0.8, color=BLOCK_COLORS[b], label=b, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segment-set", default="cuts", choices=C.SEGMENT_SETS)
    p.add_argument("--variant", default="knots_floor", choices=list(VARIANTS))
    p.add_argument("--only", default="", help="save only figures whose name contains this")
    a = p.parse_args()
    tag = {"cuts": "unchopped", "slow_chopped": "slow chopped"}[a.segment_set] + ", " + \
        {"knots_floor": "20 knots, floor", "exp_floor": "exp, floor", "knots_nofloor": "20 knots, no floor"}[a.variant]
    if C.GROUP != "adults":
        tag += f"; {C.GROUP.upper()}, {C.GROUP} reference"
    ftag = f"{a.segment_set}_{a.variant}"
    s = pd.read_csv(C.RESULTS_DIR / f"summary_{ftag}.csv", index_col=0)["value"]
    df = pd.read_parquet(C.CACHE_DIR / f"scores_{ftag}.parquet")
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    floor_txt = ("each feature model includes the floor (beta=1)" if VARIANTS[a.variant][1] == 1
                 else "feature models do NOT include the floor (beta=0); scored against the floor")
    caption = ("Combined models are fixed equal-weight mixtures of the single-feature models; "
               f"{floor_txt}. Unique = all-three minus the two-feature mixture without that feature; "
               f"shared = all-three minus the uniques. {int(s.n_participants)} {C.GROUP}, {int(s.n_segments)} segments, "
               "leave-one-clip-out. No confidence intervals yet.")
    segp = unit_partition(df, ["video", "seg_idx"])
    pplp = unit_partition(df, ["participant"])
    segp.to_csv(C.RESULTS_DIR / f"segment_partition_{ftag}.csv", index=False)
    pplp.to_csv(C.RESULTS_DIR / f"participant_partition_{ftag}.csv", index=False)
    seg_cap = caption + (" Violins/dots: one per segment (participant-mean within segment, divided by the overall ceiling). "
                         "Black line = group value (participants weighted equally), so it can differ slightly from the segment mean.")
    ppl_cap = caption + (" Violins/dots: one per participant (mean over that participant's segments, divided by the overall "
                         "ceiling). Black line = group value = mean of the dots.")
    figs = {f"overall_ig_{ftag}": headline(s, tag, seg_cap, segp),
            f"overall_ig_people_{ftag}": headline(s, tag + " (dots = participants)", ppl_cap, pplp), f"supplementary_{ftag}": supplementary(s, tag)}
    blk_cap = caption.replace("No confidence intervals yet.", "No confidence intervals yet. Each panel is scaled to its own block's ceiling (participants weighted equally within block); black line = block group value.")
    dfb = df.assign(block=df.video.map(C.BLOCKS))
    quad = quadrants(a.segment_set)
    dfq = df.merge(quad, on=["video", "seg_idx"])
    q_cap = blk_cap.replace("block", "quadrant") + (f" Quadrants from the within/between cosine metric (adult gaze), pooled {QUADRANT_Q:.0%}/{1 - QUADRANT_Q:.0%} "
                                                    f"split on each axis; {int(quad.quadrant.isna().sum())} middle-band segments excluded.")
    figs[f"overall_ig_by_quadrant_{ftag}"] = headline_by_group(dfq, ["video", "seg_idx"], tag + " (dots = segments)", q_cap, "quadrant", QUADRANT_TITLE_COLORS)
    figs[f"overall_ig_people_by_quadrant_{ftag}"] = headline_by_group(dfq, ["participant"], tag + " (dots = participants, mean within quadrant)", q_cap, "quadrant", QUADRANT_TITLE_COLORS)
    clip_colors = {v: BLOCK_COLORS[C.BLOCKS[v]] for v in C.CLIPS}
    clip_cap = blk_cap.replace("block", "clip")
    figs[f"overall_ig_by_clip_{ftag}"] = headline_by_group(df.assign(clip=df.video), ["video", "seg_idx"], tag + " (dots = segments)", clip_cap, "clip", clip_colors, ncols=3)
    figs[f"overall_ig_people_by_clip_{ftag}"] = headline_by_group(df.assign(clip=df.video), ["participant"], tag + " (dots = participants, mean within clip)", clip_cap, "clip", clip_colors, ncols=3)
    figs[f"overall_ig_by_block_{ftag}"] = headline_by_group(dfb, ["video", "seg_idx"], tag + " (dots = segments)", blk_cap, "block", BLOCK_COLORS)
    figs[f"overall_ig_people_by_block_{ftag}"] = headline_by_group(dfb, ["participant"], tag + " (dots = participants, mean within block)", blk_cap, "block", BLOCK_COLORS)
    checks = {f"information_gain_curves_{ftag}": curves_fig(pd.read_parquet(C.CACHE_DIR / f"curves_{ftag}.parquet"), tag)}

    seg = per_segment(df)
    figs[f"ceiling_vs_prop_explained_{ftag}"] = scatter(
        seg, "ceiling", "prop_SMX", "ceiling (bits)", "proportion explained by all three",
        f"Per segment: ceiling vs. proportion explained, {tag}")
    if a.segment_set == "cuts" and ENTROPY_PATH.exists():
        ent = pd.read_csv(ENTROPY_PATH).rename(columns={"segment_idx": "seg_idx"})
        m = seg.merge(ent[["video", "seg_idx", "mean_entropy_rarefied"]], on=["video", "seg_idx"]).dropna()
        figs[f"ceiling_vs_entropy_{ftag}"] = scatter(
            m, "mean_entropy_rarefied", "ceiling", "group entropy (bits, rarefied, smoothed maps)", "ceiling (bits)",
            f"Per segment: entropy vs. ceiling (n={len(m)}), r={np.corrcoef(m.mean_entropy_rarefied, m.ceiling)[0, 1]:.2f}")
    seg.to_csv(C.RESULTS_DIR / f"per_segment_{ftag}.csv", index=False)
    for folder, group in [(C.FIG_DIR, figs), (C.CHECK_FIG_DIR, checks)]:
        for name, fig in group.items():
            if not re.search(a.only, name):
                plt.close(fig)
                continue
            fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {folder / name}.png")


if __name__ == "__main__":
    main()

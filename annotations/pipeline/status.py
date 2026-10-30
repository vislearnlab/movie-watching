"""
Accuracy dashboard: how good each video's annotations are and how often each shot has been run.

Inputs: review scores in qc/reviews/*.csv (video, shot, iteration, score, issues; one row per
reviewed shot and pass) and the shard stamps written by annotate_videos.py.

  accuracy(video, pass) = Σ_shots len(shot) · score(shot) / Σ_shots len(shot) over reviewed shots,
  where score is the shot's review at that pass ("latest" = its highest reviewed iteration).

Outputs qc/status.png, qc/status.csv and a terminal table.

Usage:
    python pipeline/status.py [--watch 30]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from annotate_videos import iteration_of, shard_base
from plan import load_shots

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"
ERROR_RAMP = ["#e6f0fd", "#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # error 0 → 1
UNREVIEWED = "#d9d8d4"


def shots_of(stem) -> list[tuple[int, int]]:
    cap = cv2.VideoCapture(str(C.video_path(stem)))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return load_shots(stem, fps, n)


def load_reviews() -> pd.DataFrame:
    files = sorted((C.QC_DIR / "reviews").glob("*.csv"))
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert {"video", "shot", "iteration", "score"} <= set(df.columns)
    return df.sort_values("iteration").drop_duplicates(["video", "shot", "iteration"], keep="last")


def table(shots: dict, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-shot rows, per-video summary)."""
    rows = []
    for stem, sh in shots.items():
        for k, (s, e) in enumerate(sh):
            r = reviews[(reviews.video == stem) & (reviews.shot == k)]
            first = r[r.iteration == 1].score
            rows.append(dict(video=stem, shot=k, frames=e - s, iteration=iteration_of(shard_base(stem, k, ["face", "hand"])),
                             reviewed_iteration=r.iteration.max() if len(r) else np.nan,
                             score_pass1=first.iloc[0] if len(first) else np.nan,
                             score=r.score.iloc[-1] if len(r) else np.nan))
    per_shot = pd.DataFrame(rows)

    def wmean(g, col):
        g = g[g[col].notna()]
        return np.average(g[col], weights=g.frames) if len(g) else np.nan

    summary = pd.DataFrame([dict(video=v, shots=len(g), reviewed=int(g.score.notna().sum()),
                                 accuracy_pass1=wmean(g, "score_pass1"), accuracy=wmean(g, "score"),
                                 iter_min=int(g.iteration.min()), iter_max=int(g.iteration.max()))
                            for v, g in per_shot.groupby("video", sort=False)])
    return per_shot, summary


def plot(per_shot, summary, out: Path):
    summary = summary.sort_values("accuracy", na_position="first")
    order = list(summary.video)
    n_shots = per_shot.shot.max() + 1
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(17, 0.48 * len(order) + 2.2),
                                 gridspec_kw=dict(width_ratios=[1, 2.4], wspace=0.32), facecolor=SURFACE)
    for a in (ax, hx):
        a.set_facecolor(SURFACE)
        for s in a.spines.values():
            s.set_visible(False)
    y = np.arange(len(order))

    # left: estimated accuracy, pass 1 (hollow) → latest (filled)
    ax.hlines(y, summary.accuracy_pass1, summary.accuracy, color=BLUE_LIGHT, lw=2)
    ax.scatter(summary.accuracy_pass1, y, s=70, facecolor=SURFACE, edgecolor=BLUE, lw=2, zorder=3, label="pass 1")
    ax.scatter(summary.accuracy, y, s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=4, label="latest review")
    for yi, r in zip(y, summary.itertuples()):
        txt = f"{r.accuracy:.0%}" if np.isfinite(r.accuracy) else "–"
        ax.text(1.02, yi, txt, va="center", ha="left", color=INK, fontsize=10, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [f"{v}" for v in order], color=INK, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Estimated accuracy per video (frame-weighted review score)", loc="left", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, frameon=False,
              labelcolor=MUTED, fontsize=9)

    # right: per-shot estimated error (1 − score), with each shot's iteration count
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("err", ERROR_RAMP)
    for yi, stem in enumerate(order):
        for r in per_shot[per_shot.video == stem].itertuples():
            colour = UNREVIEWED if np.isnan(r.score) else cmap(1 - r.score)
            hx.add_patch(plt.Rectangle((r.shot + 0.05, yi - 0.4), 0.9, 0.8, color=colour, lw=0))
            if r.iteration:
                hx.text(r.shot + 0.5, yi, str(r.iteration), ha="center", va="center", fontsize=6.5,
                        color=SURFACE if not np.isnan(r.score) and r.score < 0.5 else INK)
        s = next(summary[summary.video == stem].itertuples())
        hx.text(n_shots + 0.4, yi, f"{s.reviewed}/{s.shots} reviewed · iter {s.iter_min}–{s.iter_max}",
                va="center", fontsize=8.5, color=MUTED)
    hx.set_xlim(0, n_shots)
    hx.set_ylim(-0.6, len(order) - 0.4)
    hx.set_yticks([])
    hx.set_xticks(range(0, n_shots, 5), [str(k) for k in range(0, n_shots, 5)], color=MUTED, fontsize=9)
    hx.tick_params(length=0)
    hx.set_xlabel("shot index (number = times segmented)", color=MUTED, fontsize=9)
    hx.set_title("Per-shot estimated error (darker = worse; grey = not reviewed)", loc="left", color=INK, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=hx, orientation="horizontal", fraction=0.04, pad=0.16, aspect=40)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    cb.set_label("estimated error (1 − score)", color=MUTED, fontsize=8)

    tot = summary.dropna(subset=["accuracy"])
    w = per_shot.groupby("video").frames.sum().loc[tot.video]
    overall = np.average(tot.accuracy, weights=w) if len(tot) else np.nan
    fig.suptitle(f"Annotation status — overall {overall:.0%} · {int(summary.reviewed.sum())}/{int(summary.shots.sum())} "
                 f"shots reviewed · updated {datetime.now():%Y-%m-%d %H:%M}", x=0.01, ha="left", color=INK, fontsize=13)
    tmp = out.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    tmp.replace(out)


def print_table(summary):
    print(f"\n{datetime.now():%H:%M:%S}  annotation status ({C.QC_DIR / 'status.png'})")
    print(f"{'video':16s} {'pass1':>6s} {'latest':>7s}  {'':22s} {'reviewed':>9s} {'iter':>6s}")
    for r in summary.sort_values("accuracy").itertuples():
        bar = "█" * int(round(20 * r.accuracy)) if np.isfinite(r.accuracy) else ""
        fmt = lambda v: f"{v:.0%}" if np.isfinite(v) else "–"
        print(f"{r.video:16s} {fmt(r.accuracy_pass1):>6s} {fmt(r.accuracy):>7s}  {bar:22s} "
              f"{r.reviewed:>4d}/{r.shots:<4d} {r.iter_min}–{r.iter_max:<3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, help="redraw every this many seconds")
    args = ap.parse_args()
    shots = {stem: shots_of(stem) for stem in C.VIDEOS}
    while True:
        per_shot, summary = table(shots, load_reviews())
        per_shot.to_csv(C.QC_DIR / "status.csv", index=False)
        plot(per_shot, summary, C.QC_DIR / "status.png")
        print_table(summary)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

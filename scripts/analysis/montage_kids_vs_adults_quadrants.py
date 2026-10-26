"""
Montages of segments whose within/between quadrant differs between adults and kids (median split, each group on its
own thresholds; labels from compare_quadrants_kids_adults.py). Each segment = adults panel + kids panel on the
segment's middle frame, styled like render_group_gaze_movie.py: cyan cloud = the group's duration-weighted gaze density
over the segment (scaled to each panel's own peak, so group size doesn't change brightness), yellow dots = fixations.

Output: figures/within_between/kids_vs_adults/<adult quadrant>_to_<kid quadrant(s)>.png
Usage:  python scripts/analysis/montage_kids_vs_adults_quadrants.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from calc_within_between_cosine import MAX_FIXATION_MS, load_fixations  # noqa: E402
from gaze_tiles import DISPLAY_H, DISPLAY_W, build_participant_arrays  # noqa: E402
from montage_best_examples import STIMULI_DIR  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "viz"))
from render_group_gaze_movie import CLOUD_RGB, OUT_H, OUT_W, SIGMA_PX, reader  # noqa: E402
from video_overlay import density_field, read_frame  # noqa: E402

LABELS_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "quadrants_kids_vs_adults_by_segment.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "within_between" / "kids_vs_adults"
Q = 0.5
PAIRS_PER_ROW = 2
GRID_SCALE = 4
GX = (np.arange(OUT_W // GRID_SCALE) + 0.5) * DISPLAY_W / (OUT_W // GRID_SCALE)
GY = (np.arange(OUT_H // GRID_SCALE) + 0.5) * DISPLAY_H / (OUT_H // GRID_SCALE)
DOT_FILL, DOT_EDGE = "#ffeb3b", "black"
SETS = {  # output name: (adult quadrant, kid quadrants)
    "coordinated_to_other": ("coordinated", ["concentrated", "selective", "scattered"]),
    "selective_to_scattered": ("selective", ["scattered"]),
}


def participant_arrays(video: str, group: str) -> dict:
    fix = load_fixations(video, group)
    return build_participant_arrays(fix[(fix["endT"] - fix["startT"]) <= MAX_FIXATION_MS])


def middle_frame(video: str, t: float) -> np.ndarray:
    r = reader(STIMULI_DIR / f"{video}.mp4", t, 0.1)
    img = read_frame(r, OUT_W, OUT_H)
    r.kill(); r.wait()
    return img


def draw(ax, img: np.ndarray, arrays: dict, s: float, e: float, title: str):
    field = density_field(arrays, list(arrays), s * 1000, e * 1000, SIGMA_PX, GX, GY)
    v = np.clip(field / field.max(), 0, 1)
    alpha = np.where(v > 0.04, v ** 0.6 * 0.7, 0)  # same cloud mapping as render_group_gaze_movie.cloud_overlay
    cloud = np.dstack([np.broadcast_to(np.array(CLOUD_RGB) / 255, (*v.shape, 3)), alpha])
    ax.imshow(img)
    ax.imshow(cloud, extent=(0, OUT_W, OUT_H, 0), interpolation="bilinear")
    n = 0
    for st, en, x, y in arrays.values():
        m = (np.minimum(en, e * 1000) - np.maximum(st, s * 1000)) > 0
        if m.any():
            n += 1
            ax.scatter(x[m] * OUT_W / DISPLAY_W, y[m] * OUT_H / DISPLAY_H, s=12, color=DOT_FILL, edgecolor=DOT_EDGE,
                       linewidth=0.5, alpha=0.9, zorder=3)
    ax.set_title(f"{title} (n={n})", fontsize=11)
    ax.set_xlim(0, OUT_W); ax.set_ylim(OUT_H, 0)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    times = pd.read_csv(LABELS_PATH.with_name("all_clips_by_segment.csv"))[["video", "seg_idx", "seg_start_s", "seg_end_s"]]
    lab = pd.read_csv(LABELS_PATH).merge(times, on=["video", "seg_idx"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    for name, (adult_q, kid_qs) in SETS.items():
        segs = lab[(lab[f"adults_q{Q}"] == adult_q) & lab[f"kids_q{Q}"].isin(kid_qs)]
        segs = segs.sort_values([f"kids_q{Q}", "video", "seg_idx"]).reset_index(drop=True)
        n_rows = int(np.ceil(len(segs) / PAIRS_PER_ROW))
        fig, axes = plt.subplots(n_rows, 2 * PAIRS_PER_ROW, figsize=(5 * 2 * PAIRS_PER_ROW, 3.5 * n_rows), squeeze=False)
        for k, r in segs.iterrows():
            for g in ("adults", "kids"):
                if (r.video, g) not in cache:
                    cache[(r.video, g)] = participant_arrays(r.video, g)
            img = middle_frame(r.video, (r.seg_start_s + r.seg_end_s) / 2)
            row, col = divmod(k, PAIRS_PER_ROW)
            for j, g in enumerate(("adults", "kids")):
                wv, bv = r[f"within_person_cosine_dist_{g}"], r[f"between_person_cosine_dist_{g}"]
                draw(axes[row, 2 * col + j], img, cache[(r.video, g)], r.seg_start_s, r.seg_end_s,
                     f"{r.video} seg{int(r.seg_idx)} [{r.seg_start_s:.1f}-{r.seg_end_s:.1f}s]\n"
                     f"{g.upper()}: {r[f'{g}_q{Q}']}  (within {wv:.2f}, between {bv:.2f})")
        for ax in axes.flat[2 * len(segs):]:
            ax.axis("off")
        fig.suptitle(f"Adults {adult_q} -> kids {' / '.join(kid_qs)} ({len(segs)} segments; median split, each group's own "
                     "thresholds; cyan = group gaze density, yellow = fixations)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = OUT_DIR / f"{name}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: {len(segs)} segments -> {out}")


if __name__ == "__main__":
    main()

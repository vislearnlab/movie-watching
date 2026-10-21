"""
Montage of the single BEST (most decisive) example of each of the four quadrants
(concentrated / coordinated / selective / scattered), pooled across ALL 12 clips with no
per-clip or per-block diversity constraint -- just whichever segments are most extreme.

Candidate pool: frank/pixar/sesame use their real (unchopped) segments; slow uses its
artificially-chopped ~3s subsegments (5 per real ~15.2s segment -- see METHODS_DECISIONS.md's
2026-10-20 entry), since slow's real segments have ~0% selective/scattered/concentrated
representation (duration confound pulls almost everything into coordinated) and chopping is
what exposes the rest. Chopped subsegments are treated exactly like any other segment once
built -- no special-casing in the ranking.

Segments/subsegments with more than N_CAP (12) windows available are skipped, to avoid
needing to replicate the canonical script's Monte Carlo window-rarefaction just for
illustration purposes -- this only excludes a handful of unusually long segments.

Usage:
    python scripts/analysis/montage_best_examples.py
"""

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from calc_within_between_cosine import (  # noqa: E402
    FIXATION_DIR, MAX_FIXATION_MS, N_CAP, PARTICIPANT_SUMMARY_PATH, SIGMA, WINDOW_S, cx, cy,
    within_between_for_windows,
)
from calc_concordance_group import (  # noqa: E402
    DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations,
)
from plot_within_between_timecourse import classify_quadrant, pooled_quantile_split  # noqa: E402

CANONICAL_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
OUT_PATH = PROJECT_ROOT / "figures" / "within_between" / "best_examples_all_quadrants_montage.png"
FRAME_TMP_DIR = PROJECT_ROOT / "figures" / "within_between" / ".montage_frames_tmp"

ALL_CLIPS = [
    "frank_complex", "frank_objects", "frank_play",
    "pixar_birds",
    "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2",
    "slow_animals", "slow_hands", "slow_inanimate", "slow_people",
]
CHOPPED_CLIPS = {"slow_animals", "slow_hands", "slow_inanimate", "slow_people"}
N_SUBSEGMENTS_PER_SEGMENT = 5
Q = 0.4
GROUP_COLOR = "#2196F3"
QUADRANTS = ["concentrated", "coordinated", "selective", "scattered"]
QUADRANT_MARKER_COLORS = {
    "concentrated": "#2E7D32", "coordinated": "#6A1B9A",
    "selective": "#C62828", "scattered": "#EF6C00",
}
N_EXAMPLES_PER_ROW = 4

# Manual override for examples whose segment midpoint lands on a cross-fade/transition frame
# in the raw video -- keyed by (video, seg_idx, sub_idx), value = frame time in seconds.
FRAME_TIME_OVERRIDES = {
    ("sesameus_2", 12, None): 33.5,
}


def quadrant_score(quad: str, within: float, between: float) -> float:
    """How decisively a (within, between) pair sits in its quadrant's corner -- higher is
    more extreme. Direction depends on the quadrant (e.g. concentrated wants both low)."""
    if quad == "concentrated":
        return -(within + between)
    if quad == "coordinated":
        return within - between
    if quad == "selective":
        return between - within
    if quad == "scattered":
        return within + between
    raise ValueError(quad)


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    w, h = out.stdout.strip().split(",")
    return int(w), int(h)


def extract_frame(video_path: Path, time_s: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{time_s:.3f}", "-i", str(video_path),
         "-frames:v", "1", str(out_path)],
        capture_output=True, check=True,
    )


def build_clip_vectors(video: str):
    fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
    fix = fix[(fix["endT"] - fix["startT"]) <= MAX_FIXATION_MS].copy()
    arrays = build_participant_arrays(fix)
    pids = sorted(arrays)

    canonical = pd.read_csv(CANONICAL_PATH)
    real_segs = canonical[(canonical["video"] == video)].sort_values("seg_idx")
    dur = real_segs["seg_end_s"].max()

    n_windows_total = int(dur // WINDOW_S)
    all_vecs = np.empty((len(pids), n_windows_total, len(cx)))
    for w in range(n_windows_total):
        s, e = w * WINDOW_S * 1000, (w + 1) * WINDOW_S * 1000
        for i, pid in enumerate(pids):
            startT, endT, img_x, img_y = arrays[pid]
            all_vecs[i, w] = closed_form_tile_values(startT, endT, img_x, img_y, s, e, SIGMA, cx, cy)

    return pids, arrays, all_vecs, n_windows_total, real_segs


def find_candidates(video: str, thresholds: tuple) -> list[dict]:
    chop = video in CHOPPED_CLIPS
    print(f"Processing {video} (chop={chop})...")
    pids, arrays, all_vecs, n_windows_total, real_segs = build_clip_vectors(video)
    n_sub = N_SUBSEGMENTS_PER_SEGMENT if chop else 1

    candidates = []
    for _, seg_row in real_segs.iterrows():
        s0, s1 = seg_row["seg_start_s"], seg_row["seg_end_s"]
        sub_dur = (s1 - s0) / n_sub
        for sub_idx in range(n_sub):
            sub_s = s0 + sub_idx * sub_dur
            sub_e = s0 + (sub_idx + 1) * sub_dur
            w_idx = [w for w in range(n_windows_total)
                     if w * WINDOW_S >= sub_s and (w + 1) * WINDOW_S <= sub_e]
            if len(w_idx) < 2 or len(w_idx) > N_CAP:
                continue
            result = within_between_for_windows(pids, all_vecs[:, w_idx, :])
            if len(result) < 3:
                continue
            mean_w = np.mean([v[0] for v in result.values()])
            mean_b = np.mean([v[1] for v in result.values()])
            quad = classify_quadrant(mean_w, mean_b, *thresholds)
            if quad is None:
                continue
            best_pid = max(result, key=lambda p: quadrant_score(quad, *result[p]))
            candidates.append(dict(
                video=video, seg_idx=int(seg_row["seg_idx"]), sub_idx=(sub_idx if chop else None),
                sub_start_s=sub_s, sub_end_s=sub_e, mean_within=mean_w, mean_between=mean_b,
                quad=quad, best_pid=best_pid,
                score=quadrant_score(quad, *result[best_pid]),
                arrays=arrays,
            ))
    return candidates


def draw_example_panel(ax, ex: dict, video_res: dict, color: str):
    video = ex["video"]
    vid_w, vid_h = video_res[video]
    scale_x, scale_y = vid_w / DISPLAY_W, vid_h / DISPLAY_H

    override_key = (video, ex["seg_idx"], ex["sub_idx"])
    mid_s = FRAME_TIME_OVERRIDES.get(override_key, (ex["sub_start_s"] + ex["sub_end_s"]) / 2)
    sub_tag = f"_sub{ex['sub_idx']}" if ex["sub_idx"] is not None else ""
    frame_path = FRAME_TMP_DIR / f"{video}_seg{ex['seg_idx']}{sub_tag}.png"
    extract_frame(STIMULI_DIR / f"{video}.mp4", mid_s, frame_path)
    img = plt.imread(frame_path)
    ax.imshow(img)

    sub_s_ms, sub_e_ms = ex["sub_start_s"] * 1000, ex["sub_end_s"] * 1000
    arrays = ex["arrays"]
    best_pid = ex["best_pid"]

    for pid, (startT, endT, img_x, img_y) in arrays.items():
        overlap = (np.minimum(endT, sub_e_ms) - np.maximum(startT, sub_s_ms)) > 0
        if not overlap.any():
            continue
        x, y = img_x[overlap] * scale_x, img_y[overlap] * scale_y
        if pid == best_pid:
            continue
        ax.scatter(x, y, s=10, color=GROUP_COLOR, alpha=0.5, edgecolor="none", zorder=2)

    startT, endT, img_x, img_y = arrays[best_pid]
    overlap = (np.minimum(endT, sub_e_ms) - np.maximum(startT, sub_s_ms)) > 0
    x, y = img_x[overlap] * scale_x, img_y[overlap] * scale_y
    ax.scatter(x, y, s=40, color=color, alpha=0.9, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter([0.03], [0.05], s=200, color=color, transform=ax.transAxes, clip_on=False, zorder=4)

    title_tag = f" sub{ex['sub_idx']}" if ex["sub_idx"] is not None else ""
    ax.set_title(f"{video}, seg{ex['seg_idx']}{title_tag} "
                 f"[{ex['sub_start_s']:.1f}-{ex['sub_end_s']:.1f}s]\n{best_pid} vs. group (blue)",
                 fontsize=10)
    # Clamp to the frame's own pixel bounds -- otherwise an off-screen gaze point (gaze
    # drifting past the stimulus edge) autoscales the axes and shrinks the displayed frame.
    ax.set_xlim(0, vid_w)
    ax.set_ylim(vid_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def render_montage(examples_by_quad: dict):
    video_res = {v: get_video_resolution(STIMULI_DIR / f"{v}.mp4") for v in ALL_CLIPS}
    n_cols = N_EXAMPLES_PER_ROW
    n_rows = len(QUADRANTS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for row, quad in enumerate(QUADRANTS):
        examples = examples_by_quad[quad]
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(examples):
                ax.axis("off")
                continue
            draw_example_panel(ax, examples[col], video_res, QUADRANT_MARKER_COLORS[quad])
        axes[row][0].text(-0.08, 0.5, quad.upper(), transform=axes[row][0].transAxes,
                           fontsize=14, style="italic", color="#555555",
                           ha="right", va="center", rotation=90)

    fig.suptitle("Best examples per quadrant, pooled across all 12 clips (N_CAP=12, fixation filter "
                 "applied; slow uses 5x~3s chopped subsegments)\nno per-clip/block diversity forced -- "
                 "best-fit participant highlighted, each panel its own color, marked ● top-left",
                 y=1.01)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


def main():
    canonical = pd.read_csv(CANONICAL_PATH)
    thresholds = pooled_quantile_split(canonical, Q)

    all_candidates = []
    for video in ALL_CLIPS:
        all_candidates.extend(find_candidates(video, thresholds))

    examples_by_quad = {}
    for quad in QUADRANTS:
        quad_candidates = [c for c in all_candidates if c["quad"] == quad]
        top = sorted(quad_candidates, key=lambda e: -e["score"])[:N_EXAMPLES_PER_ROW]
        examples_by_quad[quad] = top
        print(f"\n{quad}: {len(quad_candidates)} candidates, top {len(top)}:")
        for ex in top:
            sub_tag = f" sub{ex['sub_idx']}" if ex["sub_idx"] is not None else ""
            print(f"  {ex['video']} seg{ex['seg_idx']}{sub_tag} "
                  f"[{ex['sub_start_s']:.1f}-{ex['sub_end_s']:.1f}s] "
                  f"mean_within={ex['mean_within']:.3f} mean_between={ex['mean_between']:.3f} "
                  f"best_pid={ex['best_pid']}")

    render_montage(examples_by_quad)


if __name__ == "__main__":
    main()

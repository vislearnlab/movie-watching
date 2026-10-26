"""
Montage of SELECTIVE-quadrant examples drawn from the slow block's artificially-chopped
subsegments (each real ~15.2s slow segment split into 5 equal ~3s subsegments, per
METHODS_DECISIONS.md's 2026-10-20 duration-confound entry). The unchopped slow block has
~0% of its segments classified selective (duration confound pulls it toward coordinated);
chopping breaks that up and exposes selective subsegments. This script finds those, picks
examples, and renders a frame + fixation-overlay montage matching the style of the existing
figures/within_between/selective_examples_montage.png.

Usage:
    python scripts/analysis/montage_slow_chopped_selective.py
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
    FIXATION_DIR, MAX_FIXATION_MS, PARTICIPANT_SUMMARY_PATH, SIGMA, WINDOW_S, cx, cy,
    within_between_for_windows,
)
from gaze_tiles import (  # noqa: E402
    DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations,
)
from plot_within_between_timecourse import classify_quadrant, pooled_quantile_split  # noqa: E402

CANONICAL_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
OUT_PATH = PROJECT_ROOT / "figures" / "within_between" / "slow_chopped_selective_montage.png"
FRAME_TMP_DIR = PROJECT_ROOT / "figures" / "within_between" / ".montage_frames_tmp"

SLOW_CLIPS = ["slow_animals", "slow_hands", "slow_inanimate", "slow_people"]
N_SUBSEGMENTS_PER_SEGMENT = 5
Q = 0.4
GROUP_COLOR = "#2196F3"
PANEL_COLORS = ["#C2185B", "#FBC02D", "#00897B", "#6A1B9A"]


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


def find_selective_subsegments(thresholds: tuple) -> list[dict]:
    candidates = []
    for video in SLOW_CLIPS:
        print(f"Processing {video}...")
        pids, arrays, all_vecs, n_windows_total, real_segs = build_clip_vectors(video)

        for _, seg_row in real_segs.iterrows():
            s0, s1 = seg_row["seg_start_s"], seg_row["seg_end_s"]
            sub_dur = (s1 - s0) / N_SUBSEGMENTS_PER_SEGMENT
            for sub_idx in range(N_SUBSEGMENTS_PER_SEGMENT):
                sub_s = s0 + sub_idx * sub_dur
                sub_e = s0 + (sub_idx + 1) * sub_dur
                w_idx = [w for w in range(n_windows_total)
                         if w * WINDOW_S >= sub_s and (w + 1) * WINDOW_S <= sub_e]
                if len(w_idx) < 2:
                    continue
                result = within_between_for_windows(pids, all_vecs[:, w_idx, :])
                if len(result) < 3:
                    continue
                mean_w = np.mean([v[0] for v in result.values()])
                mean_b = np.mean([v[1] for v in result.values()])
                quad = classify_quadrant(mean_w, mean_b, *thresholds)
                if quad != "selective":
                    continue
                best_pid = max(result, key=lambda p: result[p][1] - result[p][0])
                candidates.append(dict(
                    video=video, seg_idx=int(seg_row["seg_idx"]), sub_idx=sub_idx,
                    sub_start_s=sub_s, sub_end_s=sub_e, mean_within=mean_w, mean_between=mean_b,
                    best_pid=best_pid, selectivity_score=result[best_pid][1] - result[best_pid][0],
                    arrays=arrays,
                ))
    return candidates


def render_montage(examples_by_clip: dict):
    video_res = {v: get_video_resolution(STIMULI_DIR / f"{v}.mp4") for v in SLOW_CLIPS}
    clips = list(examples_by_clip.keys())
    n_cols = max(len(v) for v in examples_by_clip.values())
    n_rows = len(clips)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for row, clip in enumerate(clips):
        examples = examples_by_clip[clip]
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(examples):
                ax.axis("off")
                continue
            draw_example_panel(ax, examples[col], video_res, PANEL_COLORS[row % len(PANEL_COLORS)])

    fig.suptitle("SELECTIVE examples -- slow block, chopped to 5x~3s subsegments per real segment\n"
                 "(each panel uses its own color, marked ● top-left)", y=1.01)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


def draw_example_panel(ax, ex: dict, video_res: dict, color: str):
    video = ex["video"]
    vid_w, vid_h = video_res[video]
    scale_x, scale_y = vid_w / DISPLAY_W, vid_h / DISPLAY_H

    mid_s = (ex["sub_start_s"] + ex["sub_end_s"]) / 2
    frame_path = FRAME_TMP_DIR / f"{video}_seg{ex['seg_idx']}_sub{ex['sub_idx']}.png"
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

    ax.set_title(f"{video}, seg{ex['seg_idx']} sub{ex['sub_idx']} "
                 f"[{ex['sub_start_s']:.1f}-{ex['sub_end_s']:.1f}s]\n{best_pid} vs. group (blue)",
                 fontsize=10)
    # Clamp to the frame's own pixel bounds -- otherwise an off-screen gaze point (gaze
    # drifting past the stimulus edge) autoscales the axes and shrinks the displayed frame.
    ax.set_xlim(0, vid_w)
    ax.set_ylim(vid_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    canonical = pd.read_csv(CANONICAL_PATH)
    thresholds = pooled_quantile_split(canonical, Q)

    candidates = find_selective_subsegments(thresholds)
    print(f"\nFound {len(candidates)} selective chopped subsegments across the slow block.")
    if not candidates:
        print("No selective subsegments found -- nothing to render.")
        return

    # Up to 4 examples per clip (most selective by score), one row per clip.
    examples_by_clip = {}
    for clip in ["slow_animals", "slow_hands", "slow_inanimate"]:
        clip_examples = sorted([c for c in candidates if c["video"] == clip],
                                key=lambda e: -e["selectivity_score"])[:4]
        if not clip_examples:
            continue
        examples_by_clip[clip] = clip_examples
        for ex in clip_examples:
            print(f"  {ex['video']} seg{ex['seg_idx']} sub{ex['sub_idx']} "
                  f"[{ex['sub_start_s']:.1f}-{ex['sub_end_s']:.1f}s] "
                  f"mean_within={ex['mean_within']:.3f} mean_between={ex['mean_between']:.3f} "
                  f"best_pid={ex['best_pid']}")

    render_montage(examples_by_clip)


if __name__ == "__main__":
    main()

"""
Representative-frame sampling for the caption-validation stimulus-prep pipeline (Task 3).

(Task 2, scene cut detection, was skipped by explicit request -- see METHODS_DECISIONS.md,
2026-09-25, "Task 2 (scene cut detection) skipped; Task 3 rule changed". This script
implements the replacement rule described there.)

For each block, treats its clips as one concatenated timeline (fixed alphabetical
clip order -- there is no "true" order, since real experiment playback is randomized
per participant; see METHODS_DECISIONS.md) and takes the midpoint of each of
--n_per_block equal segments as a target time. A candidate frame is rejected (and the
search moves outward within the same clip) if it's all-black, blown-out white, or too
close to a real cut (content-jump check) -- see `find_valid_local_time` below and
METHODS_DECISIONS.md, 2026-09-25 ("Added a content-jump check", "Added a blown-out/white-frame
check") for how those thresholds were calibrated.

Reads: outputs/stimulus_inventory.csv (the full, non-quick stimulus_inventory.py run --
this script asserts that file exists rather than re-deriving it from stimuli/ directly).

Part of the pipeline described in stimulus_inventory.py's docstring; see METHODS_DECISIONS.md at
the project root for the full assumptions/decisions log.

Output
------
outputs/representative_frames.csv (or .quicktest.csv with --quick) + one PNG per frame
    at display resolution, in outputs/representative_frames/ (or _quicktest/).
outputs/checks/representative_frames_timeline.png -- one timeline per block: clip
    boundaries, each target time (open circle) and chosen time (filled circle), with an
    arrow where a frame was moved. Red flags: an arrow crossing a clip boundary (shouldn't
    happen -- the search is clamped to one clip), or chosen times clustering suspiciously.
outputs/checks/representative_frames_contact_sheet.png -- every chosen frame, labeled.
    Red flags: transition frames, fades, or blurry frames -- the all-black/white check only
    catches fully black/white frames, not partial fades or motion blur, so eyeball these.

Usage
-----
    python scripts/preprocessing/stimulus_prep/representative_frames.py
    python scripts/preprocessing/stimulus_prep/representative_frames.py --quick
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import _video_utils as vu


def get_block_timeline(inventory_df: pd.DataFrame, block_id: str) -> list[dict]:
    """
    Build a canonical, fixed-order timeline for a block by concatenating its clips.

    Real experiment playback order is randomized per participant (moviewatching.py shuffles
    clips within a block), so there is no "true" order to reuse here. We pick alphabetical
    video_name order as an arbitrary but fixed, reproducible choice -- see METHODS_DECISIONS.md.
    """
    clips = (
        inventory_df[inventory_df["block_id"] == block_id]
        .sort_values("video_name")
        .to_dict("records")
    )
    assert clips, f"No clips found for block_id={block_id!r} in the inventory."

    timeline = []
    cursor = 0.0
    for clip in clips:
        timeline.append({
            "video_name": clip["video_name"],
            "video_path": clip["video_path"],
            "start": cursor,
            "end": cursor + clip["duration_sec"],
            "duration_sec": clip["duration_sec"],
        })
        cursor += clip["duration_sec"]
    return timeline


def block_time_to_clip(timeline: list[dict], block_time: float) -> dict:
    """Map a timestamp on a block's combined timeline to (clip, local_time_within_clip)."""
    total_duration = timeline[-1]["end"]
    block_time = min(max(block_time, 0.0), total_duration - 1e-6)
    for entry in timeline:
        if entry["start"] <= block_time < entry["end"]:
            return {**entry, "local_time": block_time - entry["start"]}
    return {**timeline[-1], "local_time": timeline[-1]["duration_sec"] - 1e-6}


def frame_mean_luminance(image_path: Path) -> float:
    """Mean grayscale pixel value (0-255) of a saved frame -- used for the black/white checks."""
    with Image.open(image_path) as img:
        return float(np.asarray(img.convert("L"), dtype=np.float64).mean())


def frame_diff_score(video_path: Path, t_before: float, t_after: float, tmp_dir: Path) -> float:
    """
    Mean absolute grayscale pixel difference between two small probe frames -- a cheap proxy
    for "is there a real cut between these two times". Probes are extracted small (160x90)
    since only the magnitude of change matters, not detail -- see METHODS_DECISIONS.md, 2026-09-25
    "Added a content-jump check" for how the threshold was calibrated.
    """
    before_path = tmp_dir / "_jump_probe_before.png"
    after_path = tmp_dir / "_jump_probe_after.png"
    vu.extract_frame(video_path, t_before, before_path, width=160, height=90)
    vu.extract_frame(video_path, t_after, after_path, width=160, height=90)
    a = np.asarray(Image.open(before_path).convert("L"), dtype=np.float64)
    b = np.asarray(Image.open(after_path).convert("L"), dtype=np.float64)
    return float(np.abs(a - b).mean())


def find_valid_local_time(video_path: Path, target_local_time: float, clip_duration: float,
                           tmp_dir: Path, args: argparse.Namespace) -> dict:
    """
    Reject a candidate time if it's black (mean luminance below black_frame_mean_threshold),
    blown-out white (mean luminance above white_frame_mean_threshold), or too close to a real
    cut (content-jump score above cut_jump_mean_abs_diff_threshold, from frame_diff_score).
    If the target time is rejected, search outward in frame_search_step_sec steps
    (alternating later/earlier) up to frame_search_max_offset_sec. Search is clamped to this
    clip (does not cross into a neighboring clip in the block) -- see METHODS_DECISIONS.md.

    Returns dict with: local_time, mean_luminance, jump_score, was_moved, offset_sec.
    """
    step = args.frame_search_step_sec
    max_offset = args.frame_search_max_offset_sec

    offsets = [0.0]
    n_steps = int(max_offset / step)
    for i in range(1, n_steps + 1):
        offsets += [i * step, -i * step]

    probe_path = tmp_dir / "_valid_check_probe.png"
    for offset in offsets:
        candidate = target_local_time + offset
        if not (0.0 <= candidate <= clip_duration):
            continue

        vu.extract_frame(video_path, candidate, probe_path)
        mean_lum = frame_mean_luminance(probe_path)

        t_before = max(0.0, candidate - args.cut_jump_probe_offset_sec)
        t_after = min(clip_duration, candidate + args.cut_jump_probe_offset_sec)
        jump_score = frame_diff_score(video_path, t_before, t_after, tmp_dir)

        is_valid = (
            args.black_frame_mean_threshold <= mean_lum <= args.white_frame_mean_threshold
            and jump_score <= args.cut_jump_mean_abs_diff_threshold
        )
        if is_valid:
            return {
                "local_time": candidate,
                "mean_luminance": round(mean_lum, 2),
                "jump_score": round(jump_score, 2),
                "was_moved": offset != 0.0,
                "offset_sec": round(offset, 3),
            }

    raise RuntimeError(
        f"No valid (non-black, non-white, non-cut) frame found within {max_offset}s of "
        f"t={target_local_time:.2f}s in {video_path.name} -- widen --frame_search_max_offset_sec "
        "or inspect this clip."
    )


def save_timeline_figure(rep_frames_df: pd.DataFrame, block_timelines: dict, checks_dir: Path) -> Path:
    n_blocks = len(block_timelines)
    fig, axes = plt.subplots(n_blocks, 1, figsize=(10, 1.6 * n_blocks), squeeze=False)

    for ax, block_id in zip(axes[:, 0], block_timelines):
        timeline = block_timelines[block_id]
        block_rows = rep_frames_df[rep_frames_df["block"] == block_id]

        for entry in timeline:
            ax.axvline(entry["start"], color="gray", linestyle="--", linewidth=0.8)
            ax.text(entry["start"] + 0.5, 1.15, entry["video_name"], fontsize=7, rotation=0, va="bottom")
        ax.axvline(timeline[-1]["end"], color="gray", linestyle="--", linewidth=0.8)

        ax.scatter(block_rows["target_block_time_sec"], [1] * len(block_rows),
                   facecolors="none", edgecolors="black", s=60, label="target", zorder=3)
        ax.scatter(block_rows["chosen_block_time_sec"], [1] * len(block_rows),
                   facecolors="tab:red", edgecolors="black", s=40, label="chosen", zorder=4)

        for _, r in block_rows[block_rows["was_moved"]].iterrows():
            ax.annotate(
                "", xy=(r["chosen_block_time_sec"], 1), xytext=(r["target_block_time_sec"], 1),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2), zorder=2,
            )

        ax.set_yticks([])
        ax.set_xlim(0, timeline[-1]["end"])
        ax.set_ylim(0.7, 1.4)
        ax.set_ylabel(block_id, rotation=0, ha="right", va="center", fontsize=10)

    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Time within block (s, concatenated clips)")
    fig.suptitle("Representative-frame check: target vs. chosen frame times per block")
    fig.tight_layout()

    check_path = checks_dir / "representative_frames_timeline.png"
    fig.savefig(check_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return check_path


def save_contact_sheet(rep_frames_df: pd.DataFrame, project_root: Path, checks_dir: Path) -> Path:
    n_frames = len(rep_frames_df)
    n_cols = min(6, n_frames)
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 1.8 * n_rows), squeeze=False)

    for ax, (_, r) in zip(axes.flat, rep_frames_df.iterrows()):
        img = Image.open(project_root / r["png_path"])
        ax.imshow(img)
        ax.set_title(f"{r['frame_id']}\nt={r['chosen_block_time_sec']:.1f}s (clip t={r['chosen_local_time_sec']:.1f}s)",
                     fontsize=6.5)
        ax.axis("off")

    for ax in axes.flat[n_frames:]:
        ax.axis("off")

    fig.suptitle("Representative-frame check: all chosen frames")
    fig.tight_layout()

    check_path = checks_dir / "representative_frames_contact_sheet.png"
    fig.savefig(check_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return check_path


def main():
    parser = argparse.ArgumentParser(
        description="Sample representative frames per block from outputs/stimulus_inventory.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration mode: 1 block, 2 frames. Writes to the .quicktest "
                             "CSV/PNG dir instead of the canonical ones.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for the CSV, frame PNGs, and checks/ (default: outputs/).")
    parser.add_argument("--n_per_block", type=int, default=6,
                        help="Representative frames sampled per block.")
    parser.add_argument("--n_total", type=int, default=24,
                        help="Expected total frames (n_blocks * n_per_block), asserted on a full run.")
    parser.add_argument("--display_width", type=int, default=1920)
    parser.add_argument("--display_height", type=int, default=1080)
    parser.add_argument("--black_frame_mean_threshold", type=float, default=10.0,
                        help="A frame counts as 'black' if mean grayscale luminance (0-255) is below this.")
    parser.add_argument("--white_frame_mean_threshold", type=float, default=225.0,
                        help="A frame counts as 'blown-out white' if mean luminance is above this.")
    parser.add_argument("--frame_search_step_sec", type=float, default=0.5,
                        help="Step size used when searching outward from a rejected candidate time.")
    parser.add_argument("--frame_search_max_offset_sec", type=float, default=8.0,
                        help="Max search offset in either direction from the original target time.")
    parser.add_argument("--cut_jump_probe_offset_sec", type=float, default=0.4,
                        help="Content-jump check: seconds before/after a candidate time to probe.")
    parser.add_argument("--cut_jump_mean_abs_diff_threshold", type=float, default=30.0,
                        help="Content-jump check: mean abs grayscale diff above this means 'too close to a cut'. "
                             "Calibrated against the 'slow' block's known cuts (scored 49-116) vs. ordinary "
                             "mid-shot points (scored 1-18) -- see METHODS_DECISIONS.md, 2026-09-25.")
    args = parser.parse_args()

    vu.check_ffmpeg_available()
    project_root = vu.project_root()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs"
    checks_dir = output_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / "stimulus_inventory.csv"
    if not inventory_path.exists():
        sys.exit(f"ERROR: {inventory_path} not found -- run stimulus_inventory.py (without --quick) first.")
    full_inventory_df = pd.read_csv(inventory_path)

    block_ids = sorted(full_inventory_df["block_id"].unique())
    n_per_block = args.n_per_block

    if args.quick:
        block_ids = block_ids[:1]
        n_per_block = 2
        print(f"--quick: using block(s) {block_ids}, {n_per_block} frame(s) per block.")
    else:
        assert len(block_ids) * n_per_block == args.n_total, (
            f"{len(block_ids)} blocks x {n_per_block} per block != --n_total ({args.n_total}) -- "
            "check --n_per_block/--n_total or the block list."
        )

    frame_dir = output_dir / ("representative_frames_quicktest" if args.quick else "representative_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    block_timelines = {}
    rows = []
    for block_id in block_ids:
        timeline = get_block_timeline(full_inventory_df, block_id)
        block_timelines[block_id] = timeline
        total_duration = timeline[-1]["end"]
        seg_len = total_duration / n_per_block
        target_block_times = [(i + 0.5) * seg_len for i in range(n_per_block)]

        for within_block_idx, target_bt in enumerate(target_block_times):
            loc = block_time_to_clip(timeline, target_bt)
            video_path = project_root / loc["video_path"]

            result = find_valid_local_time(video_path, loc["local_time"], loc["duration_sec"], frame_dir, args)

            frame_id = f"{block_id}_{within_block_idx:02d}"
            out_png = frame_dir / f"{frame_id}.png"
            vu.extract_display_frame(video_path, result["local_time"], out_png, args.display_width, args.display_height)

            rows.append({
                "frame_id": frame_id,
                "clip": loc["video_name"],
                "block": block_id,
                "within_block_index": within_block_idx,
                "target_block_time_sec": round(target_bt, 3),
                "chosen_block_time_sec": round(loc["start"] + result["local_time"], 3),
                "target_local_time_sec": round(loc["local_time"], 3),
                "chosen_local_time_sec": round(result["local_time"], 3),
                "mean_luminance": result["mean_luminance"],
                "jump_score": result["jump_score"],
                "was_moved": result["was_moved"],
                "move_offset_sec": result["offset_sec"],
                "png_path": str(out_png.relative_to(project_root)),
            })

    rep_frames_df = pd.DataFrame(rows)

    moved = rep_frames_df[rep_frames_df["was_moved"]]
    if not moved.empty:
        print(f"{len(moved)} frame(s) moved away from a black/white or too-close-to-a-cut target time:")
        print(moved[["frame_id", "clip", "target_local_time_sec", "chosen_local_time_sec",
                      "move_offset_sec", "mean_luminance", "jump_score"]].to_string(index=False))
    else:
        print("No frames needed to move (none landed on a black, white, or too-close-to-a-cut frame).")

    csv_out = output_dir / ("representative_frames.quicktest.csv" if args.quick else "representative_frames.csv")
    rep_frames_df.to_csv(csv_out, index=False)
    print(f"\nWrote {len(rep_frames_df)} rows to {csv_out}")

    timeline_check_path = save_timeline_figure(rep_frames_df, block_timelines, checks_dir)
    print(f"Saved check figure to {timeline_check_path}")
    contact_sheet_path = save_contact_sheet(rep_frames_df, project_root, checks_dir)
    print(f"Saved check figure to {contact_sheet_path}")


if __name__ == "__main__":
    main()

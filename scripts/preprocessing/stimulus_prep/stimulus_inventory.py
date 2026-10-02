"""
Stimulus inventory for the caption-validation stimulus-prep pipeline (Task 1).

For every `*_stripped.mp4` in stimuli/main_blocks/ (the files actually shown to
participants -- see METHODS_DECISIONS.md, 2026-09-25, "Video files used for Task 1
inventory"), records native resolution, exact fps (as a fraction, e.g.
24000/1001 for 23.976, and as a decimal), duration, its block (from filename),
and whether the video content has black bars baked in when considered at
display resolution (letterboxed / pillarboxed), via ffmpeg's cropdetect.

Part of a caption-validation stimulus-prep pipeline that reuses the movie-watching
adult eye-tracking stimuli. See METHODS_DECISIONS.md at the project root for the running
log of assumptions and deviations this pipeline is built on. This was originally
a notebook (notebooks/stimulus_prep.ipynb); it was split into standalone scripts
once RA/undergrad involvement was dropped from the project -- see METHODS_DECISIONS.md,
"Notebook retired in favor of standalone scripts".

Output
------
data/results/stimulus_prep/stimulus_inventory.csv (or .quicktest.csv with --quick) --
    NOTE: data/ is a symlink to the mounted lab server in this repo, so this writes
    through to shared storage, not just the local checkout.
figures/checks/stimulus_inventory_thumbnails.png -- one mid-clip thumbnail per clip,
    with a red rectangle outlining the detected video content area. Red flags: the
    red outline not matching the actual picture edge, real letterbox/pillarbox bars
    with no outline drawn inside them, or an outline drawn where there are no bars.

Usage
-----
    python scripts/preprocessing/stimulus_prep/stimulus_inventory.py
    python scripts/preprocessing/stimulus_prep/stimulus_inventory.py --quick
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import _video_utils as vu

# Maps a stimulus filename prefix -> block_id, matching the block_id values used in
# each participant's *_trial_order.csv. Fixed by the experiment design.
BLOCK_PREFIX_MAP = {
    "sesame": "sesame",   # sesameus_1, sesameus_2, sesameindia_1, sesameindia_2
    "slow": "slow",       # slow_animals, slow_hands, slow_inanimate, slow_people
    "frank": "frank",     # frank_complex, frank_objects, frank_play
    "pixar": "pixar",     # pixar_birds
}


def build_inventory(clip_paths: list[Path], project_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for video_path in clip_paths:
        video_name = video_path.stem.replace("_stripped", "")
        meta = vu.get_video_metadata(video_path)

        block_id = vu.block_from_filename(video_name, BLOCK_PREFIX_MAP)
        assert block_id is not None, (
            f"Could not map '{video_name}' to a block via BLOCK_PREFIX_MAP -- "
            "update it at the top of this script."
        )

        bars = vu.detect_letterbox_pillarbox(
            video_path, meta["native_width"], meta["native_height"], meta["duration_sec"],
            limit=args.cropdetect_limit, n_samples=args.cropdetect_n_samples,
            sample_sec=args.cropdetect_sample_sec, bar_frac_threshold=args.letterbox_bar_frac_threshold,
        )

        rows.append({
            "video_name": video_name,
            "block_id": block_id,
            "video_path": str(video_path.relative_to(project_root)),
            **meta,
            **bars,
        })

    return pd.DataFrame(rows)


def save_check_figure(inventory_df: pd.DataFrame, project_root: Path, checks_dir: Path) -> Path:
    thumb_dir = checks_dir / "_thumbs_stimulus_inventory"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    n_clips = len(inventory_df)
    n_cols = min(4, n_clips)
    n_rows = int(np.ceil(n_clips / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.4 * n_rows), squeeze=False)

    for ax, (_, row) in zip(axes.flat, inventory_df.iterrows()):
        video_path = project_root / row["video_path"]
        thumb_path = thumb_dir / f"{row['video_name']}_mid.png"
        vu.extract_frame(video_path, row["duration_sec"] / 2, thumb_path)

        img = Image.open(thumb_path)
        ax.imshow(img)
        scale_x = img.width / row["native_width"]
        scale_y = img.height / row["native_height"]
        rect = mpatches.Rectangle(
            (row["crop_x"] * scale_x, row["crop_y"] * scale_y),
            row["crop_w"] * scale_x, row["crop_h"] * scale_y,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title(row["video_name"], fontsize=9)
        ax.axis("off")

    for ax in axes.flat[n_clips:]:
        ax.axis("off")

    fig.suptitle("Stimulus inventory check: detected video content area (red) per clip")
    fig.tight_layout()

    check_path = checks_dir / "stimulus_inventory_thumbnails.png"
    fig.savefig(check_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return check_path


def main():
    parser = argparse.ArgumentParser(
        description="Build data/results/stimulus_prep/stimulus_inventory.csv from stimuli/main_blocks/*_stripped.mp4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration mode: 2 clips, 1 cropdetect sample per clip. "
                             "Writes to stimulus_inventory.quicktest.csv instead of the canonical file.")
    parser.add_argument("--stimuli_dir", default=None,
                        help="Directory containing *_stripped.mp4 files (default: stimuli/main_blocks).")
    parser.add_argument("--results_dir", default=None,
                        help="Output directory for the CSV (default: data/results/stimulus_prep/ -- "
                             "NOTE: data/ is a symlink to the mounted lab server in this repo).")
    parser.add_argument("--checks_dir", default=None,
                        help="Output directory for the check figure (default: figures/checks/).")
    parser.add_argument("--letterbox_bar_frac_threshold", type=float, default=0.01,
                        help="A side counts as a 'bar' if darker than cropdetect_limit for "
                             "more than this fraction of that dimension.")
    parser.add_argument("--cropdetect_limit", type=int, default=24,
                        help="ffmpeg cropdetect luma threshold (0-255).")
    parser.add_argument("--cropdetect_n_samples", type=int, default=5,
                        help="Evenly-spaced windows sampled per clip (ignored under --quick, which uses 1).")
    parser.add_argument("--cropdetect_sample_sec", type=float, default=2.0,
                        help="Seconds of video analyzed per sampled window.")
    args = parser.parse_args()

    vu.check_ffmpeg_available()
    project_root = vu.project_root()
    stimuli_dir = Path(args.stimuli_dir) if args.stimuli_dir else project_root / "stimuli" / "main_blocks"
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "data" / "results" / "stimulus_prep"
    checks_dir = Path(args.checks_dir) if args.checks_dir else project_root / "figures" / "checks"
    results_dir.mkdir(parents=True, exist_ok=True)
    checks_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.cropdetect_n_samples = 1

    clip_paths = sorted(stimuli_dir.glob("*_stripped.mp4"))
    if not clip_paths:
        sys.exit(f"ERROR: No *_stripped.mp4 files found in {stimuli_dir}")
    if args.quick:
        clip_paths = clip_paths[:2]
        print(f"--quick: using {len(clip_paths)} of the clips found.")

    inventory_df = build_inventory(clip_paths, project_root, args)

    flagged = inventory_df[inventory_df["is_letterboxed"] | inventory_df["is_pillarboxed"]]
    if not flagged.empty:
        print(
            f"ASK-ME GATE (Task 1): {len(flagged)} clip(s) are not full-screen 16:9 at display "
            "resolution -- the patch grid definition depends on this. Resolve before trusting "
            "plot_patch_grid.py's output:"
        )
        print(flagged[["video_name", "native_width", "native_height",
                        "left_bar_frac", "right_bar_frac", "top_bar_frac", "bottom_bar_frac"]]
              .to_string(index=False))
    else:
        print("All clips are full-screen (no letterbox/pillarbox bars detected).")

    inventory_out = results_dir / ("stimulus_inventory.quicktest.csv" if args.quick else "stimulus_inventory.csv")
    inventory_df.to_csv(inventory_out, index=False)
    print(f"\nWrote {len(inventory_df)} rows to {inventory_out}")

    check_path = save_check_figure(inventory_df, project_root, checks_dir)
    print(f"Saved check figure to {check_path}")


if __name__ == "__main__":
    main()

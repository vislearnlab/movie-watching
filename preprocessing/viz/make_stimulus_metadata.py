"""
Build a metadata CSV listing all stimulus videos and their locations.

Reads one representative participant's trial_order CSV to extract the
canonical (video_name, block_id, block_index, within_block_trial_index)
mapping, then cross-references with actual *_stripped.mp4 files in
stimuli/main_blocks/ to record the real path that analysis scripts should
use.

The trial_order CSV is a ground-truth record of the experiment design and
includes all 12 videos that any participant could have seen; we therefore
only need one participant's file to recover the full stimulus list.

Output
------
data/metadata/stimulus_metadata.csv
Columns:
    video_name              bare video name (e.g. "sesameus_1")
    block_id                block label (e.g. "sesame", "slow", "frank", "pixar")
    block_index             0-based numeric block index
    within_block_index      position of this video within its block (0-based)
    video_path              path to the stripped MP4, relative to project root
    video_exists            True/False — whether the stripped file was found

Usage
-----
    # From project root:
    python preprocessing/viz/make_stimulus_metadata.py

    # Specify a different trial_order file or output location:
    python preprocessing/viz/make_stimulus_metadata.py \\
        --trial_order data/raw/adults/MW002/MW002_*.._trial_order.csv \\
        --output data/metadata/stimulus_metadata.csv
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

# Project root is two levels above this script (preprocessing/viz/ → root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_any_trial_order(raw_dir: Path) -> Path | None:
    """Return the first trial_order CSV found in raw_dir (any participant group)."""
    hits = sorted(raw_dir.rglob("*_trial_order.csv"))
    return hits[0] if hits else None


def find_stripped_video(video_name: str, stimuli_dir: Path) -> str | None:
    """Return path to {video_name}_stripped.mp4 relative to project root, or None."""
    stripped = stimuli_dir / f"{video_name}_stripped.mp4"
    if stripped.exists():
        return str(stripped.relative_to(PROJECT_ROOT))
    # Fall back to plain .mp4 if stripped version is missing
    plain = stimuli_dir / f"{video_name}.mp4"
    if plain.exists():
        return str(plain.relative_to(PROJECT_ROOT))
    return None


def build_metadata(trial_order_path: Path, stimuli_dir: Path) -> pd.DataFrame:
    trial_order = pd.read_csv(trial_order_path)

    required = {"video_name", "block_id", "block_index", "within_block_trial_index"}
    missing = required - set(trial_order.columns)
    if missing:
        raise ValueError(f"trial_order CSV is missing columns: {missing}")

    # Deduplicate: each video appears once per design (all participants see the same set)
    unique = (
        trial_order[["video_name", "block_id", "block_index", "within_block_trial_index"]]
        .drop_duplicates(subset=["video_name"])
        .sort_values(["block_index", "within_block_trial_index"])
        .reset_index(drop=True)
    )

    rows = []
    for _, row in unique.iterrows():
        vname = row["video_name"]
        vpath = find_stripped_video(vname, stimuli_dir)
        rows.append({
            "video_name": vname,
            "block_id": row["block_id"],
            "block_index": int(row["block_index"]),
            "within_block_index": int(row["within_block_trial_index"]),
            "video_path": vpath if vpath else "",
            "video_exists": vpath is not None,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build stimulus_metadata.csv from a trial_order file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trial_order",
        default=None,
        help=(
            "Path to a *_trial_order.csv file (absolute or relative to project root). "
            "If omitted, the first one found in data/raw/ is used."
        ),
    )
    parser.add_argument(
        "--stimuli_dir",
        default="stimuli/main_blocks",
        help="Directory containing the stimulus MP4 files (relative to project root).",
    )
    parser.add_argument(
        "--output",
        default="data/metadata/stimulus_metadata.csv",
        help="Output path for the metadata CSV (relative to project root).",
    )
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    stimuli_dir = PROJECT_ROOT / args.stimuli_dir

    # Resolve trial_order path
    if args.trial_order:
        # Support shell-glob patterns (e.g. data/raw/adults/MW001/*.csv)
        matches = glob.glob(str(PROJECT_ROOT / args.trial_order))
        if not matches:
            sys.exit(f"ERROR: No files matched --trial_order pattern: {args.trial_order}")
        trial_order_path = Path(matches[0])
    else:
        trial_order_path = find_any_trial_order(raw_dir)
        if trial_order_path is None:
            sys.exit("ERROR: No *_trial_order.csv found in data/raw/. Use --trial_order to specify one.")

    print(f"Using trial_order: {trial_order_path.relative_to(PROJECT_ROOT)}")
    print(f"Stimuli directory: {stimuli_dir.relative_to(PROJECT_ROOT)}")

    df = build_metadata(trial_order_path, stimuli_dir)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nWrote {len(df)} videos to {output_path.relative_to(PROJECT_ROOT)}")
    missing = df[~df["video_exists"]]
    if not missing.empty:
        print(f"WARNING: {len(missing)} video(s) not found in stimuli_dir:")
        for v in missing["video_name"]:
            print(f"  {v}")
    else:
        print("All video files found.")

    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()

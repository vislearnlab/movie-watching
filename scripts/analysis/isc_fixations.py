"""
ISC (Inter-Subject Correlation) of fixation position across age groups.

Same pairwise Pearson-r-on-a-binned-time-series approach as isc_gaze.py, but
computed on I2MC fixation output (preprocessing/fixations/i2mc_fixations.py)
instead of raw gaze samples: each bin gets the (xpos, ypos) of whichever
fixation covers the largest share of that bin's duration, so a bin that
straddles a fixation boundary is assigned by overlap, not by which fixation
happened to be processed last, and a bin that falls entirely within a
saccade/gap between fixations is NaN rather than averaged across the
transition (isc_gaze.py's raw-sample bins average through saccades instead).
This measures "did two people fixate the same place", not continuous
gaze-trajectory similarity -- a different quantity from isc_gaze.py, not a
drop-in replacement for it.

Comparison groups
-----------------
  adults-adults, infants-infants, kids-kids,
  adults-infants, adults-kids, infants-kids

Outputs (written to --output_dir)
-----------------------------------
  isc_fixation_pairwise.csv    — one row per (video_name, pid_a, pid_b)
  isc_fixation_per_block.csv   — mean ISC per (block_id, comparison)
  isc_fixation_summary.csv     — mean ISC per comparison, averaged across all clips

Requires fixation CSVs already produced by
preprocessing/fixations/i2mc_fixations.py for a given parameter code (that
script prints "Parameter code for this run: <code>" -- pass that value here
via --code). --code is required, not auto-detected: a participant can have
fixation files for more than one parameter set sitting side by side (that's
the point of the hash), so "just pick the latest" would silently compare
across different exclusion thresholds if one participant was reprocessed
more recently than another.

Usage
-----
    python analysis/isc_fixations.py --code 036667f5
    python analysis/isc_fixations.py --code 036667f5 --raw_dir data/raw --fixation_dir data/preprocessed/fixations --output_dir data/results/isc --bin_ms 20
    python analysis/isc_fixations.py --code 036667f5 --overwrite   # recompute everything from scratch
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from isc_gaze import compute_pairwise_isc, aggregate_by_block, aggregate_summary

RAW_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
FIXATION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "preprocessed", "fixations")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "results", "isc")
BIN_MS       = 20  # matches isc_gaze.py's default so results stay comparable at defaults

# ── file helpers ──────────────────────────────────────────────────────────────

def find_participant_pids(group_dir: str) -> list[str]:
    """Participant ids for a group, taken from data/raw/{group}/* directory
    names. pid isn't parsed out of fixation filenames -- video_name is read
    back out of each fixation CSV's own column instead, so this only needs
    to enumerate candidates to look for."""
    return sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(group_dir, "*"))
        if os.path.isdir(d)
    )


def find_fixation_files(fix_group_dir: str, pid: str, code: str) -> list[str]:
    """All fixation CSVs for this pid at this parameter code (one per video)."""
    return sorted(glob.glob(os.path.join(fix_group_dir, f"{pid}_*_{code}.csv")))


# ── fixation time series ────────────────────────────────────────────────────────

def fixations_to_binned_series(fix_df: pd.DataFrame, bin_ms: float) -> pd.DataFrame:
    """
    Fill each bin_ms window with the (xpos, ypos) of whichever fixation
    covers the *largest share* of that window's duration -- not "last
    fixation touching this bin wins", which would arbitrarily favor
    whichever fixation is processed later when a bin straddles a fixation
    boundary (end of one fixation, saccade, start of the next). A bin
    touched by no fixation at all (fully within a saccade/gap) is NaN,
    same convention as isc_gaze.py's gaze_to_binned_series invalid-sample
    NaNs.

    Returns DataFrame with columns [bin, gaze_x, gaze_y] -- same shape as
    gaze_to_binned_series's output, so align_and_isc/compute_pairwise_isc
    need no changes to consume it.
    """
    if fix_df.empty:
        return pd.DataFrame(columns=["bin", "gaze_x", "gaze_y"])

    last_bin = int(fix_df["endT"].max() // bin_ms)
    n_bins = last_bin + 1
    x = np.full(n_bins, np.nan)
    y = np.full(n_bins, np.nan)
    best_overlap = np.zeros(n_bins)

    for _, fx in fix_df.iterrows():
        b0, b1 = int(fx["startT"] // bin_ms), int(fx["endT"] // bin_ms)
        for b in range(b0, b1 + 1):
            bin_start, bin_end = b * bin_ms, (b + 1) * bin_ms
            overlap = min(fx["endT"], bin_end) - max(fx["startT"], bin_start)
            if overlap > best_overlap[b]:
                best_overlap[b] = overlap
                x[b], y[b] = fx["xpos"], fx["ypos"]

    return pd.DataFrame({"bin": np.arange(n_bins), "gaze_x": x, "gaze_y": y})


# ── participant loading ───────────────────────────────────────────────────────

def load_group_fixations(
    raw_group_dir: str,
    fix_group_dir: str,
    group_label: str,
    code: str,
    bin_ms: float,
    skip_pids: set[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all participants in `raw_group_dir` that have at least one
    fixation file at parameter code `code` in `fix_group_dir`.

    Participants whose pid is in `skip_pids` are skipped (their rows
    already exist in isc_fixation_pairwise.csv from a previous run).
    Participants with no fixation files at this code are skipped too --
    they may never have been processed with this parameter set, or every
    one of their trials was excluded upstream (calibration/invalid_frac);
    this script doesn't need to know which, just that there's no data.

    Returns {pid: {video_name: binned_df}}.
    """
    group_data: dict[str, dict[str, pd.DataFrame]] = {}
    skip_pids = skip_pids or set()

    for pid in find_participant_pids(raw_group_dir):
        if pid in skip_pids:
            print(f"  [{group_label}] {pid}: already in pairwise CSV — skipping")
            continue

        fix_paths = find_fixation_files(fix_group_dir, pid, code)
        if not fix_paths:
            print(f"  [{group_label}] {pid}: no fixation files for code {code} — skipping")
            continue

        pid_data: dict[str, pd.DataFrame] = {}
        for path in fix_paths:
            fix_df = pd.read_csv(path)
            if fix_df.empty:
                continue
            vname = fix_df["video_name"].iloc[0]
            binned = fixations_to_binned_series(fix_df, bin_ms)
            if not binned.empty:
                pid_data[vname] = binned

        if pid_data:
            group_data[pid] = pid_data
            print(f"  [{group_label}] {pid}: {len(pid_data)} clips loaded")
        else:
            print(f"  [{group_label}] {pid}: no valid trials — skipping")

    return group_data


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute fixation-position ISC across age groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",      default=RAW_DIR,
                        help="Path to data/raw/ (used only for participant/video/block_id metadata)")
    parser.add_argument("--fixation_dir", default=FIXATION_DIR,
                        help="Path to fixation CSVs written by i2mc_fixations.py")
    parser.add_argument("--output_dir",   default=OUTPUT_DIR, help="Output directory for CSVs")
    parser.add_argument("--code", required=True,
                        help="Parameter code to load fixation files for (printed by i2mc_fixations.py "
                             "as 'Parameter code for this run: <code>'). Required -- a participant can "
                             "have fixation files for more than one parameter set, so there's no safe "
                             "default to fall back to.")
    parser.add_argument("--bin_ms",     type=float, default=BIN_MS,
                        help="Temporal bin size in ms for resampling fixation position")
    parser.add_argument("--overwrite",  action="store_true", default=False,
                        help="Ignore existing pairwise CSV and recompute everything")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairwise_path = os.path.join(args.output_dir, "isc_fixation_pairwise.csv")
    block_path    = os.path.join(args.output_dir, "isc_fixation_per_block.csv")
    summary_path  = os.path.join(args.output_dir, "isc_fixation_summary.csv")

    # ── Determine which participants / pairs to skip ───────────────────────────
    existing_pairwise: pd.DataFrame | None = None
    skip_pids:  set[str]       = set()
    skip_pairs: set[frozenset] = set()

    if not args.overwrite and os.path.exists(pairwise_path):
        existing_pairwise = pd.read_csv(pairwise_path)
        already_pids = set(existing_pairwise["pid_a"]) | set(existing_pairwise["pid_b"])
        skip_pairs = {
            frozenset((row.pid_a, row.pid_b))
            for row in existing_pairwise[["pid_a", "pid_b"]].itertuples()
        }
        print(
            f"Found existing pairwise CSV with {len(existing_pairwise)} rows "
            f"({len(already_pids)} unique pids, {len(skip_pairs)} pairs).\n"
            "Will skip already-computed pairs and append new results."
        )

    # ── Load participants ──────────────────────────────────────────────────────
    print("\nLoading adults...")
    adults = load_group_fixations(
        os.path.join(args.raw_dir, "adults"), os.path.join(args.fixation_dir, "adults"),
        "adults", args.code, args.bin_ms, skip_pids,
    )
    print("Loading infants...")
    infants = load_group_fixations(
        os.path.join(args.raw_dir, "infants"), os.path.join(args.fixation_dir, "infants"),
        "infants", args.code, args.bin_ms, skip_pids,
    )
    print("Loading kids...")
    kids = load_group_fixations(
        os.path.join(args.raw_dir, "kids"), os.path.join(args.fixation_dir, "kids"),
        "kids", args.code, args.bin_ms, skip_pids,
    )

    # Also reload already-seen participants so new × old pairs can be computed.
    if existing_pairwise is not None:
        existing_pids_by_group: dict[str, set[str]] = {}
        for col in ("pid_a", "pid_b"):
            grp_col = "group_a" if col == "pid_a" else "group_b"
            for grp, pid in zip(existing_pairwise[grp_col], existing_pairwise[col]):
                existing_pids_by_group.setdefault(grp, set()).add(pid)

        new_pids = (
            set(adults) | set(infants) | set(kids)
        ) - (set(existing_pairwise["pid_a"]) | set(existing_pairwise["pid_b"]))

        if new_pids:
            print(f"\nNew participants detected: {sorted(new_pids)}")
            print("Reloading existing participants to compute new × old pairs...")
            if "adults" in existing_pids_by_group:
                existing_adults = load_group_fixations(
                    os.path.join(args.raw_dir, "adults"), os.path.join(args.fixation_dir, "adults"),
                    "adults", args.code, args.bin_ms, skip_pids=set(adults),
                )
                adults.update(existing_adults)
            if "infants" in existing_pids_by_group:
                existing_infants = load_group_fixations(
                    os.path.join(args.raw_dir, "infants"), os.path.join(args.fixation_dir, "infants"),
                    "infants", args.code, args.bin_ms, skip_pids=set(infants),
                )
                infants.update(existing_infants)
            if "kids" in existing_pids_by_group:
                existing_kids = load_group_fixations(
                    os.path.join(args.raw_dir, "kids"), os.path.join(args.fixation_dir, "kids"),
                    "kids", args.code, args.bin_ms, skip_pids=set(kids),
                )
                kids.update(existing_kids)

    groups = {}
    if adults:  groups["adults"]  = adults
    if infants: groups["infants"] = infants
    if kids:    groups["kids"]    = kids

    if not groups:
        print("No participants loaded — exiting.")
        return

    # ── Build a trial_meta table (video_name → block_id) ─────────────────────
    all_pdirs = [
        d
        for subdir in ("adults", "infants", "kids")
        for d in glob.glob(os.path.join(args.raw_dir, subdir, "*"))
        if os.path.isdir(d)
    ]
    trial_meta = pd.DataFrame()
    for pdir in all_pdirs:
        trial_order_paths = glob.glob(os.path.join(pdir, "*_trial_order.csv"))
        if trial_order_paths:
            trial_meta = pd.read_csv(trial_order_paths[0])
            break

    # ── Compute ISC ───────────────────────────────────────────────────────────
    print("\nComputing pairwise ISC...")
    new_records = compute_pairwise_isc(groups, trial_meta, skip_pairs=skip_pairs)

    if not new_records and existing_pairwise is None:
        print("No ISC values computed — check that participants share video clips.")
        return

    new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

    if existing_pairwise is not None and not new_df.empty:
        pairwise_df = pd.concat([existing_pairwise, new_df], ignore_index=True)
        print(f"Appended {len(new_df)} new rows to {len(existing_pairwise)} existing rows.")
    elif existing_pairwise is not None:
        pairwise_df = existing_pairwise
        print("No new pairs to add; using existing pairwise CSV as-is.")
    else:
        pairwise_df = new_df

    # ── Aggregate ─────────────────────────────────────────────────────────────
    block_df   = aggregate_by_block(pairwise_df)
    summary_df = aggregate_summary(pairwise_df)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    pairwise_df.to_csv(pairwise_path, index=False)
    block_df.to_csv(block_path,       index=False)
    summary_df.to_csv(summary_path,   index=False)

    print(f"\nWrote {len(pairwise_df)} pairwise records  → {pairwise_path}")
    print(f"Wrote {len(block_df)} block rows          → {block_path}")
    print(f"Wrote {len(summary_df)} summary rows      → {summary_path}")

    print("\n=== Summary (mean ISC by comparison) ===")
    print(summary_df[["comparison", "mean_isc_xy", "n_clips", "total_pairs"]].to_string(index=False))


if __name__ == "__main__":
    main()

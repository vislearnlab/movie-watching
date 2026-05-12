"""
ISC (Inter-Subject Correlation) of gaze position across age groups.

Computes pairwise Pearson r on gaze_x / gaze_y time series (binned to a
fixed temporal grid) for every video clip, then aggregates to block and
study level.

Comparison groups
-----------------
  adults-adults, infants-infants, kids-kids,
  adults-infants, adults-kids, infants-kids

Outputs (written to --output_dir)
-----------------------------------
  isc_pairwise.csv    — one row per (video_name, pid_a, pid_b)
  isc_per_block.csv   — mean ISC per (block_id, comparison)
  isc_summary.csv     — mean ISC per comparison, averaged across all clips

Usage
-----
    python analysis/isc_gaze.py
    python analysis/isc_gaze.py --raw_dir data/raw --output_dir data/data_to_be_analyzed --bin_ms 20
    python analysis/isc_gaze.py --overwrite   # recompute everything from scratch
"""

import argparse
import glob
import os
import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "data_to_be_analyzed")
BIN_MS     = 20 # we are using 250hz so this gives up 5 samples averaged per bin ?
MIN_BINS   = 50 # minimum overlapping bins to compute ISC

# ── file helpers ──────────────────────────────────────────────────────────────

def find_participant_files(pdir: str) -> dict:
    csvs = glob.glob(os.path.join(pdir, "*.csv"))
    files = {"gaze": None, "trial_order": None}
    for f in csvs:
        bn = os.path.basename(f)
        if bn.startswith("."):
            continue
        if bn.endswith("_trial_order.csv"):
            files["trial_order"] = f
        elif bn.endswith("_validation_summary.csv") or bn.endswith("_validation.csv"):
            pass
        elif bn.endswith(".csv"):
            files["gaze"] = f
    return files


def read_gaze_csv(path: str) -> pd.DataFrame | None:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    return None


# ── trial extraction ──────────────────────────────────────────────────────────

def extract_trial_gaze(gaze_df: pd.DataFrame, video_name: str) -> pd.DataFrame | None:
    """Return gaze rows for a specific video trial with trial_time reset to 0."""
    events = gaze_df["events"].astype(str)
    start_mask = events.str.contains(f"Video_{re_escape(video_name)}$", regex=True)
    if not start_mask.any():
        return None

    start_idx = gaze_df.index[start_mask][0]
    event_str = gaze_df.loc[start_idx, "events"]
    trial_num = event_str.split("_")[2].split("|")[0]
    end_rows = gaze_df.index[events == f"Trial_End_{trial_num}"]
    if len(end_rows) == 0:
        return None

    trial_df = gaze_df.loc[start_idx : end_rows[0]].copy()
    first_t = trial_df["trial_time"].dropna()
    if not first_t.empty:
        trial_df["trial_time"] = trial_df["trial_time"] - first_t.iloc[0]
    return trial_df


def re_escape(s: str) -> str:
    import re
    return re.escape(s)


# ── gaze time series ──────────────────────────────────────────────────────────

def gaze_to_binned_series(
    trial_df: pd.DataFrame,
    bin_ms: float,
) -> pd.DataFrame:
    """
    Bin gaze_x / gaze_y at `bin_ms` resolution.

    Invalid samples (both eyes invalid, or NaN gaze) are excluded before
    binning; bins with no valid samples become NaN.

    Returns DataFrame with columns [bin, gaze_x, gaze_y].
    """
    df = trial_df.copy()

    # Keep rows within the trial (trial_time is non-null and ≥ 0)
    df = df[df["trial_time"].notna() & (df["trial_time"] >= 0)].copy()
    if df.empty:
        return pd.DataFrame(columns=["bin", "gaze_x", "gaze_y"])

    df["trial_time"] = pd.to_numeric(df["trial_time"], errors="coerce")

    # Mark invalid gaze: both eyes invalid or gaze position missing
    left_valid  = pd.to_numeric(df.get("left_valid",  0), errors="coerce").fillna(0)
    right_valid = pd.to_numeric(df.get("right_valid", 0), errors="coerce").fillna(0)
    gaze_x = pd.to_numeric(df["gaze_x"], errors="coerce")
    gaze_y = pd.to_numeric(df["gaze_y"], errors="coerce")

    no_valid_eye = (left_valid == 0) & (right_valid == 0)
    gaze_x = gaze_x.where(~no_valid_eye)
    gaze_y = gaze_y.where(~no_valid_eye)

    df["gaze_x"] = gaze_x
    df["gaze_y"] = gaze_y
    df["bin"] = (df["trial_time"] // bin_ms).astype(int)

    binned = (
        df.groupby("bin")[["gaze_x", "gaze_y"]]
        .mean()
        .reset_index()
    )
    return binned


# ── ISC core ─────────────────────────────────────────────────────────────────

def pearson_isc(
    a: np.ndarray,
    b: np.ndarray,
    min_n: int = MIN_BINS,
) -> float | None:
    """Pearson r over jointly valid samples; None if insufficient overlap."""
    both = ~(np.isnan(a) | np.isnan(b))
    if both.sum() < min_n:
        return None
    av, bv = a[both], b[both]
    if av.std() == 0 or bv.std() == 0:
        return None
    r, _ = stats.pearsonr(av, bv)
    return float(r)


def align_and_isc(
    binned_a: pd.DataFrame,
    binned_b: pd.DataFrame,
    min_bins: int = MIN_BINS,
) -> dict | None:
    """
    Merge two binned series on bin index, compute ISC for x and y separately,
    return dict with isc_x, isc_y, isc_xy (mean of x and y), n_overlap_bins.
    """
    merged = binned_a.merge(binned_b, on="bin", suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return None

    a_x = merged["gaze_x_a"].values
    b_x = merged["gaze_x_b"].values
    a_y = merged["gaze_y_a"].values
    b_y = merged["gaze_y_b"].values

    r_x = pearson_isc(a_x, b_x, min_bins)
    r_y = pearson_isc(a_y, b_y, min_bins)

    if r_x is None and r_y is None:
        return None

    both_valid = ~(np.isnan(a_x) | np.isnan(b_x) | np.isnan(a_y) | np.isnan(b_y))
    n_overlap = int(both_valid.sum())

    r_xy = np.nanmean([v for v in [r_x, r_y] if v is not None])

    return {
        "isc_x":          r_x,
        "isc_y":          r_y,
        "isc_xy":         float(r_xy),
        "n_overlap_bins": n_overlap,
    }


# ── participant loading ───────────────────────────────────────────────────────

def load_group(
    group_dir: str,
    group_label: str,
    bin_ms: float,
    skip_pids: set[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all participants in `group_dir`.

    Participants whose pid is in `skip_pids` are skipped (their rows already
    exist in isc_pairwise.csv from a previous run).

    Returns {pid: {video_name: binned_df}}.
    """
    group_data: dict[str, dict[str, pd.DataFrame]] = {}
    skip_pids = skip_pids or set()

    pdirs = sorted(
        d for d in glob.glob(os.path.join(group_dir, "*"))
        if os.path.isdir(d)
    )

    for pdir in pdirs:
        pid = os.path.basename(pdir)

        if pid in skip_pids:
            print(f"  [{group_label}] {pid}: already in pairwise CSV — skipping")
            continue

        files = find_participant_files(pdir)
        if files["gaze"] is None or files["trial_order"] is None:
            print(f"  [{group_label}] {pid}: missing files — skipping")
            continue

        gaze_df = read_gaze_csv(files["gaze"])
        if gaze_df is None:
            print(f"  [{group_label}] {pid}: could not read gaze CSV — skipping")
            continue

        trial_order = pd.read_csv(files["trial_order"])
        pid_data: dict[str, pd.DataFrame] = {}

        for _, row in trial_order.iterrows():
            vname = row["video_name"]
            trial_df = extract_trial_gaze(gaze_df, vname)
            if trial_df is None or trial_df.empty:
                continue
            binned = gaze_to_binned_series(trial_df, bin_ms)
            if not binned.empty:
                pid_data[vname] = binned

        if pid_data:
            group_data[pid] = pid_data
            print(f"  [{group_label}] {pid}: {len(pid_data)} clips loaded")
        else:
            print(f"  [{group_label}] {pid}: no valid trials — skipping")

    return group_data


# ── ISC computation across groups ────────────────────────────────────────────

def compute_pairwise_isc(
    groups: dict[str, dict[str, dict[str, pd.DataFrame]]],
    trial_meta: pd.DataFrame,
    skip_pairs: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    Compute all pairwise ISC values across and within groups.

    Pairs where both pid_a and pid_b appear in `skip_pairs` (as a frozenset)
    are skipped entirely.

    Returns list of records suitable for a DataFrame.
    """
    records = []
    skip_pairs = skip_pairs or set()

    # Build video → block_id lookup from any participant's trial_order
    video_to_block: dict[str, str] = {}
    for _, row in trial_meta.iterrows():
        video_to_block[row["video_name"]] = row["block_id"]

    group_names = list(groups.keys())

    for i, g_a in enumerate(group_names):
        for j, g_b in enumerate(group_names):
            if j < i:
                continue  # avoid duplicate (A,B) and (B,A); use (A,B) only

            pids_a = list(groups[g_a].keys())
            pids_b = list(groups[g_b].keys())

            if g_a == g_b:
                pairs = list(combinations(pids_a, 2))
                comparison = f"{g_a}-{g_b}"
            else:
                pairs = list(product(pids_a, pids_b))
                comparison = f"{g_a}-{g_b}"

            for pid_a, pid_b in pairs:
                if frozenset((pid_a, pid_b)) in skip_pairs:
                    continue

                vids_a = set(groups[g_a][pid_a].keys())
                vids_b = set(groups[g_b][pid_b].keys())
                shared_vids = vids_a & vids_b

                for vname in sorted(shared_vids):
                    result = align_and_isc(
                        groups[g_a][pid_a][vname],
                        groups[g_b][pid_b][vname],
                    )
                    if result is None:
                        continue

                    records.append(
                        dict(
                            video_name=vname,
                            block_id=video_to_block.get(vname, "unknown"),
                            comparison=comparison,
                            group_a=g_a,
                            group_b=g_b,
                            pid_a=pid_a,
                            pid_b=pid_b,
                            isc_x=result["isc_x"],
                            isc_y=result["isc_y"],
                            isc_xy=result["isc_xy"],
                            n_overlap_bins=result["n_overlap_bins"],
                        )
                    )

    return records


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_by_block(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ISC per (block_id, comparison), averaged first across clips then across pairs."""
    clip_mean = (
        pairwise_df
        .groupby(["block_id", "video_name", "comparison"])[["isc_x", "isc_y", "isc_xy"]]
        .agg(mean_isc_x=("isc_x", "mean"),
             mean_isc_y=("isc_y", "mean"),
             mean_isc_xy=("isc_xy", "mean"),
             n_pairs=("isc_xy", "count"))
        .reset_index()
    )
    block_mean = (
        clip_mean
        .groupby(["block_id", "comparison"])
        .agg(
            mean_isc_x=("mean_isc_x", "mean"),
            mean_isc_y=("mean_isc_y", "mean"),
            mean_isc_xy=("mean_isc_xy", "mean"),
            n_clips=("video_name", "count"),
            total_pairs=("n_pairs", "sum"),
        )
        .reset_index()
    )
    return block_mean


def aggregate_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ISC per comparison group, averaged across all clips and pairs."""
    clip_mean = (
        pairwise_df
        .groupby(["video_name", "comparison"])[["isc_x", "isc_y", "isc_xy"]]
        .agg(mean_isc_x=("isc_x", "mean"),
             mean_isc_y=("isc_y", "mean"),
             mean_isc_xy=("isc_xy", "mean"),
             n_pairs=("isc_xy", "count"))
        .reset_index()
    )
    summary = (
        clip_mean
        .groupby("comparison")
        .agg(
            mean_isc_x=("mean_isc_x", "mean"),
            mean_isc_y=("mean_isc_y", "mean"),
            mean_isc_xy=("mean_isc_xy", "mean"),
            n_clips=("video_name", "count"),
            total_pairs=("n_pairs", "sum"),
        )
        .reset_index()
    )
    return summary


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute gaze ISC across age groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",    default=RAW_DIR,    help="Path to data/raw/")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for CSVs")
    parser.add_argument("--bin_ms",     type=float, default=BIN_MS,
                        help="Temporal bin size in ms for resampling gaze")
    parser.add_argument("--overwrite",  action="store_true", default=False,
                        help="Ignore existing pairwise CSV and recompute everything")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairwise_path = os.path.join(args.output_dir, "isc_pairwise.csv")
    block_path    = os.path.join(args.output_dir, "isc_per_block.csv")
    # todo: fix this to be only be calculated for valid participants or not at all at this step?
    summary_path  = os.path.join(args.output_dir, "isc_summary.csv")

    # ── Determine which participants / pairs to skip ───────────────────────────
    existing_pairwise: pd.DataFrame | None = None
    skip_pids:  set[str]              = set()
    skip_pairs: set[frozenset]        = set()

    if not args.overwrite and os.path.exists(pairwise_path):
        existing_pairwise = pd.read_csv(pairwise_path)
        # A pid can be skipped from *loading* only if it appears exclusively as
        # both members of already-computed pairs — i.e. every pairing it could
        # form with any other already-loaded pid is already present.
        # The safe conservative choice: skip loading a pid only if ALL of its
        # rows in the existing file account for every partner it has been seen
        # with.  Instead of that complexity, we skip at the *pair* level:
        # load everyone who is new, and skip pairs where both pids were already
        # seen together.
        already_pids = set(existing_pairwise["pid_a"]) | set(existing_pairwise["pid_b"])
        skip_pairs = {
            frozenset((row.pid_a, row.pid_b))
            for row in existing_pairwise[["pid_a", "pid_b"]].itertuples()
        }
        # Only skip *loading* a participant if they appear in no new pairings at
        # all — i.e. they are not a new participant.  We detect new participants
        # after scanning directories, so here we just pass skip_pids as empty
        # and let compute_pairwise_isc handle pair-level skipping.
        print(
            f"Found existing pairwise CSV with {len(existing_pairwise)} rows "
            f"({len(already_pids)} unique pids, {len(skip_pairs)} pairs).\n"
            "Will skip already-computed pairs and append new results."
        )

    # ── Load participants ──────────────────────────────────────────────────────
    print("\nLoading adults...")
    adults  = load_group(os.path.join(args.raw_dir, "adults"),  "adults",  args.bin_ms, skip_pids)
    print("Loading infants...")
    infants = load_group(os.path.join(args.raw_dir, "infants"), "infants", args.bin_ms, skip_pids)
    print("Loading kids...")
    kids    = load_group(os.path.join(args.raw_dir, "kids"),    "kids",    args.bin_ms, skip_pids)

    # Also reload already-seen participants so new × old pairs can be computed.
    # We need their binned data to pair them with any newly added participants.
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
                existing_adults = load_group(
                    os.path.join(args.raw_dir, "adults"), "adults", args.bin_ms,
                    skip_pids=set(adults),   # skip ones already loaded above
                )
                adults.update(existing_adults)
            if "infants" in existing_pids_by_group:
                existing_infants = load_group(
                    os.path.join(args.raw_dir, "infants"), "infants", args.bin_ms,
                    skip_pids=set(infants),
                )
                infants.update(existing_infants)
            if "kids" in existing_pids_by_group:
                existing_kids = load_group(
                    os.path.join(args.raw_dir, "kids"), "kids", args.bin_ms,
                    skip_pids=set(kids),
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
        files = find_participant_files(pdir)
        if files["trial_order"]:
            trial_meta = pd.read_csv(files["trial_order"])
            break

    # ── Compute ISC ───────────────────────────────────────────────────────────
    print("\nComputing pairwise ISC...")
    new_records = compute_pairwise_isc(groups, trial_meta, skip_pairs=skip_pairs)

    if not new_records and existing_pairwise is None:
        print("No ISC values computed — check that participants share video clips.")
        return

    new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

    # Merge with existing results
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

    # ── Quick summary print ───────────────────────────────────────────────────
    print("\n=== Summary (mean ISC by comparison) ===")
    print(summary_df[["comparison", "mean_isc_xy", "n_clips", "total_pairs"]].to_string(index=False))


if __name__ == "__main__":
    main()
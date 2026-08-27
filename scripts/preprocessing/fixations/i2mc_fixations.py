"""
Fixation detection on raw gaze data using I2MC (Identification by 2-Means
Clustering; Hessels et al., 2016 - https://github.com/dcnieho/I2MC_Python).

Chosen over a simple velocity/dispersion threshold because I2MC is built to
tolerate the head movement, data loss, and noise typical of infant/child
eye-tracking sessions, which we have here (adults, infants, and kids).

Per trial, builds an I2MC input frame from left/right eye pixel coordinates
(NaN where left_valid/right_valid == 0), runs I2MC, and writes one row per
detected fixation.

Outputs
-------
  {output_dir}/{group}/{pid}_{video_name}_{code}.csv — one row per fixation,
      for that participant's viewing of that video. `code` is a short hash
      of the I2MC parameters used, so re-running with different parameters
      produces new files instead of overwriting the previous set of
      fixations. If --seed was passed (a reproducibility run), `code` gets
      a _REP suffix instead of changing -- the seed doesn't count as a
      parameter that changes fixation identity, it's a check that a given
      parameter set reproduces.

  {log_dir}/{group}_fix_parse_log.csv — one row per script run for that
      group, listing the code (with _REP suffix for reproducibility runs),
      timestamp, the seed used, every parameter used (I2MC options plus
      max_invalid_frac, max_calibration_deg, and max_fracinterped), which
      pids were (re)processed, how many trials were excluded this run
      (n_exclusions), and which ones (pid/video:reason_code, using
      EXCLUSION_REASON_CODES -- not the exact triggering value, just which
      rule fired). Use it to look up which parameter set a given `code`
      corresponds to, and how much exclusion it produced.

Usage
-----
    python preprocessing/fixations/i2mc_fixations.py
    python preprocessing/fixations/i2mc_fixations.py --raw_dir data/raw --output_dir data/preprocessed/fixations
    python preprocessing/fixations/i2mc_fixations.py --overwrite   # recompute even if a pid/code file already exists
    python preprocessing/fixations/i2mc_fixations.py --seed 12345  # reproducibility check against a logged seed
    python preprocessing/fixations/i2mc_fixations.py --groups adults --video pixar_birds
"""

import argparse
import glob
import hashlib
import os
import re
import secrets
import sys
import warnings
from datetime import datetime

# I2MC is vendored in scripts/I2MC (see that directory's README) rather than
# pip-installed, so this needs scripts/ on sys.path before it can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import I2MC
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "preprocessed", "fixations")
LOG_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "preprocessed", "log-files")

# Tracker/display setup (experiment/moviewatching.py: DISPSIZE = (1920, 1080);
# 250 Hz Tobii Fusion, per analysis/isc_gaze.py). Viewing distance and physical
# screen size are calibrated per-session from the Tobii display area and are
# not currently logged to the per-trial CSVs, so noise measures (RMSxy, BCEA)
# are reported in pixels rather than degrees. Pass --scr_width_cm/--dist_cm
# if you want degrees and have those values for a given session.
XRES = 1920.0
YRES = 1080.0
FREQ = 250.0

# Trials where both eyes are lost (left_valid == 0 AND right_valid == 0) for
# more than this fraction of samples are excluded before I2MC ever runs on
# them -- I2MC's own gap-interpolation/merging (see trial_to_i2mc_frame,
# run_i2mc_on_trial) already handles moderate data loss well, but a trial
# that's mostly missing shouldn't produce a fixation file at all.
MAX_INVALID_FRAC = 0.8

# Trials in a block whose calibration accuracy (mean error in degrees,
# averaged over left/right eye and the 5 validation points, from the last
# validation attempt before that block started -- see compute_block_calibration)
# exceeds this are excluded before I2MC runs, regardless of how good the gaze
# data itself looks. Tightened from 6.0 (later_validation.bad_threshold in
# experiment/config.yaml, i.e. the accuracy the live experiment treats as a
# failed calibration) to 2.0 for stricter fixation-quality screening -- no
# longer matches the live experiment's own threshold.
MAX_CALIBRATION_DEG = 2.0

# Individual fixations more than this fraction interpolated (I2MC's
# fracinterped -- see run_i2mc_on_trial) are dropped from the output CSV,
# same treatment as fixations shorter than minFixDur: I2MC already discards a
# fixation that's 100% interpolated (see the I2MC package's "all missing"
# check), but a fixation that's mostly fabricated by interpolation and just
# happens to be long enough to survive minFixDur shouldn't count either.
MAX_FRACINTERPED = 0.5

# I2MC options. These are the package/tutorial defaults (see
# https://devstart.org/CONTENT/EyeTracking/I2MC_tutorial.html), which is also
# what I2MC ships as its own defaults -- there is no separate "infant-tuned"
# preset in the tutorial. downsampFilter is turned off per the tutorial's
# explicit note that the Chebyshev filter can ring on the hard edges typical
# of eye-movement data. maxMergeDist/maxMergeTime kept at tutorial defaults --
# 30px is ~0.7-1.2 degrees visual angle across plausible viewing distances for
# this study's adult/kid/infant setups (no per-session screen geometry is
# logged yet to compute this exactly), comfortably tighter than the
# calibration accuracy this pipeline already tolerates (MAX_CALIBRATION_DEG).
# minFixDur raised from the tutorial default (40ms) to 80ms -- 40ms is too
# permissive for this study (would keep noise-driven micro-fixations), but
# full adult literature convention (100ms) risks discarding genuine short
# fixations typical of dynamic movie-watching. windowtimeInterp lowered from
# the tutorial default (100ms) to 60ms specifically to stay below minFixDur:
# dur counts wall-clock time regardless of real vs. interpolated content, so
# a gap allowed to interpolate up to (or past) minFixDur could single-
# handedly manufacture a "fixation" out of a too-short real one sitting next
# to it, while still passing max_fracinterped (e.g. 50ms real + 40ms
# interpolated = 90ms dur, clears minFixDur; fracinterped=0.44, clears
# max_fracinterped too). At 60ms, no single gap can supply enough duration
# alone to rescue a sub-threshold real fixation. edgeSampInterp raised from
# 2 to 3 samples so interpolation anchors on a slightly more stable local
# average rather than single noisy edge samples. Still revisit cutoffstd
# against real data.
I2MC_OPTIONS = dict(
    windowtimeInterp = 0.06,   # AJ's recommended default: 60ms (I2MC/tutorial default: 100ms)
    edgeSampInterp   = 3,      # AJ's recommended default: 3 samples (I2MC/tutorial default: 2)
    windowtime       = 0.2,
    steptime         = 0.02,
    downsamples      = [2, 5, 10],
    downsampFilter   = False,
    maxerrors        = 100,
    cutoffstd        = 2.0,
    onoffsetThresh   = 3.0,
    maxMergeDist     = 30.0,   # AJ's recommended default: 30px (kept at I2MC/tutorial default)
    maxMergeTime     = 30.0,   # AJ's recommended default: 30ms (kept at I2MC/tutorial default)
    minFixDur        = 80.0,   # AJ's recommended default: 80ms (I2MC/tutorial default: 40ms)
)

FIX_KEYS = [
    "cutoff", "start", "end", "startT", "endT", "dur", "xpos", "ypos",
    "flankdataloss", "fracinterped", "RMSxy", "BCEA", "fixRangeX", "fixRangeY",
]

# Short codes for the "exclusions" column in the run log, so that column
# doesn't spell out "calibration_deg" and the exact triggering value for
# every excluded trial. The console output printed during a run still shows
# the full reason and value for each exclusion as it happens.
#   1 = calibration fail  (block's calibration accuracy > max_calibration_deg)
#   2 = invalid fraction fail (trial's both-eyes-invalid fraction > max_invalid_frac)
EXCLUSION_REASON_CODES = {"calibration_deg": 1, "invalid_frac": 2}

# ── parameter code ──────────────────────────────────────────────────────────

def full_param_set(max_invalid_frac: float, max_calibration_deg: float, max_fracinterped: float) -> dict:
    """All parameters that affect fixation output, for hashing and logging."""
    return dict(I2MC_OPTIONS, xres=XRES, yres=YRES, freq=FREQ,
                max_invalid_frac=max_invalid_frac, max_calibration_deg=max_calibration_deg,
                max_fracinterped=max_fracinterped)


def compute_param_code(params: dict) -> str:
    """Short deterministic code identifying a parameter set.

    Same params -> same code, so re-running with unchanged parameters
    produces the same filename (and gets skipped by the existing-file
    check below). Any parameter change produces a new code, so old and
    new fixation sets for a participant sit side by side instead of one
    overwriting the other.
    """
    canonical = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]

# ── file helpers (mirrors analysis/isc_gaze.py) ────────────────────────────────

def find_participant_files(pdir: str) -> dict:
    csvs = glob.glob(os.path.join(pdir, "*.csv"))
    files = {"gaze": None, "trial_order": None, "validation_summary": None}
    for f in csvs:
        bn = os.path.basename(f)
        if bn.startswith("."):
            continue
        if bn.endswith("_trial_order.csv"):
            files["trial_order"] = f
        elif bn.endswith("_validation_summary.csv"):
            files["validation_summary"] = f
        elif bn.endswith("_validation.csv"):
            pass
        elif bn.endswith(".csv"):
            files["gaze"] = f
    return files


def read_csv_multi_encoding(path: str) -> pd.DataFrame | None:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    return None


# ── block calibration (mirrors scripts/preprocessing/qc/dataQC_check.py's ──
# parse_validation_summary + map_blocks_to_validations)

def compute_block_calibration(trial_order_df: pd.DataFrame, validation_summary_df: pd.DataFrame) -> dict:
    """Mean calibration accuracy (degrees, averaged over left/right eye and
    the 5 validation points) for each block_index, taken from the last
    validation attempt before that block started (block 0 uses the last
    pre_validation attempt; later blocks use the last attempt of
    block_validation_trial{first_trial_of_block}_*).

    Single validation per block, same as dataQC_check.py -- does not yet
    handle mid-block recalibration (see that file's TODO). Returns
    block_index -> float, or None for a block with no matching validation.
    """
    mean_rows = validation_summary_df[validation_summary_df["point"] == "mean"]
    val_mean_deg = {}
    for _, row in mean_rows.iterrows():
        val_mean_deg[row["validation_step"]] = (
            row["Mean_accuracy_degrees_left"] + row["Mean_accuracy_degrees_right"]
        ) / 2

    first_trial_of_block = trial_order_df.groupby("block_index")["total_trial_index"].min()

    block_calibration = {}
    for block_idx, first_trial in first_trial_of_block.items():
        if block_idx == 0:
            pre_vals = sorted(v for v in val_mean_deg if v.startswith("pre_validation"))
            block_calibration[block_idx] = val_mean_deg[pre_vals[-1]] if pre_vals else None
            continue

        matching = []
        for v in val_mean_deg:
            m = re.match(r"block_validation_trial(\d+)_(\d+)", v)
            if m and int(m.group(1)) == first_trial:
                matching.append((int(m.group(2)), v))
        block_calibration[block_idx] = val_mean_deg[max(matching)[1]] if matching else None

    return block_calibration


def extract_trial_gaze(gaze_df: pd.DataFrame, video_name: str) -> pd.DataFrame | None:
    """Return gaze rows for a specific video trial with trial_time reset to 0."""
    events = gaze_df["events"].astype(str)
    start_mask = events.str.contains(f"Video_{re.escape(video_name)}$", regex=True)
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


# ── I2MC input prep ────────────────────────────────────────────────────────────

def trial_to_i2mc_frame(trial_df: pd.DataFrame) -> tuple[pd.DataFrame | None, float | None]:
    """
    Build the [time, L_X, L_Y, R_X, R_Y] frame I2MC expects, and the fraction
    of its samples where both eyes are invalid (used for the max_invalid_frac
    exclusion check -- computed here, on the same filtered slice, so the two
    can't drift apart).

    Invalid samples (left_valid/right_valid == 0) are set to NaN per-eye,
    rather than dropped, so I2MC's own interpolation/data-loss handling
    operates on a continuous timeline.
    """
    df = trial_df[df_cols_present(trial_df)].copy()
    df = df[df["trial_time"].notna() & (df["trial_time"] >= 0)]
    if df.empty:
        return None, None

    left_valid  = pd.to_numeric(df.get("left_valid",  0), errors="coerce").fillna(0)
    right_valid = pd.to_numeric(df.get("right_valid", 0), errors="coerce").fillna(0)
    invalid_frac = float(((left_valid == 0) & (right_valid == 0)).mean())

    l_x = pd.to_numeric(df["left_x"],  errors="coerce").where(left_valid == 1)
    l_y = pd.to_numeric(df["left_y"],  errors="coerce").where(left_valid == 1)
    r_x = pd.to_numeric(df["right_x"], errors="coerce").where(right_valid == 1)
    r_y = pd.to_numeric(df["right_y"], errors="coerce").where(right_valid == 1)

    out = pd.DataFrame({
        "time": pd.to_numeric(df["trial_time"], errors="coerce"),
        "L_X":  l_x,
        "L_Y":  l_y,
        "R_X":  r_x,
        "R_Y":  r_y,
    }).dropna(subset=["time"]).reset_index(drop=True)

    return (out, invalid_frac) if not out.empty else (None, invalid_frac)


def df_cols_present(df: pd.DataFrame) -> list:
    needed = ["trial_time", "left_x", "left_y", "left_valid", "right_x", "right_y", "right_valid"]
    return [c for c in needed if c in df.columns]


# ── per-trial fixation detection ────────────────────────────────────────────────

def trial_seed(base_seed: int, pid: str, vname: str) -> int:
    """Deterministic per-(pid, video) seed derived from base_seed.

    I2MC's k-means++ initialization (its own kmeans2, not scipy's) draws
    from NumPy's global RNG with no seed parameter of its own, so
    reproducing a run means seeding that global state ourselves. Deriving
    a separate seed per trial (rather than seeding once for the whole run)
    means a reproducibility run gives identical fixations for a given
    trial regardless of what else is processed in the same run or in what
    order -- important since this pipeline is designed to be run
    incrementally over time (see the skip-existing-file logic above).
    """
    h = hashlib.sha256(f"{base_seed}:{pid}:{vname}".encode()).hexdigest()
    return int(h[:8], 16)


def run_i2mc_on_trial(i2mc_df: pd.DataFrame, max_fracinterped: float, seed: int) -> pd.DataFrame | None:
    options = dict(I2MC_OPTIONS)
    options.update(xres=XRES, yres=YRES, freq=FREQ, missingx=np.nan, missingy=np.nan)

    np.random.seed(seed)

    # Despite the docstring, this version of I2MC.I2MC returns a 3-tuple
    # (fix, data, par); it returns (False, None, None) on failure (e.g. too
    # much data loss / clustering didn't converge for this trial).
    fix, _, _ = I2MC.I2MC(i2mc_df, options, logging=False)
    if fix is False or fix is None:
        return None

    n = len(fix["start"])
    if n == 0:
        return None

    fix_df = pd.DataFrame({k: fix[k] for k in FIX_KEYS if k in fix})

    # Drop fixations that are mostly fabricated by interpolation, same
    # treatment as fixations shorter than minFixDur (silently excluded from
    # the output; not written to the CSV at all).
    if "fracinterped" in fix_df.columns:
        fix_df = fix_df[fix_df["fracinterped"] <= max_fracinterped].reset_index(drop=True)
    if fix_df.empty:
        return None

    fix_df["fixation_index"] = range(len(fix_df))
    return fix_df


# ── participant loop ────────────────────────────────────────────────────────────

def process_group(group_dir: str, group_label: str, group_output_dir: str, code: str,
                   overwrite: bool, max_invalid_frac: float, max_calibration_deg: float,
                   max_fracinterped: float, base_seed: int,
                   video_filter: str | None) -> tuple[list[str], list[tuple[str, str, str, float]], int]:
    """Write one {pid}_{video_name}_{code}.csv per participant per video.
    `code` here is whatever main() decided should appear in filenames (the
    plain param-hash, or that hash + "_REP" for a reproducibility run).
    Returns (pids that got at least one new file this run, [(pid, video_name,
    reason, value), ...] for trials excluded this run -- reason is
    "invalid_frac" or "calibration_deg", how many trial_order rows matched
    video_filter across all participants -- lets main() warn if a --video
    value never matched anything, e.g. a typo) -- all three for the run log
    / CLI feedback."""
    if not os.path.isdir(group_dir):
        print(f"  [{group_label}] group directory not found: {group_dir} — skipping")
        return [], [], 0

    os.makedirs(group_output_dir, exist_ok=True)
    processed_pids = set()
    exclusions = []
    video_matches = 0
    pdirs = sorted(d for d in glob.glob(os.path.join(group_dir, "*")) if os.path.isdir(d))

    for pdir in pdirs:
        pid = os.path.basename(pdir)
        files = find_participant_files(pdir)
        if files["gaze"] is None or files["trial_order"] is None:
            print(f"  [{group_label}] {pid}: missing files — skipping")
            continue

        trial_order = pd.read_csv(files["trial_order"])

        # calibration_unknown covers the participant-level case: no
        # validation_summary file at all, or it couldn't be read/parsed --
        # every trial's calibration is unknown, not just a specific block's
        # (see the per-block NaN case below for that). Both mean "we can't
        # verify this participant meets max_calibration_deg" and are treated
        # the same way: excluded, not silently let through.
        block_calibration = {}
        calibration_unknown = True
        if files["validation_summary"] is not None:
            val_summary_df = read_csv_multi_encoding(files["validation_summary"])
            if val_summary_df is not None and "point" in val_summary_df.columns:
                block_calibration = compute_block_calibration(trial_order, val_summary_df)
                calibration_unknown = False

        # Skip trials whose output file already exists, without paying to
        # load this participant's (potentially large) gaze CSV at all. Also
        # exclude trials whose block failed calibration here, since that
        # only needs trial_order/validation_summary, not the gaze CSV.
        pending = []
        for _, row in trial_order.iterrows():
            vname = row["video_name"]
            if video_filter is not None and vname != video_filter:
                continue
            if video_filter is not None:
                video_matches += 1

            cal_deg = block_calibration.get(row["block_index"])
            # A NaN reading (a validation attempt exists for this block but its
            # accuracy failed to compute -- e.g. an aborted/failed validation)
            # is a genuine unknown, not a known-good calibration: `cal_deg >
            # max_calibration_deg` alone would silently pass it through, since
            # NaN comparisons are always False in Python. Treat NaN -- and
            # calibration_unknown (no validation_summary at all for this
            # participant, see above) -- the same as exceeding the threshold.
            # A block with no matching validation within an otherwise-usable
            # validation_summary (cal_deg is None) is unaffected -- still not
            # excluded here.
            if calibration_unknown or (cal_deg is not None and (pd.isna(cal_deg) or cal_deg > max_calibration_deg)):
                if calibration_unknown:
                    print(f"  [{group_label}] {pid}/{vname}: no validation_summary available — "
                          f"calibration unknown, excluded, no file written")
                elif pd.isna(cal_deg):
                    print(f"  [{group_label}] {pid}/{vname}: block {row['block_index']} calibration "
                          f"unknown (validation reading missing/NaN) — excluded, no file written")
                else:
                    print(f"  [{group_label}] {pid}/{vname}: block {row['block_index']} calibration "
                          f"{cal_deg:.2f}° (> {max_calibration_deg:.2f}° threshold) — excluded, no file written")
                exclusions.append((pid, vname, "calibration_deg", cal_deg if cal_deg is not None else float("nan")))
                continue

            out_path = os.path.join(group_output_dir, f"{pid}_{vname}_{code}.csv")
            if os.path.exists(out_path) and not overwrite:
                continue
            pending.append((row, out_path))

        if not pending:
            n_considered = len(trial_order) if video_filter is None else (trial_order["video_name"] == video_filter).sum()
            print(f"  [{group_label}] {pid}: all {n_considered} considered video(s) already processed, excluded, or not present — skipping")
            continue

        gaze_df = read_csv_multi_encoding(files["gaze"])
        if gaze_df is None:
            print(f"  [{group_label}] {pid}: could not read gaze CSV — skipping")
            continue

        for row, out_path in pending:
            vname = row["video_name"]
            trial_df = extract_trial_gaze(gaze_df, vname)
            if trial_df is None or trial_df.empty:
                print(f"  [{group_label}] {pid}/{vname}: no gaze data for this trial — skipping")
                continue

            i2mc_df, invalid_frac = trial_to_i2mc_frame(trial_df)
            if i2mc_df is None:
                print(f"  [{group_label}] {pid}/{vname}: no valid samples — skipping")
                continue

            if invalid_frac > max_invalid_frac:
                print(f"  [{group_label}] {pid}/{vname}: {invalid_frac:.0%} of samples have both eyes "
                      f"invalid (> {max_invalid_frac:.0%} threshold) — excluded, no file written")
                exclusions.append((pid, vname, "invalid_frac", invalid_frac))
                continue

            fix_df = run_i2mc_on_trial(i2mc_df, max_fracinterped, trial_seed(base_seed, pid, vname))
            if fix_df is None:
                print(f"  [{group_label}] {pid}/{vname}: 0 fixations detected — no file written")
                continue

            fix_df["pid"]        = pid
            fix_df["group"]      = group_label
            fix_df["video_name"] = vname
            fix_df["block_id"]   = row.get("block_id", "unknown")
            fix_df.to_csv(out_path, index=False)
            processed_pids.add(pid)
            print(f"  [{group_label}] {pid}/{vname}: {len(fix_df)} fixations → {os.path.basename(out_path)}")

    return sorted(processed_pids), exclusions, video_matches


# ── run log ─────────────────────────────────────────────────────────────────

def write_run_log(log_dir: str, group_label: str, code: str, seed: int, params: dict, pids: list[str],
                   exclusions: list[tuple[str, str, str, float]]) -> None:
    """Append one row to {group_label}_fix_parse_log.csv documenting this
    run's parameters, which pids they were applied to, how many trials were
    excluded this run (n_exclusions), and which ones (pid/video:reason_code,
    using EXCLUSION_REASON_CODES -- the triggering value itself isn't kept
    here, just which rule fired; the console output at the time showed it).

    `seed` is logged every run (random or explicit) so a run can later be
    replayed exactly via --seed <this value>, even though seed isn't part
    of `code`/params (see trial_seed)."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{group_label}_fix_parse_log.csv")

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "code": code, "seed": seed}
    row.update(params)
    row["pids"] = "; ".join(pids)
    row["n_exclusions"] = len(exclusions)
    row["exclusions"] = "; ".join(
        f"{pid}/{vname}:{EXCLUSION_REASON_CODES[reason]}" for pid, vname, reason, _ in exclusions
    )

    row_df = pd.DataFrame([row])
    write_header = not os.path.exists(log_path)
    row_df.to_csv(log_path, mode="a", header=write_header, index=False)
    print(f"  Logged run ({code}, {len(pids)} pids, {len(exclusions)} exclusions) → {log_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect fixations from raw gaze using I2MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",    default=RAW_DIR,    help="Path to data/raw/")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for per-participant fixation CSVs")
    parser.add_argument("--log_dir",    default=LOG_DIR,    help="Output directory for per-group run logs")
    parser.add_argument("--overwrite",  action="store_true", default=False,
                        help="Recompute a participant's fixations even if a file for this pid/code already exists")
    parser.add_argument("--max_invalid_frac", type=float, default=MAX_INVALID_FRAC,
                        help="Exclude a trial (no fixation file written) if more than this fraction "
                             "of its samples have both eyes invalid")
    parser.add_argument("--max_calibration_deg", type=float, default=MAX_CALIBRATION_DEG,
                        help="Exclude a trial (no fixation file written) if its block's calibration "
                             "accuracy (mean degrees error, last attempt before that block) exceeds this")
    parser.add_argument("--max_fracinterped", type=float, default=MAX_FRACINTERPED,
                        help="Drop individual fixations from the output CSV if more than this fraction "
                             "of the fixation's samples were interpolated by I2MC")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the RNG seed I2MC's k-means clustering uses, to check that a parameter "
                             "set reproduces a prior run. Doesn't change the parameter hash code (the seed "
                             "isn't a parameter of fixation identity) -- instead the filename/log code gets "
                             "a _REP suffix. Default: a fresh random seed each run, always logged so it can "
                             "be replayed later via --seed if needed.")
    parser.add_argument("--groups", default="adults,infants,kids",
                        help="Comma-separated list of groups to process")
    parser.add_argument("--video", default=None,
                        help="Only process this video_name (e.g. pixar_birds) for each participant, "
                             "instead of every video in their trial order")
    args = parser.parse_args()

    params = full_param_set(args.max_invalid_frac, args.max_calibration_deg, args.max_fracinterped)
    code = compute_param_code(params)
    is_repro = args.seed is not None
    base_seed = args.seed if is_repro else secrets.randbits(32)
    run_code = f"{code}_REP" if is_repro else code
    print(f"Parameter code for this run: {code}"
          + (f" (reproducibility run: files use {run_code}, seed={base_seed})" if is_repro
             else f" (seed={base_seed})"))

    total_video_matches = 0
    for group_label in (g.strip() for g in args.groups.split(",")):
        print(f"\nProcessing {group_label}...")
        processed_pids, exclusions, video_matches = process_group(
            os.path.join(args.raw_dir, group_label),
            group_label,
            os.path.join(args.output_dir, group_label),
            run_code,
            args.overwrite,
            args.max_invalid_frac,
            args.max_calibration_deg,
            args.max_fracinterped,
            base_seed,
            args.video,
        )
        total_video_matches += video_matches
        if processed_pids or exclusions:
            write_run_log(args.log_dir, group_label, run_code, base_seed, params, processed_pids, exclusions)
        else:
            print(f"  [{group_label}] nothing new processed — no log row written")

    if args.video is not None and total_video_matches == 0:
        print(f"\nWarning: --video {args.video!r} matched 0 trials across all processed groups "
              f"— check for a typo against the video_name values in trial_order.csv")


if __name__ == "__main__":
    main()

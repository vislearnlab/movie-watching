"""
Windowed (time-varying) fixation-position ISC -- a Python port of
scripts/analysis/isc_analysis.qmd's Part 7 ("ISC Timecourse"), so it can be saved as a
CSV and plotted alongside other metrics (e.g. calc_concordance_group.py's output,
via plot_concordance_vs_isc.py) instead of only existing as an in-notebook R figure.

This IS the Franchak-style ISC (pairwise Pearson r on x/y position, NOT the same thing as
calc_concordance_group.py's "concordance" metric -- see that script's docstring for
why the two are named differently).

Method (same as calc_isc_fixations.py / isc_analysis.qmd Part 7):
  - Each participant's fixations are resampled onto BIN_MS-wide bins: a bin gets the
    (xpos, ypos) of whichever fixation covers the largest share of that bin's duration
    (fixations_to_binned_series, ported from calc_isc_fixations.py). A bin with no overlapping
    fixation is NaN.
  - For a WINDOW_S-wide, STEP_S-stepped sliding window, every pair of participants'
    binned x and y series are correlated (Pearson r, pairwise-complete), averaged across
    x and y, then averaged across all pairs -- one mean_r per window.
  - A window needs at least MIN_VALID jointly-valid bins for a given pair's r to count.

Restricted here to the same population as calc_concordance_group.py (valid adults,
from participant_summary.csv) and the same video, for a fair side-by-side comparison --
not because ISC itself requires this.

Reads fixation data from a LOCAL COPY only -- see METHODS_DECISIONS.md, "Local copies of
server data": never point this at the live server mount.

Output
------
data/results/concordance/isc_fixation_timecourse.csv -- one row per window:
  video_name, window_idx, window_start_s, window_end_s, mean_r, n_pairs

Usage
-----
    python scripts/analysis/calc_isc_timecourse.py \\
        --fixation_dir data/local_cache/fixations/adults \\
        --participant_summary_path data/local_cache/participant_summary.csv \\
        --param_code f329476c --duration_s 150.15
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calc_concordance_group import load_video_fixations  # noqa: E402 (same loader/population)


def fixations_to_binned_series(fix_df: pd.DataFrame, bin_ms: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ported from calc_isc_fixations.py: fill each bin_ms window with the (xpos, ypos) of
    whichever fixation covers the largest share of that bin's duration. A bin touched by
    no fixation is NaN. Returns (x, y) arrays of length n_bins (shared across all
    participants here, unlike calc_isc_fixations.py's per-participant n_bins, so pairs align
    without an extra merge step).
    """
    x = np.full(n_bins, np.nan)
    y = np.full(n_bins, np.nan)
    best_overlap = np.zeros(n_bins)

    for startT, endT, xpos, ypos in zip(fix_df["startT"], fix_df["endT"], fix_df["xpos"], fix_df["ypos"]):
        b0, b1 = int(startT // bin_ms), min(int(endT // bin_ms), n_bins - 1)
        for b in range(b0, b1 + 1):
            bin_start, bin_end = b * bin_ms, (b + 1) * bin_ms
            overlap = min(endT, bin_end) - max(startT, bin_start)
            if overlap > best_overlap[b]:
                best_overlap[b] = overlap
                x[b], y[b] = xpos, ypos

    return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Windowed fixation-position ISC (Python port of isc_analysis.qmd Part 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixation_dir", required=True,
                         help="LOCAL COPY of the adult I2MC fixation CSVs directory (never the server mount).")
    parser.add_argument("--participant_summary_path", required=True,
                         help="LOCAL COPY of participant_summary.csv.")
    parser.add_argument("--param_code", required=True)
    parser.add_argument("--video", default="pixar_birds")
    parser.add_argument("--duration_s", type=float, required=True)
    parser.add_argument("--bin_ms", type=float, default=25,
                         help="FIXED 2026-10-01 (Option B, explicit user choice) from the qmd's original 20ms "
                              "to 25ms, so a 0.25s step divides evenly (10 bins) with no rounding -- 20ms made "
                              "0.25s/0.02s=12.5 bins, which rounded to 12 (an actual 0.24s step), misaligning "
                              "this timecourse against calc_concordance_group.py's exact 0.25s grid. "
                              "User noted 0.25s is more logical and may revert bin_ms to 20 later -- this "
                              "script always prints the resulting actual window/step below so a future change "
                              "back to a non-dividing bin_ms is caught immediately, not silently reintroduced.")
    parser.add_argument("--window_s", type=float, default=2.0)
    parser.add_argument("--step_s", type=float, default=0.25)
    parser.add_argument("--min_valid", type=int, default=30,
                         help="Minimum jointly-valid bins within a window for a pair's r to count.")
    parser.add_argument("--results_dir", default=None, help="Output directory (default: data/results/concordance/ -- NOTE: data/ is a symlink to the mounted lab server in this repo).")
    parser.add_argument("--output_name", default="isc_fixation_timecourse.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "data" / "results" / "concordance"
    results_dir.mkdir(parents=True, exist_ok=True)

    fix_df = load_video_fixations(Path(args.fixation_dir), args.video, args.param_code,
                                   Path(args.participant_summary_path))
    pids = sorted(fix_df["pid"].unique())
    print(f"{len(pids)} participants, bin_ms={args.bin_ms}")

    n_bins = int(args.duration_s * 1000 // args.bin_ms) + 1

    t0 = time.perf_counter()
    series = {}
    for pid in pids:
        x, y = fixations_to_binned_series(fix_df[fix_df["pid"] == pid], args.bin_ms, n_bins)
        series[pid] = (x, y)
    print(f"Binned {len(pids)} participants in {time.perf_counter() - t0:.1f}s ({n_bins} bins each)")

    window_bins = int(round(args.window_s * 1000 / args.bin_ms))
    step_bins = int(round(args.step_s * 1000 / args.bin_ms))
    actual_window_s = window_bins * args.bin_ms / 1000
    actual_step_s = step_bins * args.bin_ms / 1000
    n_windows = max(0, (n_bins - window_bins) // step_bins + 1)

    # ALWAYS printed, every run: --window_s/--step_s only land exactly on bin boundaries
    # when they're whole multiples of --bin_ms. With bin_ms=20 (the qmd's original
    # default), a 0.25s step rounds to 12 bins = an ACTUAL 0.24s step, not 0.25s -- this
    # silently misaligned this script's output against calc_concordance_group.py's
    # exact-0.25s grid (METHODS_DECISIONS.md, 2026-10-01). bin_ms is now 25 so 0.25s
    # divides evenly; this check stays so a future change back to bin_ms=20 (or any other
    # non-dividing value) is caught immediately instead of silently reintroducing the drift.
    print(f"Requested window={args.window_s}s, step={args.step_s}s  |  "
          f"ACTUAL window={actual_window_s}s, step={actual_step_s}s  (bin_ms={args.bin_ms})")
    if not np.isclose(actual_window_s, args.window_s) or not np.isclose(actual_step_s, args.step_s):
        print(f"  WARNING: requested window/step does not divide evenly by bin_ms={args.bin_ms} -- "
              f"rounded to the ACTUAL values above. This will misalign against any other timecourse "
              f"computed on the requested (not actual) grid, e.g. calc_concordance_group.py.")
    print(f"window_bins={window_bins}, step_bins={step_bins}, n_windows={n_windows}")

    pairs = list(combinations(pids, 2))
    t0 = time.perf_counter()
    rows = []
    for w in range(n_windows):
        b0 = w * step_bins
        b1 = b0 + window_bins
        start_s, end_s = b0 * args.bin_ms / 1000, b1 * args.bin_ms / 1000

        rs = []
        for pid_a, pid_b in pairs:
            xa, ya = series[pid_a]
            xb, yb = series[pid_b]
            xa, ya, xb, yb = xa[b0:b1], ya[b0:b1], xb[b0:b1], yb[b0:b1]

            valid_x = ~(np.isnan(xa) | np.isnan(xb))
            valid_y = ~(np.isnan(ya) | np.isnan(yb))
            r_vals = []
            if valid_x.sum() >= args.min_valid and xa[valid_x].std() > 0 and xb[valid_x].std() > 0:
                r_vals.append(np.corrcoef(xa[valid_x], xb[valid_x])[0, 1])
            if valid_y.sum() >= args.min_valid and ya[valid_y].std() > 0 and yb[valid_y].std() > 0:
                r_vals.append(np.corrcoef(ya[valid_y], yb[valid_y])[0, 1])
            if r_vals:
                rs.append(np.mean(r_vals))

        rows.append(dict(video_name=args.video, window_idx=w, window_start_s=round(start_s, 3),
                          window_end_s=round(end_s, 3), mean_r=np.mean(rs) if rs else np.nan,
                          n_pairs=len(rs)))

        if w % 100 == 0 or w == n_windows - 1:
            print(f"  window {w}/{n_windows - 1} ({time.perf_counter() - t0:.1f}s elapsed)")

    out_df = pd.DataFrame(rows)
    out_path = results_dir / args.output_name
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path} ({time.perf_counter() - t0:.1f}s)")
    print(f"mean_r: mean={out_df['mean_r'].mean():.3f}, NaN rate={out_df['mean_r'].isna().mean():.1%}")


if __name__ == "__main__":
    main()

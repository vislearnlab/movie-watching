"""
Fixation density concordance: individual-to-group fixation density similarity metric.

- Called "concordance", not "ISC" -- ISC is reserved for the existing Franchak-style x/y
  time-series analysis (calc_isc_rawgaze.py, calc_isc_fixations.py, isc_analysis.qmd).
- For every participant and sliding window, computes a duration-weighted Gaussian density
  value at each of 120 scene-tile centers (plot_patch_grid.py's grid), for that
  participant and for the leave-one-out group, then correlates the two 120-value vectors.
- Computed in CLOSED FORM (no raster image / no `gaussian_filter`): each tile's value is
  `sum over overlapping fixations of overlap_weight_ms * Gaussian(distance_to_center, sigma)`
  -- mathematically identical to blurring the whole image and sampling that point, verified
  against the raster approach (figures/checks/closed_form_vs_raster_blur_verification.png).
- Leave-one-out group vectors come from `total_vec - individual_vec` (exact, by linearity),
  not by resumming every other participant's fixations per participant -- see
  METHODS_DECISIONS.md, 2026-10-01, for why.
- Confirmed parameters (METHODS_DECISIONS.md): sigma=46px fixed, window=1.0s, step=0.25s,
  no minimum-data gate, no bound/percentile filtering, no 0-1 rescaling.
- Reads fixation data from a LOCAL COPY only -- never the live server mount.
- See diagnostic_concordance.py for the visual sanity-check this was
  validated against before being built as a batch script.

Output
------
data/results/concordance/fixation_density_concordance.csv -- one row per (pid, window):
  pid, video_name, window_idx, window_start_s, window_end_s, r

Usage
-----
    python scripts/analysis/calc_concordance_group.py \\
        --fixation_dir data/local_cache/fixations/adults \\
        --participant_summary_path data/local_cache/participant_summary.csv \\
        --param_code f329476c \\
        --duration_s 150.15
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[4]

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
from gaze_tiles import DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations  # noqa: E402,F401


def main():
    parser = argparse.ArgumentParser(
        description="Batch fixation density concordance: individual vs. leave-one-out group, "
                    "per sliding window, per participant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixation_dir", required=True,
                         help="LOCAL COPY of the adult I2MC fixation CSVs directory (never the server mount).")
    parser.add_argument("--participant_summary_path", required=True,
                         help="LOCAL COPY of participant_summary.csv.")
    parser.add_argument("--param_code", required=True,
                         help="I2MC parameter code suffix on the fixation CSV filenames (e.g. f329476c).")
    parser.add_argument("--video", default="pixar_birds")
    parser.add_argument("--duration_s", type=float, required=True,
                         help="Clip duration in seconds (e.g. from data/results/stimulus_prep/stimulus_inventory.csv) -- "
                              "determines how many windows span the clip.")
    parser.add_argument("--participants", nargs="*", default=None,
                         help="Restrict to these pids (default: all valid pids with data for --video).")
    parser.add_argument("--window_s", type=float, default=1.0)
    parser.add_argument("--step_s", type=float, default=0.25)
    parser.add_argument("--sigma_px", type=float, default=46.0)
    parser.add_argument("--patch_diameter_px", type=float, default=240)  # 2026-10-22: moved from 180px to 240px (50% overlap)
    parser.add_argument("--patch_n_cols", type=int, default=15)
    parser.add_argument("--results_dir", default=None, help="Output directory (default: data/results/concordance/ -- NOTE: data/ is a symlink to the mounted lab server in this repo).")
    parser.add_argument("--output_name", default="fixation_density_concordance.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "data" / "results" / "concordance"
    results_dir.mkdir(parents=True, exist_ok=True)

    fix_df = load_video_fixations(Path(args.fixation_dir), args.video, args.param_code,
                                   Path(args.participant_summary_path))

    all_pids = sorted(fix_df["pid"].unique())
    pids = args.participants if args.participants else all_pids
    missing = set(pids) - set(all_pids)
    assert not missing, f"pid(s) {missing} have no fixations for {args.video}."
    print(f"Running on {len(pids)} of {len(all_pids)} available participants.")

    patch_centers_df, grid_info = compute_patch_grid(args.patch_diameter_px, args.patch_n_cols,
                                                       DISPLAY_W, DISPLAY_H)
    cx = patch_centers_df["center_x"].to_numpy()
    cy = patch_centers_df["center_y"].to_numpy()
    n_tiles = len(patch_centers_df)
    print(f"Scene tiles: {grid_info['n_cols']}x{grid_info['n_rows']}={n_tiles} "
          f"(diameter={args.patch_diameter_px}px)")

    pid_arrays = build_participant_arrays(fix_df[fix_df["pid"].isin(pids)])

    n_windows = max(0, int((args.duration_s - args.window_s) // args.step_s) + 1)
    print(f"Clip duration {args.duration_s}s -> {n_windows} windows "
          f"(window={args.window_s}s, step={args.step_s}s)")

    t_start = time.perf_counter()
    rows = []
    for w in range(n_windows):
        start_s = w * args.step_s
        end_s = start_s + args.window_s
        start_ms, end_ms = start_s * 1000, end_s * 1000

        per_pid_vec = {}
        total_vec = np.zeros(n_tiles)
        for pid, (startT, endT, img_x, img_y) in pid_arrays.items():
            vec = closed_form_tile_values(startT, endT, img_x, img_y, start_ms, end_ms,
                                           args.sigma_px, cx, cy)
            per_pid_vec[pid] = vec
            total_vec += vec

        for pid, vec in per_pid_vec.items():
            group_vec = total_vec - vec  # leave-one-out, exact via linearity
            if vec.std() == 0 or group_vec.std() == 0:
                r = float("nan")
            else:
                r, _ = pearsonr(vec, group_vec)
            rows.append(dict(pid=pid, video_name=args.video, window_idx=w,
                              window_start_s=round(start_s, 3), window_end_s=round(end_s, 3), r=r))

        if w % 100 == 0 or w == n_windows - 1:
            print(f"  window {w}/{n_windows - 1} ({time.perf_counter() - t_start:.1f}s elapsed)")

    elapsed = time.perf_counter() - t_start
    out_df = pd.DataFrame(rows)
    out_path = results_dir / args.output_name
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(out_df)} rows ({len(pids)} participants x {n_windows} windows) to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s total, {1000 * elapsed / (len(pids) * n_windows):.2f}ms/(participant-window)")
    print(f"NaN rate: {out_df['r'].isna().mean():.1%}")


if __name__ == "__main__":
    main()

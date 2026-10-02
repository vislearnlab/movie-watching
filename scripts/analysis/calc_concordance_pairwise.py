"""
Fixation density concordance, PAIRWISE variant -- same metric as
calc_concordance_group.py, but structured like the real ISC scripts (every pair of
participants, not individual-vs-leave-one-out-pooled-group).

Still called "concordance", not "ISC" -- it's the same spatial density quantity as
calc_concordance_group.py, just pairwise-structured. See that script's docstring
for the naming rule and the closed-form/leave-one-out methodology this builds on.

- For every window, for every pair of participants, computes each person's OWN 120-tile
  closed-form density vector (just their own fixations -- no pooling), then Pearson r
  between the pair's two vectors directly.
- Averages all pairs' r into one mean_r per window -- same aggregation (and output
  schema) as calc_isc_timecourse.py, so the two are directly comparable/overlay-able.
- All pairs' correlations for a window are computed in one `np.corrcoef` call on the
  stacked (n_participants x 120) matrix of that window's vectors, rather than looping
  over pairs in Python -- the matrix's pairwise correlations are exactly the 820 pair
  values we want, just computed together.
- Same parameters as calc_concordance_group.py: sigma=46px, window=1.0s, step=0.25s,
  no minimum-data gate (METHODS_DECISIONS.md).

Reads fixation data from a LOCAL COPY only -- see METHODS_DECISIONS.md, "Local copies of
server data": never point this at the live server mount.

Output
------
data/results/concordance/fixation_density_concordance_pairwise.csv -- one row per window:
  video_name, window_idx, window_start_s, window_end_s, mean_r, n_pairs

Usage
-----
    python scripts/analysis/calc_concordance_pairwise.py \\
        --fixation_dir data/local_cache/fixations/adults \\
        --participant_summary_path data/local_cache/participant_summary.csv \\
        --param_code f329476c --duration_s 150.15
"""

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calc_concordance_group import (  # noqa: E402
    load_video_fixations, build_participant_arrays, closed_form_tile_values, DISPLAY_W, DISPLAY_H,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise fixation density concordance, windowed over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixation_dir", required=True,
                         help="LOCAL COPY of the adult I2MC fixation CSVs directory (never the server mount).")
    parser.add_argument("--participant_summary_path", required=True,
                         help="LOCAL COPY of participant_summary.csv.")
    parser.add_argument("--param_code", required=True)
    parser.add_argument("--video", default="pixar_birds")
    parser.add_argument("--duration_s", type=float, required=True)
    parser.add_argument("--window_s", type=float, default=1.0)
    parser.add_argument("--step_s", type=float, default=0.25)
    parser.add_argument("--sigma_px", type=float, default=46.0)
    parser.add_argument("--patch_diameter_px", type=float, default=180)
    parser.add_argument("--patch_n_cols", type=int, default=15)
    parser.add_argument("--results_dir", default=None, help="Output directory (default: data/results/concordance/ -- NOTE: data/ is a symlink to the mounted lab server in this repo).")
    parser.add_argument("--output_name", default="fixation_density_concordance_pairwise.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "data" / "results" / "concordance"
    results_dir.mkdir(parents=True, exist_ok=True)

    fix_df = load_video_fixations(Path(args.fixation_dir), args.video, args.param_code,
                                   Path(args.participant_summary_path))
    pids = sorted(fix_df["pid"].unique())
    n_pids = len(pids)
    n_possible_pairs = n_pids * (n_pids - 1) // 2
    print(f"{n_pids} participants -> {n_possible_pairs} pairs per window")

    patch_centers_df, grid_info = compute_patch_grid(args.patch_diameter_px, args.patch_n_cols,
                                                       DISPLAY_W, DISPLAY_H)
    cx = patch_centers_df["center_x"].to_numpy()
    cy = patch_centers_df["center_y"].to_numpy()
    n_tiles = len(patch_centers_df)
    print(f"Scene tiles: {grid_info['n_cols']}x{grid_info['n_rows']}={n_tiles}")

    pid_arrays = build_participant_arrays(fix_df)

    n_windows = max(0, int((args.duration_s - args.window_s) // args.step_s) + 1)
    print(f"Clip duration {args.duration_s}s -> {n_windows} windows "
          f"(window={args.window_s}s, step={args.step_s}s)")

    tri_rows, tri_cols = np.triu_indices(n_pids, k=1)  # upper triangle, excluding diagonal

    t_start = time.perf_counter()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # zero-variance rows -> NaN, expected
        for w in range(n_windows):
            start_s = w * args.step_s
            end_s = start_s + args.window_s
            start_ms, end_ms = start_s * 1000, end_s * 1000

            vecs = np.stack([
                closed_form_tile_values(startT, endT, img_x, img_y, start_ms, end_ms, args.sigma_px, cx, cy)
                for pid, (startT, endT, img_x, img_y) in ((p, pid_arrays[p]) for p in pids)
            ])  # (n_pids, n_tiles)

            with np.errstate(invalid="ignore", divide="ignore"):
                corr_matrix = np.corrcoef(vecs)  # (n_pids, n_pids), NaN for zero-variance rows

            pair_r = corr_matrix[tri_rows, tri_cols]
            valid = ~np.isnan(pair_r)
            mean_r = float(pair_r[valid].mean()) if valid.any() else float("nan")

            rows.append(dict(video_name=args.video, window_idx=w, window_start_s=round(start_s, 3),
                              window_end_s=round(end_s, 3), mean_r=mean_r, n_pairs=int(valid.sum())))

            if w % 100 == 0 or w == n_windows - 1:
                print(f"  window {w}/{n_windows - 1} ({time.perf_counter() - t_start:.1f}s elapsed)")

    elapsed = time.perf_counter() - t_start
    out_df = pd.DataFrame(rows)
    out_path = results_dir / args.output_name
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(out_df)} rows to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s total")
    print(f"mean_r: mean={out_df['mean_r'].mean():.3f}, NaN rate={out_df['mean_r'].isna().mean():.1%}")


if __name__ == "__main__":
    main()

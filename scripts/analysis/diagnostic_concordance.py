"""
Diagnostic figure for the individual-to-group fixation density concordance metric (first
build step -- see METHODS_DECISIONS.md, 2026-10-01 entries, for the full parameter
rationale). Called "concordance", not "ISC" -- ISC is reserved in this project for the
existing Franchak-style x/y time-series analysis (calc_isc_rawgaze.py, calc_isc_fixations.py,
isc_analysis.qmd); this is a different, spatial metric.

For one sample participant and a handful of consecutive sliding windows, shows:
  1. that participant's duration-weighted, Gaussian-smoothed fixation density
  2. the leave-one-out group's density (same participants minus the sample one) for the
     same windows
  3. the 120 scene-tile centers (the caption-validation patch grid) as red dots, where
     this metric samples heat values -- overlaid on every panel
  4. the Pearson r between the two tile-value vectors, per window

This is a sanity check / visual companion to the batch script
(calc_concordance_group.py) -- it does not write any CSV of tile values, only the
figure. The displayed heatmap images are still built by rasterizing + Gaussian-blurring
the whole frame (purely for the picture), but the r values use the same closed-form
`closed_form_tile_values` the batch script uses -- the two were verified equivalent (see
figures/checks/closed_form_vs_raster_blur_verification.png and METHODS_DECISIONS.md)
before the batch script switched to closed-form for speed.

Confirmed parameters used here (METHODS_DECISIONS.md, 2026-10-01):
  - Scene tiles: the 120-patch grid from plot_patch_grid.py (180px diameter, 15x8,
    edge-touching).
  - Heatmap: duration-weighted (each fixation's contribution to a window is its overlap
    duration with that window, in ms -- not a plain fixation count, and not discretized
    into time bins first).
  - Smoothing sigma: 46px, fixed for everyone.
  - Group heatmap: leave-one-out (sample participant excluded from "group").
  - Similarity metric: Pearson r across the 120 tile values.
  - Window/step: 1.0s / 0.25s, no minimum-data gate on a window.

Reads fixation data from a LOCAL COPY only -- see METHODS_DECISIONS.md, "Local copies of
server data": never point this at the live server mount.

Output
------
figures/checks/fixation_density_concordance_diagnostic.png

Usage
-----
    python scripts/analysis/diagnostic_concordance.py \\
        --fixation_dir data/local_cache/fixations/adults \\
        --participant_summary_path data/local_cache/participant_summary.csv \\
        --param_code f329476c
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402
from calc_concordance_group import load_video_fixations, closed_form_tile_values  # noqa: E402

DISPLAY_W, DISPLAY_H = 1920, 1080


def window_density(fix_df: pd.DataFrame, window_start_ms: float, window_end_ms: float,
                    sigma_px: float) -> np.ndarray:
    """
    Duration-weighted, Gaussian-smoothed density for every fixation row in `fix_df` that
    overlaps [window_start_ms, window_end_ms). Each fixation's weight is its overlap
    duration with the window (ms), not its full duration and not a bare count -- a
    fixation that only partly falls in the window contributes only that part.
    """
    start = fix_df["startT"].to_numpy()
    end = fix_df["endT"].to_numpy()
    overlap_ms = np.minimum(end, window_end_ms) - np.maximum(start, window_start_ms)
    mask = overlap_ms > 0
    if not mask.any():
        return gaussian_filter(np.zeros((DISPLAY_H, DISPLAY_W)), sigma=sigma_px)

    x = np.clip(fix_df["xpos"].to_numpy()[mask], 0, DISPLAY_W - 1).astype(int)
    y = np.clip(DISPLAY_H - fix_df["ypos"].to_numpy()[mask], 0, DISPLAY_H - 1).astype(int)
    w = overlap_ms[mask]

    density = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.float64)
    np.add.at(density, (y, x), w)
    return gaussian_filter(density, sigma=sigma_px)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic figure: sample participant vs. leave-one-out group fixation "
                    "density, at the scene-tile centers, over a few sliding windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixation_dir", required=True,
                         help="LOCAL COPY of the adult I2MC fixation CSVs directory (never the server mount).")
    parser.add_argument("--participant_summary_path", required=True,
                         help="LOCAL COPY of participant_summary.csv.")
    parser.add_argument("--param_code", required=True,
                         help="I2MC parameter code suffix on the fixation CSV filenames (e.g. f329476c).")
    parser.add_argument("--video", default="pixar_birds")
    parser.add_argument("--participant", default=None,
                         help="pid to use as the sample participant (default: first valid pid, sorted).")
    parser.add_argument("--n_windows", type=int, default=5)
    parser.add_argument("--first_window_idx", type=int, default=0,
                         help="Window index to start from (window 0 starts at t=0).")
    parser.add_argument("--window_s", type=float, default=1.0)
    parser.add_argument("--step_s", type=float, default=0.25)
    parser.add_argument("--sigma_px", type=float, default=46.0)
    parser.add_argument("--patch_diameter_px", type=float, default=180)
    parser.add_argument("--patch_n_cols", type=int, default=15)
    parser.add_argument("--checks_dir", default=None, help="Output directory (default: figures/checks/).")
    args = parser.parse_args()

    checks_dir = Path(args.checks_dir) if args.checks_dir else PROJECT_ROOT / "figures" / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    fix_df = load_video_fixations(Path(args.fixation_dir), args.video, args.param_code,
                                   Path(args.participant_summary_path))

    sample_pid = args.participant or sorted(fix_df["pid"].unique())[0]
    assert sample_pid in fix_df["pid"].values, f"pid {sample_pid!r} has no fixations for {args.video}."
    print(f"Sample participant: {sample_pid}")

    indiv_fix = fix_df[fix_df["pid"] == sample_pid]
    group_fix = fix_df[fix_df["pid"] != sample_pid]  # leave-one-out
    print(f"Leave-one-out group: {group_fix['pid'].nunique()} participants.")

    patch_centers_df, grid_info = compute_patch_grid(args.patch_diameter_px, args.patch_n_cols,
                                                       DISPLAY_W, DISPLAY_H)
    print(f"Scene tiles: {grid_info['n_cols']}x{grid_info['n_rows']}={len(patch_centers_df)} "
          f"(diameter={args.patch_diameter_px}px)")

    window_bins = range(args.first_window_idx, args.first_window_idx + args.n_windows)
    results = []
    for w in window_bins:
        start_s = w * args.step_s
        end_s = start_s + args.window_s
        start_ms, end_ms = start_s * 1000, end_s * 1000

        # Displayed heatmaps: raster + Gaussian blur, for the picture only.
        indiv_density = window_density(indiv_fix, start_ms, end_ms, args.sigma_px)
        group_density = window_density(group_fix, start_ms, end_ms, args.sigma_px)

        # r itself: closed-form at the tile centers, same function the batch script uses
        # (verified equivalent to sampling the rasterized/blurred image above).
        cx, cy = patch_centers_df["center_x"].to_numpy(), patch_centers_df["center_y"].to_numpy()
        indiv_vec = closed_form_tile_values(indiv_fix["startT"].to_numpy(), indiv_fix["endT"].to_numpy(),
                                             indiv_fix["xpos"].to_numpy(), DISPLAY_H - indiv_fix["ypos"].to_numpy(),
                                             start_ms, end_ms, args.sigma_px, cx, cy)
        group_vec = closed_form_tile_values(group_fix["startT"].to_numpy(), group_fix["endT"].to_numpy(),
                                             group_fix["xpos"].to_numpy(), DISPLAY_H - group_fix["ypos"].to_numpy(),
                                             start_ms, end_ms, args.sigma_px, cx, cy)

        if indiv_vec.std() == 0 or group_vec.std() == 0:
            r = float("nan")
        else:
            r, _ = pearsonr(indiv_vec, group_vec)

        results.append(dict(window_idx=w, start_s=start_s, end_s=end_s,
                             indiv_density=indiv_density, group_density=group_density, r=r))
        print(f"window {w}: t=[{start_s:.2f}, {end_s:.2f})s  r={r:.3f}")

    # ── Figure: row 1 = sample participant, row 2 = leave-one-out group ────────────────
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 2 * (DISPLAY_H / DISPLAY_W) * 4 + 1.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, res in enumerate(results):
        for row, (density, label) in enumerate([(res["indiv_density"], f"participant {sample_pid}"),
                                                  (res["group_density"], "group (leave-one-out)")]):
            ax = axes[row, col]
            vmax = density.max()
            ax.imshow(density, cmap="inferno", vmin=0, vmax=vmax if vmax > 0 else None,
                      extent=[0, DISPLAY_W, DISPLAY_H, 0])
            ax.scatter(patch_centers_df["center_x"], patch_centers_df["center_y"],
                       s=4, c="red", marker="o", linewidths=0)
            ax.set_xlim(0, DISPLAY_W)
            ax.set_ylim(DISPLAY_H, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t=[{res['start_s']:.2f},{res['end_s']:.2f})s\nr={res['r']:.3f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    fig.suptitle(f"Individual vs. leave-one-out group fixation density, {args.video} "
                 f"(sigma={args.sigma_px}px, window={args.window_s}s, step={args.step_s}s)\n"
                 f"red dots: {len(patch_centers_df)} scene-tile centers (where Pearson r is computed)",
                 fontsize=11)
    fig.tight_layout()

    out_path = checks_dir / "fixation_density_concordance_diagnostic.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

"""
Patch grid preview for the caption-validation stimulus-prep pipeline (Task 5).

FINAL, CONFIRMED patch grid (2026-09-29 -- see METHODS_DECISIONS.md, "FINAL: patch grid
confirmed at 180px diameter, 15x8, edge-touching", which supersedes an earlier 192px
choice and an even earlier 200px/144-patch "rings removed" grid): diameter=180px,
15 columns x 8 rows = 120 patches. The outermost patches' circle perimeter sits
exactly on the display edge rather than overshooting it (`compute_patch_grid` in preprocessing/stimulus_prep/patch_grid.py),
so a thin strip near the edges/corners is left untiled by design -- that tradeoff was
confirmed visually against all 24 representative frames before being adopted.

This script only previews the grid overlay on the representative frames -- it is NOT
the rest of Task 5 (per-frame patch PNGs, patch_index.csv, neighbor table, coverage
checks), which has not been built yet.

Reads: all PNGs in data/results/stimulus_prep/representative_frames/ (expects 24, from
representative_frames.py).

Part of the pipeline described in stimulus_inventory.py's docstring; see METHODS_DECISIONS.md at
the project root for the full assumptions/decisions log, including why the screen
geometry (--screen_width_cm/--screen_height_cm/--viewing_distance_cm) is still an
unconfirmed estimate rather than a measured constant.

Output
------
figures/checks/patch_grid_preview/{frame_id}_patch_grid.png -- the grid (green circles +
    red '+' markers) overlaid on each representative frame, with the display boundary
    outlined in yellow. Confirm visually that the outermost circles touch (not cross) the
    yellow boundary.

Usage
-----
    python scripts/decision_making/plot_patch_grid.py
    python scripts/decision_making/plot_patch_grid.py --patch_diameter_px 200 --patch_n_cols 12
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing", "stimulus_prep"))
import _video_utils as vu  # noqa: E402
from patch_grid import compute_patch_grid, px_per_degree  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Overlay the confirmed patch grid on all representative frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default=None,
                        help="Directory to read representative_frames/ from (default: "
                             "data/results/stimulus_prep/ -- NOTE: data/ is a symlink to the mounted "
                             "lab server in this repo).")
    parser.add_argument("--checks_dir", default=None,
                        help="Output directory for the check figures (default: figures/checks/).")
    parser.add_argument("--patch_diameter_px", type=float, default=240,
                        help="Patch circle diameter in pixels. Moved from 180px to 240px 2026-10-22 (50% overlap) -- see METHODS_DECISIONS.md.")
    parser.add_argument("--patch_n_cols", type=int, default=15,
                        help="Number of columns (row count is derived). FINAL value confirmed 2026-09-29.")
    parser.add_argument("--display_width", type=int, default=1920)
    parser.add_argument("--display_height", type=int, default=1080)
    parser.add_argument("--screen_width_cm", type=float, default=53.14,
                        help="Still an unconfirmed estimate -- see METHODS_DECISIONS.md, 'Screen geometry for "
                             "visual-angle calculations'.")
    parser.add_argument("--viewing_distance_cm", type=float, default=70.0)
    parser.add_argument("--n_frames_expected", type=int, default=24,
                        help="Expected number of representative frames; mismatches warn but don't block.")
    args = parser.parse_args()

    project_root = vu.project_root()
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "data" / "results" / "stimulus_prep"
    checks_dir = Path(args.checks_dir) if args.checks_dir else project_root / "figures" / "checks"

    grid_df, grid_info = compute_patch_grid(args.patch_diameter_px, args.patch_n_cols,
                                             args.display_width, args.display_height)
    radius = grid_info["radius"]
    deg = args.patch_diameter_px / px_per_degree(args.display_width, args.screen_width_cm, args.viewing_distance_cm)

    print(f"diameter={args.patch_diameter_px}px ({deg:.2f} deg)  radius={radius}px")
    print(f"n_cols={grid_info['n_cols']}  spacing_x={grid_info['spacing_x']:.2f}px  ratio_x={grid_info['ratio_x']:.3f}")
    print(f"n_rows={grid_info['n_rows']}  spacing_y={grid_info['spacing_y']:.2f}px  ratio_y={grid_info['ratio_y']:.3f}")
    print(f"n_patches={len(grid_df)}")

    rep_frames_dir = results_dir / "representative_frames"
    frame_paths = sorted(p for p in rep_frames_dir.glob("*.png") if not p.name.startswith("_"))
    if not frame_paths:
        sys.exit(f"ERROR: No representative frames found in {rep_frames_dir} -- run representative_frames.py first.")
    if len(frame_paths) != args.n_frames_expected:
        print(f"WARNING: expected {args.n_frames_expected} representative frames, found {len(frame_paths)}.")

    preview_dir = checks_dir / "patch_grid_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in frame_paths:
        frame_id = frame_path.stem
        img = Image.open(frame_path)

        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.imshow(img)

        for _, r in grid_df.iterrows():
            ax.add_patch(mpatches.Circle((r["center_x"], r["center_y"]), radius,
                                         edgecolor="lime", facecolor="none", linewidth=1))
            ax.plot(r["center_x"], r["center_y"], marker="+", color="red", markersize=5, mew=1)

        # Outline the display boundary so it's easy to confirm the outermost circles touch it.
        ax.add_patch(mpatches.Rectangle((0, 0), args.display_width, args.display_height,
                                        fill=False, edgecolor="yellow", linewidth=1.5))

        ax.set_xlim(0, args.display_width)
        ax.set_ylim(args.display_height, 0)
        ax.axis("off")
        ax.set_title(f"{frame_id}  (diameter={args.patch_diameter_px}px, "
                     f"{grid_info['n_cols']}x{grid_info['n_rows']}={len(grid_df)} patches)", fontsize=10)

        preview_path = preview_dir / f"{frame_id}_patch_grid.png"
        fig.savefig(preview_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved {len(frame_paths)} previews to {preview_dir}")


if __name__ == "__main__":
    main()

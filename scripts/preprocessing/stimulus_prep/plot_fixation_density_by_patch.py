"""
Adult fixation density by circular patch (overlap-aware), on the current patch grid.

How many adult fixations fall in each patch of the pipeline's one confirmed spatial grid
-- circular, overlapping patches, not a plain non-overlapping grid tile (the earlier
16x9-tile version of this question, fixation_density_by_tile.py, is retired; see
METHODS_DECISIONS.md, 2026-10-01, "Removed the 16x9 plain-grid-tile script"). Because patches
overlap, a single fixation can count toward more than one patch, so per-patch counts do
not sum to the total fixation count (expected, not a bug).

Uses the FINAL, confirmed patch grid from plot_patch_grid.py: 180px diameter, 15
columns x 8 rows = 120 patches, edge-touching convention (METHODS_DECISIONS.md, 2026-09-29,
"FINAL: patch grid confirmed..."). This used to be frozen to an earlier, retired
200px/144-patch "rings removed" snapshot (the grid an earlier ring-exclusion decision
actually ran against) -- see METHODS_DECISIONS.md, 2026-10-01, "Un-froze fixation_density_by_patch
to the current grid" for why that snapshot was dropped in favor of tracking the live grid.

Reads fixation data from a LOCAL COPY only -- see METHODS_DECISIONS.md, "Local copies of server
data": never point this at the live server mount.

Output
------
outputs/patch_fixation_counts.csv (row, col, center_x, center_y, n_fixations)
outputs/checks/fixation_density_by_patch.png -- the current grid's circular patches,
    filled by fixation count.

Usage
-----
    python scripts/preprocessing/stimulus_prep/plot_fixation_density_by_patch.py \\
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _video_utils as vu
from plot_patch_grid import compute_patch_grid, px_per_degree


def load_adult_fixations(fixation_dir: Path, param_code: str, participant_summary_path: Path) -> pd.DataFrame:
    """
    Load every adult I2MC fixation CSV at `param_code` from `fixation_dir`, keep only
    participants flagged valid_data=True in participant_summary.csv (same convention as
    preprocessing/viz/visualize_heatmap.py), and return one concatenated DataFrame -- one
    row per fixation, across all participants and videos.
    """
    fix_paths = sorted(fixation_dir.glob(f"*_{param_code}.csv"))
    assert fix_paths, (
        f"No fixation CSVs found in {fixation_dir} for param_code {param_code!r} -- copy the "
        "adult fixation CSVs for this parameter code from data/preprocessed/fixations/adults first."
    )

    summary = pd.read_csv(participant_summary_path)
    valid_ids = set(
        summary.loc[
            (summary["participant_type"] == "adults") & (summary["valid_data"] == True),
            "participant_id",
        ]
    )

    all_fix = pd.concat([pd.read_csv(p) for p in fix_paths], ignore_index=True)

    n_before = all_fix["pid"].nunique()
    all_fix = all_fix[all_fix["pid"].isin(valid_ids)].copy()
    n_after = all_fix["pid"].nunique()
    print(f"Loaded {len(all_fix)} fixations from {len(fix_paths)} files, {n_before} participants; "
          f"{n_after} remain after the valid_data filter.")

    return all_fix


def compute_patch_fixation_membership(fix_df: pd.DataFrame, patch_centers_df: pd.DataFrame,
                                       display_height: int, radius_px: float) -> np.ndarray:
    """
    Boolean (n_fixations x n_patches) membership matrix: True where a fixation's (x, y)
    falls within `radius_px` of a patch center -- i.e. inside that circular patch.

    Patches overlap by design (diameter > center spacing), so a single fixation can be
    True for more than one patch. Fixation coordinates are used as-is (not clipped to the
    display). Same y-flip as compute_tile_fixation_counts in fixation_density_by_tile.py.
    """
    img_x = fix_df["xpos"].to_numpy()[:, None]
    img_y = (display_height - fix_df["ypos"].to_numpy())[:, None]
    centers_x = patch_centers_df["center_x"].to_numpy()[None, :]
    centers_y = patch_centers_df["center_y"].to_numpy()[None, :]

    dist_sq = (img_x - centers_x) ** 2 + (img_y - centers_y) ** 2
    return dist_sq <= radius_px ** 2


def save_check_figure(patch_centers_df: pd.DataFrame, radius_px: float,
                       display_width: int, display_height: int, checks_dir: Path) -> Path:
    cmap = plt.get_cmap("Blues")
    vmax = patch_centers_df["n_fixations"].max()
    norm = plt.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_facecolor("white")

    for _, r in patch_centers_df.iterrows():
        color = cmap(norm(r["n_fixations"]))
        ax.add_patch(mpatches.Circle((r["center_x"], r["center_y"]), radius_px,
                                     facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.6))
        text_color = "white" if r["n_fixations"] > vmax * 0.6 else "#333333"
        ax.text(r["center_x"], r["center_y"], f"{int(r['n_fixations'])}",
                ha="center", va="center", fontsize=6, color=text_color)

    ax.add_patch(mpatches.Rectangle((0, 0), display_width, display_height,
                                    fill=False, edgecolor="yellow", linewidth=1.5))

    ax.set_xlim(0, display_width)
    ax.set_ylim(display_height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Adult fixation count per circular patch (across all videos) -- current grid")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="n_fixations (overlap-counted)", fraction=0.03, pad=0.02)
    fig.tight_layout()

    check_path = checks_dir / "fixation_density_by_patch.png"
    fig.savefig(check_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return check_path


def main():
    parser = argparse.ArgumentParser(
        description="Adult fixation density by circular patch, on the current (final) patch grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixation_dir", required=True,
                         help="LOCAL COPY of the adult I2MC fixation CSVs directory (never the server mount).")
    parser.add_argument("--participant_summary_path", required=True,
                         help="LOCAL COPY of participant_summary.csv.")
    parser.add_argument("--param_code", required=True,
                         help="I2MC parameter code suffix on the fixation CSV filenames (e.g. f329476c).")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: outputs/).")
    parser.add_argument("--display_width", type=int, default=1920)
    parser.add_argument("--display_height", type=int, default=1080)
    parser.add_argument("--patch_diameter_px", type=float, default=180,
                        help="FINAL value confirmed 2026-09-29 -- see METHODS_DECISIONS.md.")
    parser.add_argument("--patch_n_cols", type=int, default=15,
                        help="FINAL value confirmed 2026-09-29.")
    parser.add_argument("--screen_width_cm", type=float, default=53.14)
    parser.add_argument("--viewing_distance_cm", type=float, default=70.0)
    args = parser.parse_args()

    project_root = vu.project_root()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs"
    checks_dir = output_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    fix_df = load_adult_fixations(Path(args.fixation_dir), args.param_code, Path(args.participant_summary_path))

    patch_centers_df, grid_info = compute_patch_grid(args.patch_diameter_px, args.patch_n_cols,
                                                       args.display_width, args.display_height)
    radius_px = grid_info["radius"]
    deg = args.patch_diameter_px / px_per_degree(args.display_width, args.screen_width_cm, args.viewing_distance_cm)
    print(f"Grid: diameter={args.patch_diameter_px}px ({deg:.2f} deg)  "
          f"{grid_info['n_cols']}x{grid_info['n_rows']}={len(patch_centers_df)} patches")

    membership = compute_patch_fixation_membership(fix_df, patch_centers_df, args.display_height, radius_px)
    patch_centers_df["n_fixations"] = membership.sum(axis=0)

    patch_counts_path = output_dir / "patch_fixation_counts.csv"
    patch_centers_df.to_csv(patch_counts_path, index=False)

    total = len(fix_df)
    n_covered = membership.any(axis=1).sum()
    print(f"Radius: {radius_px}px (diameter={args.patch_diameter_px}px)")
    print(f"{n_covered}/{total} fixations ({100 * n_covered / total:.2f}%) fall inside at least one "
          f"of the {len(patch_centers_df)} current-grid patches.")
    print(f"Sum of per-patch counts: {patch_centers_df['n_fixations'].sum()} "
          f"(> {total} is expected -- patches overlap, so fixations in overlap zones are counted more than once).")

    print(f"\nWrote {patch_counts_path}")

    check_path = save_check_figure(patch_centers_df, radius_px, args.display_width, args.display_height, checks_dir)
    print(f"Saved check to {check_path}")


if __name__ == "__main__":
    main()

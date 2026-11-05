"""
Shared patch-grid geometry: the confirmed 180px, 15x8 edge-touching grid (compute_patch_grid) and px/deg.
Imported by the tile, entropy, within/between and schematic scripts; the grid preview lives in
scripts/decision_making/plot_patch_grid.py.
"""

import math

import pandas as pd


def px_per_degree(display_width_px: int, screen_width_cm: float, viewing_distance_cm: float) -> float:
    """Pixels per degree of visual angle for the given screen geometry (METHODS_DECISIONS.md,
    'Screen geometry for visual-angle calculations' -- still an unconfirmed assumption)."""
    full_width_deg = 2 * math.degrees(math.atan((screen_width_cm / 2) / viewing_distance_cm))
    return display_width_px / full_width_deg


def compute_patch_grid(diameter_px: float, n_cols: int, display_width: int, display_height: int) -> tuple[pd.DataFrame, dict]:
    """
    Edge-touching grid: the OUTERMOST patches' circle edge sits exactly on the display
    boundary rather than overshooting it, so centers are inset by exactly `radius` (not
    half a cell). Some display area near the edges/corners -- wherever no circle reaches
    -- is left untiled by design.

    n_cols fixes the column count; row count is chosen so vertical spacing stays close to
    horizontal spacing while also hitting the top/bottom edges exactly, so spacing_x and
    spacing_y (and the resulting overlap ratio) can differ slightly between axes -- see the
    returned info dict.

    row/col are 0-indexed grid positions, used for patch_id = f"{frame_id}_r{row}_c{col}".
    """
    assert n_cols >= 2, "Need at least 2 columns to define a spacing between centers."
    radius = diameter_px / 2
    usable_w = display_width - diameter_px
    usable_h = display_height - diameter_px
    assert usable_w > 0 and usable_h > 0, (
        f"diameter_px={diameter_px} is too large for a {display_width}x{display_height} display."
    )

    spacing_x = usable_w / (n_cols - 1)
    n_rows = max(2, round(usable_h / spacing_x) + 1)
    spacing_y = usable_h / (n_rows - 1)

    rows = [
        {"row": row, "col": col,
         "center_x": radius + col * spacing_x, "center_y": radius + row * spacing_y}
        for row in range(n_rows) for col in range(n_cols)
    ]
    info = {
        "n_cols": n_cols, "n_rows": n_rows, "spacing_x": spacing_x, "spacing_y": spacing_y,
        "ratio_x": diameter_px / spacing_x, "ratio_y": diameter_px / spacing_y,
        "diameter_px": diameter_px, "radius": radius,
    }
    return pd.DataFrame(rows), info

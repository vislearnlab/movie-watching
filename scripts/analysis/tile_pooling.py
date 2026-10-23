"""
Box-mean pooling of a dense per-pixel map onto the 120-tile grid, using the exact square
footprint DeepMeaning crops for each fine patch (240px on the 1920x1080 canvas, centered on
the tile center), so motion/saliency/meaning/gaze all share tile centers.
"""

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402

DISPLAY_W, DISPLAY_H = 1920, 1080
PATCH_PX = 240  # matches DeepMeaning FINE_PATCH_SIZE
N_COLS = 15

CENTERS, _GRID_INFO = compute_patch_grid(PATCH_PX, N_COLS, DISPLAY_W, DISPLAY_H)
assert len(CENTERS) == 120


@lru_cache(maxsize=None)
def _box_bounds(src_w: int, src_h: int) -> tuple:
    # Same rounding as DeepMeaning's crop (round(c - size/2)), then mapped canvas -> source.
    sx, sy = src_w / DISPLAY_W, src_h / DISPLAY_H
    x0c = np.round(CENTERS["center_x"].to_numpy() - PATCH_PX / 2)
    y0c = np.round(CENTERS["center_y"].to_numpy() - PATCH_PX / 2)
    x0, x1 = np.round(x0c * sx).astype(int), np.round((x0c + PATCH_PX) * sx).astype(int)
    y0, y1 = np.round(y0c * sy).astype(int), np.round((y0c + PATCH_PX) * sy).astype(int)
    assert x0.min() >= 0 and y0.min() >= 0 and x1.max() <= src_w and y1.max() <= src_h
    return x0, x1, y0, y1


def tile_box_means(img_map: np.ndarray) -> np.ndarray:
    """(H, W) map in source-frame pixels -> (120,) row-major tile means."""
    h, w = img_map.shape
    x0, x1, y0, y1 = _box_bounds(w, h)
    ii = cv2.integral(np.ascontiguousarray(img_map, dtype=np.float64))
    sums = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return sums / ((x1 - x0) * (y1 - y0))

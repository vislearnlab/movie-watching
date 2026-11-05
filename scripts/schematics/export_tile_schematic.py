"""
Data for the tile schematic figure (plot_tile_schematic.R): one video frame per time point plus the group's 120-tile
gaze vector over a 0.5s window centered on it (closed-form Gaussian, sigma=46, 240px patches).

Output:
    figures/presentations/tile_schematic/frames/<video>_<t>s.png
    data/results/presentations/tile_schematic_<video>.csv   t, tile, col, row, center_x, center_y, value (sums to 1 per t)

Usage: python scripts/schematics/export_tile_schematic.py --video pixar_birds --times 10 30 60
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from gaze_tiles import DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations  # noqa: E402
from patch_grid import compute_patch_grid  # noqa: E402

FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / "adults"
STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
FRAME_DIR = PROJECT_ROOT / "figures" / "presentations" / "tile_schematic" / "frames"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "presentations"
SIGMA, WINDOW_S = 46.0, 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="pixar_birds")
    p.add_argument("--times", type=float, nargs="+", required=True)
    a = p.parse_args()

    centers, _ = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)
    cx, cy = centers.center_x.to_numpy(), centers.center_y.to_numpy()
    arrays = build_participant_arrays(load_video_fixations(FIXATION_DIR, a.video, "f329476c"))
    FRAME_DIR.mkdir(parents=True, exist_ok=True); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in a.times:
        s, e = (t - WINDOW_S / 2) * 1000, (t + WINDOW_S / 2) * 1000
        v = sum(closed_form_tile_values(*arr, s, e, SIGMA, cx, cy) for arr in arrays.values())
        v = v / v.sum()
        rows.append(pd.DataFrame(dict(t=t, tile=np.arange(1, len(v) + 1), col=((cx - 120) // 120).astype(int) + 1,
                                      row=((cy - 120) // 120).astype(int) + 1, center_x=cx, center_y=cy, value=v)))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(STIMULI_DIR / f"{a.video}_stripped.mp4"),
                        "-frames:v", "1", "-vf", f"scale={DISPLAY_W}:{DISPLAY_H}", str(FRAME_DIR / f"{a.video}_{t:g}s.png")], check=True)
    out = RESULTS_DIR / f"tile_schematic_{a.video}.csv"
    pd.concat(rows).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()

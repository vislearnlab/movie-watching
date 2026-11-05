"""
Data for the leave-one-out schematic (plot_loo_schematic.R): one participant's 120-tile gaze vector and the group's
vector without them (leave-one-out), in consecutive non-overlapping 0.5s windows starting at --t.

Output: data/results/presentations/loo_schematic_<video>.csv   who, t, tile, center_x, center_y, value, cosine_dist
        figures/presentations/tile_schematic/frames/<video>_<t>s.png

Usage: python scripts/schematics/export_loo_schematic.py --video sesameus_1 --t 30 [--n-windows 3] [--pid ...]
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
from export_tile_schematic import FIXATION_DIR, FRAME_DIR, RESULTS_DIR, SIGMA, STIMULI_DIR, WINDOW_S  # noqa: E402


def cos_dist(a, b):
    return 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="sesameus_1")
    p.add_argument("--t", type=float, required=True, help="center of the first window (s)")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--pid", help="participant to show; default = median distance to the group among those with gaze in every window")
    a = p.parse_args()

    centers, _ = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)
    cx, cy = centers.center_x.to_numpy(), centers.center_y.to_numpy()
    arrays = build_participant_arrays(load_video_fixations(FIXATION_DIR, a.video, "f329476c"))
    pids = sorted(arrays)
    times = a.t + WINDOW_S * np.arange(a.n_windows)
    V = np.array([[closed_form_tile_values(*arrays[pid], (t - WINDOW_S / 2) * 1000, (t + WINDOW_S / 2) * 1000, SIGMA, cx, cy)
                   for t in times] for pid in pids])  # (people, windows, tiles)
    total = V.sum(0)
    ok = [i for i in range(len(pids)) if (V[i].sum(1) > 0).all()]
    dists = {i: np.mean([cos_dist(V[i, w], total[w] - V[i, w]) for w in range(len(times))]) for i in ok}
    i = pids.index(a.pid) if a.pid else sorted(ok, key=dists.get)[len(ok) // 2]
    print(f"participant {pids[i]}: mean cosine distance to group {dists[i]:.3f} (median of {len(ok)} with gaze in all windows)")

    rows = []
    for w, t in enumerate(times):
        for who, v in [("individual", V[i, w]), ("group_loo", total[w] - V[i, w])]:
            rows.append(pd.DataFrame(dict(who=who, t=t, tile=np.arange(1, 121), center_x=cx, center_y=cy, value=v / v.sum(),
                                          cosine_dist=cos_dist(V[i, w], total[w] - V[i, w]))))
    out = RESULTS_DIR / f"loo_schematic_{a.video}.csv"
    pd.concat(rows).to_csv(out, index=False)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.t), "-i", str(STIMULI_DIR / f"{a.video}_stripped.mp4"),
                    "-frames:v", "1", "-vf", f"scale={DISPLAY_W}:{DISPLAY_H}", str(FRAME_DIR / f"{a.video}_{a.t:g}s.png")], check=True)
    print("Wrote", out)


if __name__ == "__main__":
    main()

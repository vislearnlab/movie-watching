"""
Segment-level entropy across a window-size and grid-size sweep -- robustness checks for
"pixar is less entropic than frank/sesame" (METHODS_DECISIONS.md, calc_entropy_by_segment.py).

Window sweep: WINDOW_S in [0.5,1.0,1.5,2.0]s, 120-tile grid. N_FIX_FIXED = floor(p5) of
each window size's own fixation-count distribution (53/78/105/132), not one fixed N.

Grid sweep: n_tiles in [6,15,66,120], WINDOW_S=1.0s. Diameters (806/510/243/240px) matched
to the 120-tile grid's overlap ratio (~1.448) -- mismatched ratios gave a spurious null at
6 tiles in an earlier pass.

Shared condition (WINDOW_S=1.0, 120 tiles, N=78) tagged by `sweep` in both outputs.

Output: entropy_sensitivity_sweep.csv -- sweep, window_s, n_tiles, diameter_px,
n_fix_fixed, video, block, segment_idx, mean_entropy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from calc_concordance_group import DISPLAY_H, DISPLAY_W, build_participant_arrays, load_video_fixations  # noqa: E402
from plot_patch_grid import compute_patch_grid  # noqa: E402

STEP_S, SIGMA = 0.25, 46.0
N_DRAWS, RNG_SEED = 100, 0

WINDOW_SWEEP = {0.5: 53, 1.0: 78, 1.5: 105, 2.0: 132}  # window_s -> N_FIX_FIXED (floor(p5))
# 2026-10-22: 120-tile entry's diameter moved 180->240px to match the current tile-center grid;
# the 6/15/66-tile diameters are still ratio-matched (~1.448) to the OLD 180px grid, not
# re-derived for the new 240px/ratio-2.0 grid. Not re-run; kept for historical record only.
GRID_SWEEP = [(806, 3, 6), (510, 5, 15), (243, 11, 66), (240, 15, 120)]  # (diameter, n_cols, n_tiles)
BASELINE_WINDOW_S, BASELINE_N_FIX = 1.0, 78

DURATIONS = {
    "frank_complex": 60.32, "frank_objects": 60.12, "frank_play": 60.92,
    "pixar_birds": 150.15,
    "sesameindia_1": 60.00, "sesameindia_2": 60.84, "sesameus_1": 60.5605, "sesameus_2": 60.827433,
    "slow_animals": 60.727333, "slow_hands": 60.72, "slow_inanimate": 60.727333, "slow_people": 60.7607,
}
BLOCKS = {
    "frank_complex": "frank", "frank_objects": "frank", "frank_play": "frank",
    "pixar_birds": "pixar",
    "sesameindia_1": "sesame", "sesameindia_2": "sesame", "sesameus_1": "sesame", "sesameus_2": "sesame",
    "slow_animals": "slow", "slow_hands": "slow", "slow_inanimate": "slow", "slow_people": "slow",
}

FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / "adults"
PARTICIPANT_SUMMARY_PATH = PROJECT_ROOT / "data" / "local_cache" / "participant_summary.csv"
CUTS_PATH = PROJECT_ROOT / "annotations" / "scene-cuts" / "all_clips_cuts.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "entropy_by_segment" / "entropy_sensitivity_sweep.csv"


def gauss_density(x, y, w, cx, cy, sigma):
    dist_sq = (x[:, None] - cx[None, :]) ** 2 + (y[:, None] - cy[None, :]) ** 2
    gauss = np.exp(-dist_sq / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return (w[:, None] * gauss).sum(axis=0)


def entropy_from_vec(vec):
    total = vec.sum()
    if total <= 0:
        return np.nan
    p = vec / total
    nz = p[p > 0]
    return -(nz * np.log2(nz)).sum()


def segment_entropy(video, dur, arrays, window_s, n_fix_fixed, cx, cy, rng, cuts_df):
    """Rarefied entropy per window, then averaged (full-containment only) per segment."""
    pids = sorted(arrays)
    n_windows = max(0, int((dur - window_s) // STEP_S) + 1)
    ent = np.full(n_windows, np.nan)
    for w in range(n_windows):
        s, e = w * STEP_S * 1000, (w * STEP_S + window_s) * 1000
        xs, ys, ws = [], [], []
        for p in pids:
            startT, endT, img_x, img_y = arrays[p]
            overlap = np.minimum(endT, e) - np.maximum(startT, s)
            m = overlap > 0
            if m.any():
                xs.append(img_x[m]); ys.append(img_y[m]); ws.append(overlap[m])
        if not xs:
            continue
        x, y, wt = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        if len(x) < n_fix_fixed:
            continue
        draw_ent = np.empty(N_DRAWS)
        for k in range(N_DRAWS):
            idx = rng.choice(len(x), size=n_fix_fixed, replace=False)
            draw_ent[k] = entropy_from_vec(gauss_density(x[idx], y[idx], wt[idx], cx, cy, SIGMA))
        ent[w] = draw_ent.mean()

    t = np.arange(n_windows) * STEP_S
    cut_times = sorted(cuts_df[cuts_df.video_name == video].cut_time_s.tolist())
    bounds = [0.0] + cut_times + [dur]
    rows = []
    for seg_idx, (s0, s1) in enumerate(zip(bounds[:-1], bounds[1:])):
        if s1 - s0 <= 0:
            continue
        m = (t >= s0) & (t + window_s <= s1)  # full containment, see calc_entropy_by_segment.py
        mean_ent = np.nanmean(ent[m]) if m.sum() > 0 else np.nan
        if not np.isnan(mean_ent):
            rows.append(dict(video=video, block=BLOCKS[video], segment_idx=seg_idx, mean_entropy=mean_ent))
    return rows


def main():
    cuts_df = pd.read_csv(CUTS_PATH)
    all_arrays = {}
    for video in DURATIONS:
        fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
        all_arrays[video] = build_participant_arrays(fix)
    print("fixations loaded for all 12 clips\n")

    all_rows = []

    # Window-size sweep: fixed 120-tile grid, N matched per window size.
    centers120, _ = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)  # 2026-10-22: moved from 180px to 240px (50% overlap)
    cx120, cy120 = centers120["center_x"].to_numpy(), centers120["center_y"].to_numpy()
    for window_s, n_fix in WINDOW_SWEEP.items():
        rng = np.random.default_rng(RNG_SEED)
        for video, dur in DURATIONS.items():
            rows = segment_entropy(video, dur, all_arrays[video], window_s, n_fix, cx120, cy120, rng, cuts_df)
            for r in rows:
                r.update(sweep="window", window_s=window_s, n_tiles=120, diameter_px=240, n_fix_fixed=n_fix)
            all_rows.extend(rows)
        print(f"window sweep: window_s={window_s} done")

    # Grid-size sweep: fixed WINDOW_S=1.0s, N=78 throughout.
    for diameter, n_cols, n_tiles in GRID_SWEEP:
        centers, info = compute_patch_grid(diameter, n_cols, DISPLAY_W, DISPLAY_H)
        assert len(centers) == n_tiles
        cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()
        rng = np.random.default_rng(RNG_SEED)
        for video, dur in DURATIONS.items():
            rows = segment_entropy(video, dur, all_arrays[video], BASELINE_WINDOW_S, BASELINE_N_FIX,
                                   cx, cy, rng, cuts_df)
            for r in rows:
                r.update(sweep="grid", window_s=BASELINE_WINDOW_S, n_tiles=n_tiles,
                         diameter_px=diameter, n_fix_fixed=BASELINE_N_FIX)
            all_rows.extend(rows)
        print(f"grid sweep: n_tiles={n_tiles} done")

    out = pd.DataFrame(all_rows)
    out["mean_entropy"] = out["mean_entropy"].round(4)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")
    print(f"  window sweep rows: {(out.sweep=='window').sum()}, grid sweep rows: {(out.sweep=='grid').sum()}")


if __name__ == "__main__":
    main()

"""
Mean entropy of the pooled group fixation density, per cut-to-cut segment, for all 12 clips.

"Entropy" here measures how SPREAD OUT the group's collective attention is in a given
analysis window: normalize the pooled (duration-weighted Gaussian) density across the 120
scene tiles to a probability distribution, then take its Shannon entropy in bits. A single
tight target (everyone looking at the same spot) gives low entropy; several simultaneously
salient things (attention genuinely split) gives high entropy. See METHODS_DECISIONS.md for
the fuller writeup of this line of investigation, including the montage/video spot-checks
that confirmed low/high-entropy windows look exactly like this intuition predicts.

Tile grid
---------
120 tiles: diameter=180px, 15 cols x 8 rows -- the SAME grid used throughout this project's
concordance work (calc_concordance_group.py etc.), not a separate/coarser grid. Asserted in
code, not just assumed.

Fixed-N rarefaction -- and why these specific numbers
-------------------------------------------------------
Raw entropy is sensitive to how many fixations happen to fall in a window: more fixations
mechanically tends to inflate entropy a little, independent of content (pooled correlation
between window fixation count and raw entropy was r=0.33 at this grid). To stop that from
confounding any comparison across windows/clips/blocks, each window's entropy is computed
from a FIXED number of fixations, randomly subsampled (without replacement) from everyone
who fixated in that window, repeated across many draws and averaged -- the standard
"rarefaction" fix for sample-size bias in diversity/entropy indices (same idea as
species-rarefaction curves in ecology).

- N_FIX_FIXED = 75 fixations per window. The pooled distribution of raw per-window fixation
  counts across all 12 clips has percentiles p5=78, p25=94, median=109, p75=124, p95=144 (see
  METHODS_DECISIONS.md, entropy fixation-count diagnostic). 75 sits just under that p5, so
  only a small minority of windows -- the ones with the fewest fixations available -- don't
  have enough to rarefy and are left as NaN (reported per-row via n_windows_dropped_low_fix,
  not silently dropped). An earlier pass used N=60, picked as "comfortably below p5" without
  a stated reason to prefer 60 specifically over other values well under 78; 75 was chosen
  deliberately instead, closer to the actual p5 cutoff, directly motivated and documented
  rather than picked as an arbitrary round number.
- N_DRAWS = 100 random subsamples per window, averaged, to control Monte Carlo noise in the
  rarefied entropy estimate. An earlier pass used 40 draws chosen for runtime convenience,
  with no convergence check -- i.e. no verification that the per-window estimate had actually
  stabilized by 40 draws rather than still drifting. 100 draws is a more conservative choice
  for the same reason: it isn't independently proven sufficient either (no formal convergence
  check has been run at 100 vs. 200+ draws), but it meaningfully reduces that risk relative to
  40 and was requested explicitly over re-deriving a data-driven cutoff.

Reads fixation data from a LOCAL COPY only -- never the live server mount.

Output
------
data/results/entropy_by_segment/entropy_by_segment.csv -- one row per (video, segment):
  video, block, segment_idx, seg_start_s, seg_end_s, seg_dur_s, n_windows,
  n_windows_dropped_low_fix, mean_entropy_rarefied

Usage
-----
    python scripts/analysis/calc_entropy_by_segment.py
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

WINDOW_S, STEP_S, SIGMA = 1.0, 0.25, 46.0
N_FIX_FIXED = 75   # see module docstring for why this value, not another, was chosen
N_DRAWS = 100       # see module docstring for why this value, not another, was chosen
RNG_SEED = 0

# Exact clip durations via ffprobe -- NOT the rounded estimates (e.g. 60.3 vs the true 60.32)
# used earlier in this project's exploration, which produced a negative-duration trailing
# segment for frank_complex.
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
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "entropy_by_segment" / "entropy_by_segment.csv"

centers, grid_info = compute_patch_grid(180, 15, DISPLAY_W, DISPLAY_H)
cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()
n_tiles = len(cx)
assert n_tiles == 120 and grid_info["n_cols"] == 15 and grid_info["n_rows"] == 8, \
    "Expected the standard 15x8=120-tile grid, same as the rest of the concordance work."


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


def rarefied_entropy_series(video: str, dur: float, rng: np.random.Generator):
    """One entry per window: mean entropy across N_DRAWS random subsamples of exactly
    N_FIX_FIXED pooled fixations (no replacement, across all participants together)."""
    fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
    arrays = build_participant_arrays(fix)
    pids = sorted(arrays)
    n_windows = max(0, int((dur - WINDOW_S) // STEP_S) + 1)

    ent = np.full(n_windows, np.nan)
    n_fix_avail = np.zeros(n_windows, dtype=int)
    for w in range(n_windows):
        s, e = w * STEP_S * 1000, (w * STEP_S + WINDOW_S) * 1000
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
        n_fix_avail[w] = len(x)
        if len(x) < N_FIX_FIXED:
            continue  # too few fixations to rarefy to N_FIX_FIXED; leave as NaN
        draw_ent = np.empty(N_DRAWS)
        for k in range(N_DRAWS):
            idx = rng.choice(len(x), size=N_FIX_FIXED, replace=False)
            draw_ent[k] = entropy_from_vec(gauss_density(x[idx], y[idx], wt[idx], cx, cy, SIGMA))
        ent[w] = draw_ent.mean()
    return ent, n_fix_avail


def main():
    rng = np.random.default_rng(RNG_SEED)
    cuts_df = pd.read_csv(CUTS_PATH)

    rows = []
    for video, dur in DURATIONS.items():
        ent, n_fix_avail = rarefied_entropy_series(video, dur, rng)
        t = np.arange(len(ent)) * STEP_S

        cut_times = sorted(cuts_df[cuts_df.video_name == video].cut_time_s.tolist())
        bounds = [0.0] + cut_times + [dur]

        for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            m = (t >= s) & (t < e)
            n_windows = int(m.sum())
            n_dropped = int(((n_fix_avail[m] < N_FIX_FIXED) & (n_fix_avail[m] > 0)).sum())
            mean_ent = np.nanmean(ent[m]) if n_windows > 0 else np.nan
            rows.append(dict(
                video=video, block=BLOCKS[video], segment_idx=seg_idx,
                seg_start_s=round(s, 2), seg_end_s=round(e, 2), seg_dur_s=round(e - s, 2),
                n_windows=n_windows, n_windows_dropped_low_fix=n_dropped,
                mean_entropy_rarefied=round(mean_ent, 4) if not np.isnan(mean_ent) else np.nan,
            ))
        print(f"{video:16s} done, {len(cut_times)} cuts -> {len(bounds) - 1} segments")

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    total_windows = out["n_windows"].sum()
    total_dropped = out["n_windows_dropped_low_fix"].sum()
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")
    print(f"n_tiles: {n_tiles} (15x8 grid, diameter=180px)")
    print(f"rarefaction: N={N_FIX_FIXED} fixed fixations/window, {N_DRAWS} random draws "
          f"averaged per window, seed={RNG_SEED}")
    print(f"windows with fewer than {N_FIX_FIXED} fixations available (entropy=NaN for that "
          f"window): {total_dropped}/{total_windows} ({100 * total_dropped / total_windows:.1f}%)")


if __name__ == "__main__":
    main()

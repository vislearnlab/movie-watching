"""
Mean entropy of the pooled group fixation density, per cut-to-cut segment, for all 12 clips.

Entropy = Shannon entropy (bits) of the pooled duration-weighted Gaussian density, normalized
across the 120 scene tiles. Low = everyone looks at one spot; high = attention genuinely
split. See METHODS_DECISIONS.md for the full writeup and montage spot-checks.

Tile grid: 120 tiles (diameter=180px, 15x8), same grid as the concordance work. Asserted
in code.

Fixed-N rarefaction: raw entropy correlates with per-window fixation count (r=0.33),
independent of content. Fix: subsample a FIXED N fixations per window (no replacement),
repeat over many draws, average -- standard rarefaction for sample-size bias.
- N_FIX_FIXED=75: just under the pooled per-window fixation-count p5 (78; percentiles
  p5=78/p25=94/median=109/p75=124/p95=144). Windows below N get NaN, tracked via
  n_windows_dropped_low_fix. Supersedes an earlier N=60 that had no stated reason over
  other values below 78.
- N_DRAWS=100: reduces Monte Carlo noise vs. an earlier N=40 (picked for runtime, no
  convergence check). 100 isn't convergence-proven either, just more conservative.

Reads fixation data from a LOCAL COPY only -- never the live server mount.

Window-to-segment assignment requires full containment (start >= seg_start AND
start+WINDOW_S <= seg_end), not just start < seg_end -- otherwise windows bleed past the
segment boundary into the next segment's data (100% bleed for segments < WINDOW_S; same
bug class `visualize_entropy_segment.py` fixed on the video side). This also sets the
minimum segment length at WINDOW_S (1.0s): 15/183 segments (8.2%) get NaN entropy.

Output: data/results/entropy_by_segment/entropy_by_segment.csv -- one row per (video,
segment): video, block, segment_idx, seg_start_s, seg_end_s, seg_dur_s, n_windows,
n_windows_dropped_low_fix, mean_entropy_rarefied (NaN if segment < WINDOW_S).

Usage: python scripts/analysis/calc_entropy_by_segment.py
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

# Exact durations via ffprobe, not rounded estimates (e.g. 60.3 vs true 60.32).
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

centers, grid_info = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)  # 2026-10-22: moved from 180px to 240px (50% overlap)
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
        n_written = 0

        for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            if e - s <= 0:
                continue  # zero-length trailing segment (cut lands exactly at clip end)
            # Require full containment (t+WINDOW_S <= e), not just t < e, so windows can't
            # bleed past the segment boundary -- see module docstring.
            m = (t >= s) & (t + WINDOW_S <= e)
            n_windows = int(m.sum())
            n_dropped = int(((n_fix_avail[m] < N_FIX_FIXED) & (n_fix_avail[m] > 0)).sum())
            mean_ent = np.nanmean(ent[m]) if n_windows > 0 else np.nan
            rows.append(dict(
                video=video, block=BLOCKS[video], segment_idx=seg_idx,
                seg_start_s=round(s, 2), seg_end_s=round(e, 2), seg_dur_s=round(e - s, 2),
                n_windows=n_windows, n_windows_dropped_low_fix=n_dropped,
                mean_entropy_rarefied=round(mean_ent, 4) if not np.isnan(mean_ent) else np.nan,
            ))
            n_written += 1
        print(f"{video:16s} done, {len(cut_times)} cuts -> {n_written} segments")

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    total_windows = out["n_windows"].sum()
    total_dropped = out["n_windows_dropped_low_fix"].sum()
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")
    print(f"n_tiles: {n_tiles} (15x8 grid, diameter=240px)")
    print(f"rarefaction: N={N_FIX_FIXED} fixed fixations/window, {N_DRAWS} random draws "
          f"averaged per window, seed={RNG_SEED}")
    print(f"windows with fewer than {N_FIX_FIXED} fixations available (entropy=NaN for that "
          f"window): {total_dropped}/{total_windows} ({100 * total_dropped / total_windows:.1f}%)")


if __name__ == "__main__":
    main()

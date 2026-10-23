"""
Tile-peak ISC: inter-subject correlation of gaze position over time, built from each
person's density-vector PEAK (argmax tile center, same closed-form Gaussian/120-tile
machinery as concordance/entropy) instead of raw gaze or fixation coordinates -- a third
sibling to calc_isc_rawgaze.py (raw samples) and calc_isc_fixations.py (fixation events),
named the same way, by input representation.

Two structural differences from both of those scripts, not just the input type:
- GROUP-NATIVE, not pairwise: each person is correlated against the LEAVE-ONE-OUT group's
  peak (everyone else's pooled density, same construction as concordance), giving one
  value per PERSON, not one per dyad. This was a deliberate fix, not a convenience --
  pairwise tile-peak ISC was tried first and was worse on every measure (lower baseline,
  more attenuation under segmentation, far less data survived); leave-one-out fixed it
  because the group side is built from ~35-41 people and is almost never undefined in a
  given window, unlike a single other person's trajectory. See METHODS_DECISIONS.md,
  2026-10-15, for the full comparison.
- Non-overlapping windows (STEP_S=WINDOW_S): the existing concordance/entropy convention
  of 75%-overlapping windows is fine when scoring each window independently, but would
  artificially smooth (and inflate) a correlation computed ACROSS windows here.

WINDOW_S=0.5s: just past this project's typical fixation duration (~300ms median), so a
window usually spans more than one fixation -- real pooling, not just raw position at a
coarser grid. MIN_WINDOWS=3 is the floor for a non-degenerate correlation (n=2 is always
exactly +-1, pure math, not signal).

Reads fixation data from a LOCAL COPY only -- never the live server mount.

Output: data/results/isc/isc_tilepeak.csv -- one row per (video, pid): video, block, pid,
isc_tilepeak_x, isc_tilepeak_y, isc_tilepeak_xy. Prefixed to stay visually distinct from
isc_fixation_pairwise.csv's isc_x/isc_y/isc_xy -- different construction, not a drop-in
replacement.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from calc_concordance_group import DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations  # noqa: E402
from plot_patch_grid import compute_patch_grid  # noqa: E402

WINDOW_S = STEP_S = 0.5
SIGMA = 46.0
MIN_WINDOWS = 3
TOL = 1e-6  # tolerance, not exact ==0: pooled-minus-own leaves ~1e-13 float noise on axes
            # that are mathematically exactly constant; scipy's own near-constant check
            # still returns NaN on that, so this has to be caught first.

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
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "isc" / "isc_tilepeak.csv"

centers, _ = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)  # 2026-10-22: moved from 180px to 240px (50% overlap)
cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()


def axis_isc(a, b):
    """One axis's correlation, or None if either side is (near-)constant -- same
    per-axis independence as calc_isc_rawgaze.py's pearson_isc/align_and_isc."""
    if a.std() < TOL or b.std() < TOL:
        return None
    r, _ = stats.pearsonr(a, b)
    return r


def clip_tilepeak_isc(video: str, dur: float) -> list[dict]:
    """Own peak vs. leave-one-out GROUP peak (not another person -- see module
    docstring), tracked across non-overlapping windows spanning the whole clip."""
    fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
    arrays = build_participant_arrays(fix)
    pids = sorted(arrays)
    n_windows = int(dur // WINDOW_S)

    own_x = np.full((len(pids), n_windows), np.nan); own_y = np.full((len(pids), n_windows), np.nan)
    grp_x = np.full((len(pids), n_windows), np.nan); grp_y = np.full((len(pids), n_windows), np.nan)
    for w in range(n_windows):
        s, e = w * WINDOW_S * 1000, (w + 1) * WINDOW_S * 1000
        vecs = np.empty((len(pids), len(cx)))
        for i, pid in enumerate(pids):
            startT, endT, img_x, img_y = arrays[pid]
            vecs[i] = closed_form_tile_values(startT, endT, img_x, img_y, s, e, SIGMA, cx, cy)
        total = vecs.sum(axis=0)
        for i in range(len(pids)):
            if vecs[i].sum() > 0:
                k = vecs[i].argmax(); own_x[i, w], own_y[i, w] = cx[k], cy[k]
            grp_vec = total - vecs[i]  # leave-one-out: everyone else, not one other person
            if grp_vec.sum() > 0:
                k = grp_vec.argmax(); grp_x[i, w], grp_y[i, w] = cx[k], cy[k]

    rows = []
    for i, pid in enumerate(pids):
        xa, xb = own_x[i], grp_x[i]
        ya, yb = own_y[i], grp_y[i]
        both = ~(np.isnan(xa) | np.isnan(xb) | np.isnan(ya) | np.isnan(yb))
        if both.sum() < MIN_WINDOWS:
            continue
        rx = axis_isc(xa[both], xb[both])
        ry = axis_isc(ya[both], yb[both])
        if rx is None and ry is None:
            continue
        rxy = np.nanmean([v for v in [rx, ry] if v is not None])
        rows.append(dict(video=video, block=BLOCKS[video], pid=pid,
                          isc_tilepeak_x=rx, isc_tilepeak_y=ry, isc_tilepeak_xy=rxy))
    return rows


def main():
    all_rows = []
    for video, dur in DURATIONS.items():
        rows = clip_tilepeak_isc(video, dur)
        all_rows.extend(rows)
        print(f"{video:16s} {len(rows)} participants scored")

    out = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

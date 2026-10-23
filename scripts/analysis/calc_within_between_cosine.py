"""
Within-person vs. between-person gaze dispersion, per cut-to-cut segment, for all 12 clips
(pilot -- not yet promoted to METHODS_DECISIONS.md; started as a pixar_birds-only pilot,
extended to all clips once the construction below was validated against known examples and
cross-checked against tile-peak ISC -- see conversation history / METHODS_DECISIONS.md).

Supersedes calc_entropy_within_between.py's approach. That script tried to decompose
within/between via Shannon entropy of each person's own tile vector, which broke down at
the fixation counts this data actually has (entropy's log-transform saturates near its
ceiling once a sample has even 2 well-separated points -- see METHODS_DECISIONS.md /
conversation history for the full diagnosis). This script never takes an entropy of
anything; it only ever compares two tile vectors to each other via cosine distance, which
has no such saturation.

Same closed-form Gaussian/120-tile machinery as concordance and calc_isc_tilepeak.py
(sigma=46px, 15x8 grid) -- no new spatial parameters. Windowing also matches
calc_isc_tilepeak.py: WINDOW_S=STEP_S=0.5s (non-overlapping, since within-person compares
windows to each other and overlapping windows would artificially inflate similarity).

Definitions -- deliberately PARALLEL constructions (both are "distance from a mean"),
deliberately NOT an additive decomposition (see conversation/METHODS_DECISIONS.md for why
total=within+between was dropped: the quadrant framework only needs two comparable axes,
not a literal partition of one total, and cosine distance doesn't decompose additively the
way squared Euclidean distance would):
- within_person_cosine_dist: for one person, mean cosine distance between their own
  per-window tile vectors, over all pairs of windows within the segment (or within the
  sampled window set -- see N_CAP below). Variance of that person's OWN windows around
  their OWN mean, in substance. Low = this person looks at roughly the same thing
  throughout the segment; high = they shift between different things over time. Captures
  switching ACROSS windows, not two targets fixated simultaneously WITHIN one 0.5s window
  (which blurs into one bimodal vector for that window instead) -- a known, accepted
  limitation, not a bug.
- between_person_cosine_dist: first collapse each person's own per-window vectors to ONE
  mean vector for the segment (same windows used for within_person_cosine_dist). Then, for
  each person, cosine distance between their own mean vector and the LEAVE-ONE-OUT grand
  mean vector (the mean of every OTHER eligible person's own mean vector -- excluding
  themselves, same reasoning as concordance/tilepeak ISC's leave-one-out group: comparing
  someone to a reference that includes their own data inflates apparent agreement).
  Averaged over people, per segment. This is the PARALLEL counterpart to within-person:
  within = each person's windows vs. their own mean; between = each person's mean vs. the
  group's mean. Deliberately NOT time-matched / synchrony-sensitive -- two people who
  visited the same targets at different moments within the segment score as agreeing here
  (unlike an ISC-style per-window leave-one-out comparison, which an earlier version of
  this script used and which conflated "same moment" with "same overall pattern"; see
  conversation history, this was corrected after the first version shipped).

MIN_WINDOWS=2 per person per segment for both quantities (lowered from 3 -- see inline
comment at the constant for why). MIN_PEOPLE=3 per segment for the segment-level aggregate.

MAX_FIXATION_MS=2000 fixation-duration sanity filter (see inline comment at the constant):
a small number of fixations in this dataset run 5-19 seconds, 15-60x the ~300ms median --
almost certainly eye-tracker/fixation-classifier artifacts (one participant, MW028, alone
accounts for 4 of the 10 longest fixations in the whole corpus), not real sustained gaze.
Because within-person distance rewards near-zero values most strongly for exactly this kind
of degenerate, unmoving multi-window fixation, picking "most extreme" illustrative examples
per quadrant was specifically selecting for this artifact -- discovered via the
selective-quadrant GIFs, where several "textbook decisive" examples turned out to be a
single absurdly-long fixation, not genuine behavior. See conversation history.

N_CAP=12 adaptive window cap (segment duration confound fix): a long segment gives
between-person distance more time to converge (people's overall patterns agree more just
because there's more time to sample the same places) even with no real moment-to-moment
agreement, and gives within-person distance more time to drift -- demonstrated, not just
suspected, by artificially chopping slow's real ~15.2s segments into ~3s chunks: quadrant
distribution flipped from 75% COORDINATED/0% SELECTIVE to a roughly even spread across all
four. Fix: a segment with <=N_CAP windows available uses all of them (unchanged, exact);
a segment with MORE than N_CAP windows gets N_DRAWS random draws of exactly N_CAP windows
(no replacement), averaged -- same Monte Carlo rarefaction convention the project already
uses for fixation-count confounds in the entropy work. N_CAP=12 (6s) chosen over a strict
fixed-N (same N for every segment, which controls the confound best but costs the most
coverage -- fixed N=6 took coverage from 162/183 to 89/183) because it preserves full
coverage (every segment that had >=2 windows before still qualifies) at a real but smaller
cost to confound control: corr(duration, between) improves from an uncontrolled baseline to
-0.33 at N_CAP=12, worse than tighter caps (-0.23 at N_CAP=8, -0.14 at fixed N=6) because
caps only intervene on the minority of segments above the cap -- most of the corpus (63% at
N_CAP=8-10) has fewer than 8 real windows and is untouched by capping regardless of the cap
value. This is a known, accepted partial fix, not a full solution -- see conversation
history for the fixed-N vs. adaptive-cap comparison across N_CAP in {6, 8, 10, 12}.

Output: data/results/within_between_cosine/all_clips_by_segment.csv -- one row per
(video, segment): video, block, seg_idx, seg_start_s, seg_end_s, seg_dur_s,
n_windows_available, n_windows_used, n_people, within_person_cosine_dist,
between_person_cosine_dist.

Usage: python scripts/analysis/calc_within_between_cosine.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from calc_concordance_group import (  # noqa: E402
    DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations,
)
from plot_patch_grid import compute_patch_grid  # noqa: E402

WINDOW_S = STEP_S = 0.5
SIGMA = 46.0
MIN_WINDOWS = 2  # was 3; lowered because segment-level averaging over ~35-40 people per
                 # segment should wash out the single-pair noise a lone person's estimate
                 # carries at MIN_WINDOWS=2 -- unlike calc_isc_tilepeak.py, which reports one
                 # number per PERSON (where single-pair noise is exposed directly), this script
                 # only ever reports a segment-level mean over people. Recovers 18/183 segments
                 # (144->162 valid). Checked against pixar's 3 known example segments (seg7/10/33,
                 # all already >=3 windows -- unchanged) and the tile-peak ISC cross-correlation
                 # (-0.278 -> -0.269, negligible shift) -- but that check only covers the 25
                 # segments already in both datasets, NOT the newly-recovered 2-window segments
                 # specifically, since those aren't in the old ISC scratch file. The newly
                 # recovered segments' plausibility is therefore not independently validated,
                 # just not shown to disturb what was already validated. See conversation history.
MIN_PEOPLE = 3
MAX_FIXATION_MS = 2000.0  # drop fixations longer than this -- see module docstring. Covers
                          # ~98.8% of real fixations (median ~296ms); excludes the ~1.2% of
                          # events that are almost certainly tracker/classifier artifacts,
                          # not genuine sustained gaze.
N_CAP = 12       # use all available windows if <=N_CAP; random-sample N_CAP if more -- see
                 # module docstring for why this value and why adaptive-cap over fixed-N.
N_DRAWS = 100
RNG_SEED = 0

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
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"

centers, grid_info = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)  # 2026-10-22: moved from 180px (~29-31% overlap) to 240px (exactly 50% overlap on both axes), see METHODS_DECISIONS.md
cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()
assert len(cx) == 120


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - np.dot(a, b) / (na * nb)


def within_between_for_windows(pids: list[str], seg_vecs: np.ndarray) -> dict[str, tuple[float, float]]:
    """One pass of the within/between computation over a FIXED set of windows (either all of
    a short segment's windows, or one Monte Carlo draw of N_CAP windows from a long one).
    Returns {pid: (within, between)} for people with enough data in this window set."""
    within_vals, own_mean_vecs, eligible = [], [], []
    for i, pid in enumerate(pids):
        own_by_window = seg_vecs[i]
        valid_w = np.where(own_by_window.sum(axis=1) > 0)[0]
        if len(valid_w) < MIN_WINDOWS:
            continue
        pair_dists = [cosine_dist(own_by_window[a], own_by_window[b])
                      for a, b in combinations(valid_w, 2)]
        pair_dists = [d for d in pair_dists if not np.isnan(d)]
        if not pair_dists:
            continue
        within_vals.append(np.mean(pair_dists))
        own_mean_vecs.append(own_by_window[valid_w].mean(axis=0))
        eligible.append(pid)

    n_eligible = len(eligible)
    if n_eligible < MIN_PEOPLE:
        return {}
    own_mean_vecs = np.array(own_mean_vecs)
    total_mean = own_mean_vecs.sum(axis=0)
    result = {}
    for j, pid in enumerate(eligible):
        grand_mean_loo = (total_mean - own_mean_vecs[j]) / (n_eligible - 1)
        d = cosine_dist(own_mean_vecs[j], grand_mean_loo)
        if not np.isnan(d):
            result[pid] = (within_vals[j], d)
    return result


def clip_within_between(video: str, dur: float, cuts_df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
    n_before = len(fix)
    fix = fix[(fix["endT"] - fix["startT"]) <= MAX_FIXATION_MS].copy()
    n_dropped = n_before - len(fix)
    if n_dropped:
        print(f"  dropped {n_dropped}/{n_before} fixations > {MAX_FIXATION_MS:.0f}ms")
    arrays = build_participant_arrays(fix)
    pids = sorted(arrays)
    cut_times = sorted(cuts_df[cuts_df.video_name == video].cut_time_s.tolist())
    bounds = [0.0] + cut_times + [dur]

    n_windows_total = int(dur // WINDOW_S)
    # Precompute every person's tile vector for every window up front (reused across segments).
    all_vecs = np.empty((len(pids), n_windows_total, len(cx)))
    for w in range(n_windows_total):
        s, e = w * WINDOW_S * 1000, (w + 1) * WINDOW_S * 1000
        for i, pid in enumerate(pids):
            startT, endT, img_x, img_y = arrays[pid]
            all_vecs[i, w] = closed_form_tile_values(startT, endT, img_x, img_y, s, e, SIGMA, cx, cy)

    rows = []
    for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        if e - s <= 0:
            continue
        w_idx = [w for w in range(n_windows_total) if w * WINDOW_S >= s and (w + 1) * WINDOW_S <= e]
        base_row = dict(video=video, block=BLOCKS[video], seg_idx=seg_idx,
                         seg_start_s=round(s, 2), seg_end_s=round(e, 2),
                         seg_dur_s=round(e - s, 2), n_windows_available=len(w_idx))

        if len(w_idx) < MIN_WINDOWS:
            rows.append(dict(base_row, n_windows_used=0, n_people=0,
                              within_person_cosine_dist=np.nan, between_person_cosine_dist=np.nan))
            continue

        if len(w_idx) <= N_CAP:
            result = within_between_for_windows(pids, all_vecs[:, w_idx, :])
            n_windows_used = len(w_idx)
        else:
            within_acc = {p: [] for p in pids}
            between_acc = {p: [] for p in pids}
            for _ in range(N_DRAWS):
                draw_w = rng.choice(w_idx, size=N_CAP, replace=False)
                draw_result = within_between_for_windows(pids, all_vecs[:, draw_w, :])
                for pid, (wv, bv) in draw_result.items():
                    within_acc[pid].append(wv)
                    between_acc[pid].append(bv)
            result = {p: (np.mean(within_acc[p]), np.mean(between_acc[p]))
                      for p in pids if within_acc[p]}
            n_windows_used = N_CAP

        n_people = len(result)
        if n_people < MIN_PEOPLE:
            rows.append(dict(base_row, n_windows_used=n_windows_used, n_people=n_people,
                              within_person_cosine_dist=np.nan, between_person_cosine_dist=np.nan))
            continue

        rows.append(dict(
            base_row, n_windows_used=n_windows_used, n_people=n_people,
            within_person_cosine_dist=round(float(np.mean([v[0] for v in result.values()])), 4),
            between_person_cosine_dist=round(float(np.mean([v[1] for v in result.values()])), 4),
        ))

    return rows


def main():
    cuts_df = pd.read_csv(CUTS_PATH)
    rng = np.random.default_rng(RNG_SEED)
    all_rows = []
    for video, dur in DURATIONS.items():
        rows = clip_within_between(video, dur, cuts_df, rng)
        n_valid = sum(1 for r in rows if not np.isnan(r["within_person_cosine_dist"]))
        print(f"{video:16s} {len(rows)} segments, {n_valid} with a valid within/between pair")
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    n_valid = int(out["within_person_cosine_dist"].notna().sum())
    print(f"\nWrote {len(out)} rows ({n_valid} with a valid within/between pair) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

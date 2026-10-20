"""
Within-person vs. between-person gaze entropy, per cut-to-cut segment, for pixar_birds only
(pilot -- not yet validated across clips or promoted to METHODS_DECISIONS.md).

Decomposes pooled group entropy (diversity-partitioning framework from ecology):
  group_entropy = within_person_entropy + between_person_entropy
- within_person_entropy: mean of each person's own entropy, computed separately.
- group_entropy: entropy of everyone's fixations pooled together (same quantity
  calc_entropy_by_segment.py calls "entropy", but NOT numerically comparable -- see
  rarefaction note below).
- between_person_entropy: group_entropy - within_person_entropy. Not a standalone
  estimator, a residual: how much group-level spread comes from people disagreeing
  about WHERE to look, beyond what any one person's own spread already accounts for.

Four-way read (see figures/entropy/pixar_within_vs_between.png):
  low within + low between  = everyone shares one spot ("concentrated")
  high within + low between = everyone individually splits attention the same way
                               ("coordinated", e.g. slow_people's two co-moving targets)
  low within + high between = each person is decisive, but people disagree on WHAT
                               ("selective")
  high within + high between = individually unfocused AND disagreeing ("scattered")

Rarefaction: calc_entropy_by_segment.py rarefies to a fixed N=75 fixations per POOLED
window, which assumes ~40 people's fixations pooled together -- not reachable per
person. Per-person-per-segment fixation counts here have p5=2, median=6. So:
rarefy each person to a fixed N_FIX_PERSON=2 fixations per segment (at that p5, same
floor(p5) convention as the canonical pipeline, just applied to this metric's own
distribution), repeat N_DRAWS times, average. Same procedure as the canonical
pipeline, different N because the unit of rarefaction (one person vs. the pooled
group) is different.

within_person_entropy and group_entropy are computed from the SAME per-draw rarefied
sample (each eligible person contributes exactly N_FIX_PERSON fixations to both their
own entropy and the pooled group vector, per draw), so the decomposition is exact per
draw, not reconstructed after the fact from two different procedures. This means
group_entropy here will NOT match entropy_by_segment.csv's mean_entropy_rarefied for
the same segment -- different rarefaction unit, built for a different question (this
one decomposes cleanly into within/between; that one is the best single group-level
entropy estimate).

Segments need >=3 eligible people (>=N_FIX_PERSON fixations in that segment) for a
decomposition; others get NaN.

Output: data/results/entropy_within_between/pixar_birds_by_segment.csv -- one row per
segment: seg_idx, seg_start_s, seg_end_s, seg_dur_s, n_people_eligible,
within_person_entropy, group_entropy, between_person_entropy.

Usage: python scripts/analysis/calc_entropy_within_between.py
"""

import sys
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

SIGMA = 46.0
N_FIX_PERSON = 2   # per-person-per-segment p5, floor(p5) convention -- see module docstring
N_DRAWS = 100
RNG_SEED = 0
MIN_PEOPLE = 3      # need at least this many eligible people for a meaningful decomposition

VIDEO = "pixar_birds"
DUR = 150.15

FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / "adults"
PARTICIPANT_SUMMARY_PATH = PROJECT_ROOT / "data" / "local_cache" / "participant_summary.csv"
CUTS_PATH = PROJECT_ROOT / "annotations" / "scene-cuts" / "all_clips_cuts.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "entropy_within_between" / "pixar_birds_by_segment.csv"

centers, grid_info = compute_patch_grid(180, 15, DISPLAY_W, DISPLAY_H)
cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()
assert len(cx) == 120


def entropy_from_vec(vec):
    total = vec.sum()
    if total <= 0:
        return np.nan
    p = vec / total
    nz = p[p > 0]
    return -(nz * np.log2(nz)).sum()


def main():
    fix = load_video_fixations(FIXATION_DIR, VIDEO, "f329476c", PARTICIPANT_SUMMARY_PATH)
    arrays = build_participant_arrays(fix)
    cuts_df = pd.read_csv(CUTS_PATH)
    cut_times = sorted(cuts_df[cuts_df.video_name == VIDEO].cut_time_s.tolist())
    bounds = [0.0] + cut_times + [DUR]

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        if e - s <= 0:
            continue
        s_ms, e_ms = s * 1000, e * 1000

        eligible = {}
        for pid, (startT, endT, img_x, img_y) in arrays.items():
            overlap = np.minimum(endT, e_ms) - np.maximum(startT, s_ms)
            m = overlap > 0
            if m.sum() >= N_FIX_PERSON:
                eligible[pid] = (startT[m], endT[m], img_x[m], img_y[m])

        n_eligible = len(eligible)
        base_row = dict(seg_idx=seg_idx, seg_start_s=round(s, 2), seg_end_s=round(e, 2),
                         seg_dur_s=round(e - s, 2), n_people_eligible=n_eligible)
        if n_eligible < MIN_PEOPLE:
            rows.append(dict(base_row, within_person_entropy=np.nan, group_entropy=np.nan,
                              between_person_entropy=np.nan))
            continue

        within_draws = np.empty(N_DRAWS)
        group_draws = np.empty(N_DRAWS)
        for k in range(N_DRAWS):
            person_ents = []
            total_vec = np.zeros(120)
            for startT, endT, img_x, img_y in eligible.values():
                idx = rng.choice(len(startT), size=N_FIX_PERSON, replace=False)
                vec = closed_form_tile_values(startT[idx], endT[idx], img_x[idx], img_y[idx],
                                               s_ms, e_ms, SIGMA, cx, cy)
                person_ents.append(entropy_from_vec(vec))
                total_vec += vec
            within_draws[k] = np.nanmean(person_ents)
            group_draws[k] = entropy_from_vec(total_vec)

        within_mean, group_mean = within_draws.mean(), group_draws.mean()
        rows.append(dict(
            base_row,
            within_person_entropy=round(within_mean, 4),
            group_entropy=round(group_mean, 4),
            between_person_entropy=round(group_mean - within_mean, 4),
        ))

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    n_valid = int(out["within_person_entropy"].notna().sum())
    print(f"Wrote {len(out)} rows ({n_valid} with a valid decomposition, "
          f"{len(out) - n_valid} below {MIN_PEOPLE}-person minimum) to {OUTPUT_PATH}")
    print(f"rarefaction: N={N_FIX_PERSON} fixed fixations/person/segment, {N_DRAWS} draws, seed={RNG_SEED}")


if __name__ == "__main__":
    main()

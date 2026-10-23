"""
Trial exclusion rule: a trial (participant x clip) is excluded if its mean ceiling across segments is
< 0, i.e. the other viewers predict that person's gaze worse than the floor (likely calibration/tracking
failure). Floor and ceiling are computed here on the unfiltered data; downstream steps drop the trials
via common.load_gaze.

Output: data/results/information_gain/excluded_trials_<segment_set>.csv

Usage: python scripts/analysis/information_gain/exclusions.py
"""

import baseline
import ceiling
import config as C
from common import excluded_trials_path, load_gaze


def main():
    for segment_set in C.SEGMENT_SETS:
        gaze = load_gaze(segment_set, apply_exclusions=False)
        floors = baseline.build(segment_set, gaze).set_index("video")
        ceil = ceiling.build(segment_set, gaze, floors)
        trials = ceil.groupby(["video", "participant"]).agg(mean_ceiling=("ig_ceiling", "mean"),
                                                            n_segments=("ig_ceiling", "size")).reset_index()
        ex = trials[trials.mean_ceiling < 0]
        ex.round(4).to_csv(excluded_trials_path(segment_set), index=False)
        print(f"{segment_set}: excluding {len(ex)} of {len(trials)} trials")
        print(ex.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

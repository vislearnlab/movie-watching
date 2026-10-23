"""
Within/between-person gaze cosine distance for the `slow` block's clips, computed on
artificially-chopped ~3s subsegments (5 per real ~15.2s segment) instead of real cut-to-cut
segments -- reconstructed 2026-10-22 since the original generator for
data/results/within_between_cosine/slow_chopped_by_subsegment.csv could not be found anywhere
in the repo (local or server-side) or in any notebook. Schema and row count (80 = 4 clips x
20 subsegments) matched exactly against the existing file before this was written; logic
mirrors montage_best_examples.py's N_SUBSEGMENTS_PER_SEGMENT=5 chopping and
calc_within_between_cosine.py's within_between_for_windows computation (reused directly, not
reimplemented) -- see METHODS_DECISIONS.md 2026-10-20 entry on why slow needs chopping at all
(real segments are too long and duration-confounded to show quadrant diversity).

Usage:
    python scripts/analysis/calc_within_between_cosine_slowchopped.py
"""

import numpy as np
import pandas as pd

from calc_within_between_cosine import (
    BLOCKS, CUTS_PATH, DURATIONS, FIXATION_DIR, MIN_WINDOWS, N_CAP, N_DRAWS, PARTICIPANT_SUMMARY_PATH,
    PROJECT_ROOT, RNG_SEED, SIGMA, WINDOW_S, cx, cy, within_between_for_windows,
)
from calc_concordance_group import build_participant_arrays, closed_form_tile_values, load_video_fixations

SLOW_CLIPS = ["slow_animals", "slow_hands", "slow_inanimate", "slow_people"]
N_SUBSEGMENTS_PER_SEGMENT = 5
MAX_FIXATION_MS = 2000.0
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "slow_chopped_by_subsegment.csv"


def clip_chopped_rows(video: str, dur: float, cuts_df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
    fix = fix[(fix["endT"] - fix["startT"]) <= MAX_FIXATION_MS].copy()
    arrays = build_participant_arrays(fix)
    pids = sorted(arrays)
    cut_times = sorted(cuts_df[cuts_df.video_name == video].cut_time_s.tolist())
    bounds = [0.0] + cut_times + [dur]

    n_windows_total = int(dur // WINDOW_S)
    all_vecs = np.empty((len(pids), n_windows_total, len(cx)))
    for w in range(n_windows_total):
        s, e = w * WINDOW_S * 1000, (w + 1) * WINDOW_S * 1000
        for i, pid in enumerate(pids):
            startT, endT, img_x, img_y = arrays[pid]
            all_vecs[i, w] = closed_form_tile_values(startT, endT, img_x, img_y, s, e, SIGMA, cx, cy)

    rows = []
    sub_counter = 0
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        if s1 - s0 <= 0:
            continue
        sub_dur = (s1 - s0) / N_SUBSEGMENTS_PER_SEGMENT
        for sub_idx in range(N_SUBSEGMENTS_PER_SEGMENT):
            sub_s = s0 + sub_idx * sub_dur
            sub_e = s0 + (sub_idx + 1) * sub_dur
            w_idx = [w for w in range(n_windows_total) if w * WINDOW_S >= sub_s and (w + 1) * WINDOW_S <= sub_e]
            base_row = dict(video=video, block=BLOCKS[video], seg_idx=sub_counter,
                             seg_start_s=round(sub_s, 2), seg_end_s=round(sub_e, 2),
                             seg_dur_s=round(sub_e - sub_s, 2))
            sub_counter += 1

            if len(w_idx) < MIN_WINDOWS:
                rows.append(dict(base_row, n_windows=len(w_idx), n_people=0,
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
            if n_people < 3:
                rows.append(dict(base_row, n_windows=n_windows_used, n_people=n_people,
                                  within_person_cosine_dist=np.nan, between_person_cosine_dist=np.nan))
                continue

            rows.append(dict(
                base_row, n_windows=n_windows_used, n_people=n_people,
                within_person_cosine_dist=round(float(np.mean([v[0] for v in result.values()])), 4),
                between_person_cosine_dist=round(float(np.mean([v[1] for v in result.values()])), 4),
            ))

    return rows


def main():
    cuts_df = pd.read_csv(CUTS_PATH)
    rng = np.random.default_rng(RNG_SEED)
    all_rows = []
    for video in SLOW_CLIPS:
        rows = clip_chopped_rows(video, DURATIONS[video], cuts_df, rng)
        n_valid = sum(1 for r in rows if not np.isnan(r["within_person_cosine_dist"]))
        print(f"{video:16s} {len(rows)} subsegments, {n_valid} with a valid within/between pair")
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

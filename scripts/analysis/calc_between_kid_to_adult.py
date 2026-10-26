"""
Kid-to-adult between-subject distance per segment: cosine distance between each kid's segment-mean tile vector and
the mean of all adults' segment-mean vectors, averaged over kids. Same windows, eligibility (MIN_WINDOWS), fixation
filter and N_CAP window draws as calc_within_between_cosine.py; in each draw kids and adults use the same windows.

Output: data/results/within_between_cosine/all_clips_by_segment_kid_to_adult.csv
        (video, seg_idx, seg_start_s, seg_end_s, n_kids, n_adults, kid_to_adult_cosine_dist)
Usage:  python scripts/analysis/calc_between_kid_to_adult.py
"""

import numpy as np
import pandas as pd

from calc_within_between_cosine import (
    CUTS_PATH, DURATIONS, MAX_FIXATION_MS, MIN_PEOPLE, MIN_WINDOWS, N_CAP, N_DRAWS, OUTPUT_PATH, RNG_SEED, SIGMA,
    WINDOW_S, cosine_dist, cx, cy, load_fixations,
)
from gaze_tiles import build_participant_arrays, closed_form_tile_values

OUT_PATH = OUTPUT_PATH.with_name("all_clips_by_segment_kid_to_adult.csv")


def window_vectors(video: str, group: str, n_windows: int) -> np.ndarray:
    fix = load_fixations(video, group)
    arrays = build_participant_arrays(fix[(fix["endT"] - fix["startT"]) <= MAX_FIXATION_MS])
    vecs = np.empty((len(arrays), n_windows, len(cx)))
    for i, (startT, endT, x, y) in enumerate(arrays.values()):
        for w in range(n_windows):
            vecs[i, w] = closed_form_tile_values(startT, endT, x, y, w * WINDOW_S * 1000, (w + 1) * WINDOW_S * 1000, SIGMA, cx, cy)
    return vecs


def own_means(seg_vecs: np.ndarray) -> np.ndarray:
    """Segment-mean vector per eligible person (>= MIN_WINDOWS windows with gaze)."""
    out = [v[v.sum(1) > 0].mean(0) for v in seg_vecs if (v.sum(1) > 0).sum() >= MIN_WINDOWS]
    return np.array(out)


def kid_to_adult(kid_vecs: np.ndarray, adult_vecs: np.ndarray) -> tuple[float, int, int]:
    k, a = own_means(kid_vecs), own_means(adult_vecs)
    if len(k) < MIN_PEOPLE or len(a) < MIN_PEOPLE:
        return np.nan, len(k), len(a)
    ref = a.mean(0)
    return float(np.nanmean([cosine_dist(v, ref) for v in k])), len(k), len(a)


def main():
    cuts = pd.read_csv(CUTS_PATH)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for video, dur in DURATIONS.items():
        n_win = int(dur // WINDOW_S)
        kv, av = window_vectors(video, "kids", n_win), window_vectors(video, "adults", n_win)
        bounds = [0.0] + sorted(cuts[cuts.video_name == video].cut_time_s) + [dur]
        for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            w_idx = [w for w in range(n_win) if w * WINDOW_S >= s and (w + 1) * WINDOW_S <= e]
            d, nk, na = np.nan, 0, 0
            if len(w_idx) >= MIN_WINDOWS:
                draws = [w_idx] if len(w_idx) <= N_CAP else [rng.choice(w_idx, N_CAP, replace=False) for _ in range(N_DRAWS)]
                res = [kid_to_adult(kv[:, dw], av[:, dw]) for dw in draws]
                d = np.nanmean([r[0] for r in res]) if any(np.isfinite(r[0]) for r in res) else np.nan
                nk, na = max(r[1] for r in res), max(r[2] for r in res)
            rows.append(dict(video=video, seg_idx=seg_idx, seg_start_s=round(s, 2), seg_end_s=round(e, 2),
                             n_kids=nk, n_adults=na, kid_to_adult_cosine_dist=round(d, 4) if np.isfinite(d) else np.nan))
        print(f"{video}: done")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"{out.kid_to_adult_cosine_dist.notna().sum()} valid segments -> {OUT_PATH}")


if __name__ == "__main__":
    main()

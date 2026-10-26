"""
Segments where kids look least adultlike: kids agree with each other more than they agree with adults
(mismatch = kid-to-adult cosine distance - kid-to-kid between distance). Ranks all segments and draws the top ones
as adults/kids panel pairs (same style as montage_kids_vs_adults_quadrants.py).

Output: data/results/within_between_cosine/kid_adult_mismatch_by_segment.csv,
        figures/within_between/kids_vs_adults/least_adultlike_segments.png
Usage:  python scripts/analysis/montage_kids_vs_adults_mismatch.py [--top 6]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from montage_kids_vs_adults_quadrants import OUT_DIR, PROJECT_ROOT, draw, middle_frame, participant_arrays

WB_DIR = PROJECT_ROOT / "data" / "results" / "within_between_cosine"
PAIRS_PER_ROW = 2


def mismatch_table() -> pd.DataFrame:
    cols = ["video", "seg_idx", "between_person_cosine_dist"]
    a = pd.read_csv(WB_DIR / "all_clips_by_segment.csv")[["block", "seg_start_s", "seg_end_s"] + cols]
    k = pd.read_csv(WB_DIR / "all_clips_by_segment_kids.csv")[cols + ["n_people"]]
    x = pd.read_csv(WB_DIR / "all_clips_by_segment_kid_to_adult.csv")[["video", "seg_idx", "kid_to_adult_cosine_dist"]]
    d = (a.rename(columns={"between_person_cosine_dist": "adult_adult"})
         .merge(k.rename(columns={"between_person_cosine_dist": "kid_kid", "n_people": "n_kids"}), on=["video", "seg_idx"])
         .merge(x.rename(columns={"kid_to_adult_cosine_dist": "kid_adult"}), on=["video", "seg_idx"]).dropna())
    d["mismatch"] = d.kid_adult - d.kid_kid
    d["mismatch_z"] = (d.mismatch - d.mismatch.mean()) / d.mismatch.std()
    return d.sort_values("mismatch", ascending=False).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=6)
    a = p.parse_args()
    d = mismatch_table()
    d.to_csv(WB_DIR / "kid_adult_mismatch_by_segment.csv", index=False)
    print(d.head(12).round(3).to_string(index=False))

    segs = d.head(a.top)
    n_rows = int(np.ceil(len(segs) / PAIRS_PER_ROW))
    fig, axes = plt.subplots(n_rows, 2 * PAIRS_PER_ROW, figsize=(10 * PAIRS_PER_ROW, 3.5 * n_rows), squeeze=False)
    cache = {}
    for i, r in segs.iterrows():
        img = middle_frame(r.video, (r.seg_start_s + r.seg_end_s) / 2)
        row, col = divmod(i, PAIRS_PER_ROW)
        for j, g in enumerate(("adults", "kids")):
            if (r.video, g) not in cache:
                cache[(r.video, g)] = participant_arrays(r.video, g)
            sub = (f"ADULTS (adult-adult {r.adult_adult:.2f})" if g == "adults"
                   else f"KIDS (kid-kid {r.kid_kid:.2f}, kid-adult {r.kid_adult:.2f})")
            draw(axes[row, 2 * col + j], img, cache[(r.video, g)], r.seg_start_s, r.seg_end_s,
                 f"#{i + 1} {r.video} seg{int(r.seg_idx)} [{r.seg_start_s:.1f}-{r.seg_end_s:.1f}s]\n{sub}")
    for ax in axes.flat[2 * len(segs):]:
        ax.axis("off")
    fig.suptitle(f"Least adultlike segments: kid-adult distance minus kid-kid distance, top {len(segs)} of {len(d)} "
                 "(cyan = group gaze density, yellow = fixations)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "least_adultlike_segments.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

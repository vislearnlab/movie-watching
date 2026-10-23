"""
Kids scored against adult-derived models: floor, lambda, feature curves and the ceiling's "others" all come
from the adult pipeline (no kid data fit anywhere). Ceiling = (1 - lam_c) * mean adult gaze in the segment
+ lam_c * adult floor_c. Same cell rule (>= 3 fixations) and trial exclusion rule (participant x clip mean
ceiling < 0) as adults.

Output: gaze_kids_<set>.parquet, scores_<set>_kids_adultref.parquet (cache);
        summary_<set>_kids_adultref.csv, excluded_trials_kids_adultref_<set>.csv (results)

Usage: python scripts/analysis/information_gain/kids_adult_reference.py [--segment-set cuts]
"""

import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import config as C
import load_gaze
from common import TILE_COLS, load_gaze as load_adult_gaze, log_lik
from partition import summarize
from plots import BLOCK_COLORS, QUADRANT_Q, QUADRANT_TITLE_COLORS, headline, headline_by_group, quadrants, unit_partition
from scoring import ABBR, MODELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segment-set", default="cuts", choices=C.SEGMENT_SETS)
    ss = p.parse_args().segment_set
    gaze_path = C.CACHE_DIR / f"gaze_kids_{ss}.parquet"
    if not gaze_path.exists():
        load_gaze.build(ss, group="kids").to_parquet(gaze_path, index=False)
    kids = pd.read_parquet(gaze_path)
    print(f"kids: {kids.participant.nunique()} participants, {len(kids)} cells, "
          f"{(kids.n_fix < C.MIN_FIXATIONS).mean():.1%} below {C.MIN_FIXATIONS} fixations")
    kids = kids[kids.n_fix >= C.MIN_FIXATIONS]

    ceil = pd.read_parquet(C.CACHE_DIR / f"ceiling_{ss}.parquet")
    adults = load_adult_gaze(ss).merge(ceil[["video", "seg_idx", "participant"]])
    adult_mean = {k: g[TILE_COLS].to_numpy().mean(0) for k, g in adults.groupby(["video", "seg_idx"])}
    lam = ceil.groupby("video").lam.first()
    floors = pd.read_parquet(C.CACHE_DIR / f"floor_{ss}.parquet").set_index("video")
    preds = pd.read_parquet(C.CACHE_DIR / f"predictions_{ss}_knots_floor.parquet")
    q = {(r.video, r.seg_idx, ABBR[r.feature], r.shuffled): np.array([getattr(r, t) for t in TILE_COLS])
         for r in preds.itertuples()}

    out = []
    for (clip, seg), g in kids.groupby(["video", "seg_idx"]):
        if (clip, seg) not in adult_mean:
            continue
        G = g[TILE_COLS].to_numpy()
        floor = floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
        ll_floor = log_lik(G, np.broadcast_to(floor, G.shape))
        gold = (1 - lam[clip]) * adult_mean[(clip, seg)] + lam[clip] * floor
        row = g[["video", "seg_idx", "participant", "n_fix", "look_ms"]].assign(
            ig_ceiling=log_lik(G, np.broadcast_to(gold, G.shape)) - ll_floor)
        for name, parts in MODELS.items():
            if all((clip, seg, m, False) in q for m in parts):
                mix = np.mean([q[(clip, seg, m, False)] for m in parts], axis=0)
                row[f"ig_{name}"] = log_lik(G, np.broadcast_to(mix, G.shape)) - ll_floor
        for m in "SMX":
            if (clip, seg, m, True) in q:
                row[f"ig_{m}_shuffled"] = log_lik(G, np.broadcast_to(q[(clip, seg, m, True)], G.shape)) - ll_floor
        out.append(row)
    df = pd.concat(out, ignore_index=True)

    trials = df.groupby(["video", "participant"]).agg(mean_ceiling=("ig_ceiling", "mean"),
                                                      n_segments=("ig_ceiling", "size")).reset_index()
    ex = trials[trials.mean_ceiling < 0]
    ex.round(4).to_csv(C.RESULTS_DIR / f"excluded_trials_kids_adultref_{ss}.csv", index=False)
    print(f"excluding {len(ex)} of {len(trials)} kid trials (mean ceiling < 0)")
    print(ex.round(3).to_string(index=False))
    df = df[~df.set_index(["video", "participant"]).index.isin(ex.set_index(["video", "participant"]).index)]

    df.to_parquet(C.CACHE_DIR / f"scores_{ss}_kids_adultref.parquet", index=False)
    s = pd.Series(summarize(df))
    s.to_frame("value").to_csv(C.RESULTS_DIR / f"summary_{ss}_kids_adultref.csv")
    print(s.round(4).to_string())

    tag = {"cuts": "unchopped", "slow_chopped": "slow chopped"}[ss] + ", 20 knots, floor; KIDS vs adult models"
    cap = (f"Kids scored against adult-derived models: floor, lambda, feature curves and the ceiling's other viewers are all adults; "
           f"ceiling = how well the adult group predicts each kid. {int(s.n_participants)} kids, {int(s.n_segments)} segments. "
           "Combined models are fixed equal-weight mixtures; unique = all-three minus the two-feature mixture without that feature; "
           "shared = all-three minus the uniques. No confidence intervals yet.")
    for name, by, label, extra in [(f"overall_ig_{ss}_knots_floor", ["video", "seg_idx"], "dots = segments",
                                    " Violins/dots: one per segment (kid-mean within segment, divided by the overall ceiling)."),
                                   (f"overall_ig_people_{ss}_knots_floor", ["participant"], "dots = kids",
                                    " Violins/dots: one per kid (mean over that kid's segments, divided by the overall ceiling).")]:
        figs = {name: headline(s, f"{tag} ({label})", cap + extra, unit_partition(df, by))}
        blk = cap.replace("No confidence intervals yet.", "No confidence intervals yet. Each panel is scaled to its own block's ceiling "
                          "(kids weighted equally within block); black line = block group value.")
        figs[name.replace(f"{ss}_", f"by_block_{ss}_")] = headline_by_group(
            df.assign(block=df.video.map(C.BLOCKS)), by, f"{tag} ({label})", blk, "block", BLOCK_COLORS)
        quad = quadrants(ss)
        qcap = blk.replace("block", "quadrant") + (f" Quadrants from ADULT within/between cosine dispersion, pooled {QUADRANT_Q:.0%}/{1 - QUADRANT_Q:.0%} "
                                                   f"split; {int(quad.quadrant.isna().sum())} middle-band segments excluded.")
        figs[name.replace(f"{ss}_", f"by_quadrant_{ss}_")] = headline_by_group(
            df.merge(quad, on=["video", "seg_idx"]), by, f"{tag} ({label})", qcap, "quadrant", QUADRANT_TITLE_COLORS)
        for n, fig in figs.items():
            C.KIDS_ADULTREF_FIG_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(C.KIDS_ADULTREF_FIG_DIR / f"{n}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {C.KIDS_ADULTREF_FIG_DIR / n}.png")


if __name__ == "__main__":
    main()

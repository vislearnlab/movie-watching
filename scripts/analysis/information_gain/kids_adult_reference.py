"""
Kids scored against adult-derived models: floor, lambda, feature curves and the ceiling's "others" all come
from the adult pipeline (no kid data fit anywhere). Ceiling = (1 - lam_c) * mean adult gaze in the segment
+ lam_c * adult floor_c. Same cell rule (>= 3 fixations) and trial exclusion rule (participant x clip mean
ceiling < 0) as adults.

Output: gaze_kids_<set>.parquet, scores_<set>_kids_adultref.parquet (cache); summary_, excluded_trials_kids_adultref_,
        plotdata_*_<set>_kids_adultref.csv (results)

Usage: python scripts/analysis/information_gain/kids_adult_reference.py [--segment-set cuts]
"""

import argparse

import numpy as np
import pandas as pd


import config as C
import load_gaze
from common import TILE_COLS, load_gaze as load_adult_gaze, log_lik
from partition import summarize
from plot_data import export
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
        for m in C.ALL:
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
    for suffix, (feats, clips) in C.SUBSETS.items():
        s = pd.Series(summarize(df, feats, clips))
        s.to_frame("value").to_csv(C.RESULTS_DIR / f"summary_{ss}_kids_adultref{suffix}.csv")
        print(s.round(4).to_string())
        export(df, ss, C.RESULTS_DIR, f"{ss}_kids_adultref{suffix}", feats=feats, clips=clips)  # plot_information_gain.R --group kids-adultref


if __name__ == "__main__":
    main()

"""
Center-bias check: uniform baseline, with the old floor entered as its own feature ("spatial", P). Features are the
no-floor models (knots_nofloor: f(x) alone, beta=0) for saliency, motion, meaning; spatial = the leave-one-clip-out
floor. All equal-weight mixtures of P, S, M, X are scored as IG over uniform; the ceiling is the same model as
ceiling.py, re-expressed over uniform. "Shared" then excludes overlap that comes only from every model carrying the floor.

Output: scores_<set>_uniform.parquet (cache); summary_<set>_uniform.csv, plotdata_*_<set>_uniform.csv (results)
Usage:  python scripts/analysis/information_gain/uniform_baseline.py
"""

import numpy as np
import pandas as pd

import config as C
from common import TILE_COLS, UNIFORM, load_gaze, log_lik
from partition import models_of, summarize
from plot_data import export

FEATS = "PSMX"


def score(segment_set: str) -> pd.DataFrame:
    ceil = pd.read_parquet(C.CACHE_DIR / f"ceiling_{segment_set}.parquet")
    gaze = load_gaze(segment_set).merge(ceil[["video", "seg_idx", "participant"]], on=["video", "seg_idx", "participant"])
    preds = pd.read_parquet(C.CACHE_DIR / f"predictions_{segment_set}_knots_nofloor.parquet")
    preds = preds[~preds.shuffled & preds.feature.isin(["saliency", "motion", "meaning"])]
    floors = pd.read_parquet(C.CACHE_DIR / f"floor_{segment_set}.parquet").set_index("video")
    q = {(r.video, r.seg_idx, C.ABBR[r.feature]): np.array([getattr(r, t) for t in TILE_COLS]) for r in preds.itertuples()}

    out = []
    for (clip, seg), g in gaze.groupby(["video", "seg_idx"]):
        G = g[TILE_COLS].to_numpy()
        floor = floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
        q[(clip, seg, "P")] = floor
        ll_unif = log_lik(G, np.broadcast_to(UNIFORM, G.shape))
        ll_floor = log_lik(G, np.broadcast_to(floor, G.shape))
        row = g[["video", "seg_idx", "participant"]].copy()
        for m in models_of(FEATS):
            if all((clip, seg, a) in q for a in m):
                mix = np.mean([q[(clip, seg, a)] for a in m], axis=0)
                row[f"ig_{m}"] = log_lik(G, np.broadcast_to(mix, G.shape)) - ll_unif
        row["ll_floor_minus_unif"] = ll_floor - ll_unif
        out.append(row)
    df = ceil.merge(pd.concat(out, ignore_index=True), on=["video", "seg_idx", "participant"])
    df["ig_ceiling"] = df.ig_ceiling + df.ll_floor_minus_unif  # same ceiling model, now over uniform
    assert np.allclose(df.ig_P, df.ll_floor_minus_unif)
    return df.drop(columns="ll_floor_minus_unif")


def main():
    for ss in C.SEGMENT_SETS:
        df = score(ss)
        df.to_parquet(C.CACHE_DIR / f"scores_{ss}_uniform.parquet", index=False)
        s = pd.Series(summarize(df, FEATS))
        s.to_frame("value").to_csv(C.RESULTS_DIR / f"summary_{ss}_uniform.csv")
        print(f"\n== {ss} uniform ==\n{s.round(4).to_string()}")
        export(df, ss, C.RESULTS_DIR, f"{ss}_uniform", pd.read_parquet(C.CACHE_DIR / f"curves_{ss}_knots_nofloor.parquet"), FEATS)


if __name__ == "__main__":
    main()

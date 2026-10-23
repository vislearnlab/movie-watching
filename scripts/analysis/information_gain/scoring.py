"""
Scores the 7 models (single features + equal-weight mixtures) and the shuffled-feature controls as
information gain over the floor, on exactly the ceiling's participant x segment cells.

Output: scores_<segment_set>_<variant>.parquet (ceiling columns + ig_<model> per cell)

Usage: python scripts/analysis/information_gain/scoring.py
"""

import numpy as np
import pandas as pd

import config as C
from common import TILE_COLS, load_gaze, log_lik
from feature_models import VARIANTS

ABBR = {"saliency": "S", "motion": "M", "meaning": "X"}
MODELS = {"S": ["S"], "M": ["M"], "X": ["X"], "SM": ["S", "M"], "SX": ["S", "X"], "MX": ["M", "X"], "SMX": ["S", "M", "X"]}


def score(segment_set: str, variant: str) -> pd.DataFrame:  # variant = predictions file tag
    ceil = pd.read_parquet(C.CACHE_DIR / f"ceiling_{segment_set}.parquet")
    gaze = load_gaze(segment_set).merge(ceil[["video", "seg_idx", "participant"]], on=["video", "seg_idx", "participant"])
    preds = pd.read_parquet(C.CACHE_DIR / f"predictions_{segment_set}_{variant}.parquet")
    floors = pd.read_parquet(C.CACHE_DIR / f"floor_{segment_set}.parquet").set_index("video")
    q = {(r.video, r.seg_idx, ABBR[r.feature], r.shuffled): np.array([getattr(r, t) for t in TILE_COLS])
         for r in preds.itertuples()}

    out = []
    for (clip, seg), g in gaze.groupby(["video", "seg_idx"]):
        G = g[TILE_COLS].to_numpy()
        ll_floor = log_lik(G, np.broadcast_to(floors.loc[clip, TILE_COLS].to_numpy(dtype=float), G.shape))
        row = g[["video", "seg_idx", "participant"]].copy()
        for name, parts in MODELS.items():
            if all((clip, seg, p, False) in q for p in parts):
                mix = np.mean([q[(clip, seg, p, False)] for p in parts], axis=0)  # fixed equal weights
                row[f"ig_{name}"] = log_lik(G, np.broadcast_to(mix, G.shape)) - ll_floor
        for p in "SMX":
            if (clip, seg, p, True) in q:
                row[f"ig_{p}_shuffled"] = log_lik(G, np.broadcast_to(q[(clip, seg, p, True)], G.shape)) - ll_floor
        out.append(row)
    return ceil.merge(pd.concat(out, ignore_index=True), on=["video", "seg_idx", "participant"])


def main():
    for segment_set in C.SEGMENT_SETS:
        for variant in VARIANTS:
            df = score(segment_set, variant)
            df.to_parquet(C.CACHE_DIR / f"scores_{segment_set}_{variant}.parquet", index=False)
            print(f"{segment_set} {variant}: {len(df)} cells")


if __name__ == "__main__":
    main()

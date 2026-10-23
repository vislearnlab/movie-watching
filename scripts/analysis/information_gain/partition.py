"""
Group-average information gains, proportion explained, unique/shared partition, and full
commonality decomposition. Cells restricted to those where all 7 models are defined.

Output: data/results/information_gain/summary_<segment_set>_<variant>.csv

Usage: python scripts/analysis/information_gain/partition.py
"""

import pandas as pd

import config as C
from common import group_average
from feature_models import VARIANTS
from scoring import MODELS


def partition(ig: dict) -> dict:
    u_S, u_M, u_X = ig["SMX"] - ig["MX"], ig["SMX"] - ig["SX"], ig["SMX"] - ig["SM"]
    out = dict(unique_saliency=u_S, unique_motion=u_M, unique_meaning=u_X,
               shared=ig["SMX"] - (u_S + u_M + u_X),
               common_all3=ig["S"] + ig["M"] + ig["X"] - ig["SM"] - ig["SX"] - ig["MX"] + ig["SMX"],
               common_SM=ig["SX"] + ig["MX"] - ig["X"] - ig["SMX"],
               common_SX=ig["SM"] + ig["MX"] - ig["M"] - ig["SMX"],
               common_MX=ig["SM"] + ig["SX"] - ig["S"] - ig["SMX"])
    assert abs(out["shared"] - (out["common_all3"] + out["common_SM"] + out["common_SX"] + out["common_MX"])) < 1e-9
    return out


def summarize(df: pd.DataFrame) -> dict:
    cols = [f"ig_{m}" for m in MODELS]
    df = df.dropna(subset=cols)
    ig = {m: group_average(df, f"ig_{m}") for m in MODELS}
    ceiling = group_average(df, "ig_ceiling")
    out = dict(n_cells=len(df), n_segments=df.groupby(["video", "seg_idx"]).ngroups,
               n_participants=df.participant.nunique(), ceiling=ceiling,
               **{f"ig_{m}": v for m, v in ig.items()},
               **{f"prop_{m}": v / ceiling for m, v in ig.items()},  # ratio of averages, not average of ratios
               **{f"ig_{p}_shuffled": group_average(df, f"ig_{p}_shuffled") for p in "SMX"},
               **partition(ig))
    return out


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for segment_set in C.SEGMENT_SETS:
        for variant in VARIANTS:
            df = pd.read_parquet(C.CACHE_DIR / f"scores_{segment_set}_{variant}.parquet")
            s = pd.Series(summarize(df))
            s.to_frame("value").to_csv(C.RESULTS_DIR / f"summary_{segment_set}_{variant}.csv")
            print(f"\n== {segment_set} / {variant} ==\n{s.round(4).to_string()}")


if __name__ == "__main__":
    main()

"""
Group-average information gains, proportion explained, unique/shared partition, and full
commonality decomposition, per C.SUBSETS (all clips with saliency/motion/meaning; face clips with all four).
Cells restricted to those where every model in the subset is defined.

Output: data/results/information_gain/summary_<segment_set>_<variant>[_faceclips].csv

Usage: python scripts/analysis/information_gain/partition.py
"""

from itertools import combinations

import pandas as pd

import config as C
from common import group_average
from feature_models import VARIANTS
NAMES = {a: f for f, a in C.ABBR.items()} | {"P": "spatial"}  # P = spatial prior (the old floor), uniform_baseline.py


def ig_of(ig: dict, letters, feats: str) -> float:
    key = "".join(a for a in feats if a in letters)
    return ig[key] if key else 0.0


def models_of(feats: str) -> list:
    # Model names keep the letter order of feats, e.g. "SX" for feats "SMX".
    return ["".join(c) for n in range(1, len(feats) + 1) for c in combinations(feats, n)]


def partition(ig: dict, feats: str = C.ALL) -> dict:
    # unique_k = all-features minus the mixture without k; shared = the rest. Commonality of subset S (Seibold &
    # McDonald 1974): -sum_{T in S} (-1)^|T| IG((ALL - S) + T); common_<S> for |S| >= 2 sums to shared.
    full = ig[feats]
    out = {f"unique_{NAMES[a]}": full - ig_of(ig, set(feats) - {a}, feats) for a in feats}
    out["shared"] = full - sum(out.values())
    common = {}
    for n in range(2, len(feats) + 1):
        for S in combinations(feats, n):
            rest = set(feats) - set(S)
            common["common_" + "".join(S)] = -sum((-1) ** len(T) * ig_of(ig, rest | set(T), feats)
                                                  for k in range(len(S) + 1) for T in combinations(S, k))
    assert abs(out["shared"] - sum(common.values())) < 1e-9
    return {**out, **common}


def summarize(df: pd.DataFrame, feats: str = C.ALL, clips: list | None = None) -> dict:
    models = models_of(feats)
    df = df.dropna(subset=[f"ig_{m}" for m in models])
    if clips is not None:
        df = df[df.video.isin(clips)]
    ig = {m: group_average(df, f"ig_{m}") for m in models}
    ceiling = group_average(df, "ig_ceiling")
    out = dict(n_cells=len(df), n_segments=df.groupby(["video", "seg_idx"]).ngroups,
               n_participants=df.participant.nunique(), ceiling=ceiling,
               **{f"ig_{m}": v for m, v in ig.items()},
               **{f"prop_{m}": v / ceiling for m, v in ig.items()},  # ratio of averages, not average of ratios
               ig_all=ig[feats], **{f"ig_{p}_shuffled": group_average(df, f"ig_{p}_shuffled") for p in feats
                                     if f"ig_{p}_shuffled" in df},
               **partition(ig, feats))
    return out


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for segment_set in C.SEGMENT_SETS:
        for variant in VARIANTS:
            df = pd.read_parquet(C.CACHE_DIR / f"scores_{segment_set}_{variant}.parquet")
            for suffix, (feats, clips) in C.SUBSETS.items():
                s = pd.Series(summarize(df, feats, clips))
                s.to_frame("value").to_csv(C.RESULTS_DIR / f"summary_{segment_set}_{variant}{suffix}.csv")
                print(f"\n== {segment_set} / {variant}{suffix} ==\n{s.round(4).to_string()}")


if __name__ == "__main__":
    main()

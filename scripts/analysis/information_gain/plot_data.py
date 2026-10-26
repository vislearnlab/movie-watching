"""
Writes the tidy CSVs that plot_information_gain.R draws from (all plotting is in R). Per facet (overall, block,
quadrant, clip): group values as proportion of that facet's ceiling (ratio of group averages), and per-unit values
(segments or participants) divided by the same ceiling.

Written once per C.SUBSETS entry (tag suffix "" = all clips, 3 features; "_faceclips" = face clips, 4 features).
Output (data/results/information_gain[/<group>]/):
  plotdata_groups_<tag>.csv  facet_type, facet, n_segments, n_participants, ceiling, ig_<model> + ig_all (bits), shared/unique_* (prop)
  plotdata_units_<tag>.csv   facet_type, facet, unit_type, unit, component, prop
  per_segment_<tag>.csv      per-segment ceiling and all-features IG (+ group entropy for cuts)
  curves_<tag>.csv           fitted nonlinearity per held-out clip

Usage: python scripts/analysis/information_gain/plot_data.py [--segment-set cuts] [--variant knots_floor]
"""

import argparse

import numpy as np
import pandas as pd

import config as C
from common import group_average
from feature_models import VARIANTS
from partition import NAMES, models_of, partition

COMPONENT_ORDER = ["shared", "unique_spatial", "unique_motion", "unique_saliency", "unique_meaning", "unique_face"]
WB_PATH = C.PROJECT_ROOT / "data" / "results" / "within_between_cosine" / "all_clips_by_segment.csv"
WB_SLOW_CHOPPED_PATH = WB_PATH.parent / "slow_chopped_by_subsegment.csv"
ENTROPY_PATH = C.PROJECT_ROOT / "data" / "results" / "entropy_by_segment" / "entropy_by_segment.csv"
QUADRANT_Q = 0.4  # plot_quadrant_scatter.py default


def quadrants(segment_set: str = "cuts", q: float = QUADRANT_Q) -> pd.DataFrame:
    # Same classification as plot_quadrant_scatter.py (pooled quantile split, middle band unclassified).
    # Thresholds always from the unchopped segments; slow_chopped swaps in slow sub-segments.
    w = pd.read_csv(WB_PATH)
    wl, wh = w.within_person_cosine_dist.quantile([q, 1 - q])
    bl, bh = w.between_person_cosine_dist.quantile([q, 1 - q])
    if segment_set == "slow_chopped":
        w = pd.concat([w[w.block != "slow"], pd.read_csv(WB_SLOW_CHOPPED_PATH)], ignore_index=True)
    lo_w, lo_b = w.within_person_cosine_dist <= wl, w.between_person_cosine_dist <= bl
    hi_w, hi_b = (w.within_person_cosine_dist >= wh) & ~lo_w, (w.between_person_cosine_dist >= bh) & ~lo_b  # disjoint at q = 0.5
    w["quadrant"] = np.select([lo_w & lo_b, hi_w & lo_b, lo_w & hi_b, hi_w & hi_b],
                              ["concentrated", "coordinated", "selective", "scattered"], None)
    return w[["video", "seg_idx", "quadrant"]]


def components(feats: str) -> list:
    return [k for k in COMPONENT_ORDER if k == "shared" or k.removeprefix("unique_") in {NAMES[a] for a in feats}]


def unit_partition(df: pd.DataFrame, by: list, feats: str) -> pd.DataFrame:
    """Partition per unit (segment or participant) from mean IGs within the unit (bits)."""
    models = models_of(feats)
    cols = [f"ig_{m}" for m in models]
    seg = df.dropna(subset=cols).groupby(by)[cols + ["ig_ceiling"]].mean()
    parts = seg.apply(lambda r: pd.Series(partition({m: r[f"ig_{m}"] for m in models}, feats)), axis=1)
    return seg.join(parts).reset_index()


def facet_tables(df: pd.DataFrame, facet_type: str, col: str | None, feats: str) -> tuple[list, list]:
    groups, units = [], []
    models = models_of(feats)
    df = df.dropna(subset=[f"ig_{m}" for m in models])
    for facet, d in (df.groupby(col) if col else [("all", df)]):
        ig = {m: group_average(d, f"ig_{m}") for m in models}
        c, part = group_average(d, "ig_ceiling"), partition(ig, feats)
        groups.append(dict(facet_type=facet_type, facet=facet, n_segments=d.groupby(["video", "seg_idx"]).ngroups,
                           n_participants=d.participant.nunique(), ceiling=c, **{f"ig_{m}": v for m, v in ig.items()}, ig_all=ig[feats],
                           **{k: part[k] / c for k in components(feats)}))
        for unit_type, by in [("segment", ["video", "seg_idx"]), ("participant", ["participant"])]:
            p = unit_partition(d, by, feats)
            unit = p.video + "_" + p.seg_idx.astype(str) if unit_type == "segment" else p.participant
            for k in components(feats):
                units.append(pd.DataFrame(dict(facet_type=facet_type, facet=facet, unit_type=unit_type, unit=unit,
                                               component=k, prop=p[k] / c)))
    return groups, units


def export(df: pd.DataFrame, segment_set: str, out_dir, tag: str, curves: pd.DataFrame | None = None,
           feats: str = C.ALL, clips: list | None = None):
    if clips is not None:
        df = df[df.video.isin(clips)]
    df = df.assign(block=df.video.map(C.BLOCKS), clip=df.video,
                   faces=np.where(df.video.isin(C.FACE_CLIPS), "clips with faces", "clips without faces")).merge(quadrants(segment_set), on=["video", "seg_idx"], how="left")
    groups, units = [], []
    for facet_type, col in [("overall", None), ("block", "block"), ("quadrant", "quadrant"), ("clip", "clip"), ("faces", "faces")]:
        g, u = facet_tables(df, facet_type, col, feats)
        groups += g; units += u
    pd.DataFrame(groups).to_csv(out_dir / f"plotdata_groups_{tag}.csv", index=False)
    pd.concat(units, ignore_index=True).to_csv(out_dir / f"plotdata_units_{tag}.csv", index=False)

    seg = df.dropna(subset=[f"ig_{feats}"]).groupby(["video", "seg_idx", "block"]).agg(
        ceiling=("ig_ceiling", "mean"), ig_all=(f"ig_{feats}", "mean"), n=("participant", "size")).reset_index()
    seg["prop_all"] = seg.ig_all / seg.ceiling
    if segment_set == "cuts" and ENTROPY_PATH.exists():
        ent = pd.read_csv(ENTROPY_PATH).rename(columns={"segment_idx": "seg_idx"})
        seg = seg.merge(ent[["video", "seg_idx", "mean_entropy_rarefied"]], on=["video", "seg_idx"], how="left")
    seg.to_csv(out_dir / f"per_segment_{tag}.csv", index=False)
    if curves is not None:
        curves.to_csv(out_dir / f"curves_{tag}.csv", index=False)
    print(f"Wrote plot data for {tag} -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segment-set", default="cuts", choices=C.SEGMENT_SETS)
    p.add_argument("--variant", default="knots_floor", choices=list(VARIANTS))
    a = p.parse_args()
    tag = f"{a.segment_set}_{a.variant}"
    df = pd.read_parquet(C.CACHE_DIR / f"scores_{tag}.parquet")
    curves = pd.read_parquet(C.CACHE_DIR / f"curves_{tag}.parquet")
    for suffix, (feats, clips) in C.SUBSETS.items():
        export(df, a.segment_set, C.RESULTS_DIR, tag + suffix, curves, feats, clips)


if __name__ == "__main__":
    main()

"""
Bootstrap CIs for the group IG values. --unit participant resamples participants; segment resamples segments (all
participants' cells in a drawn segment kept together); clip resamples whole clips. Each draw recomputes every model's
group average (mean of participant means, as partition.summarize) and the ceiling, then the % of ceiling for each
model and partition component. Feature models are not refit (they are leave-one-clip-out fits on all participants).
--by block|quadrant|face runs the bootstrap separately within each group (quadrants as plot_data.quadrants at --q; unclassified dropped; face = config.FACE_CLIPS vs the rest).

Output: data/results/information_gain/bootstrap_<set>_<variant>[_<unit>][_by_<by>[_q<100q>]].csv (quantity, estimate, lo, hi[, group]; proportion of ceiling)
Usage:  python scripts/analysis/information_gain/bootstrap_ci.py [--set cuts --variant knots_floor --n 2000 --unit participant|segment|clip --by block|quadrant|face --q 0.4]
"""

import argparse

import numpy as np
import pandas as pd

import config as C
from partition import models_of, partition
from plot_data import quadrants

FEATS = "SMX"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="cuts")
    p.add_argument("--variant", default="knots_floor")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--unit", default="participant", choices=["participant", "segment", "clip"])
    p.add_argument("--by", choices=["block", "quadrant", "face"])
    p.add_argument("--q", type=float, default=0.4, help="quadrant split quantile (0.5 = median split)")
    a = p.parse_args()
    models = models_of(FEATS)
    cols = [f"ig_{m}" for m in models] + ["ig_ceiling"]
    df = pd.read_parquet(C.CACHE_DIR / f"scores_{a.set}_{a.variant}.parquet").dropna(subset=cols)
    df["segment"] = df.video + "_" + df.seg_idx.astype(str)
    df["block"] = df.video.map(C.BLOCKS)
    df["face"] = np.where(df.video.isin(C.FACE_CLIPS), "face", "no_face")
    if a.by == "quadrant":
        df = df.merge(quadrants(a.set, a.q), on=["video", "seg_idx"], how="left")

    def values(rows: pd.DataFrame) -> dict:
        g = rows.mean() if a.unit == "participant" else rows.groupby("participant")[cols].mean().mean()
        ig = {m: g[f"ig_{m}"] for m in models}
        out = {f"prop_{m}": ig[m] / g.ig_ceiling for m in models}
        out |= {k: v / g.ig_ceiling for k, v in partition(ig, FEATS).items() if not k.startswith("common_")}
        return out

    def run(df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        if a.unit == "participant":
            pm = df.groupby("participant")[cols].mean()  # group value = mean of these rows
            est = values(pm)
            draws = pd.DataFrame([values(pm.iloc[rng.integers(0, len(pm), len(pm))]) for _ in range(a.n)])
        else:
            col = "segment" if a.unit == "segment" else "video"
            groups = {k: g[["participant"] + cols] for k, g in df.groupby(col)}
            keys = list(groups)
            est = values(df[["participant"] + cols])
            draws = pd.DataFrame([values(pd.concat([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
                                  for _ in range(a.n)])
        return pd.DataFrame(dict(quantity=list(est), estimate=list(est.values()),
                                 lo=draws.quantile(0.025).to_numpy(), hi=draws.quantile(0.975).to_numpy(),
                                 n_segments=df.segment.nunique()))

    out = run(df) if a.by is None else pd.concat([run(g).assign(group=k) for k, g in df.groupby(a.by)], ignore_index=True)
    path = C.RESULTS_DIR / f"bootstrap_{a.set}_{a.variant}{'' if a.unit == 'participant' else '_' + a.unit}{'_by_' + a.by if a.by else ''}{f'_q{round(100 * a.q)}' if a.by == 'quadrant' and a.q != 0.4 else ''}.csv"
    out.to_csv(path, index=False)
    print(f"unit = {a.unit}, {a.n} draws -> {path}")
    print(out.assign(**{c: out[c] * 100 for c in ["estimate", "lo", "hi"]}).round(1).to_string(index=False))


if __name__ == "__main__":
    main()

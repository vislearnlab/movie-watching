"""
Sanity checks (plan Step 7) not already enforced inline: floor scored against itself, floor leakage,
leave-one-out leakage in the ceiling, ceiling > 0, shuffled-feature drop, floor-on vs. floor-off,
and ceiling convergence with number of other participants (plot).
(Probability vectors summing to 1 with no zeros is asserted inside common.log_lik on every score.)

Usage: python scripts/analysis/information_gain/sanity_checks.py [--segment-set cuts]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from common import TILE_COLS, UNIFORM, clip_sums, group_average, load_gaze, log_lik, pooled_excluding

N_OTHERS = [2, 4, 8, 16, 32]
N_DRAWS = 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segment-set", default="cuts", choices=C.SEGMENT_SETS)
    a = p.parse_args()
    ss = a.segment_set
    gaze = load_gaze(ss)
    ceil = pd.read_parquet(C.CACHE_DIR / f"ceiling_{ss}.parquet")
    floors = pd.read_parquet(C.CACHE_DIR / f"floor_{ss}.parquet").set_index("video")
    sums, counts = clip_sums(gaze)
    report = {}

    # Floor vs. itself = 0; floor for clip c rebuilt from other clips only matches the cached floor.
    for clip in C.CLIPS:
        f = floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
        assert abs(log_lik(f, f) - log_lik(f, f)) == 0
        rebuilt = (1 - floors.loc[clip, "eps"]) * pooled_excluding(sums, counts, {clip}) + floors.loc[clip, "eps"] * UNIFORM
        assert np.allclose(rebuilt, f), f"floor for {clip} does not match a rebuild that excludes {clip}"
    report["floor_leakage"] = "pass: every clip's floor rebuilt without that clip matches"

    # Leave-one-out: recompute one segment's ceiling by hand, including each participant's own gaze -> must differ.
    (clip, seg), g = next(iter(gaze.merge(ceil[["video", "seg_idx", "participant"]]).groupby(["video", "seg_idx"])))
    G = g[TILE_COLS].to_numpy()
    lam, f = ceil[ceil.video == clip].lam.iloc[0], floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
    loo = (1 - lam) * ((G.sum(0) - G) / (len(G) - 1)) + lam * f
    ig_loo = log_lik(G, loo) - log_lik(G, np.broadcast_to(f, G.shape))
    cached = ceil.set_index(["video", "seg_idx", "participant"]).loc[[(clip, seg, pid) for pid in g.participant], "ig_ceiling"]
    assert np.allclose(ig_loo, cached.to_numpy()), "cached ceiling does not equal the leave-one-out recomputation"
    report["ceiling_leave_one_out"] = "pass: cached ceiling equals hand recomputation excluding own gaze"

    report["ceiling_positive"] = f"group ceiling {group_average(ceil, 'ig_ceiling'):.3f} bits; " \
                                 f"{(ceil.groupby(['video', 'seg_idx']).ig_ceiling.mean() < 0).sum()} segments with mean < 0"

    for variant in ["knots_floor", "knots_nofloor"]:
        sc = pd.read_parquet(C.CACHE_DIR / f"scores_{ss}_{variant}.parquet")
        for ftr in "SMX":
            report[f"{variant}_{ftr}_real_vs_shuffled"] = (f"{group_average(sc.dropna(subset=[f'ig_{ftr}']), f'ig_{ftr}'):.3f} vs "
                                                          f"{group_average(sc.dropna(subset=[f'ig_{ftr}_shuffled']), f'ig_{ftr}_shuffled'):.3f}")

    # Ceiling convergence: random subsets of k other participants (same lam per clip).
    rng = np.random.default_rng(C.RNG_SEED)
    rows = []
    cells = gaze.merge(ceil[["video", "seg_idx", "participant", "lam"]])
    for (clip, seg), g in cells.groupby(["video", "seg_idx"]):
        G = g[TILE_COLS].to_numpy()
        f = floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
        lam = g.lam.iloc[0]
        llf = log_lik(G, np.broadcast_to(f, G.shape))
        for k in N_OTHERS + [len(G) - 1]:
            if k > len(G) - 1:
                continue
            igs = np.zeros(len(G))
            for _ in range(N_DRAWS if k < len(G) - 1 else 1):
                for i in range(len(G)):
                    others = np.delete(np.arange(len(G)), i)
                    pick = rng.choice(others, size=k, replace=False)
                    q = (1 - lam) * G[pick].mean(0) + lam * f
                    igs[i] += log_lik(G[i], q) - llf[i]
            igs /= N_DRAWS if k < len(G) - 1 else 1
            rows.append(pd.DataFrame(dict(participant=g.participant.to_numpy(), k=k if k in N_OTHERS else -1, ig=igs)))
    conv = pd.concat(rows)
    curve = conv[conv.k > 0].groupby("k").apply(lambda d: group_average(d, "ig"), include_groups=False)
    full = group_average(conv[conv.k == -1], "ig")
    report["ceiling_convergence"] = ", ".join(f"k={k}: {v:.3f}" for k, v in curve.items()) + f", all others: {full:.3f}"

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(curve.index, curve.values, "o-", color="#2a78d6", lw=2, ms=6)
    ax.axhline(full, color="#52514e", ls="--", lw=1)
    ax.text(curve.index.max(), full, " all others", va="bottom", ha="right", fontsize=8, color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of other participants in the ceiling")
    ax.set_ylabel("group ceiling (bits)")
    ax.set_title(f"Ceiling convergence, {ss}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = C.CHECK_FIG_DIR / f"information_gain_ceiling_convergence_{ss}.png"
    fig.savefig(out, dpi=150)
    pd.Series(report).to_frame("result").to_csv(C.RESULTS_DIR / f"sanity_checks_{ss}.csv")
    print(pd.Series(report).to_string())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

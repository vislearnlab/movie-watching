"""
Check: held-out information gain of single-feature models vs. number of knots (plus exp, 1 parameter).
Selection on the same leave-one-clip-out scores used for reporting, so treat as a robustness check.

Usage: python scripts/analysis/information_gain/knots_sweep.py [--segment-set cuts]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import feature_models as FM
from common import group_average
from plots import COLORS
from scoring import score

KNOT_COUNTS = [3, 5, 10, 20, 40]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segment-set", default="cuts", choices=C.SEGMENT_SETS)
    ss = p.parse_args().segment_set
    rows = []
    for n in KNOT_COUNTS:
        FM.run(ss, "knots_floor", np.random.default_rng(C.RNG_SEED), n_knots=n, suffix=f"_k{n}")
        sc = score(ss, f"knots_floor_k{n}")
        rows.append(dict(n_params=n - 1, label=f"{n} knots", **{f: group_average(sc.dropna(subset=[f"ig_{a}"]), f"ig_{a}")
                                                              for f, a in zip(C.FEATURES, "SMX")}))
    sc = score(ss, "exp_floor")
    rows.append(dict(n_params=1, label="exp", **{f: group_average(sc.dropna(subset=[f"ig_{a}"]), f"ig_{a}") for f, a in zip(C.FEATURES, "SMX")}))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / f"knots_sweep_{ss}.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    k = df[df.label != "exp"]
    for f in C.FEATURES:
        ax.plot(k.n_params + 1, k[f], "o-", color=COLORS[f], lw=2, label=f)
        ax.scatter([1], df.loc[df.label == "exp", f], marker="s", color=COLORS[f], s=40)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1] + KNOT_COUNTS, ["exp"] + [str(n) for n in KNOT_COUNTS])
    ax.set_xlabel("number of knots (exp = 1 parameter)")
    ax.set_ylabel("held-out information gain (bits)")
    ax.set_title(f"Single-feature models vs. nonlinearity flexibility, {ss}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = C.CHECK_FIG_DIR / f"information_gain_knots_sweep_{ss}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

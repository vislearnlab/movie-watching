"""
Check: held-out information gain of single-feature models vs. number of knots (plus exp, 1 parameter).
Selection on the same leave-one-clip-out scores used for reporting, so treat as a robustness check.

Usage: python scripts/decision_making/knots_sweep.py [--segment-set cuts]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "information_gain"))
import config as C  # noqa: E402
import feature_models as FM  # noqa: E402
from common import group_average  # noqa: E402
from scoring import score  # noqa: E402

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
                                                              for f, a in C.ABBR.items()}))
    sc = score(ss, "exp_floor")
    rows.append(dict(n_params=1, label="exp", **{f: group_average(sc.dropna(subset=[f"ig_{a}"]), f"ig_{a}") for f, a in C.ABBR.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / f"knots_sweep_{ss}.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""
Do kids and adults put the same segments in the same within/between quadrant? Each group is classified with its own
pooled thresholds (q=0.4: middle band unclassified, as plot_quadrant_scatter.py / IG; q=0.5: median split, all
classified), plus kids on adult thresholds. Writes per-segment labels and adult x kid cross-tabs.

Output: data/results/within_between_cosine/quadrants_kids_vs_adults_by_segment.csv, quadrants_kids_vs_adults_crosstab.csv
Usage:  python scripts/analysis/compare_quadrants_kids_adults.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

WB_DIR = Path(__file__).resolve().parents[2] / "data" / "results" / "within_between_cosine"
QUADS = ["concentrated", "coordinated", "selective", "scattered"]
COLS = ["within_person_cosine_dist", "between_person_cosine_dist"]


def thresholds(df: pd.DataFrame, q: float) -> dict:
    return {c: df[c].quantile([q, 1 - q]).to_numpy() for c in COLS}


def classify(df: pd.DataFrame, th: dict) -> pd.Series:
    (wl, wh), (bl, bh) = th[COLS[0]], th[COLS[1]]
    w, b = df[COLS[0]], df[COLS[1]]
    lo_w, hi_w, lo_b, hi_b = w <= wl, w >= wh, b <= bl, b >= bh
    lab = np.select([lo_w & lo_b, hi_w & lo_b, lo_w & hi_b, hi_w & hi_b], QUADS, "unclassified")
    return pd.Series(np.where(w.isna() | b.isna(), None, lab), index=df.index)


def main():
    keys = ["video", "block", "seg_idx", "seg_dur_s"]
    a = pd.read_csv(WB_DIR / "all_clips_by_segment.csv")
    k = pd.read_csv(WB_DIR / "all_clips_by_segment_kids.csv")
    m = a[keys + COLS + ["n_people"]].merge(k[keys + COLS + ["n_people"]], on=keys, suffixes=("_adults", "_kids"))
    m = m.dropna(subset=[f"{c}_{g}" for c in COLS for g in ("adults", "kids")]).reset_index(drop=True)
    ad = m[[f"{c}_adults" for c in COLS]].set_axis(COLS, axis=1)
    kd = m[[f"{c}_kids" for c in COLS]].set_axis(COLS, axis=1)
    print(f"{len(m)} segments valid in both groups; kids per segment median {m.n_people_kids.median():.0f} "
          f"(range {m.n_people_kids.min()}-{m.n_people_kids.max()}), adults median {m.n_people_adults.median():.0f}")
    for c in COLS:
        print(f"  {c}: adult-kid r = {np.corrcoef(ad[c], kd[c])[0, 1]:.2f}; mean adults {ad[c].mean():.3f}, kids {kd[c].mean():.3f}")

    tabs = []
    for q in (0.4, 0.5):
        m[f"adults_q{q}"] = classify(ad, thresholds(ad, q))
        m[f"kids_q{q}"] = classify(kd, thresholds(kd, q))
        m[f"kids_adultth_q{q}"] = classify(kd, thresholds(ad, q))
        for kid_col, label in [(f"kids_q{q}", "own thresholds"), (f"kids_adultth_q{q}", "adult thresholds")]:
            both = m[(m[f"adults_q{q}"] != "unclassified") & (m[kid_col] != "unclassified")]
            same = (both[f"adults_q{q}"] == both[kid_col]).sum()
            pa, pk = both[f"adults_q{q}"].value_counts(normalize=True), both[kid_col].value_counts(normalize=True)
            chance = (pa * pk.reindex(pa.index, fill_value=0)).sum()  # agreement expected from the marginals alone
            order = QUADS + (["unclassified"] if q < 0.5 else [])
            ct = pd.crosstab(m[f"adults_q{q}"], m[kid_col]).reindex(index=order, columns=order, fill_value=0)
            print(f"\n== q={q}, kids on {label}: {same}/{len(both)} same ({same / len(both):.0%}) among segments "
                  f"classified in both (chance {chance:.0%})")
            print("rows = adults, cols = kids\n" + ct.to_string())
            tabs.append(ct.rename_axis("adults").reset_index().assign(q=q, kid_thresholds=label))

    m.round(4).to_csv(WB_DIR / "quadrants_kids_vs_adults_by_segment.csv", index=False)
    pd.concat(tabs, ignore_index=True).to_csv(WB_DIR / "quadrants_kids_vs_adults_crosstab.csv", index=False)


if __name__ == "__main__":
    main()

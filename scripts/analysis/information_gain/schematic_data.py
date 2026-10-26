"""
Data for the floor -> model -> ceiling IG schematic (plot_ig_schematic.R): one viewer in one segment, the three
probability maps they are scored under (floor, a feature model, the ceiling built from everyone else's gaze), their
fixations, and the per-map log-likelihoods. Recomputed from the caches and checked against the stored scores.

Output: data/results/information_gain/schematic/<clip>_seg<idx>_<pid>_{maps,fixations,scores}.csv,
        figures/presentations/ig_schematic/frames/<clip>_seg<idx>.png
Usage:  python scripts/analysis/information_gain/schematic_data.py [--clip sesameus_1 --seg 15 --pid MW012 --feature motion]
"""

import argparse
import subprocess

import numpy as np
import pandas as pd

import config as C
from common import TILE_COLS, load_gaze, log_lik
from load_features import TILE_XY
from load_gaze import load_fixations
from segments import segment_table

OUT_DIR = C.RESULTS_DIR / "schematic"
FRAME_DIR = C.PROJECT_ROOT / "figures" / "presentations" / "ig_schematic" / "frames"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="sesameus_1")
    p.add_argument("--seg", type=int, default=15)
    p.add_argument("--pid", default="MW012")
    p.add_argument("--feature", default="motion")
    a = p.parse_args()
    tag = f"{a.clip}_seg{a.seg}_{a.pid}"

    scores = pd.read_parquet(C.CACHE_DIR / "scores_cuts_knots_floor.parquet").query("video == @a.clip and seg_idx == @a.seg")
    me = scores.set_index("participant").loc[a.pid]
    gaze = load_gaze("cuts").merge(scores[["video", "seg_idx", "participant"]])  # exactly the ceiling's cells
    G_all = gaze[TILE_COLS].to_numpy()
    i = gaze.participant.tolist().index(a.pid)
    G = G_all[i]
    floor = pd.read_parquet(C.CACHE_DIR / "floor_cuts.parquet").set_index("video").loc[a.clip, TILE_COLS].to_numpy(float)
    preds = pd.read_parquet(C.CACHE_DIR / "predictions_cuts_knots_floor.parquet")
    model = preds.query("video == @a.clip and seg_idx == @a.seg and feature == @a.feature and not shuffled")[TILE_COLS].to_numpy(float)[0]
    others = (G_all.sum(0) - G) / (len(G_all) - 1)
    ceiling = (1 - me.lam) * others + me.lam * floor

    ll = {k: float(log_lik(G, q)) for k, q in [("floor", floor), ("model", model), ("ceiling", ceiling)]}
    abbr = C.ABBR[a.feature]
    assert np.isclose(ll["ceiling"] - ll["floor"], me.ig_ceiling) and np.isclose(ll["model"] - ll["floor"], me[f"ig_{abbr}"])
    seg_mean = scores[[f"ig_{abbr}", "ig_ceiling"]].mean()
    out_scores = pd.DataFrame(dict(
        map=["floor", "model", "ceiling"], ll_bits=[ll["floor"], ll["model"], ll["ceiling"]],
        ig_bits=[0, ll["model"] - ll["floor"], ll["ceiling"] - ll["floor"]],
        feature=a.feature, participant=a.pid, n_others=len(G_all) - 1, lam=me.lam,
        segment_mean_ig_model=seg_mean.iloc[0], segment_mean_ig_ceiling=seg_mean.iloc[1]))

    fix = load_fixations(a.clip, "adults").query("pid == @a.pid")
    st = segment_table("cuts").query("video == @a.clip and seg_idx == @a.seg").iloc[0]
    s, e = st.seg_start_s, st.seg_end_s
    fix = fix[(fix.endT > s * 1000) & (fix.startT < e * 1000)]
    fixations = pd.DataFrame(dict(x=fix.xpos.to_numpy(), y=1080 - fix.ypos.to_numpy(),  # Tobii bottom-left -> top-left
                                  dur_ms=(np.minimum(fix.endT, e * 1000) - np.maximum(fix.startT, s * 1000)).to_numpy()))
    maps = pd.DataFrame(dict(tile=np.arange(120), x=TILE_XY[:, 0], y=TILE_XY[:, 1], floor=floor, model=model,
                             ceiling=ceiling, viewer=G))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    maps.to_csv(OUT_DIR / f"{tag}_maps.csv", index=False)
    fixations.to_csv(OUT_DIR / f"{tag}_fixations.csv", index=False)
    out_scores.to_csv(OUT_DIR / f"{tag}_scores.csv", index=False)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{(s + e) / 2:.3f}", "-i",
                    str(C.PROJECT_ROOT / "stimuli" / "main_blocks" / f"{a.clip}.mp4"), "-frames:v", "1",
                    "-vf", "scale=1920:1080", str(FRAME_DIR / f"{a.clip}_seg{a.seg}.png")], check=True)
    print(out_scores.round(3).to_string(index=False))
    print(f"{len(fixations)} fixations; wrote {OUT_DIR}/{tag}_*.csv")


if __name__ == "__main__":
    main()

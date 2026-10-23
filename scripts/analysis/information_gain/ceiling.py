"""
Ceiling: how well everyone else predicts each viewer, per segment. gold = (1 - lam) * mean of the
other participants' gaze in the segment + lam * floor. lam_c for held-out clip c is chosen by inner
leave-one-clip-out on the other clips (inner floors exclude both c and the inner held-out clip).

Output: ceiling_<segment_set>.parquet, one row per participant x segment: ll_floor, ll_gold, ig_ceiling, lam, n_others.

Usage: python scripts/analysis/information_gain/ceiling.py
"""

import numpy as np
import pandas as pd

import config as C
from common import TILE_COLS, UNIFORM, clip_sums, group_average, load_gaze, log_lik, pooled_excluding

LAM_GRID = np.logspace(-4, -0.005, 40)


def segment_blocks(gaze: pd.DataFrame) -> list[dict]:
    """Per segment: cell rows, gaze matrix, and leave-one-out mean of the others (cells with too few others dropped)."""
    blocks = []
    for (clip, seg), g in gaze.groupby(["video", "seg_idx"]):
        n = len(g)
        if n - 1 < C.MIN_OTHERS_CEILING:
            continue
        G = g[TILE_COLS].to_numpy()
        others = (G.sum(0) - G) / (n - 1)  # excludes each participant's own gaze
        blocks.append(dict(video=clip, seg_idx=seg, rows=g.drop(columns=TILE_COLS), G=G, others=others))
    return blocks


def floor_vec(sums, counts, exclude, eps):
    return (1 - eps) * pooled_excluding(sums, counts, exclude) + eps * UNIFORM


def choose_lam(blocks, sums, counts, held_out, eps):
    train = [b for b in blocks if b["video"] != held_out]
    lls, meta = [], []
    for b in train:
        f = floor_vec(sums, counts, {held_out, b["video"]}, eps)
        q = (1 - LAM_GRID[:, None, None]) * b["others"][None] + LAM_GRID[:, None, None] * f[None, None]
        lls.append(log_lik(b["G"][None], q))  # (n_lam, n_cells)
        meta.append(b["rows"][["participant"]])
    ll = np.concatenate(lls, axis=1)
    meta = pd.concat(meta, ignore_index=True)
    scores = [group_average(meta.assign(ll=ll[i]), "ll") for i in range(len(LAM_GRID))]
    return LAM_GRID[int(np.argmax(scores))]


def build(segment_set: str, gaze: pd.DataFrame | None = None, floors: pd.DataFrame | None = None) -> pd.DataFrame:
    gaze = load_gaze(segment_set) if gaze is None else gaze
    sums, counts = clip_sums(gaze)
    if floors is None:
        floors = pd.read_parquet(C.CACHE_DIR / f"floor_{segment_set}.parquet").set_index("video")
    blocks = segment_blocks(gaze)
    out = []
    for clip in C.CLIPS:
        eps = floors.loc[clip, "eps"]
        lam = choose_lam(blocks, sums, counts, clip, eps)
        floor = floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
        for b in (b for b in blocks if b["video"] == clip):
            gold = (1 - lam) * b["others"] + lam * floor
            ll_floor = log_lik(b["G"], np.broadcast_to(floor, b["G"].shape))
            ll_gold = log_lik(b["G"], gold)
            out.append(b["rows"].assign(n_others=len(b["G"]) - 1, lam=lam, ll_floor=ll_floor,
                                        ll_gold=ll_gold, ig_ceiling=ll_gold - ll_floor))
        print(f"  {clip:15s} lam={lam:.3f}")
    return pd.concat(out, ignore_index=True)


def main():
    for segment_set in C.SEGMENT_SETS:
        print(segment_set)
        df = build(segment_set)
        out = C.CACHE_DIR / f"ceiling_{segment_set}.parquet"
        df.to_parquet(out, index=False)
        per_seg = df.groupby(["video", "seg_idx"]).ig_ceiling.mean()
        print(f"  group ceiling = {group_average(df, 'ig_ceiling'):.3f} bits; per-segment median "
              f"{per_seg.median():.3f} (range {per_seg.min():.3f} to {per_seg.max():.3f}); "
              f"{len(df)} cells, {per_seg.size} segments -> {out}")


if __name__ == "__main__":
    main()

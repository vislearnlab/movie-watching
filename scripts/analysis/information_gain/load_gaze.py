"""
Per participant x segment gaze vectors over the 120 tiles, normalized to sum to 1. Cached as
gaze_<segment_set>.parquet: group, segment_set, video, seg_idx, participant, n_fix, look_ms, tile_000..tile_119.

Usage: python scripts/analysis/information_gain/load_gaze.py
"""

import sys

import numpy as np
import pandas as pd

import config as C
from segments import segment_table

sys.path.insert(0, str(C.PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(C.PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from gaze_tiles import build_participant_arrays, closed_form_tile_values, load_video_fixations  # noqa: E402
from tile_pooling import CENTERS  # noqa: E402

TILE_COLS = [f"tile_{i:03d}" for i in range(120)]
CX, CY = CENTERS["center_x"].to_numpy(), CENTERS["center_y"].to_numpy()


def load_fixations(clip: str, group: str) -> pd.DataFrame:
    if group == "adults":
        return load_video_fixations(C.FIXATION_DIR, clip, C.FIXATION_PARAM_CODE, C.PARTICIPANT_SUMMARY_PATH)
    fix = pd.concat([pd.read_csv(p) for p in sorted(C.KIDS_FIXATION_DIR.glob(f"*_{clip}_{C.FIXATION_PARAM_CODE}.csv"))],
                    ignore_index=True)
    return fix[~fix.pid.isin(C.KIDS_EXCLUDE)]


def build(segment_set: str, group: str = C.GROUP) -> pd.DataFrame:
    segs = segment_table(segment_set)
    rows, vecs = [], []
    for clip in C.CLIPS:
        fix = load_fixations(clip, group)
        fix = fix[(fix.endT - fix.startT) <= C.MAX_FIXATION_MS]
        arrays = build_participant_arrays(fix)
        for seg in segs[segs.video == clip].itertuples():
            s_ms, e_ms = seg.seg_start_s * 1000, seg.seg_end_s * 1000
            for pid, (startT, endT, x, y) in sorted(arrays.items()):
                overlap = np.minimum(endT, e_ms) - np.maximum(startT, s_ms)
                n_fix = int((overlap > 0).sum())
                v = closed_form_tile_values(startT, endT, x, y, s_ms, e_ms, C.SIGMA_PX, CX, CY)
                total = v.sum()
                if total <= 0:
                    continue
                rows.append(dict(group=group, segment_set=segment_set, video=clip, seg_idx=seg.seg_idx,
                                 participant=pid, n_fix=n_fix, look_ms=float(overlap.clip(min=0).sum())))
                vecs.append(v / total)
    df = pd.DataFrame(rows)
    return pd.concat([df, pd.DataFrame(np.array(vecs), columns=TILE_COLS)], axis=1)


def main():
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for segment_set in C.SEGMENT_SETS:
        df = build(segment_set)
        out = C.CACHE_DIR / f"gaze_{segment_set}.parquet"
        df.to_parquet(out, index=False)
        keep = df.n_fix >= C.MIN_FIXATIONS
        n_cells = df.groupby(["video", "seg_idx"]).ngroups
        print(f"\n{segment_set}: {len(df)} participant x segment cells with gaze, {n_cells} segments -> {out}")
        print(f"  excluded (< {C.MIN_FIXATIONS} fixations): {(~keep).sum()} ({(~keep).mean():.1%})")
        print(f"  look_ms per cell: median {df.look_ms.median():.0f}, IQR {df.look_ms.quantile(.25):.0f}-{df.look_ms.quantile(.75):.0f}")
        print(f"  fixations per cell: median {df.n_fix.median():.0f}")
        assert np.allclose(df[TILE_COLS].sum(axis=1), 1)


if __name__ == "__main__":
    main()

"""Segment table for either segment set: cut-to-cut, or slow clips chopped into equal subsegments."""

import pandas as pd

import config as C


def segment_table(segment_set: str) -> pd.DataFrame:
    assert segment_set in C.SEGMENT_SETS
    cuts = pd.read_csv(C.CUTS_PATH)
    rows = []
    for clip in C.CLIPS:
        dur = C.DURATIONS[clip]
        bounds = [0.0] + sorted(cuts.loc[cuts.video_name == clip, "cut_time_s"]) + [dur]
        n_sub = C.N_SUBSEGMENTS_SLOW if (segment_set == "slow_chopped" and clip in C.CHOPPED_CLIPS) else 1
        seg_idx = 0
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            if s1 - s0 <= 0:
                continue
            step = (s1 - s0) / n_sub
            for k in range(n_sub):
                rows.append(dict(video=clip, block=C.BLOCKS[clip], seg_idx=seg_idx,
                                 seg_start_s=s0 + k * step, seg_end_s=s0 + (k + 1) * step))
                seg_idx += 1
    df = pd.DataFrame(rows)
    df["seg_dur_s"] = df.seg_end_s - df.seg_start_s
    df["segment_set"] = segment_set
    return df

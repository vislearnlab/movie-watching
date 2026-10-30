"""
Snap reference cut times to the frame where the picture actually changes.

About half the cuts in scene-cuts/all_clips_cuts.csv map one frame late. For
each cut, the frame-to-frame change is measured within ±WINDOW frames; if one
frame's change is at least SHARPNESS × every other's, that frame is the cut
("hard"). A fade through black has no such frame: if the darkest frame within
±FADE_WINDOW has mean intensity below FADE_BLACK, the cut is the frame after it,
the first of the fade-in ("fade"). Other cuts keep the reference frame ("kept").

Cuts checked by eye that this rule gets wrong are set in scene-cuts/cut_overrides.csv
(video_name, ref_frame, cut_frame, note) and marked "manual".

Output: scene-cuts/all_clips_cuts_frames.csv (video_name, cut_time_s, ref_frame, cut_frame, method)

Usage:
    python pipeline/refine_cuts.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

WINDOW    = 3
SHARPNESS = 2.0
FADE_WINDOW = 5
FADE_BLACK  = 20


def refine_cut(cap, f: int) -> tuple[int, str]:
    """(cut frame, "hard" | "kept") for a reference cut at frame f."""
    a = max(f - WINDOW - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    small = []
    for _ in range(a, f + WINDOW + 1):
        ok, img = cap.read()
        if not ok:
            break
        small.append(cv2.resize(img, (160, 90)).astype(np.int16))
    change = {a + i: np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))}  # change entering frame
    best = max(change, key=change.get)
    rest = [v for k, v in change.items() if k != best]
    if change[best] >= SHARPNESS * max(rest):
        return best, "hard"
    cap.set(cv2.CAP_PROP_POS_FRAMES, f - FADE_WINDOW)
    level = {}
    for g in range(f - FADE_WINDOW, f + FADE_WINDOW + 1):
        ok, img = cap.read()
        if not ok:
            break
        level[g] = img.mean()
    dark = min(level, key=level.get)
    if level[dark] < FADE_BLACK:
        return dark + 1, "fade"
    return f, "kept"


def main():
    ref = pd.read_csv(C.CUTS_CSV)
    rows = []
    for stem, g in ref.groupby("video_name"):
        cap = cv2.VideoCapture(str(C.video_path(stem)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for t in sorted(g.cut_time_s):
            f = round(t * fps)
            cut, method = refine_cut(cap, f)
            rows.append(dict(video_name=stem, cut_time_s=t, ref_frame=f, cut_frame=cut, method=method))
        cap.release()
    df = pd.DataFrame(rows)
    for o in pd.read_csv(C.CUT_OVERRIDES_CSV).itertuples():
        hit = (df.video_name == o.video_name) & (df.ref_frame == o.ref_frame)
        assert hit.sum() == 1, f"override {o.video_name} ref_frame {o.ref_frame} matches {hit.sum()} cuts"
        df.loc[hit, ["cut_frame", "method"]] = [o.cut_frame, "manual"]
    df.to_csv(C.CUT_FRAMES_CSV, index=False)
    moved = df[df.cut_frame != df.ref_frame]
    print(f"{len(df)} cuts: {(df.method == 'hard').sum()} hard, {(df.method == 'fade').sum()} fade, {(df.method == 'kept').sum()} kept, {(df.method == 'manual').sum()} manual; "
          f"{len(moved)} moved, offsets {moved.eval('cut_frame - ref_frame').value_counts().to_dict()}")
    print(f"→ {C.CUT_FRAMES_CSV}")


if __name__ == "__main__":
    main()

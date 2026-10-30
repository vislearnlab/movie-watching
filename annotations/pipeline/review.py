"""
Images for reviewing annotations shot by shot.

  sheets  qc/review/{stem}_shot{k}.jpg — N_FRAMES frames spread over each shot with
          every mask, label and click (all shots, including ones with no prompt).
  frame   qc/review/frames/{stem}_f{frame}.jpg — one full-size frame with masks and a
          GRID_PX coordinate grid, for reading off box / click coordinates.

Both draw each shot from its newest shard, so they show the latest iteration even
before the pass ends and merges.

Usage:
    python pipeline/review.py sheets --video sesameus_1 [--shot 3 4]
    python pipeline/review.py frame --video sesameus_1 --frame 120 [240 …]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plan import load_shots
from annotate_videos import shard_base
from render import Overlay, sheet

N_FRAMES = 8
GRID_PX  = 100
REVIEW_DIR = C.QC_DIR / "review"


def overlay(stem, k: int) -> Overlay:
    """Overlay of shot k's newest shard (the merged output lags until a pass ends); no shard: no masks."""
    base = shard_base(stem, k, ["face", "hand"])
    npz = Path(f"{base}.npz")
    return Overlay(stem, npz if npz.exists() else None, Path(f"{base}.clicks.csv"))


def shot_of(stem, f: int) -> int:
    return next(k for k, (s, e) in enumerate(shots_of(stem)) if s <= f < e)


def shots_of(stem) -> list[tuple[int, int]]:
    cap = cv2.VideoCapture(str(C.video_path(stem)))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return load_shots(stem, fps, n)


def shot_sheet(ov: Overlay, k: int, title: str = "") -> np.ndarray:
    s, e = load_shots(ov.stem, ov.fps, ov.n)[k]
    frames = sorted({int(f) for f in np.linspace(s, e - 1, N_FRAMES)})
    img = sheet(ov, frames, cols=4, width=480)
    bar = np.zeros((30, img.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{ov.stem} shot {k}  frames {s}-{e - 1}  ({s / ov.fps:.2f}-{e / ov.fps:.2f}s)  {title}",
                (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def grid_frame(ov: Overlay, f: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(C.video_path(ov.stem)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    assert ok, f"{ov.stem}: could not read frame {f}"
    cap.release()
    img = ov.draw(f, img)
    H, W = img.shape[:2]
    for x in range(0, W, GRID_PX):
        cv2.line(img, (x, 0), (x, H), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, H, GRID_PX):
        cv2.line(img, (0, y), (W, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sheets", "frame"])
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--shot", nargs="+", type=int)
    ap.add_argument("--frame", nargs="+", type=int)
    args = ap.parse_args()
    for stem in args.video:
        if args.what == "sheets":
            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            for k in args.shot or range(len(shots_of(stem))):
                out = REVIEW_DIR / f"{stem}_shot{k:03d}.jpg"
                cv2.imwrite(str(out), shot_sheet(overlay(stem, k), k), [cv2.IMWRITE_JPEG_QUALITY, 88])
                print(out)
        else:
            (REVIEW_DIR / "frames").mkdir(parents=True, exist_ok=True)
            for f in args.frame:
                out = REVIEW_DIR / "frames" / f"{stem}_f{f}.jpg"
                cv2.imwrite(str(out), grid_frame(overlay(stem, shot_of(stem, f)), f), [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(out)


if __name__ == "__main__":
    main()

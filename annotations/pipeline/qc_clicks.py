"""
Draw every annotator click on the frame it was made on, one contact sheet per
video. Use this to check the screen→video coordinate mapping in config.VIDEOS:
include clicks (green) should sit on the named object, exclude clicks (red) on
whatever SAM3 should not segment.

Usage:
    python pipeline/qc_clicks.py [--video sesameus_1 …] [--out qc/clicks]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TILE_W, TILE_H, COLS = 384, 216, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+")
    ap.add_argument("--out", default=str(C.QC_DIR / "clicks"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.PROMPTS_CSV)
    for stem, g in df.groupby("video_stem"):
        if args.video and stem not in args.video:
            continue
        clicks = [(c, r.prompt_idx_raw, r.prompt_name) for r in g.itertuples() for c in json.loads(r.clicks)]
        if not clicks:
            continue
        cap = cv2.VideoCapture(str(C.video_path(stem))); fps = cap.get(cv2.CAP_PROP_FPS)
        tiles = []
        for c, pid, name in sorted(clicks, key=lambda t: t[0]["t_ms"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(c["t_ms"] / 1000 * fps)))
            ok, f = cap.read()
            if not ok:
                continue
            col = (0, 220, 0) if c["label"] else (0, 0, 255)
            cv2.circle(f, (c["x"], c["y"]), 14, col, -1); cv2.circle(f, (c["x"], c["y"]), 14, (255, 255, 255), 3)
            f = cv2.resize(f, (TILE_W, TILE_H))
            cv2.rectangle(f, (0, 0), (TILE_W, 20), (0, 0, 0), -1)
            cv2.putText(f, f"{c['t_ms'] / 1000:.2f}s #{pid} {name}"[:48], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            tiles.append(f)
        cap.release()
        while len(tiles) % COLS:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
        cv2.imwrite(str(out / f"{stem}_clicks.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{stem}: {len(clicks)} clicks → {out / f'{stem}_clicks.jpg'}")


if __name__ == "__main__":
    main()

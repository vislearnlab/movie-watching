"""
Review sheets as shots finish: every POLL_S seconds, each shard whose stamp is newer
than its sheet gets qc/review/{video}_shot{k}.jpg (review.shot_sheet, drawn from that
shard alone, titled with its iteration) and qc/review/latest.jpg. Runs until killed.

Usage (cycle.sh starts it in its own tmux window):
    python pipeline/live_preview.py
"""

import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from annotate_videos import stamp
from render import Overlay
from review import REVIEW_DIR, shot_sheet

POLL_S = 20
PREFIX = "face_hand"


def main():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for js in sorted(C.OUTPUT_DIR.glob(f"*/shards/{PREFIX}_shot*.json"), key=lambda p: p.stat().st_mtime):
            if not re.fullmatch(rf"{PREFIX}_shot\d+", js.stem):           # skip *.tmp.json
                continue
            stem, k = js.parent.parent.name, int(js.stem.rsplit("shot", 1)[1])
            out = REVIEW_DIR / f"{stem}_shot{k:03d}.jpg"
            if out.exists() and out.stat().st_mtime > js.stat().st_mtime:
                continue
            base = js.with_suffix("")
            st = stamp(base)
            ov = Overlay(stem, Path(f"{base}.npz"), Path(f"{base}.clicks.csv"))
            tmp = out.with_suffix(".tmp.jpg")
            cv2.imwrite(str(tmp), shot_sheet(ov, k, f"iteration {st['iteration']}  finished {st['finished']}"),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            os.replace(tmp, out)
            shutil.copyfile(out, REVIEW_DIR / "latest.tmp.jpg")
            os.replace(REVIEW_DIR / "latest.tmp.jpg", REVIEW_DIR / "latest.jpg")
            print(f"{datetime.now():%H:%M:%S} {stem} shot {k} iteration {st['iteration']} → {out}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()

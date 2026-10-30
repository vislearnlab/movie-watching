"""
Render merged annotations for inspection.

  video  {output}/{stem}/annotated.mp4 — full length, original size and exact
         source frame rate (ffmpeg, so 30000/1001 stays 30000/1001), every
         prompt's mask with a "{prompt_idx_raw}. {name}" label (the id corrections use), cut marks, and each
         annotator click drawn on the frame it was matched to.
  sheet  {qc_dir}/{stem}_sheet_{t0}-{t1}.jpg — contact sheet of the same overlay.

Usage:
    python pipeline/render.py video --video sesameus_1
    python pipeline/render.py sheet --video sesameus_1 --t0 0 --t1 30 --step 1
"""

import argparse
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plan import load_shots

FILL_ALPHA = 0.25   # light fills keep the object under a mask visible to reviewers
COLOURS = [(60, 60, 255), (255, 180, 60), (60, 220, 60), (0, 165, 255), (255, 60, 160), (0, 210, 190),
           (255, 60, 220), (0, 210, 210), (200, 0, 200), (100, 220, 255), (255, 128, 0), (0, 200, 100)]


class Overlay:
    """Draws masks, labels, clicks and shot index for one video."""

    def __init__(self, stem, masks_npz: Path | None, clicks_csv: Path):
        """Overlay from one masks.npz (merged or a single shard; None: no masks) and its clicks.csv (may be absent)."""
        self.stem = stem
        self.masks = {}
        if masks_npz is not None:
            with np.load(masks_npz) as z:
                for k in z.files:
                    prefix, f = k.rsplit("|", 1)
                    self.masks.setdefault(int(f), []).append((prefix, z[k]))
        prompts = pd.read_csv(C.PROMPTS_CSV)
        prompts = prompts[prompts.video_stem == stem]
        self.label = {}
        for i, p in enumerate(prompts.itertuples()):
            self.label[int(p.prompt_idx)] = (COLOURS[i % len(COLOURS)], f"{p.prompt_idx_raw}. {p.prompt_name}")
        self.clicks = pd.read_csv(clicks_csv) if clicks_csv.exists() else pd.DataFrame(columns=["frame_used"])
        cap = cv2.VideoCapture(str(C.video_path(stem)))
        self.fps, self.n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.shot_of = np.zeros(self.n, int)
        for k, (s, e) in enumerate(load_shots(stem, self.fps, self.n)):
            self.shot_of[s:e] = k

    def draw(self, f: int, img: np.ndarray) -> np.ndarray:
        for prefix, m in self.masks.get(f, []):
            colour, label = self.label[int(prefix[1:].split("_", 1)[0])]
            over = img.copy(); over[m] = colour
            cv2.addWeighted(over, FILL_ALPHA, img, 1 - FILL_ALPHA, 0, img)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx, ty = min(max(cx - tw // 2, 0), img.shape[1] - tw), min(max(cy, th + 4), img.shape[0] - 4)
            cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        for c in self.clicks[self.clicks.frame_used == f].itertuples():
            colour = (0, 220, 0) if c.label == 1 else (0, 0, 255)
            cv2.drawMarker(img, (int(c.x), int(c.y)), colour, cv2.MARKER_TILTED_CROSS, 22, 3)
            cv2.putText(img, f"{c.prompt_idx}:{c.status}", (int(c.x) + 12, int(c.y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        cv2.putText(img, f"f{f} {f / self.fps:.2f}s shot {self.shot_of[f]}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img


def merged_overlay(stem) -> Overlay:
    vdir = Path(C.OUTPUT_DIR) / stem
    return Overlay(stem, vdir / "masks.npz", vdir / "clicks.csv")


def render_video(stem):
    ov = merged_overlay(stem)
    W, H = C.VIDEO_SIZE
    out = Path(C.OUTPUT_DIR) / stem / "annotated.mp4"
    rate = Fraction(ov.fps).limit_denominator(1001)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                          stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(C.video_path(stem)))
    for f in range(ov.n):
        ok, img = cap.read()
        assert ok, f"{stem}: could not read frame {f}"
        ff.stdin.write(ov.draw(f, img).tobytes())
    cap.release()
    ff.stdin.close()
    assert ff.wait() == 0, f"ffmpeg failed for {stem}"
    print(f"{stem}: {ov.n} frames @ {rate} → {out}")


def sheet(ov: Overlay, frames, cols=4, width=480) -> np.ndarray:
    """Contact sheet of ov drawn on the given frame indices."""
    cap = cv2.VideoCapture(str(C.video_path(ov.stem)))
    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok, f"{ov.stem}: could not read frame {f}"
        tiles.append(cv2.resize(ov.draw(f, img), (width, width * 9 // 16)))
    cap.release()
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def render_sheet(stem, t0, t1, step, out_dir):
    ov = merged_overlay(stem)
    frames = [int(t * ov.fps) for t in np.arange(t0, t1, step) if int(t * ov.fps) < ov.n]
    out = Path(out_dir) / f"{stem}_sheet_{t0:g}-{t1:g}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet(ov, frames), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["video", "sheet"])
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--out", default=str(C.QC_DIR / "sheets"))
    args = ap.parse_args()
    for stem in args.video:
        if args.what == "video":
            render_video(stem)
        else:
            render_sheet(stem, args.t0, args.t1, args.step, args.out)


if __name__ == "__main__":
    main()

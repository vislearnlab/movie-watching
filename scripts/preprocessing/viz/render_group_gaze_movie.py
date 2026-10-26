"""
Presentation group-gaze movie: full-color clip with every adult's current fixation as a dot and/or the group's
density cloud (0.5s window centered on each frame). --frame renders single-frame style previews instead.

Usage:
    python scripts/preprocessing/viz/render_group_gaze_movie.py --video pixar_birds --start 45 --dur 10  # none/dots/cloud/both
    python scripts/preprocessing/viz/render_group_gaze_movie.py --video pixar_birds --frame 47.3
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from gaze_tiles import DISPLAY_H, DISPLAY_W, build_participant_arrays, load_video_fixations  # noqa: E402
from video_overlay import density_field, open_frame_writer, probe_video, read_frame  # noqa: E402

FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / "adults"
SUMMARY_PATH = PROJECT_ROOT / "data" / "local_cache" / "participant_summary.csv"
STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
OUT_DIR = PROJECT_ROOT / "figures" / "presentations" / "group_gaze"
OUT_W, OUT_H = 1280, 720
SIGMA_PX = 46.0
CLOUD_WINDOW_S = 0.5
GRID_SCALE = 4  # density grid at 1/4 output resolution

STYLES = {  # file name -> overlay
    "none": dict(cloud=False, dots=False),
    "dots": dict(cloud=False, dots=True),
    "cloud": dict(cloud=True, dots=False),
    "both": dict(cloud=True, dots=True),
}
CLOUD_RGB = (0, 230, 255)  # cyan: contrasts with warm scenes and pixar's blues
DOT_FILL, DOT_EDGE, DOT_R = (255, 235, 59, 235), (0, 0, 0, 255), 9


def cloud_overlay(field: np.ndarray, vmax: float) -> Image.Image:
    v = np.clip(field / vmax, 0, 1)
    alpha = np.where(v > 0.04, v ** 0.6 * 0.7, 0)
    rgba = np.dstack([np.broadcast_to(np.array(CLOUD_RGB, float), (*v.shape, 3)), alpha * 255]).astype(np.uint8)
    return Image.fromarray(rgba).resize((OUT_W, OUT_H), Image.BILINEAR)


def current_fixations(arrays: dict, pids: list, t_ms: float) -> list:
    pts = []
    for pid in pids:
        startT, endT, x, y = arrays[pid]
        m = (startT <= t_ms) & (endT > t_ms)
        if m.any():
            pts.append((x[m][0], y[m][0]))
    return pts


def compose(bg: np.ndarray, t_s: float, arrays: dict, pids: list, gx, gy, vmax: float, style: dict) -> Image.Image:
    img = Image.fromarray(bg).convert("RGBA")
    if style["cloud"]:
        h = CLOUD_WINDOW_S * 500
        field = density_field(arrays, pids, t_s * 1000 - h, t_s * 1000 + h, SIGMA_PX, gx, gy)
        img = Image.alpha_composite(img, cloud_overlay(field, vmax))
    if style["dots"]:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for x, y in current_fixations(arrays, pids, t_s * 1000):
            px, py = x * OUT_W / DISPLAY_W, y * OUT_H / DISPLAY_H
            d.ellipse([px - DOT_R, py - DOT_R, px + DOT_R, py + DOT_R], fill=DOT_FILL, outline=DOT_EDGE, width=2)
        img = Image.alpha_composite(img, layer)
    return img.convert("RGB")


def reader(path: Path, start: float, dur: float) -> subprocess.Popen:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-t", str(dur), "-vf", f"scale={OUT_W}:{OUT_H}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float)
    p.add_argument("--dur", type=float, default=10.0)
    p.add_argument("--frame", type=float, help="render style previews of this one time point instead of a movie")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--margin-frames", type=int, default=0,
                   help="trim this many frames off each end; cut annotations are only frame-accurate, so use 2 for segment clips")
    a = p.parse_args()

    path = STIMULI_DIR / f"{a.video}_stripped.mp4"
    fps = probe_video(str(path))["fps"]
    if a.frame is None and a.margin_frames:
        a.start, a.dur = a.start + a.margin_frames / fps, a.dur - 2 * a.margin_frames / fps
    arrays = build_participant_arrays(load_video_fixations(FIXATION_DIR, a.video, "f329476c", SUMMARY_PATH))
    pids = sorted(arrays)
    gx = (np.arange(OUT_W // GRID_SCALE) + 0.5) * DISPLAY_W / (OUT_W // GRID_SCALE)
    gy = (np.arange(OUT_H // GRID_SCALE) + 0.5) * DISPLAY_H / (OUT_H // GRID_SCALE)
    t0 = a.frame if a.frame is not None else a.start
    span = 1.0 if a.frame is not None else a.dur
    h = CLOUD_WINDOW_S * 500
    peaks = [density_field(arrays, pids, t * 1000 - h, t * 1000 + h, SIGMA_PX, gx, gy).max() for t in np.arange(t0, t0 + span, 0.25)]
    vmax = float(np.percentile(peaks, 90))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.frame is not None:
        r = reader(path, a.frame, 0.5)
        bg = read_frame(r, OUT_W, OUT_H); r.kill(); r.wait()
        for name, style in STYLES.items():
            out = a.out_dir / "previews" / f"{a.video}_{a.frame:.2f}s_{name}.png"
            out.parent.mkdir(exist_ok=True)
            compose(bg, a.frame, arrays, pids, gx, gy, vmax, style).save(out)
            print("Saved", out)
        return

    folder = a.out_dir / f"{a.video}_{a.start:g}-{a.start + a.dur:g}s"
    folder.mkdir(exist_ok=True)
    r = reader(path, a.start, a.dur)
    writers = {n: open_frame_writer(str(folder / f"{n}.mp4"), OUT_W, OUT_H, fps) for n in STYLES}
    i = 0
    while (bg := read_frame(r, OUT_W, OUT_H)) is not None:
        for n, style in STYLES.items():
            writers[n].stdin.write(np.asarray(compose(bg, a.start + i / fps, arrays, pids, gx, gy, vmax, style)).tobytes())
        i += 1
    r.stdout.close(); r.wait()
    for w in writers.values():
        w.stdin.close(); w.wait()
    print(f"Saved {folder}/{{{','.join(STYLES)}}}.mp4 ({i} frames)")


if __name__ == "__main__":
    main()

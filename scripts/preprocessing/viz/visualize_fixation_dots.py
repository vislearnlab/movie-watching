"""
Kid-vs-adult fixation dot video: one dot per participant, visible for exactly
that participant's real fixation duration -- red = kids, blue = adults.

Replaces visualize_fixation_diff.py's duration-weighted, calibration-blurred
density-map approach. That approach doesn't match how this project's ISC
analysis actually treats fixations (calc_isc_fixations.py / isc_analysis.qmd
correlate raw fixation positions, not smoothed density), so this script
shows the same underlying data the ISC numbers are computed from: discrete
points, no blur, no duration weighting, no windowing.

Per-frame fixation lookup
----------------------------
For each output video frame (native stimulus fps), and for each participant,
a dot is drawn iff that participant has a fixation whose [startT, endT)
contains this frame's timestamp -- i.e. the dot is on screen for exactly as
long as the real fixation lasted, then disappears (or jumps to the next
fixation's position) the instant it ends, same semantics as
i2mc_fixations.py's own fixation boundaries. Each participant's fixation
list is time-ordered, and the video plays forward monotonically, so a
per-participant pointer advances through their fixations as playback
proceeds rather than rescanning all fixations every frame.

Dots, not density
--------------------
Each dot is a fixed-radius (--dot_radius) filled circle at the fixation's
(xpos, ypos), drawn at moderate opacity (--dot_alpha, default 0.8) so that
several same-group participants fixating the same spot are still visible as
a stacked/darker cluster rather than becoming indistinguishable -- but nothing
is blurred, weighted, or spatially smoothed. No calibration-derived sigma is
used here at all (contrast with visualize_fixation_diff.py): a dot's exact
plotted position is the fixation's raw (xpos, ypos) as detected by I2MC.

Color / background
----------------------
Kids = red, adults = blue (project convention, unchanged from the density
version). Background stimulus video is dimmed to --bg_opacity (default 0.4,
same value used previously) so dots stay legible. A static red/blue legend
is drawn in the top-right corner of every frame (built once, not redrawn
per frame, since its content never changes).

Video I/O
-----------
Uses the system `ffmpeg`/`ffprobe` binaries via subprocess (raw RGB24 frames
piped in/out), same as the density version -- this machine has no
opencv-python for any available Python interpreter. Circles are drawn with
Pillow (already installed), not OpenCV.

Usage
-----
    python preprocessing/viz/visualize_fixation_dots.py --video pixar_birds --code f329476c
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── defaults ─────────────────────────────────────────────────────────────────

DISPLAY_WIDTH  = 1920
DISPLAY_HEIGHT = 1080
OUTPUT_WIDTH   = 1280
OUTPUT_HEIGHT  = 720

DOT_RADIUS = 6      # output px
DOT_ALPHA  = 0.8    # per-dot fill opacity (0-1)
BG_OPACITY = 0.4    # background video brightness fraction, everywhere

KID_COLOR    = (220, 30, 30)    # red
ADULT_COLOR  = (30, 60, 220)    # blue


# ── ffmpeg/ffprobe helpers (same approach as visualize_fixation_diff.py used) ──

def probe_video(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]

    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    duration_s = float(stream["duration"])
    n_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") not in (None, "N/A") \
        else int(round(duration_s * fps))

    return {"fps": fps, "n_frames": n_frames, "duration_s": duration_s}


def open_frame_reader(path: str, output_width: int, output_height: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"scale={output_width}:{output_height}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)


def open_frame_writer(path: str, width: int, height: int, fps: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def read_frame(proc: subprocess.Popen, width: int, height: int) -> np.ndarray | None:
    n_bytes = width * height * 3
    buf = proc.stdout.read(n_bytes)
    if len(buf) < n_bytes:
        return None
    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)


# ── fixation loading ────────────────────────────────────────────────────────

def load_group_fixations(fixation_dir: Path, group: str, video_name: str, code: str) -> list[dict]:
    """One entry per participant: {pid, startT, endT, xpos, ypos} arrays, sorted by startT."""
    paths = sorted(glob.glob(str(fixation_dir / group / f"*_{video_name}_{code}.csv")))
    if not paths:
        print(f"  WARNING: no fixation files for group={group} video={video_name} code={code}")
        return []

    participants = []
    for p in paths:
        df = pd.read_csv(p, usecols=["pid", "startT", "endT", "xpos", "ypos"]).sort_values("startT")
        participants.append(dict(
            pid=df["pid"].iloc[0],
            startT=df["startT"].to_numpy(),
            endT=df["endT"].to_numpy(),
            xpos=df["xpos"].to_numpy(),
            ypos=df["ypos"].to_numpy(),
            ptr=0,  # index of the fixation to check next, advances monotonically with t_ms
        ))
    print(f"  {group}: {len(participants)} participants")
    return participants


def current_position(participant: dict, t_ms: float) -> tuple[float, float] | None:
    """Advance participant['ptr'] past any fixations that have already ended,
    then return (xpos, ypos) if the fixation now at ptr contains t_ms, else None.
    Assumes t_ms is non-decreasing across successive calls (video plays forward)."""
    startT, endT = participant["startT"], participant["endT"]
    n = len(startT)
    ptr = participant["ptr"]

    while ptr < n and endT[ptr] <= t_ms:
        ptr += 1
    participant["ptr"] = ptr

    if ptr < n and startT[ptr] <= t_ms:
        return participant["xpos"][ptr], participant["ypos"][ptr]
    return None


# ── frame compositing ───────────────────────────────────────────────────────

def to_output_px(x: float, y: float, display_width: int, display_height: int,
                  output_width: int, output_height: int) -> tuple[int, int]:
    px = x * output_width / display_width
    py = (display_height - y) * output_height / display_height
    return px, py


def build_legend_overlay(output_width: int, output_height: int, dot_radius: int) -> Image.Image:
    """Static 'Kids' / 'Adults' color key, top-right corner. Built once and
    alpha-composited onto every frame, rather than redrawn from scratch each
    frame, since its content never changes."""
    overlay = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    entries = [("Kids", KID_COLOR), ("Adults", ADULT_COLOR)]
    swatch_r   = max(dot_radius, 5)
    row_h      = swatch_r * 2 + 8
    pad        = 10
    text_gap   = 8
    margin     = 14

    max_text_w = max(draw.textbbox((0, 0), label, font=font)[2] for label, _ in entries)
    box_w = pad * 2 + swatch_r * 2 + text_gap + max_text_w
    box_h = pad * 2 + row_h * len(entries)

    box_x0 = output_width - margin - box_w
    box_y0 = margin
    draw.rounded_rectangle(
        [box_x0, box_y0, box_x0 + box_w, box_y0 + box_h],
        radius=6, fill=(0, 0, 0, 150),
    )

    for i, (label, color) in enumerate(entries):
        cy = box_y0 + pad + row_h * i + row_h // 2
        cx = box_x0 + pad + swatch_r
        draw.ellipse([cx - swatch_r, cy - swatch_r, cx + swatch_r, cy + swatch_r],
                     fill=color + (255,), outline=(255, 255, 255, 255), width=1)
        text_x = box_x0 + pad + swatch_r * 2 + text_gap
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_y = cy - (text_bbox[3] - text_bbox[1]) // 2
        draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255, 255))

    return overlay


def compose_frame(
    bg_frame: np.ndarray,
    kids: list[dict], adults: list[dict], t_ms: float,
    display_width: int, display_height: int, output_width: int, output_height: int,
    dot_radius: int, dot_alpha: float, bg_opacity: float,
    legend_overlay: Image.Image,
) -> np.ndarray:
    dimmed_bg = (bg_frame.astype(np.float64) * bg_opacity).astype(np.uint8)
    base = Image.fromarray(dimmed_bg, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    alpha_255 = int(round(dot_alpha * 255))
    for group, color in ((adults, ADULT_COLOR), (kids, KID_COLOR)):
        fill = color + (alpha_255,)
        for participant in group:
            pos = current_position(participant, t_ms)
            if pos is None:
                continue
            px, py = to_output_px(pos[0], pos[1], display_width, display_height, output_width, output_height)
            draw.ellipse(
                [px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius],
                fill=fill, outline=(255, 255, 255, 255), width=1,
            )

    composited = Image.alpha_composite(base, overlay)
    composited = Image.alpha_composite(composited, legend_overlay).convert("RGB")
    return np.array(composited)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Kid-vs-adult fixation dot video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default="pixar_birds", help="video_name to process")
    parser.add_argument("--code", required=True,
                         help="I2MC parameter code (printed by i2mc_fixations.py).")
    parser.add_argument("--fixation_dir", default="data/preprocessed/fixations",
                         help="Directory containing fixation CSVs, relative to project root.")
    parser.add_argument("--stimuli_dir", default="stimuli/main_blocks",
                         help="Directory containing stimulus MP4s, relative to project root.")
    parser.add_argument("--output_dir", default="figures/heat-viz",
                         help="Output directory, relative to project root.")
    parser.add_argument("--display_width", type=int, default=DISPLAY_WIDTH)
    parser.add_argument("--display_height", type=int, default=DISPLAY_HEIGHT)
    parser.add_argument("--output_width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--output_height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument("--dot_radius", type=int, default=DOT_RADIUS, help="Dot radius in output px.")
    parser.add_argument("--dot_alpha", type=float, default=DOT_ALPHA, help="Per-dot fill opacity (0-1).")
    parser.add_argument("--bg_opacity", type=float, default=BG_OPACITY,
                         help="Background video brightness fraction (0-1).")
    return parser.parse_args()


def main():
    args = parse_args()

    fixation_dir = PROJECT_ROOT / args.fixation_dir
    stimuli_dir  = PROJECT_ROOT / args.stimuli_dir
    output_dir   = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = stimuli_dir / f"{args.video}_stripped.mp4"
    if not video_path.exists():
        video_path = stimuli_dir / f"{args.video}.mp4"
    if not video_path.exists():
        sys.exit(f"ERROR: no video found for '{args.video}' in {stimuli_dir}")

    info = probe_video(str(video_path))
    fps, n_frames = info["fps"], info["n_frames"]
    print(f"Video: {video_path.name}  fps={fps:.3f}  frames={n_frames}  duration={info['duration_s']:.1f}s")

    print("Loading fixations...")
    kids   = load_group_fixations(fixation_dir, "kids",   args.video, args.code)
    adults = load_group_fixations(fixation_dir, "adults", args.video, args.code)
    if not kids and not adults:
        sys.exit("ERROR: no fixation data for either group -- nothing to render.")

    legend_overlay = build_legend_overlay(args.output_width, args.output_height, args.dot_radius)

    reader = open_frame_reader(str(video_path), args.output_width, args.output_height)
    out_path = output_dir / f"{args.video}_kids_vs_adults_dots_{args.code}.mp4"
    writer = open_frame_writer(str(out_path), args.output_width, args.output_height, fps)

    for frame_idx in range(n_frames):
        bg_frame = read_frame(reader, args.output_width, args.output_height)
        if bg_frame is None:
            print(f"  Video stream ended early at frame {frame_idx}/{n_frames}")
            break

        t_ms = frame_idx / fps * 1000.0
        out_frame = compose_frame(
            bg_frame, kids, adults, t_ms,
            args.display_width, args.display_height, args.output_width, args.output_height,
            args.dot_radius, args.dot_alpha, args.bg_opacity, legend_overlay,
        )
        writer.stdin.write(out_frame.tobytes())

        if frame_idx % int(round(fps)) == 0:
            print(f"  frame {frame_idx}/{n_frames}", end="\r")

    print()
    reader.stdout.close()
    reader.wait()
    writer.stdin.close()
    writer.wait()

    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

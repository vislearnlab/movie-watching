"""
Render a 0.5s motion window as video, with the per-tile motion overlay held static on top.

Usage:
    python scripts/analysis/render_motion_overlay_video.py --clip frank_complex --window 79
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402

STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
MOTION_DIR = PROJECT_ROOT / "data" / "results" / "motion"
OUT_DIR = PROJECT_ROOT / "figures" / "checks" / "motion_overlay_videos"
WINDOW_S = 0.5
N_COLS = 15


def get_video_info(video_path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "default=noprint_wrappers=1",
         str(video_path)],
        capture_output=True, text=True, check=True,
    )
    info = {}
    for line in out.stdout.strip().splitlines():
        k, v = line.split("=")
        info[k] = v
    num, den = info["r_frame_rate"].split("/")
    return dict(width=int(info["width"]), height=int(info["height"]), fps=float(num) / float(den))


def render_window_video(video: str, window: int):
    video_path = STIMULI_DIR / f"{video}.mp4"
    info = get_video_info(video_path)
    motion_df = pd.read_csv(MOTION_DIR / f"{video}_motion_tiles.csv")
    tile_cols = [c for c in motion_df.columns if c.startswith("tile_")]
    motion_vec = motion_df.loc[motion_df["window_idx"] == window, tile_cols].to_numpy()[0]

    centers, _ = compute_patch_grid(180, N_COLS, 1920, 1080)
    scale_x, scale_y = info["width"] / 1920, info["height"] / 1080

    t0, t1 = window * WINDOW_S, (window + 1) * WINDOW_S
    work_dir = OUT_DIR / f".{video}_w{window}_frames"
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_pattern = work_dir / "raw_%04d.png"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(video_path),
         str(raw_pattern)],
        capture_output=True, check=True,
    )
    raw_frames = sorted(work_dir.glob("raw_*.png"))
    assert raw_frames, f"No frames extracted for {video} window {window}"

    vmax = max(motion_vec.max(), 1e-6)
    for i, raw_path in enumerate(raw_frames):
        img = plt.imread(raw_path)
        fig, ax = plt.subplots(figsize=(info["width"] / 100, info["height"] / 100), dpi=100)
        ax.imshow(img)
        ax.scatter(centers["center_x"] * scale_x, centers["center_y"] * scale_y,
                   c=motion_vec, s=300, cmap="inferno", vmin=0, vmax=vmax,
                   alpha=0.6, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, info["width"])
        ax.set_ylim(info["height"], 0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(work_dir / f"out_{i:04d}.png", dpi=100)
        plt.close(fig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{video}_w{window}_{t0:.1f}-{t1:.1f}s.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(info["fps"]), "-i", str(work_dir / "out_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        capture_output=True, check=True,
    )
    shutil.rmtree(work_dir)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()
    render_window_video(args.clip, args.window)


if __name__ == "__main__":
    main()

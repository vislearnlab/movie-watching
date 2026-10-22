"""
Per-tile optical-flow motion magnitude, on the same 120-tile (15x8) grid as the gaze tile
vectors. See METHODS_DECISIONS.md for method rationale (global-motion subtraction, box- vs.
Gaussian-pooling, fps normalization).

Usage:
    python scripts/analysis/calc_motion_tiles.py --clip pixar_birds --sanity-check
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402

STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "motion"
FIGURE_DIR = PROJECT_ROOT / "figures" / "checks"

WINDOW_S = 0.5  # matches calc_within_between_cosine.py
N_COLS, N_ROWS = 15, 8  # matches the 120-tile gaze grid (compute_patch_grid(180, 15, ...))


def get_video_info(video_path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    info = {}
    for line in out.stdout.strip().splitlines():
        k, v = line.split("=")
        info[k] = v
    num, den = info["r_frame_rate"].split("/")
    return dict(width=int(info["width"]), height=int(info["height"]),
                fps=float(num) / float(den), n_frames=int(info["nb_frames"]),
                duration=float(info["duration"]))


def compute_motion_windows(video: str) -> tuple[np.ndarray, dict]:
    """Returns (motion_vecs, info); motion_vecs is (n_windows, 120), row-major tile order."""
    video_path = STIMULI_DIR / f"{video}.mp4"
    info = get_video_info(video_path)
    fps = info["fps"]

    cap = cv2.VideoCapture(str(video_path))
    ok, prev = cap.read()
    assert ok, f"Could not read first frame of {video_path}"
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    n_windows_total = int(info["duration"] // WINDOW_S)
    window_sums = np.zeros((n_windows_total, N_ROWS * N_COLS))
    window_counts = np.zeros(n_windows_total)

    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                             pyr_scale=0.5, levels=3, winsize=15,
                                             iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        global_dx, global_dy = np.median(flow[..., 0]), np.median(flow[..., 1])
        residual_dx, residual_dy = flow[..., 0] - global_dx, flow[..., 1] - global_dy
        mag = np.sqrt(residual_dx ** 2 + residual_dy ** 2) * fps  # px/frame -> px/second
        tile_mag = cv2.resize(mag, (N_COLS, N_ROWS), interpolation=cv2.INTER_AREA)

        t_s = frame_idx / fps
        w = int(t_s // WINDOW_S)
        if w < n_windows_total:
            window_sums[w] += tile_mag.flatten()
            window_counts[w] += 1

        prev_gray = gray
        frame_idx += 1

    cap.release()
    window_counts[window_counts == 0] = 1  # avoid div-by-zero for any empty window
    motion_vecs = window_sums / window_counts[:, None]
    return motion_vecs, info


def sanity_check_frame(video: str, motion_vecs: np.ndarray, info: dict):
    """Pick the highest-motion window, grab its frame, overlay the 120-tile heatmap."""
    window_totals = motion_vecs.sum(axis=1)
    best_w = int(np.argmax(window_totals))
    t_s = (best_w + 0.5) * WINDOW_S

    frame_path = FIGURE_DIR / f".motion_sanity_{video}_frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t_s:.3f}", "-i", str(STIMULI_DIR / f"{video}.mp4"),
         "-frames:v", "1", str(frame_path)],
        capture_output=True, check=True,
    )
    img = plt.imread(frame_path)

    centers, _ = compute_patch_grid(180, N_COLS, 1920, 1080)
    scale_x, scale_y = info["width"] / 1920, info["height"] / 1080

    tile_vals = motion_vecs[best_w].reshape(N_ROWS, N_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"{video}, window {best_w} [{best_w*WINDOW_S:.1f}-{(best_w+1)*WINDOW_S:.1f}s]\n"
                       f"(highest total motion window)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(img)
    sc = axes[1].scatter(centers["center_x"] * scale_x, centers["center_y"] * scale_y,
                          c=motion_vecs[best_w], s=300, cmap="inferno", alpha=0.75,
                          edgecolor="white", linewidth=0.5)
    axes[1].set_title("Per-tile motion magnitude overlay")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="mean flow magnitude (px/s)")

    fig.tight_layout()
    out_path = FIGURE_DIR / f"motion_sanity_check_{video}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    frame_path.unlink()
    print(f"Saved sanity check: {out_path}")
    print(f"Highest-motion window: {best_w} [{best_w*WINDOW_S:.1f}-{(best_w+1)*WINDOW_S:.1f}s], "
          f"total={window_totals[best_w]:.2f}, mean_window_total={window_totals.mean():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    print(f"Computing optical flow for {args.clip}...")
    motion_vecs, info = compute_motion_windows(args.clip)
    print(f"{args.clip}: {motion_vecs.shape[0]} windows, {motion_vecs.shape[1]} tiles, "
          f"fps={info['fps']:.3f}, n_frames={info['n_frames']}")

    n_windows = motion_vecs.shape[0]
    tile_cols = [f"tile_{i:03d}" for i in range(N_ROWS * N_COLS)]
    df = pd.DataFrame(motion_vecs, columns=tile_cols)
    df.insert(0, "window_end_s", (np.arange(n_windows) + 1) * WINDOW_S)
    df.insert(0, "window_start_s", np.arange(n_windows) * WINDOW_S)
    df.insert(0, "window_idx", np.arange(n_windows))
    df.insert(0, "video", args.clip)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.clip}_motion_tiles.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    if args.sanity_check:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        sanity_check_frame(args.clip, motion_vecs, info)


if __name__ == "__main__":
    main()

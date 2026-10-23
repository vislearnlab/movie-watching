"""
Per-tile low-level visual saliency (cv2.saliency.StaticSaliencySpectralResidual), on the same
120-tile grid, box-pooled over each DeepMeaning 240px patch footprint (tile_pooling.py). See METHODS_DECISIONS.md.

Usage:
    python scripts/analysis/calc_saliency_tiles.py --clip pixar_birds --sanity-check
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))
from plot_patch_grid import compute_patch_grid  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
from tile_pooling import tile_box_means  # noqa: E402

STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "saliency"
FIGURE_DIR = PROJECT_ROOT / "figures" / "checks"

WINDOW_S = 0.5
N_COLS, N_ROWS = 15, 8


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


ALGOS = {
    "spectral_residual": cv2.saliency.StaticSaliencySpectralResidual_create,
    "fine_grained": cv2.saliency.StaticSaliencyFineGrained_create,
}


def compute_saliency_windows(video: str, algo: str) -> tuple[np.ndarray, dict]:
    video_path = STIMULI_DIR / f"{video}.mp4"
    info = get_video_info(video_path)
    fps = info["fps"]

    sal = ALGOS[algo]()
    cap = cv2.VideoCapture(str(video_path))

    n_windows_total = int(info["duration"] // WINDOW_S)
    window_sums = np.zeros((n_windows_total, N_ROWS * N_COLS))
    window_counts = np.zeros(n_windows_total)
    frame_t, frame_vecs = [], []  # per-frame output, for exact segment averaging

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, sal_map = sal.computeSaliency(frame)
        tile_sal = tile_box_means(sal_map)

        t_s = frame_idx / fps
        frame_t.append(t_s)
        frame_vecs.append(tile_sal)
        w = int(t_s // WINDOW_S)
        if w < n_windows_total:
            window_sums[w] += tile_sal.flatten()
            window_counts[w] += 1
        frame_idx += 1

    cap.release()
    window_counts[window_counts == 0] = 1
    return window_sums / window_counts[:, None], info, np.array(frame_t), np.array(frame_vecs)


def sanity_check_frame(video: str, sal_vecs: np.ndarray, info: dict):
    window_totals = sal_vecs.sum(axis=1)
    best_w = int(np.argmax(window_totals))
    t_s = (best_w + 0.5) * WINDOW_S

    frame_path = FIGURE_DIR / f".saliency_sanity_{video}_frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t_s:.3f}", "-i", str(STIMULI_DIR / f"{video}.mp4"),
         "-frames:v", "1", str(frame_path)],
        capture_output=True, check=True,
    )
    img = plt.imread(frame_path)

    centers, _ = compute_patch_grid(240, N_COLS, 1920, 1080)
    scale_x, scale_y = info["width"] / 1920, info["height"] / 1080

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"{video}, window {best_w} [{best_w*WINDOW_S:.1f}-{(best_w+1)*WINDOW_S:.1f}s]")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(img)
    sc = axes[1].scatter(centers["center_x"] * scale_x, centers["center_y"] * scale_y,
                          c=sal_vecs[best_w], s=300, cmap="inferno", alpha=0.75,
                          edgecolor="white", linewidth=0.5)
    axes[1].set_title("Per-tile saliency overlay")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="saliency")

    fig.tight_layout()
    out_path = FIGURE_DIR / f"saliency_sanity_check_{video}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    frame_path.unlink()
    print(f"Saved sanity check: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--algo", choices=list(ALGOS), default="fine_grained")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    print(f"Computing saliency ({args.algo}) for {args.clip}...")
    sal_vecs, info, frame_t, frame_vecs = compute_saliency_windows(args.clip, args.algo)
    print(f"{args.clip}: {sal_vecs.shape[0]} windows, {sal_vecs.shape[1]} tiles")

    n_windows = sal_vecs.shape[0]
    tile_cols = [f"tile_{i:03d}" for i in range(N_ROWS * N_COLS)]
    df = pd.DataFrame(sal_vecs, columns=tile_cols)
    df.insert(0, "window_end_s", (np.arange(n_windows) + 1) * WINDOW_S)
    df.insert(0, "window_start_s", np.arange(n_windows) * WINDOW_S)
    df.insert(0, "window_idx", np.arange(n_windows))
    df.insert(0, "video", args.clip)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    algo_tag = "" if args.algo == "fine_grained" else f"_{args.algo}"
    out_path = args.out_dir / f"{args.clip}_saliency_tiles{algo_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    fdf = pd.DataFrame(frame_vecs, columns=tile_cols)
    fdf.insert(0, "frame_time_s", frame_t)
    fdf.insert(0, "video", args.clip)
    frame_path = args.out_dir / f"{args.clip}_saliency_tiles{algo_tag}_frames.csv"
    fdf.to_csv(frame_path, index=False)
    print(f"Saved {frame_path}")

    if args.sanity_check:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        sanity_check_frame(args.clip, sal_vecs, info)


if __name__ == "__main__":
    main()

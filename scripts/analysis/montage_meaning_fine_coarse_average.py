"""
Visual check: fine, coarse, and their combined average, sampled at the fine grid's own 120
tile positions -- 4 rows per frame (raw frame, fine, coarse, average). Each scale is
normalized to its own max BEFORE averaging (see chat discussion 2026-10-22: raw scales
differ with patch size, confirmed empirically when 128px->240px widened the raw value
range), so the average reflects an actual 50/50 contribution rather than being dominated by
whichever scale happens to have more raw spread.

The "average at fine positions" value for a fine tile is (fine_value + coarse_value)/2,
where coarse_value is the mean of whichever coarse patches' footprints contain that fine
tile's position (point-in-box containment against native-pixel-space patch footprints, not a
full raster reconstruction -- see chat discussion 2026-10-22 on why this is mathematically
equivalent to building the dense overlap-averaged map and sampling it, for this box-average
case).

Usage:
    python scripts/analysis/montage_meaning_fine_coarse_average.py --clip frank_complex \
        --fine-csv data/local_cache/deepmeaning_frank_complex_fine240/frank_complex.csv \
        --coarse-csv data/local_cache/deepmeaning_frank_complex_coarse479/frank_complex.csv \
        --fine-patch-size 240 --coarse-patch-size 479 --n 8
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
OUT_DIR = PROJECT_ROOT / "figures" / "checks"
CANVAS_TO_NATIVE_SCALE = 1.5  # all 12 clips are 1280x720 native -> 1920x1080 canvas


def dot_size_for_spacing(ax, spacing_px: float, fig) -> float:
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((spacing_px, 0))
    diameter_pts = abs(p1[0] - p0[0]) / fig.dpi * 72
    return diameter_pts ** 2


def extract_frame(video_path: Path, t_s: float, out_path: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ss", f"{t_s:.3f}", "-frames:v", "1", str(out_path)],
        capture_output=True, check=True,
    )


def tile_spacing(sub: pd.DataFrame) -> float:
    return np.median(np.diff(np.sort(sub["patch_center_x"].unique())))


def coarse_value_at_points(fine_xy: np.ndarray, coarse_sub: pd.DataFrame, coarse_half: float) -> np.ndarray:
    """Mean of whichever coarse patches' footprints contain each fine point (point-in-box
    containment in native-pixel space), i.e. the coarse map's value at each fine position."""
    cx = coarse_sub["patch_center_x"].to_numpy()
    cy = coarse_sub["patch_center_y"].to_numpy()
    cv = coarse_sub["predicted_meaning_norm"].to_numpy()
    out = np.full(len(fine_xy), np.nan)
    for i, (px, py) in enumerate(fine_xy):
        mask = (np.abs(cx - px) <= coarse_half) & (np.abs(cy - py) <= coarse_half)
        if mask.any():
            out[i] = cv[mask].mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--fine-csv", required=True)
    parser.add_argument("--coarse-csv", required=True)
    parser.add_argument("--fine-patch-size", type=float, required=True)
    parser.add_argument("--coarse-patch-size", type=float, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    fine = pd.read_csv(args.fine_csv)
    coarse = pd.read_csv(args.coarse_csv)
    video_path = STIMULI_DIR / f"{args.clip}.mp4"

    # Normalize each scale to its own max BEFORE combining (see header).
    fine["predicted_meaning_norm"] = fine["predicted_meaning"] / fine["predicted_meaning"].max()
    coarse["predicted_meaning_norm"] = coarse["predicted_meaning"] / coarse["predicted_meaning"].max()

    coarse_half = args.coarse_patch_size / (2 * CANVAS_TO_NATIVE_SCALE)

    frame_idxs = sorted(set(fine["frame_index"].unique()) & set(coarse["frame_index"].unique()))
    assert frame_idxs, "No frame_index values in common -- were fine/coarse run at the same sample_fps?"
    chosen = [frame_idxs[i] for i in np.linspace(0, len(frame_idxs) - 1, args.n).round().astype(int)]
    chosen = sorted(set(chosen))

    vmin, vmax = 0.0, 1.0  # both already normalized to [0, max=1]

    fig, axes = plt.subplots(4, len(chosen), figsize=(3 * len(chosen), 12.0))
    fig.subplots_adjust(right=0.96)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for col, fidx in enumerate(chosen):
            f_sub = fine[fine["frame_index"] == fidx].sort_values("patch_index")
            c_sub = coarse[coarse["frame_index"] == fidx].sort_values("patch_index")
            t_s = f_sub["frame_time_sec"].iloc[0]
            frame_path = tmp / f"frame_{fidx:04d}.png"
            extract_frame(video_path, t_s, frame_path)
            img = plt.imread(frame_path)

            ax_frame, ax_fine, ax_coarse, ax_avg = axes[0, col], axes[1, col], axes[2, col], axes[3, col]
            ax_frame.imshow(img)
            ax_frame.set_title(f"f{fidx} [{t_s:.1f}s]", fontsize=9)
            ax_frame.axis("off")

            for ax, sub, cmap in [(ax_fine, f_sub, "inferno"), (ax_coarse, c_sub, "inferno")]:
                ax.imshow(img)
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
                s = dot_size_for_spacing(ax, tile_spacing(sub), fig)
                sc = ax.scatter(sub["patch_center_x"], sub["patch_center_y"], c=sub["predicted_meaning_norm"],
                                 s=s, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.65, edgecolor="white", linewidth=0.4)
                ax.axis("off")

            fine_xy = f_sub[["patch_center_x", "patch_center_y"]].to_numpy()
            coarse_at_fine = coarse_value_at_points(fine_xy, c_sub, coarse_half)
            avg = (f_sub["predicted_meaning_norm"].to_numpy() + coarse_at_fine) / 2

            ax_avg.imshow(img)
            ax_avg.set_xlim(0, img.shape[1])
            ax_avg.set_ylim(img.shape[0], 0)
            s = dot_size_for_spacing(ax_avg, tile_spacing(f_sub), fig)
            sc_avg = ax_avg.scatter(f_sub["patch_center_x"], f_sub["patch_center_y"], c=avg,
                                     s=s, cmap="inferno", vmin=vmin, vmax=vmax, alpha=0.65,
                                     edgecolor="white", linewidth=0.4)
            ax_avg.axis("off")

    axes[0, 0].set_ylabel("frame")
    for i, label in enumerate(["fine (240px, 120 tiles)", "coarse (479px, 28 tiles)", "average @ fine positions"], start=1):
        axes[i, 0].axis("on"); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(label, fontsize=8)

    fig.suptitle(f"{args.clip}: fine vs coarse vs average, both normalized to max=1 before combining")
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.6])
    fig.colorbar(sc_avg, cax=cbar_ax, label="predicted_meaning (normalized)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"meaning_fine_coarse_average_{args.clip}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
Montage of raw frame (top) + per-tile predicted_meaning dot overlay (bottom), for N frames
evenly spaced across a clip -- same visual convention as the saliency algorithm sanity check
(figures/checks/saliency_sanity_check_pixar_birds_multi_finegrained.png) and the per-tile dot
style in render_motion_overlay_video.py.

Usage:
    python scripts/analysis/montage_meaning_tiles.py --clip pixar_birds \
        --csv data/local_cache/deepmeaning_pixar_review/pixar_birds.csv --n 10
"""

import argparse
import subprocess
import sys
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


def dot_size_for_spacing(ax, spacing_px: float, fig) -> float:
    """Marker `s` (points^2) so a dot's diameter matches `spacing_px` data pixels on this
    axes -- independent of how many columns squeeze the frame into the figure."""
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((spacing_px, 0))
    diameter_pts = abs(p1[0] - p0[0]) / fig.dpi * 72
    return diameter_pts ** 2


def extract_frame(video_path: Path, t_s: float, out_path: Path):
    # Output-seek (-ss after -i) rather than input-seek: slower, but input-seeking to exactly
    # t=0 returns a black/garbage frame on this video (a known ffmpeg keyframe-seek quirk).
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ss", f"{t_s:.3f}", "-frames:v", "1", str(out_path)],
        capture_output=True, check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--tag", default=None, help="Suffix for the output filename (e.g. patch size), to avoid overwriting other runs for the same clip")
    parser.add_argument("--normalize", action="store_true",
                         help="Divide predicted_meaning by its own max (over the full CSV, not just the shown frames) so the color scale always tops out at 1 -- makes different patch-size runs visually comparable.")
    parser.add_argument("--dot-scale", type=float, default=1.0,
                         help="Shrink dot diameter relative to tile pitch (e.g. 0.6). At 1.0, dots touch edge-to-edge, which makes clamped-edge-tile position shifts visually obvious; <1.0 leaves gaps everywhere so that's less noticeable. Cosmetic only -- doesn't change the underlying data.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    video_path = STIMULI_DIR / f"{args.clip}.mp4"

    value_col = "predicted_meaning"
    if args.normalize:
        df = df.copy()
        df[value_col] = df["predicted_meaning"] / df["predicted_meaning"].max()

    frame_idxs = sorted(df["frame_index"].unique())
    chosen = [frame_idxs[i] for i in np.linspace(0, len(frame_idxs) - 1, args.n).round().astype(int)]
    chosen = sorted(set(chosen))

    vmin, vmax = df[value_col].min(), df[value_col].max()

    fig, axes = plt.subplots(2, len(chosen), figsize=(3 * len(chosen), 6.4))
    fig.subplots_adjust(right=0.96)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for col, fidx in enumerate(chosen):
            sub = df[df["frame_index"] == fidx]
            t_s = sub["frame_time_sec"].iloc[0]
            category = sub["category_used"].iloc[0]
            frame_path = tmp / f"frame_{fidx:04d}.png"
            extract_frame(video_path, t_s, frame_path)
            img = plt.imread(frame_path)

            ax_top, ax_bot = axes[0, col], axes[1, col]
            ax_top.imshow(img)
            ax_top.set_title(f"f{fidx} [{t_s:.1f}s]\n{category}", fontsize=9)
            ax_top.axis("off")

            ax_bot.imshow(img)
            ax_bot.set_xlim(0, img.shape[1])
            ax_bot.set_ylim(img.shape[0], 0)

            # median, not min: patch_center_x/y is the center of the actual (possibly
            # clamped-to-canvas-edge) crop, not the nominal tile center -- clamped edge tiles
            # shrink their gap to a neighbor, so the smallest gap understates true spacing
            # once patch_size is large enough to need clamping (confirmed 2026-10-22: 128px
            # unaffected, 240px/300px compressed to 62.7px/42.7px vs the true 82.7px).
            tile_spacing_px = np.median(np.diff(np.sort(sub["patch_center_x"].unique())))
            s = dot_size_for_spacing(ax_bot, tile_spacing_px, fig) * args.dot_scale ** 2
            sc = ax_bot.scatter(
                sub["patch_center_x"], sub["patch_center_y"], c=sub[value_col],
                s=s, cmap="inferno", vmin=vmin, vmax=vmax, alpha=0.65,
                edgecolor="white", linewidth=0.4,
            )
            ax_bot.axis("off")

    norm_tag = " normalized to max=1" if args.normalize else ""
    fig.suptitle(f"{args.clip}: predicted_meaning by tile ({len(chosen)} frames, shared color scale{norm_tag})")
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.6])
    fig.colorbar(sc, cax=cbar_ax, label="predicted_meaning" + (" (normalized)" if args.normalize else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"meaning_sanity_check_{args.clip}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

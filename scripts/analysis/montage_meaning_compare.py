"""
Direct visual comparison of two DeepMeaning runs for the same clip/frames (e.g. fp32 vs.
fp16) -- 4 rows per frame: raw frame, run A's tile overlay, run B's tile overlay, and their
per-tile difference (diverging colormap centered at 0). A/B share one color scale so they're
directly comparable; the diff row uses its own symmetric scale.

Usage:
    python scripts/analysis/montage_meaning_compare.py --clip frank_complex \
        --csv-a data/local_cache/deepmeaning_frank_review_patch240/frank_complex.csv --label-a fp32 \
        --csv-b data/local_cache/deepmeaning_frank_review_patch240_fp16/frank_complex.csv --label-b fp16 \
        --n 10
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
    # median, not min -- see montage_meaning_tiles.py's note on clamped-edge-tile duplicates.
    return np.median(np.diff(np.sort(sub["patch_center_x"].unique())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--csv-a", required=True)
    parser.add_argument("--csv-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    video_path = STIMULI_DIR / f"{args.clip}.mp4"

    # Frames common to both runs (same sample_fps/clip means identical frame_index values).
    frame_idxs = sorted(set(df_a["frame_index"].unique()) & set(df_b["frame_index"].unique()))
    assert frame_idxs, "No frame_index values in common between the two CSVs -- were they run at the same sample_fps?"
    chosen = [frame_idxs[i] for i in np.linspace(0, len(frame_idxs) - 1, args.n).round().astype(int)]
    chosen = sorted(set(chosen))

    vmin = min(df_a["predicted_meaning"].min(), df_b["predicted_meaning"].min())
    vmax = max(df_a["predicted_meaning"].max(), df_b["predicted_meaning"].max())

    diffs_all = []
    for fidx in chosen:
        a = df_a[df_a["frame_index"] == fidx].sort_values("patch_index")
        b = df_b[df_b["frame_index"] == fidx].sort_values("patch_index")
        diffs_all.append(b["predicted_meaning"].to_numpy() - a["predicted_meaning"].to_numpy())
    diff_absmax = max(1e-9, np.abs(np.concatenate(diffs_all)).max())

    fig, axes = plt.subplots(4, len(chosen), figsize=(3 * len(chosen), 12.0))
    fig.subplots_adjust(right=0.96)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for col, fidx in enumerate(chosen):
            a = df_a[df_a["frame_index"] == fidx].sort_values("patch_index")
            b = df_b[df_b["frame_index"] == fidx].sort_values("patch_index")
            t_s = a["frame_time_sec"].iloc[0]
            frame_path = tmp / f"frame_{fidx:04d}.png"
            extract_frame(video_path, t_s, frame_path)
            img = plt.imread(frame_path)

            ax_frame, ax_a, ax_b, ax_diff = axes[0, col], axes[1, col], axes[2, col], axes[3, col]
            ax_frame.imshow(img)
            ax_frame.set_title(f"f{fidx} [{t_s:.1f}s]", fontsize=9)
            ax_frame.axis("off")

            for ax, sub, label, cmap, vlo, vhi in [
                (ax_a, a, args.label_a, "inferno", vmin, vmax),
                (ax_b, b, args.label_b, "inferno", vmin, vmax),
            ]:
                ax.imshow(img)
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
                s = dot_size_for_spacing(ax, tile_spacing(a), fig)
                sc = ax.scatter(sub["patch_center_x"], sub["patch_center_y"], c=sub["predicted_meaning"],
                                 s=s, cmap=cmap, vmin=vlo, vmax=vhi, alpha=0.65, edgecolor="white", linewidth=0.4)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(label)

            diff = b["predicted_meaning"].to_numpy() - a["predicted_meaning"].to_numpy()
            ax_diff.imshow(img)
            ax_diff.set_xlim(0, img.shape[1])
            ax_diff.set_ylim(img.shape[0], 0)
            s = dot_size_for_spacing(ax_diff, tile_spacing(a), fig)
            sc_diff = ax_diff.scatter(b["patch_center_x"], b["patch_center_y"], c=diff,
                                       s=s, cmap="RdBu_r", vmin=-diff_absmax, vmax=diff_absmax,
                                       alpha=0.65, edgecolor="white", linewidth=0.4)
            ax_diff.axis("off")

    axes[1, 0].axis("on"); axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    axes[2, 0].axis("on"); axes[2, 0].set_xticks([]); axes[2, 0].set_yticks([])
    axes[0, 0].set_ylabel("frame")
    axes[3, 0].axis("on"); axes[3, 0].set_xticks([]); axes[3, 0].set_yticks([])
    axes[3, 0].set_ylabel(f"{args.label_b} - {args.label_a}")

    fig.suptitle(f"{args.clip}: {args.label_a} vs {args.label_b} ({len(chosen)} frames)")
    cbar_ax1 = fig.add_axes([0.97, 0.52, 0.01, 0.35])
    fig.colorbar(sc, cax=cbar_ax1, label="predicted_meaning")
    cbar_ax2 = fig.add_axes([0.97, 0.08, 0.01, 0.35])
    fig.colorbar(sc_diff, cax=cbar_ax2, label=f"{args.label_b} - {args.label_a}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"meaning_compare_{args.label_a}_vs_{args.label_b}_{args.clip}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

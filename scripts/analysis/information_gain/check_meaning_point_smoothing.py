"""
Check figure: meaning plotted at patch centers, Gaussian-smoothed (fine sigma=46, coarse sigma=92,
normalized weighted average), sampled at the 120 tile centers. Rows: frame, fine, coarse, average.

Usage: python scripts/analysis/information_gain/check_meaning_point_smoothing.py
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from load_features import load_meaning_raw

SIGMA_FINE, SIGMA_COARSE = 46.0, 92.0
EXAMPLES = [("pixar_birds", 1.33), ("pixar_birds", 60.0), ("frank_complex", 20.0), ("sesameus_1", 30.0)]
STIMULI_DIR = C.PROJECT_ROOT / "stimuli" / "main_blocks"
STEP = 4  # dense-map render resolution, display px


def smooth_at(points_xy: np.ndarray, vals: np.ndarray, query_xy: np.ndarray, sigma: float) -> np.ndarray:
    # Normalized Gaussian-weighted average of point values, evaluated at query points.
    d2 = ((query_xy[:, None, :] - points_xy[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma ** 2))
    return (w @ vals) / w.sum(1)


def frame_rgb(clip: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(STIMULI_DIR / f"{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1920, 1080))


def main():
    gx, gy = np.meshgrid(np.arange(0, 1920, STEP) + STEP / 2, np.arange(0, 1080, STEP) + STEP / 2)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    fig, axes = plt.subplots(4, len(EXAMPLES), figsize=(4.6 * len(EXAMPLES), 11))
    for col, (clip, t) in enumerate(EXAMPLES):
        fine_df, coarse_df = load_meaning_raw(clip)
        fi = fine_df.loc[(fine_df.frame_time_sec - t).abs().idxmin(), "frame_index"]
        f = fine_df[fine_df.frame_index == fi].sort_values("patch_index")
        c = coarse_df[coarse_df.frame_index == fi].sort_values("patch_index")
        to_disp = 1 / C.CANVAS_TO_SOURCE
        f_xy = f[["patch_center_x", "patch_center_y"]].to_numpy() * to_disp
        c_xy = c[["patch_center_x", "patch_center_y"]].to_numpy() * to_disp
        # Each scale rescaled to [0, 1] over the whole clip, as a stand-in for the corpus-wide rescale.
        fv = (f.predicted_meaning - fine_df.predicted_meaning.min()) / np.ptp(fine_df.predicted_meaning)
        cv = (c.predicted_meaning - coarse_df.predicted_meaning.min()) / np.ptp(coarse_df.predicted_meaning)

        dense_f = smooth_at(f_xy, fv.to_numpy(), grid, SIGMA_FINE).reshape(gx.shape)
        dense_c = smooth_at(c_xy, cv.to_numpy(), grid, SIGMA_COARSE).reshape(gx.shape)
        samp_f = smooth_at(f_xy, fv.to_numpy(), f_xy, SIGMA_FINE)
        samp_c = smooth_at(c_xy, cv.to_numpy(), f_xy, SIGMA_COARSE)
        rows = [("fine, sigma=46", dense_f, samp_f), ("coarse, sigma=92", dense_c, samp_c),
                ("average", (dense_f + dense_c) / 2, (samp_f + samp_c) / 2)]

        img = frame_rgb(clip, f.frame_time_sec.iloc[0])
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{clip}, t={f.frame_time_sec.iloc[0]:.2f}s", fontsize=10)
        for r, (label, dense, samp) in enumerate(rows, start=1):
            ax = axes[r, col]
            ax.imshow(img, alpha=0.45)
            ax.imshow(dense, cmap="inferno", vmin=0, vmax=1, alpha=0.7, extent=(0, 1920, 1080, 0))
            sc = ax.scatter(f_xy[:, 0], f_xy[:, 1], c=samp, cmap="inferno", vmin=0, vmax=1,
                            s=22, edgecolor="white", linewidth=0.5)
            if label.startswith("coarse"):
                ax.scatter(c_xy[:, 0], c_xy[:, 1], marker="+", color="cyan", s=40, linewidth=1)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
        for ax in axes[:, col]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 1920); ax.set_ylim(1080, 0)

    fig.colorbar(sc, ax=axes[1:, :], fraction=0.015, pad=0.01, label="meaning (rescaled 0-1 within clip)")
    fig.suptitle("Meaning plotted at patch centers, smoothed, sampled at 120 tile centers (dots); "
                 "cyan + = coarse patch centers", fontsize=12)
    out = C.CHECK_FIG_DIR / "meaning_point_smoothing_fine46_coarse92.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

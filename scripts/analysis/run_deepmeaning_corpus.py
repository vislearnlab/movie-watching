"""
Run the rebuilt DeepMeaning pipeline (120-tile grid, 128px patches, manual indoor/outdoor
labels) across the full 12-clip corpus. One row per (frame, patch) CSV per clip, matching
DeepMeaning's own output schema (patch_center_x/y already in native clip pixel coords).

Must run with the DeepMeaning repo's own venv (torch/open_clip/opencv installed there, not
this project's conda env):
    research-projects/DeepMeaning/.venv/bin/python scripts/analysis/run_deepmeaning_corpus.py
    research-projects/DeepMeaning/.venv/bin/python scripts/analysis/run_deepmeaning_corpus.py \
        --clips frank_complex,frank_objects,frank_play
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPMEANING_ROOT = PROJECT_ROOT.parent / "DeepMeaning"
# The DeepMeaning venv's editable install doesn't self-register via site.py's automatic .pth
# processing on this Python 3.14 build (confirmed: works if imported manually, not at startup).
sys.path.insert(0, str(DEEPMEANING_ROOT))

from deepmeaning.coca_model import CocaEmbedder  # noqa: E402
from deepmeaning.pipeline import process_video  # noqa: E402

STIMULI_DIR = PROJECT_ROOT / "stimuli" / "main_blocks"
LABELS_PATH = PROJECT_ROOT / "annotations" / "indoor-outdoor" / "all_clips_indoor_outdoor.csv"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "meaning"

SAMPLE_FPS = 6.0  # 2026-10-22: moved from 1.0fps (so each 0.5s window gets ~3 samples)

CLIPS = [
    "frank_complex", "frank_objects", "frank_play", "pixar_birds",
    "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2",
    "slow_animals", "slow_hands", "slow_inanimate", "slow_people",
]


def compute_patch_centers(diameter_px: float, n_cols: int, display_width: int, display_height: int):
    """Same edge-touching grid formula as plot_patch_grid.compute_patch_grid, inlined to avoid
    that module's matplotlib/PIL dependency (not installed in the DeepMeaning venv)."""
    radius = diameter_px / 2
    usable_w = display_width - diameter_px
    usable_h = display_height - diameter_px
    spacing_x = usable_w / (n_cols - 1)
    n_rows = max(2, round(usable_h / spacing_x) + 1)
    spacing_y = usable_h / (n_rows - 1)
    return [
        (radius + col * spacing_x, radius + row * spacing_y)
        for row in range(n_rows) for col in range(n_cols)
    ]


def build_label_lookup(labels_df: pd.DataFrame, video: str):
    segs = labels_df[labels_df.video == video].sort_values("seg_idx")
    bounds = list(zip(segs.seg_start_s, segs.seg_end_s, segs.label))

    def lookup(t: float) -> str:
        for s, e, label in bounds:
            if t < e:
                return label
        return bounds[-1][2]

    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", default=None, help="Comma-separated subset of CLIPS; default is all 12.")
    args = parser.parse_args()
    clips = args.clips.split(",") if args.clips else CLIPS

    patch_centers = compute_patch_centers(240, 15, 1920, 1080)  # 2026-10-22: moved from 180px to 240px (50% overlap)
    labels_df = pd.read_csv(LABELS_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embedder = CocaEmbedder(precision="fp16")  # 2026-10-22: ~16% faster than fp32, validated numerically equivalent (0.08% mean relative diff)
    for video in clips:
        lookup = build_label_lookup(labels_df, video)
        csv_path = process_video(
            video_path=str(STIMULI_DIR / f"{video}.mp4"),
            sample_fps=SAMPLE_FPS,
            output_dir=str(OUT_DIR),
            patch_centers=patch_centers,
            patch_size=240,  # 2026-10-22: user's explicit call -- staying with 240px despite 128px being the only size the regression weights were actually trained on; see chat discussion on why that's a real-but-not-disqualifying distribution-shift concern, not a hard validation constraint
            manual_label_lookup=lookup,
            save_visuals=False,
            embedder=embedder,
        )
        print(f"{video} -> {csv_path}")


if __name__ == "__main__":
    main()

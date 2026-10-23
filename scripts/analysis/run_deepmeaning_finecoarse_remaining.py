"""
Run fine (240px/15x8/120 tiles) and coarse (479px/7x4/28 tiles) DeepMeaning passes for all
clips except frank_complex (already done), overnight -- see METHODS_DECISIONS.md 2026-10-22.
One shared fp16 CocaEmbedder reused across every clip and both scales.

Usage:
    research-projects/DeepMeaning/.venv/bin/python3 scripts/analysis/run_deepmeaning_finecoarse_remaining.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts/analysis")
from run_deepmeaning_corpus import build_label_lookup, compute_patch_centers, STIMULI_DIR, LABELS_PATH
from deepmeaning.coca_model import CocaEmbedder
from deepmeaning.pipeline import process_video

CLIPS = [
    "frank_objects", "frank_play", "pixar_birds",
    "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2",
    "slow_animals", "slow_hands", "slow_inanimate", "slow_people",
]

FINE_PATCH_SIZE = 240
FINE_N_COLS = 15
COARSE_PATCH_SIZE = 479
COARSE_N_COLS = 7
SAMPLE_FPS = 6.0


def main():
    labels_df = pd.read_csv(LABELS_PATH)
    embedder = CocaEmbedder(precision="fp16")
    fine_centers = compute_patch_centers(FINE_PATCH_SIZE, FINE_N_COLS, 1920, 1080)
    coarse_centers = compute_patch_centers(COARSE_PATCH_SIZE, COARSE_N_COLS, 1920, 1080)

    for video in CLIPS:
        lookup = build_label_lookup(labels_df, video)

        for label, centers, patch_size, out_dir in [
            ("fine", fine_centers, FINE_PATCH_SIZE, f"data/local_cache/deepmeaning_{video}_fine240"),
            ("coarse", coarse_centers, COARSE_PATCH_SIZE, f"data/local_cache/deepmeaning_{video}_coarse479"),
        ]:
            t0 = time.time()
            csv_path = process_video(
                video_path=str(STIMULI_DIR / f"{video}.mp4"),
                sample_fps=SAMPLE_FPS,
                output_dir=out_dir,
                patch_centers=centers,
                patch_size=patch_size,
                manual_label_lookup=lookup,
                save_visuals=False,
                embedder=embedder,
            )
            print(f"{video} [{label}] -> {csv_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

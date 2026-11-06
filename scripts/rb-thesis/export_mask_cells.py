# Converts SAM3 face masks (masks.npz, not readable in R) to CSVs of face-covered 8px cells for the RB thesis R scripts.
# Input: data/local_cache/face_masks/<clip>/{frames.csv, masks.npz}
# Output: data/results/rb-thesis/mask_cells/<clip>.csv.gz (frame, face_type, cell_x, cell_y), one row per face-covered cell
# Usage: .venv-stimulus-prep/bin/python scripts/rb-thesis/export_mask_cells.py

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_DIR = PROJECT_ROOT / "data" / "local_cache" / "face_masks"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "rb-thesis" / "mask_cells"
CLIPS = ["sesameus_1", "sesameus_2", "sesameindia_1", "sesameindia_2"]
CELL = 8  # video px per cell (1280x720 -> 160x90 cells)
# same prompt -> type mapping as face_tools/make_face_tracks.R
TYPES = {
    "human": ["child face", "children faces", "child faces", "adult face", "adult faces", "child and adult faces",
              "person face", "people faces"],
    "muppet": ["muppet face", "muppet Face", "sheep muppet face"],
    "animal": ["chicken faces", "sheep face", "goat face", "turkey face", "cow face", "llama face"],
    "unlabelled": ["faces", "face (unlabelled)"],
}
PROMPT_TYPE = {p: t for t, ps in TYPES.items() for p in ps}

OUT_DIR.mkdir(parents=True, exist_ok=True)
for clip in CLIPS:
    prompts = pd.read_csv(MASK_DIR / clip / "frames.csv").drop_duplicates("prompt_idx").set_index("prompt_idx").prompt
    npz = np.load(MASK_DIR / clip / "masks.npz")
    rows = []
    for key in npz.files:
        name, frame = key.rsplit("|", 1)
        face_type = PROMPT_TYPE.get(prompts[int(name[1:].split("_", 1)[0])])
        if face_type is None:
            continue
        # a cell counts as face if its centre pixel is in the mask
        cy, cx = np.nonzero(npz[key][CELL // 2::CELL, CELL // 2::CELL])
        rows.append(pd.DataFrame({"frame": int(frame), "face_type": face_type, "cell_x": cx, "cell_y": cy}))
    cells = pd.concat(rows).drop_duplicates()
    cells.to_csv(OUT_DIR / f"{clip}.csv.gz", index=False)
    print(clip, len(cells), cells.face_type.value_counts().to_dict())

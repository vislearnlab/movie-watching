"""
Shared paths and per-video settings for the annotation pipeline.

Everything is resolved relative to this file, so scripts work from any cwd.
"""

from pathlib import Path

ANN_DIR     = Path(__file__).resolve().parent.parent          # annotations/
REPO_ROOT   = ANN_DIR.parent                                  # movie-watching/
STIMULI_DIR = REPO_ROOT / "stimuli" / "main_blocks"
SAM3_REPO   = ANN_DIR / "sam3"
ENV_FILE    = ANN_DIR / ".env"                                # holds HF_TOKEN
INPUTS_DIR  = ANN_DIR / "inputs"
EXPORTS_DIR = INPUTS_DIR / "datavyu_exports"                  # round1/, round2/ Datavyu CSVs
ROUND0_CSV  = INPUTS_DIR / "spreadsheets" / "stimuli_prompts_round0.csv"   # legacy spreadsheet prompts
CORRECTIONS_DIR = INPUTS_DIR / "corrections"                  # {video}.csv: fixes found while reviewing outputs
CUTS_CSV    = ANN_DIR / "scene-cuts" / "all_clips_cuts.csv"
CUT_OVERRIDES_CSV = ANN_DIR / "scene-cuts" / "cut_overrides.csv"     # cuts fixed by eye
CUT_FRAMES_CSV = ANN_DIR / "scene-cuts" / "all_clips_cuts_frames.csv"  # output of refine_cuts.py
OUTPUT_DIR  = ANN_DIR / "output" / "current"                  # the one live output; old runs are in output/archive/
PROMPTS_CSV = OUTPUT_DIR / "prompts.csv"                      # output of prepare_prompts.py
QC_DIR      = OUTPUT_DIR / "qc"
TMP_DIR     = Path("/Scratch") / Path.home().name / "annotation_tmp"   # session JPEGs on Tversky's local NVMe
LOG_DIR     = OUTPUT_DIR / "logs"

# All stimuli are 1280×720 and decode to their full header frame count; outputs keep this size.
VIDEO_SIZE = (1280, 720)

# Datavyu screen-space → video-space mapping.
#   fullscreen: video filled the 1920×1080 screen while annotating.
#   letterbox : video was shown inside LETTERBOX on a 1920×1080 screen
#               (Remora Mac Mini, round 1 only).
SCREEN_W, SCREEN_H = 1920, 1080
LETTERBOX = dict(left=46, top=53, right=1873, bottom=1079)

# stem → Datavyu export name and the coordinate system used *per round*.
# auto: categories that get an "(unlabelled)" prompt on every shot (default face and hand);
# reviewers found no hands in the animal/object videos and no faces in frank_objects.
# A video with no Datavyu export falls back to the round-0 spreadsheet.
VIDEOS = {
    "sesameus_1":     dict(block="Sesame US 1",    datavyu="Sesame_US_1.csv",    coords={"round1": "fullscreen"}),
    "sesameus_2":     dict(block="Sesame US 2",    datavyu="Sesame_US_2.csv",    coords={"round1": "fullscreen"}),
    "sesameindia_1":  dict(block="Sesame India 1", datavyu="Sesame_India_1.csv", coords={"round1": "fullscreen"}),
    "sesameindia_2":  dict(block="Sesame India 2", datavyu="Sesame_India_2.csv", coords={"round1": "letterbox"}),
    "pixar_birds":    dict(block="Pixar Birds",    datavyu="Pixar_Birds.csv",    coords={"round1": "letterbox"}, auto=("face",)),
    "frank_complex":  dict(block="Frank Complex",  datavyu="Frank_Complex.csv",  coords={"round1": "fullscreen", "round2": "fullscreen"}),
    "frank_play":     dict(block="Frank Play",     datavyu="Frank_Play.csv",     coords={"round1": "fullscreen"}),
    "slow_hands":     dict(block="Slow Hands",     datavyu="Slow_Hands.csv",     coords={"round1": "fullscreen", "round2": "fullscreen"}),
    "slow_people":    dict(block="Slow People",    datavyu="Slow_People.csv",    coords={"round1": "letterbox",  "round2": "fullscreen"}),
    "frank_objects":  dict(block="Frank Objects",  datavyu=None, coords={}, auto=()),
    "slow_animals":   dict(block="Slow Animals",   datavyu=None, coords={}, auto=("face",)),
    "slow_inanimate": dict(block="Slow Inanimate", datavyu=None, coords={}, auto=("face",)),
}

# Clicks whose prompt_idx has no prompt row in the export, reassigned after
# checking the clicked frame by eye: (stem, click prompt_idx) → prompt_idx.
CLICK_REMAP = {
    ("sesameus_2", "69a"): "69",       # same window as #69 "muppet face"
    ("sesameindia_1", "269"): "267A",  # both clicks on the seated adults' faces
}

# Datavyu rounds, newest first. The newest round that has a video's export wins.
ROUNDS = ["round2", "round1"]


def video_path(stem: str) -> Path:
    return STIMULI_DIR / f"{stem}_stripped.mp4"

"""Settings for the information-gain analysis (Kümmerer, Wallis & Bethge 2015). See METHODS_DECISIONS.md."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GROUP = os.environ.get("IG_GROUP", "adults")  # IG_GROUP=kids runs the whole pipeline kids-only (kid floor/ceiling/curves)
SEGMENT_SETS = ["cuts", "slow_chopped"]  # run both; slow_chopped splits each slow segment into 5
N_SUBSEGMENTS_SLOW = 5
CHOPPED_CLIPS = {"slow_animals", "slow_hands", "slow_inanimate", "slow_people"}

CLIPS = ["frank_complex", "frank_objects", "frank_play", "pixar_birds",
         "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2",
         "slow_animals", "slow_hands", "slow_inanimate", "slow_people"]
BLOCKS = {c: c.split("_")[0].replace("sesameindia", "sesame").replace("sesameus", "sesame") for c in CLIPS}
DURATIONS = {  # ffprobe, same as calc_entropy_by_segment.py
    "frank_complex": 60.32, "frank_objects": 60.12, "frank_play": 60.92, "pixar_birds": 150.15,
    "sesameindia_1": 60.00, "sesameindia_2": 60.84, "sesameus_1": 60.5605, "sesameus_2": 60.827433,
    "slow_animals": 60.727333, "slow_hands": 60.72, "slow_inanimate": 60.727333, "slow_people": 60.7607,
}

CV_UNIT = "clip"  # leave-one-clip-out for floor, epsilon, lambda, feature parameters
FEATURES = ["saliency", "motion", "meaning"]
FEATURE_SCALING = "corpus"  # joint [0, 1] rescale across all segments (Kümmerer 2015); "segment_z" = alternative
NONLINEARITY = "knots"  # 20 equidistant knots, monotone piecewise-linear (Kümmerer 2015); "exp" = comparison
N_KNOTS = 20
SHUFFLE_ACROSS_CLIPS = True  # shuffled-feature control pairs each segment with one from a different clip
MIN_OTHERS_CEILING = 3
EXCLUDE_NEGATIVE_CEILING = True  # drop participant x clip trials with mean ceiling < 0 (exclusions.py, unfiltered data)
FLOOR_BETA = 1.0
LOG_BASE = 2

# Gaze vectors (as-is: same closed-form density as concordance / within-between)
SIGMA_PX = 46.0
MAX_FIXATION_MS = 2000.0
MIN_FIXATIONS = 3  # per participant x segment; fewer -> excluded (counts reported)
FIXATION_PARAM_CODE = "f329476c"

# Inputs
FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / GROUP
KIDS_FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / "kids"
KIDS_EXCLUDE = ["MWK005"]  # valid_data=False in participant_summary; MWK011-021 have no summary row yet, all included
PARTICIPANT_SUMMARY_PATH = PROJECT_ROOT / "data" / "local_cache" / "participant_summary.csv"
CUTS_PATH = PROJECT_ROOT / "annotations" / "scene-cuts" / "all_clips_cuts.csv"
FEATURE_TILE_DIR = PROJECT_ROOT / "data" / "results"  # <feature>/<clip>_<feature>_tiles_frames.csv
MEANING_DIR = PROJECT_ROOT / "data" / "local_cache"
MEANING_FINE_PATCH, MEANING_COARSE_PATCH = 240, 479
SIGMA_FEATURE_FINE = 46.0  # features plotted at patch centers, smoothed, sampled at tile centers (same sigma as gaze)
SIGMA_FEATURE_COARSE = 92.0  # coarse meaning: kernel scaled to ~2x spacing
CANVAS_TO_SOURCE = 1280 / 1920

# Outputs
_SUB = "" if GROUP == "adults" else GROUP  # non-adult groups write to their own subfolders
CACHE_DIR = PROJECT_ROOT / "data" / "local_cache" / "information_gain" / _SUB
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "information_gain" / _SUB
FIG_ROOT = PROJECT_ROOT / "figures" / "information_gain"
FIG_DIR = FIG_ROOT / ("adults" if GROUP == "adults" else f"{GROUP}-{GROUP[:-1]}ref")  # adults/, kids-kidref/
KIDS_ADULTREF_FIG_DIR = FIG_ROOT / "kids-adultref"
CHECK_FIG_DIR = PROJECT_ROOT / "figures" / "checks" / (f"information_gain_{_SUB}" if _SUB else "")
RNG_SEED = 0

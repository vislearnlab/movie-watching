# Archived: feature–gaze correlation

**Superseded by the information-gain analysis (`scripts/analysis/information_gain/`), which now addresses this question.**

These scripts asked which content features (motion, saliency, meaning, center) explain where people look, by correlating
the *group* gaze tile-vector with each feature tile-vector per segment (Pearson r, partial r, and unique variance /
R² partitioning). Information gain answers the same question per viewer, in bits, against an empirical floor and a
between-viewer ceiling (Kümmerer, Wallis & Bethge 2015), with a unique/shared partition across features.

- `calc_motion_gaze_correlation.py`, `calc_saliency_gaze_correlation.py` — per-segment r, averaged by quadrant
- `calc_motion_saliency_partial_correlation.py`, `calc_feature_gaze_partial_correlation.py` — partial r
- `calc_feature_gaze_variance_partition.R` — unique variance per feature (results in `data/results/variance_partition/`)

Kept for the record; not maintained. They read the motion/saliency `*_tiles.csv` files, which were re-pooled onto
the DeepMeaning patch footprints on 2026-10-23 (old versions in `legacy_resize_grid/`), so rerunning will not
reproduce earlier numbers exactly. See METHODS_DECISIONS.md (2026-10-21 to 2026-10-23).

# Archived: fixation density concordance

Individual-vs-group fixation density similarity (per participant, per sliding window: Pearson r between that
viewer's 120-tile density vector and the leave-one-out group's), plus its pairwise/dyadic variant, diagnostics,
and the comparisons against windowed ISC. Never called "ISC" -- see `calc_concordance_group.py`'s header.

- `calc_concordance_group.py` / `.R`, `calc_concordance_pairwise.py` / `.R` -- the metric (results in `data/results/concordance/`)
- `diagnostic_concordance.py` -- the visual sanity check it was validated against
- `visualize_concordance_video.py` -- high/low-concordance viewer overlay video
- `plot_concordance_isc_timecourses.qmd`, `plot_individual_differences.qmd` -- concordance vs ISC timecourses and individual differences
- `calc_center_bias.R` -- cut-locked center-bias diagnostic for those timecourses

Kept for the record; not maintained. The shared gaze-loading code that used to live in `calc_concordance_group.py`
(`load_video_fixations`, `build_participant_arrays`, `closed_form_tile_values`, `DISPLAY_W/H`) is now
`scripts/analysis/gaze_tiles.py`; the shared video helpers from `visualize_concordance_video.py` are now
`scripts/preprocessing/viz/video_overlay.py`. These archived scripts import from there. `isc_analysis.qmd` (kept)
still writes `isc_fixation_timecourse.csv` into `data/results/concordance/` for the archived comparison notebook.

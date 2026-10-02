# Methods decisions

Running log of methods decisions and assumptions for this project: parameter choices,
deviations from a spec, and the reasoning behind them -- not scoped to any one
script or sub-project. Originally started for the caption-validation stimulus-prep
pipeline (now `scripts/preprocessing/stimulus_prep/`); also covers methods decisions
for other analyses (e.g. `scripts/analysis/`) going forward. Newest at bottom.

## 2026-09-25 — Screen geometry for visual-angle calculations

- **Assumption:** screen width = 53.14 cm, height = 29.89 cm, viewing distance = 70 cm.
- **Basis:** user confirmed a 24" monitor (16:9, matching `DISPSIZE = (1920, 1080)` in
  `experiment/moviewatching.py`) and 70 cm viewing distance. Width/height are computed from
  24" diagonal assuming a 16:9 panel (`width_in = 24 * 16/√337`, `height_in = 24 * 9/√337`),
  not read from a manufacturer spec sheet — panels marketed as "24-inch" vary slightly
  (e.g. active area 52.7–53.3 cm wide across common 24" 16:9 models).
- **Not used:** the exact per-session value logged live by the Tobii (`display_area`) in
  `experiment/moviewatching.py:383-398) is only written to run logs, not to any CSV, and
  may vary slightly by session/machine. We're using one fixed nominal value for all
  degrees-of-visual-angle calculations rather than per-session values.
- **Open question for user:** if you have the exact monitor model (e.g. ASUS ProArt PA24xx)
  or a session log with the logged `screen_width_cm`/`viewing_distance_cm`, let me know and
  I'll correct this constant.

## 2026-09-25 — Video files used for Task 1 inventory

- **Decision:** using `stimuli/main_blocks/{video_name}_stripped.mp4` (not the plain
  `{video_name}.mp4`) as the ground truth for "as displayed" video properties.
- **Basis:** `experiment/moviewatching.py:511-521` loads `stripped_video_path` at
  `size=[1920, 1080]` during the actual experiment — the plain `.mp4` is the pre-strip
  source, not what participants saw.

## 2026-09-25 — Section 1 results: no letterbox/pillarbox gate triggered

- Full run (`outputs/stimulus_inventory.csv`, all 12 clips, 5-sample cropdetect) found no
  letterboxing or pillarboxing in any clip — all report `crop_x=crop_y=0` and
  `crop_w/h == native_w/h`. The Task 1 "ask me" gate does not apply; proceeding to later
  tasks with the assumption that the full native frame is the displayed content area
  (subject to uniform stretch to 1920x1080, not aspect-preserved).
- Native resolution for all 12 clips is 1280x720 (below the 1920x1080 display size, so
  PsychoPy upscales rather than crops/pads).
- **QUICK_TEST caveat confirmed by testing:** with only 1 cropdetect sample, `frank_objects`
  falsely reported as pillarboxed (a bright near-white scene at the ~10%-duration sample
  point was misread). The full 5-sample run does not reproduce this. Do not trust
  letterbox/pillarbox flags from a `QUICK_TEST=True` run.
- **Observation, not yet a decision:** exact fps varies across clips —
  `25/1` (frank_*, sesameindia_*), `24000/1001` = 23.976 (pixar_birds, slow_animals,
  slow_inanimate), `30000/1001` = 29.97 (sesameus_*, slow_people). Relevant to Task 4's
  off-by-one timing red flag; frame-timestamp math must use each clip's own exact fps,
  not a single assumed value.
- **Observation, not yet a decision:** the Section 1 check figure
  (`outputs/checks/section1_inventory_thumbnails.fullrun_12clips.png`) shows 3 of the 4
  `slow` block clips (`slow_hands`, `slow_inanimate`, `slow_people`) as nearly all-black at
  their exact midpoint frame. Could be genuine dim content, or these clips may pause/fade to
  black partway through. Flagging for the PI's review rather than assuming either way —
  worth a look before Task 3 picks representative frames from these clips.

## 2026-09-25 — Task 2 (scene cut detection) skipped; Task 3 rule changed

- **Decision (explicit user override of the TODO spec):** skipping Task 2 (scene cut
  detection) entirely for now. Task 3's original rule required excluding timepoints near
  cuts, which needs cut times — since we don't have them, the user replaced that exclusion
  criterion with a simpler one: skip an all-black frame and move to the nearest non-black
  time instead.
- **This explicitly drops the TODO's "Selection must not use frame content (rule only)"
  constraint for Task 3** — the user confirmed this tradeoff directly (asked, and picked
  "drop the content-blind constraint" over two content-blind alternatives). Selection is no
  longer purely timestamp-based; it now inspects mean frame luminance to reject black frames.
- **New rule implemented:** 24 frames total, 6 per block, timestamps evenly spaced across
  each block's *combined* duration (not per clip) at the midpoint of 6 equal segments. If a
  candidate frame's mean luminance is below `black_frame_mean_threshold`, search outward in
  `black_frame_search_step_sec` steps (up to `black_frame_search_max_offset_sec`) for the
  nearest non-black frame and log the move.
- **Open sub-decision, not yet confirmed:** "evenly spaced across each block" is read
  literally — a block's clips are concatenated into one timeline for spacing purposes.
  Since real experiment playback order is randomized per participant (`moviewatching.py`
  shuffles clips within a block), there's no canonical clip order to concatenate in; I used
  alphabetical `video_name` order as an arbitrary but fixed, reproducible choice. Flagging
  this in case a different canonical order matters to you.
- `cut_exclusion_after_sec` / `cut_exclusion_before_sec` remain in `CONFIG` but are unused
  by Section 3 now; left in place in case cut-based exclusion is reinstated later.
- **Implementation detail:** the black-frame search (`find_valid_local_time`) is clamped
  to stay within the same clip as the original target time -- it will not search across a
  clip boundary into the next clip in the block's concatenated timeline. If a clip has a
  black run longer than `frame_search_max_offset_sec` (5s default) centered near a
  target time close to the clip's edge, this raises rather than silently crossing into a
  different clip's content.

## 2026-09-25 — Added a content-jump check (the "slow" block has real cuts, not just black frames)

- **Evidence:** sampled luminance every 0.5s across all 4 `slow` clips end-to-end. Every one
  has a hard cut at t≈30.0s (an ordinary scene change, NOT black -- e.g. `slow_hands` goes
  ~130 -> ~120 luminance) and a genuinely near-black single frame at t≈45.5s (mean luminance
  0-6). Matches the TODO's own description of the `slow` block as "cuts ~every 15s" -- these
  are compilations, not single continuous shots. The original all-black check only catches
  the 45.5s-style cut, not the 30s-style one.
- **User's call:** add a lightweight content-jump check alongside the black-frame check,
  rather than reinstating full scene-cut detection (Task 2) or accepting the risk as-is.
- **Method:** at each candidate time, compare two small probe frames offset
  `cut_jump_probe_offset_sec` before/after it (mean absolute grayscale pixel difference). A
  candidate is only accepted if it's both non-black AND this jump score is below
  `cut_jump_mean_abs_diff_threshold`.
- **Threshold calibration:** measured jump scores (offset=0.4s) at the known cut points above
  vs. ordinary mid-shot points, across all 4 `slow` clips: cut points scored 49-116; ordinary
  points scored 1-18. Set `cut_jump_mean_abs_diff_threshold = 30.0` for a clean margin on both
  sides. Not validated against `frank`/`sesame`/`pixar` cut styles specifically, but the same
  check applies uniformly to all blocks.
- Renamed `black_frame_search_step_sec` -> `frame_search_step_sec` and
  `black_frame_search_max_offset_sec` -> `frame_search_max_offset_sec` since the search now
  covers both checks, not just blackness. `find_nonblack_local_time` renamed to
  `find_valid_local_time`.

## 2026-09-25 — Added a blown-out/white-frame check (mirrors the black-frame check)

- **Finding:** `frank_03` (t=45.973s local in `frank_objects`) was picked as a nearly pure
  white frame (mean luminance 251.6/255) -- a real, uncaptionable macro/close-up shot on a
  white background, not a bug. Sampling luminance across that clip shows the bright region
  spans roughly t=46-53s (mean luminance 230-251), with more normal content before (~40s,
  ~179) and after (~54-58s, ~113-156).
- **Decision:** added a symmetric `white_frame_mean_threshold` (225.0) check next to the
  existing `black_frame_mean_threshold` (10.0) in `find_valid_local_time` -- a candidate is
  now rejected if it's too dark OR too bright, same search-and-move mechanism as before.
- **Also widened `frame_search_max_offset_sec` from 5.0 to 8.0s**, because escaping this
  particular white region requires searching back to ~t=40s (~5.5s from the original
  target) -- the old 5.0s cap left almost no margin. Forward isn't an option here (the
  bright region extends ~8s forward before recovering).

## 2026-09-25 — Changed patch grid from 2 rings removed (60) to 1 ring removed (98) to 0 rings removed (144)

- **User's explicit override of a fixed TODO_week1.md design parameter:** the spec states
  "60 per frame" (12x5 grid, outer two rings of centers removed from the full 16x9 grid).
  User first asked to keep one of those two rings (14x7 = 98 centers), then asked to keep
  both, i.e. remove no rings at all.
- **Current grid: full 16x9 = 144 centers per frame** (x=60-1860, y=60-1020, 120px apart,
  covering the entire display area edge to edge). This is a real deviation from the spec's
  stated patch count (60) -- flagging in case it affects downstream assumptions (RA
  templates, list generation, etc. reference "60" in TODO_week1.md, though those are out of
  scope this week).
- `compute_patch_centers`'s `rings_removed` is now 0; the function still asserts an exact
  expected count so a future change is caught immediately rather than silently producing the
  wrong number of patches. With 0 rings removed, the outermost circles' edges will extend
  past the frame boundary (circle radius > ring-to-edge distance) -- expected, not a bug.

## 2026-09-25 — Bug: QUICK_TEST was clobbering full-run representative-frame PNGs

- **Found while preparing a patch-grid preview on `frank_00`:** Section 3 wrote its PNGs to
  a single `outputs/representative_frames/` directory regardless of `QUICK_TEST`, using
  `frame_id`-based filenames (`frank_00.png`, etc.) that collide between quick and full runs
  even though the CSV outputs were already correctly split
  (`representative_frames.csv` vs `representative_frames.quicktest.csv`). Running the
  notebook in `QUICK_TEST` mode after a full run had silently overwritten `frank_00.png` and
  `frank_01.png` with different (quick-test-target) frames, while `representative_frames.csv`
  still described the old (full-run) timestamps -- image and metadata went out of sync.
- **Fix:** `QUICK_TEST` now writes PNGs to `outputs/representative_frames_quicktest/`,
  mirroring the CSV split. Re-ran the full pass afterward to regenerate correct
  `frank_00.png`/`frank_01.png`.

## 2026-09-25 — Section 6 preview: tile-level fixation density (ring-exclusion decision)

- **Purpose:** ad hoc analysis (not the full Task 6 gaze check) to answer one specific
  question -- across all clips, how many adult fixations land in each tile of the full
  16x9 grid, to help decide whether to drop the outermost ring(s) of tiles (see the
  2026-09-25 "Changed patch grid..." entry above).
- **Data source and parameter code:** used the existing precomputed I2MC fixation CSVs in
  `data/preprocessed/fixations/adults`, not raw gaze -- at parameter code `f329476c`,
  the most recent full-adult-group run per `data/preprocessed/log-files/adults_fix_parse_log.csv`
  (`MAX_CALIBRATION_DEG=2.0`, matching the current script default). Not asked/confirmed with
  the user first since this run's parameters are already the current default; flagging in case
  a different code/run is preferred.
- **Participant filter:** kept only `participant_type == "adults"` with `valid_data == True`
  in `participant_summary.csv`, same convention as `preprocessing/viz/visualize_heatmap.py`.
  49 participants had fixation files at this code; 45 remained after the valid_data filter;
  63,081 fixations total.
- **Coordinate convention reused (not a new assumption):** `xpos`/`ypos` in the fixation CSVs
  are raw Tobii pixel coordinates (origin bottom-left), same as raw `gaze_x`/`gaze_y`. Flipped
  to image/top-left origin (`img_y = display_height - ypos`) before binning into tiles --
  this is the same flip already used by `preprocessing/viz/visualize_heatmap.py` and
  `visualize_fixation_dots.py` elsewhere in this repo, not a new choice made for this analysis.
- **239/63,081 fixations (0.4%) fell outside the 1920x1080 display** (calibration drift/edge
  noise) and were clipped to the nearest edge tile rather than dropped, so they still count
  toward the total; noted as a printed warning in the notebook rather than silently discarded.
- **Result:** dropping the outermost 1 ring costs 2.05% of all adult fixations (98 tiles
  kept); dropping 2 rings costs 10.79% (60 tiles kept, matching the TODO's original spec).
  See `outputs/tile_fixation_counts.csv` and `outputs/checks/section6_tile_fixation_density.png`.
- **Flagging for review, not resolved:** row 0 (top edge), columns 14-15 (top-right corner)
  has an anomalous local spike (78 and 54 fixations vs. ~4-35 in neighboring row-0 tiles) --
  confirmed genuine (in-bounds, spread across ~25 different participants, not a coordinate
  bug), and concentrated almost entirely in `sesameindia_1`/`sesameindia_2` (121 of 130
  fixations in that corner). Likely a persistent on-screen element (e.g. a channel logo) in
  those clips' top-right corner draws real fixations there -- doesn't change the ring-drop
  recommendation but is worth a look if those clips matter for hard-frame selection.
- **Local copy:** copied only the adult `f329476c` fixation CSVs (490 files) and
  `participant_summary.csv` from the server into `data/local_cache/` (gitignored via the
  existing `data/` rule) rather than pointing the notebook at the live server mount --
  see "Local copies of server data" below.

## 2026-09-25 — Section 6 preview, part 2: fixation density by circular patch

- **Follow-up to the grid-tile version above,** per request: same question, counted against
  the actual overlapping circular patches (144, radius 100px = `patch_diameter_px`/2, still
  an unconfirmed placeholder per Task 5) instead of plain non-overlapping grid tiles. A
  fixation can fall inside more than one patch, so per-patch counts sum to more than the
  total fixation count (137,126 vs. 63,081 fixations) -- expected, not a bug.
- **Sanity check:** 99.77% of fixations fall inside at least one of the current 144 patches
  (radius reaches essentially the whole display) -- the 0.23% outside all patches are
  presumably the most extreme off-display/calibration-drift points.
- **Result (overlap-aware, the more correct number for this decision):** dropping the
  outermost 1 ring of patches (98 kept) leaves 1.23% of all fixations outside every
  surviving patch; dropping 2 rings (60 kept) leaves 6.97% -- both lower than the plain-
  grid-tile estimates (2.05% / 10.79%) above, since a fixation dropped from a removed outer
  patch can still be caught by a surviving neighbor's overlap. See
  `outputs/patch_fixation_counts.csv` and `outputs/checks/section6b_patch_fixation_density.png`.
- Same top-right corner hotspot (row 0, cols 14-15) reappears here, now clearly spread
  across neighboring overlapping patches too -- reinforces that it's a real, spatially
  localized signal (likely on-screen content in `sesameindia_1`/`sesameindia_2`), not a
  binning artifact of the grid-tile version.

## 2026-09-25 — Our patches vs. Henderson & Hayes fine/coarse scales

Requested comparison from Task 5 ("write a short comparison of our patches vs. theirs
-- scales, diameter, spacing, overlap, size in degrees"). Source for H&H's numbers:
Peacock, Hayes & Henderson (2019), PMC6690792 -- "Participants sat 85 cm away from a 21"
monitor, so that scenes subtended approximately 26.5° x 20° of visual angle at 1024x768
pixels"; fine patches 87px diameter (300/image), coarse patches 205px diameter (108/image).

|                     | H&H fine | H&H coarse | Ours (current: 200px diameter, 120px spacing) |
|---------------------|----------|------------|-------------------------------------------------|
| px/deg of their/our screen | 38.6 | 38.6 | 46.2 (using the unconfirmed 53.14cm/70cm assumption above) |
| diameter (px)       | 87       | 205        | 200 |
| diameter (deg)      | 2.25°    | 5.31°      | **4.33°** |
| est. center spacing (px) | 51.2 (derived from n) | 85.3 (derived from n) | 120 (actual) |
| diameter:spacing ratio (overlap) | 1.70 | 2.40 | **1.67** |
| n patches/image     | 300      | 108        | 144 (0 rings removed) |

**Answer: closer to their COARSE scale in absolute size** (4.33° vs. 5.31° coarse, vs. 2.25°
fine -- almost 2x their fine diameter), **but our overlap ratio matches their FINE scale**
(1.67 vs. 1.70 fine, vs. 2.40 coarse) -- i.e. we're using coarse-sized patches packed at
fine-scale density (denser overlap than either of H&H's individual scales). This matches
the original TODO_week1.md note that 200px was chosen as "~1.7x the center spacing, similar
to H&H's fine-scale diameter-to-spacing ratio" -- that ratio choice was intentional, but
nobody had yet checked where the resulting absolute size (in degrees) landed relative to
H&H's two scales until now. Flagging since H&H used two separate scales specifically
because fine and coarse capture different information (local detail vs. broader regions);
a single scale that's coarse-sized-but-finely-packed doesn't map cleanly onto either of
their published scales, which may matter if this dataset is ever compared to meaning-map
literature.
**Caveat:** the "ours" column depends on the still-unconfirmed 53.14cm/29.89cm/70cm screen
geometry assumption above -- if the real geometry differs, degrees-of-visual-angle numbers
shift accordingly (px/deg is directly proportional to screen width and inversely
proportional to viewing distance).

## 2026-09-29 — FINAL: patch grid confirmed at 180px diameter, 15x8, edge-touching

**This is the final answer, superseding the 2026-09-28 192px choice below.**
`CONFIG["patch_diameter_px"] = 180` (was 192), `CONFIG["patch_n_cols"] = 15` (unchanged).

- **Grid:** 15 columns x 8 rows = **120 patches** (unchanged count from the 192px choice --
  `compute_patch_grid`'s row-count derivation happens to land on 8 rows at both diameters).
  Horizontal spacing 124.29px (overlap ratio 1.448); vertical spacing 128.57px (ratio 1.400).
  180px = 3.90deg of visual angle at the current (still-unconfirmed) screen-geometry
  assumption.
- **Arrived at through further live exploration** after the 192px choice: the user asked
  for a fixed 15x8 grid (rather than deriving row count) at a few candidate diameters (4deg,
  3.75deg), then asked whether a diameter existed that would make ratio_x exactly equal
  ratio_y -- solved algebraically: with n_cols=15, n_rows=8 fixed and edge-tangency on both
  axes, `spacing_x=(W-d)/14` and `spacing_y=(H-d)/7`; setting them equal and solving for d
  gives the unique solution **d=240px (5.196deg)**, where spacing_x=spacing_y=120px exactly
  (ratio=2.0 on both axes). That value was previewed too, then the user asked to see 180px
  specifically.
- **"Do 2x2 blocks of tiles meet at exactly one point?" check:** at 180px (radius 90px), the
  half-diagonal of the (124.29 x 128.57) cell is 89.41px -- since 90 > 89.41, four
  neighboring circles actually overlap very slightly at the center of each 2x2 block (a
  ~0.59px-radius lens-shaped overlap), not a single point. The exact diameter for a true
  single-point meeting was calculated as **178.98px** (solving `d = sqrt(spacing_x(d)^2 +
  spacing_y(d)^2)` self-consistently, since spacing itself depends on d under edge-tangency).
  **User's call: the ~0.6px discrepancy is negligible and not worth chasing** -- 180px is
  fine as specified, not 178.98px.
- **This value (180px) is now what `compute_patch_grid(CONFIG)` produces** -- no new grid
  function was needed; the existing edge-touching helper (adopted 2026-09-28) already derives
  exactly 8 rows at 15 columns for a 180px diameter, so only the CONFIG value changed.
- **Cleanup:** removed the exploratory cells/helpers used to arrive at this decision (the
  ratio=1.5-centered-grid variant and the fixed-15x8-multiple-diameters variant, including
  4deg, 3.75deg, 5.196deg, and 178.98/180px-adjacent previews) and their
  `outputs/checks/section5_patch_grid_preview_*` image folders. The canonical
  `outputs/checks/section5_patch_grid_preview/` (24 frames) was regenerated at 180px and is
  the only patch-grid preview folder remaining.
- **`compute_patch_centers` and Section 6 preview part 2 are unaffected** -- that analysis
  stays frozen to its own 200px/144-patch snapshot regardless of this change (see the
  2026-09-28 entry's note on that decoupling); still correct/reproducible, verified by
  re-running it after this change (identical 99.77% / 1.23% / 6.97% figures as before).
- **Not yet built:** the rest of Task 5 (per-frame patch PNGs, `patch_index.csv`, neighbor
  table, coverage-map/reassembly/gallery checks).

## 2026-09-28 — Adopted patch grid: edge-touching, 192px diameter, 15 columns (superseded)

**CONFIRMED** (Task 5's "confirm the diameter with me before generating final patches"
gate is now cleared): `CONFIG["patch_diameter_px"] = 192`, `CONFIG["patch_n_cols"] = 15`,
using a new convention -- the outermost patches' circle perimeter sits exactly on the
display edge rather than overshooting it (as the earlier 200px/144-patch grid did).

- **Convention change:** previous grids (see the 2026-09-25 "Changed patch grid..." entry)
  offset centers by half a cell from the edge, so outer circles typically extended past the
  display boundary. This grid instead insets the outermost centers by exactly `radius`, so
  the outer circles are tangent to the edge -- by design, this leaves a thin strip near the
  edges and full corners untiled (no circle reaches there, since round patches can't stay
  tangent to two edges at a corner). Implemented in the new `compute_patch_grid(config)`
  helper (Section 5 helpers cell), which replaces `compute_patch_centers` for all NEW work.
- **Resulting grid:** 15 columns (fixed) x 8 rows (chosen so vertical spacing stays close to
  horizontal spacing while also hitting the top/bottom edges exactly) = **120 patches**.
  Horizontal spacing 123.4px (overlap ratio 1.556); vertical spacing 126.9px (ratio 1.514) --
  the two axes differ slightly because 8 rows was the nearest integer count satisfying both
  "near-square cells" and "exact edge touch," not a free choice.
  192px = 4.16deg of visual angle at the current (still-unconfirmed) screen-geometry
  assumption -- still closer to H&H's coarse scale (5.31deg) than fine (2.25deg), though
  less extreme than the previous 200px/4.33deg choice.
- **Arrived at through a live exploration** (diameter in degrees vs. H&H fine/coarse,
  diameter x ratio patch-count table, several rendered previews at 3deg/1.667, 4deg/1.5,
  and this edge-touching option) -- all confirmed visually against all 24 representative
  frames before adopting this one. The intermediate options were previewed, not built into
  the pipeline, and have been removed (see "Cleanup" below).
- **`compute_patch_centers` (the old 200px/144-patch, rings-removed grid) is RETAINED**,
  unchanged, solely because Section 6 preview part 2's already-reported ring-exclusion
  numbers (1.23% / 6.97% fixation loss at 1/2 rings dropped) were computed against that
  specific grid; that cell now uses a locally frozen `SNAPSHOT_PATCH_DIAMETER_PX = 200`
  instead of reading the (now different) live `CONFIG["patch_diameter_px"]`, so its numbers
  stay correct and reproducible rather than silently drifting when the config changed. If an
  up-to-date ring/coverage number against the new 192px/120-patch grid is needed later, that
  section needs to be explicitly re-run against `compute_patch_grid`, not just re-executed.
- **Cleanup:** removed the exploratory Section 5 preview cells/outputs for the intermediate
  diameter/ratio options considered along the way (3deg @ ratio 1.667, 4deg @ ratio 1.5) and
  the original 200px/144-patch preview (superseded by this grid) -- both code and their
  `outputs/checks/section5_patch_grid_preview_*` image folders. The current
  `outputs/checks/section5_patch_grid_preview/` (24 frames) reflects only this adopted grid.
- **Not yet built:** the rest of Task 5 (per-frame patch PNGs, `patch_index.csv`, neighbor
  table, coverage-map/reassembly/gallery checks) -- still pending, now that the diameter is
  confirmed.

## 2026-10-01 — Notebook retired in favor of standalone scripts; Task 7 (RA templates) dropped

- **Decision (explicit user request):** an undergrad RA is no longer involved in this
  project. Task 7 of `TODO_week1.md` ("RA templates" -- onboarding materials for an RA to
  select 24 additional "hard" frames) is dropped from scope entirely, not just deferred.
  Mentions of a later RA hard-frame-selection step in the retired notebook's intro text no
  longer apply to this pipeline.
- **Decision (explicit user request):** `notebooks/stimulus_prep.ipynb` is retired and
  deleted. Every section that had working code was ported, logic-for-logic, into standalone
  CLI scripts under `scripts/preprocessing/stimulus_prep/`, each independently runnable
  (not notebook/Colab-dependent) with argparse flags replacing the notebook's `CONFIG` dict
  and `QUICK_TEST` toggle:
  - `stimulus_inventory.py` -- Task 1 (stimulus inventory, letterbox/pillarbox check).
  - `representative_frames.py` -- Task 3 (representative-frame sampling with the
    black/white/content-jump checks; reads `stimulus_inventory.py`'s output).
  - `patch_grid_preview.py` -- Task 5's grid-overlay preview at the FINAL confirmed
    180px/15x8 grid (2026-09-29 entry below); reads `representative_frames.py`'s output.
    The rest of Task 5 (per-frame patch PNGs, `patch_index.csv`, neighbor table, coverage
    checks) is still not built -- this script is only the grid preview, same as the
    notebook's Section 5 was.
  - `fixation_density_by_tile.py` / `fixation_density_by_patch.py` -- the two Section 6
    ad hoc ring-exclusion previews (still not the full Task 6 gaze check). Each takes
    `--fixation_dir`/`--participant_summary_path` as explicit CLI args rather than a
    hardcoded local-cache path, per the "Local copies of server data" rule below.
  - Shared ffmpeg/ffprobe plumbing (used by both the inventory and representative-frames
    scripts) lives in `_video_utils.py`, a non-CLI helper module in the same directory.
- **Verified equivalent, not just ported:** `patch_grid_preview.py`,
  `fixation_density_by_tile.py`, and `fixation_density_by_patch.py` were each run against
  the existing `outputs/representative_frames/` PNGs and the local fixation cache and
  reproduced the exact same numbers already on record below (120 patches at
  180px/3.90deg/ratio 1.448x1.400; 63,081 fixations, 2.05%/10.79% ring loss by tile;
  99.77%/137,126/1.23%/6.97% by patch). `stimulus_inventory.py` and
  `representative_frames.py` were only smoke-tested (`--help`, syntax) since the raw
  `stimuli/main_blocks/*_stripped.mp4` files aren't present in this local checkout --
  re-run those two for real the next time the videos are available locally, to confirm end
  to end before trusting a fresh `outputs/stimulus_inventory.csv` or
  `outputs/representative_frames.csv`.
- **Cleanup:** removed the now-redundant notebook-named duplicate check outputs
  (`outputs/checks/section5_patch_grid_preview/`, `section6_tile_fixation_density.png`,
  `section6b_patch_fixation_density.png`) after confirming the new scripts' outputs cover
  the same frames/numbers. Left `section1_inventory_thumbnails*.png` and
  `section3_*.png` in place (not regenerated this session, since the inventory/
  representative-frames scripts need the stimulus videos to re-run).

## 2026-10-01 — Un-froze fixation_density_by_patch.py to the current grid

- **Decision (explicit user request):** `fixation_density_by_patch.py` no longer uses the
  frozen 200px/144-patch "rings removed" snapshot described in the entry above. It now
  computes fixation density directly against the live, FINAL patch grid from
  `patch_grid_preview.compute_patch_grid` (180px diameter, 15x8 = 120 patches,
  edge-touching) -- reverses the explicit 2026-09-28 "keep this decoupled" decision, since
  the ring-exclusion question that justified freezing it is already resolved and the user
  now wants an up-to-date view on the grid that's actually in use.
- Dropped the "N rings dropped" coverage printout and the ring-boundary overlay rectangles
  from the check figure -- they answered the (now-resolved) ring-exclusion question and
  don't map cleanly onto the edge-touching grid's row/col indices as a meaningful decision
  point. The script now just reports overall coverage (99.59% of fixations fall inside at
  least one of the 120 patches) and the per-patch density figure.
- `outputs/patch_fixation_counts.csv` and `outputs/checks/fixation_density_by_patch.png`
  are now against the 180px/120-patch grid, not the old 200px/144-patch one -- re-running
  this script will no longer reproduce the 2026-09-25 snapshot numbers (99.77% / 137,126 /
  1.23% / 6.97%); that's expected.
- Same corner hotspot as before (top-right, now landing in the corner-most patches) is
  still visible at similar relative magnitude -- consistent with the earlier finding that
  it's a real, spatially localized signal (likely on-screen content in
  `sesameindia_1`/`sesameindia_2`), not a grid-dependent artifact.

## 2026-10-01 — Removed the 16x9 plain-grid-tile script

- **Decision (explicit user request):** deleted `fixation_density_by_tile.py` (the
  non-overlapping 16x9/120px-tile version of the fixation-density ring-exclusion preview)
  along with its stale outputs (`outputs/tile_fixation_counts.csv`,
  `outputs/checks/fixation_density_by_tile.png`). Only the circular, overlapping-patch
  version (`fixation_density_by_patch.py`, now on the current 15x8/120-patch grid per the
  entry above) remains -- the project's one confirmed spatial grid is 15 columns x 8 rows,
  so a script built around the old 16x9 tile count no longer has a reason to exist.
- `load_adult_fixations` (the only piece `fixation_density_by_patch.py` imported from the
  deleted script) was inlined directly into `fixation_density_by_patch.py` rather than
  kept in a shared module, since it now has exactly one caller.

## 2026-09-25 — Local copies of server data

- Per user's request, no notebook/script in this project points directly at the mounted
  server (`/Volumes/vislearnlab/...`). Files needed for development/testing are copied
  into a local scratch location first; only specific files needed for a given step are
  copied (not full participant/stimulus directories).

## 2026-10-01 — Smoothing scale for individual-to-group fixation-density comparison

- **Naming:** this new metric is called **fixation density concordance** (or just
  "concordance") in code/filenames -- explicitly NOT "ISC", which this user reserves for
  the existing Franchak-style x/y time-series analysis (`isc_gaze.py`, `isc_fixations.py`,
  `isc_analysis.qmd`). Don't reuse "ISC" for this metric anywhere (variable names, file
  names, docstrings, plots).
- **Context:** planning a new analysis (not yet built) comparing an individual
  participant's duration-weighted, Gaussian-smoothed fixation density to the rest of
  their group's (leave-one-out) density, both sampled at the 120 scene-tile centers
  from the caption-validation patch grid (180px/15x8, `patch_grid_preview.py`), compared
  via Pearson r. Adults only, `pixar_birds` only, to start.
- **Decision:** Gaussian smoothing sigma = **46px** (~1.0 deg of visual angle at the
  project's screen-geometry assumption, 46.186 px/deg). Chosen by rendering a single
  synthetic 500ms fixation at screen center blurred at several candidate sigmas
  (`outputs/checks/sigma_preview_single_fixation.png`) and the real adult group density
  for `pixar_birds` at sigma 50/75/100px (`outputs/checks/sigma_preview.png`), then
  picking 46px directly.
- **Basis offered (not the only one considered):** the pipeline's existing calibration-
  accuracy exclusion threshold (`MAX_CALIBRATION_DEG = 2.0`, `i2mc_fixations.py`) is an
  upper bound on included trials' validation error, not a measured typical error. Reading
  that 2 deg bound as an approximate 95% (2-sigma) bound on gaze-position error gives
  1 sigma ~= 1.0 deg ~= 46px -- the value picked.
- **Not yet decided:** whether this fixed sigma is the final choice once the real
  individual-vs-group comparison is built (e.g. it may need revisiting per-group if kids
  or infants are added later, since their calibration-accuracy distribution may differ
  from adults').

## 2026-10-01 — Sliding-window size/step for the time-varying version (correction)

- **Correction:** an earlier note here claimed there was "no existing precedent" for
  windowed gaze-position ISC in this repo and pointed at `run_all_clips.py`'s
  `compute_windowed_isc` (2.0s window / 1.0s step) as the closest thing available. That
  was wrong on two counts: (1) `run_all_clips.py`'s windowed ISC is for a different
  metric entirely (binary face-looking proportion, frame-indexed), and (2) there IS a
  direct precedent for windowed *fixation-position* ISC that was missed: `isc_analysis.qmd`
  Part 7 ("ISC Timecourse"), which recomputes windowed Pearson r straight from each
  participant's fixation CSV (same overlap-weighted `fixations_to_binned_series` logic as
  `isc_fixations.py`) at `BIN_MS=20`, `WINDOW_S=2`, `STEP_S=0.25`, `MIN_VALID=30` bins/window.
- **Decision:** the new individual-to-group metric uses **WINDOW_S = 1.0s, STEP_S = 0.25s**
  -- shorter window than the qmd's 2.0s precedent (explicit user choice), same 0.25s step.
- **No time-binning (`BIN_MS`) in this metric:** `isc_analysis.qmd`'s `BIN_MS=20` exists
  only to align two continuous x/y sample streams onto a shared grid so they can be
  correlated point-by-point. This metric doesn't correlate point-by-point time series at
  all -- it builds a duration-weighted density per window directly, so a fixation's
  contribution to a given window is weighted by its actual overlap duration with that
  window (continuous), not discretized into bins first.
- **Decision: no `MIN_VALID` equivalent.** Unlike the qmd's windowed ISC (which drops a
  window's r if a pair has fewer than 30 valid bins), this metric applies no minimum-data
  gate on a window -- every window's tile vector/correlation is computed and kept
  regardless of how little fixation data falls in it.

## 2026-10-01 — First diagnostic build: individual vs. leave-one-out group heatmap figure

- **Reference code reviewed:** the user shared a prior Code Ocean capsule (own-vs-other
  gaze-prediction analysis on static scene-viewing data, stacked ridge regression on
  image-feature spaces). Confirms "sample heat value at each scene-tile center into a
  per-participant vector" (their `heatByTile_<subject>.csv`, `tileCenterValueCol`) is an
  established convention in the user's prior work, consistent with this project's plan.
  Their "other" is a mean of pairwise individual-vs-individual correlations (plus partial
  correlation controlling for a group-level model), not one correlation against a single
  pooled group map. **User's explicit call: keep this project's pooled leave-one-out
  group approach (one group heatmap, not pairwise-averaged) rather than switching to
  match.** No heatmap-generation/smoothing code was in that capsule (heat files are a
  precomputed input there), so nothing in it conflicts with the sigma=46px choice above.
- **Built:** `scripts/analysis/fixation_density_concordance_diagnostic.py` -- first concrete build
  step, not the batch pipeline. For one sample participant (`MW001`, first valid pid
  sorted) and 5 consecutive windows (window_idx 0-4, i.e. t=[0,1), [0.25,1.25),
  [0.5,1.5), [0.75,1.75), [1.0,2.0)s), renders: the participant's density, the
  leave-one-out group's density, the 120 scene-tile centers overlaid as red dots, and the
  Pearson r per window. Output: `outputs/checks/fixation_density_concordance_diagnostic.png`.
- **Observed consequence of the "no minimum-data gate" decision:** window 0 (t=[0,1)s)
  has `r=nan` for MW001 -- they have zero fixations overlapping that window (density is
  all-zero, undefined correlation), while the leave-one-out group already has usable
  density there. This is expected given the no-minimum decision, not a bug; downstream
  consumers of the real pipeline's per-window r values need to handle NaN windows.
  r rises smoothly across windows 1-4 (-0.005, 0.191, 0.565, 0.667) as the sample
  participant's gaze converges toward the same region the group is looking at --
  behaves as expected for a sanity check.

## 2026-10-01 — No bound filtering, no 0-1 rescaling, no histogram matching for concordance

- **Reference reviewed:** the user shared the Robertson-Lab `vrGazeCore-Toolbox` repo
  (VR free-viewing heatmap pipeline). Its `scaleFixationDurations.m` caps each fixation's
  duration at the 95th/0.1th percentile (computed across all fixations going into one
  heatmap) before accumulating, then rescales to [0.1, 1] -- guards against one very long
  fixation dominating a heatmap pooled over a whole (unwindowed) viewing period.
- **Decision: no equivalent bound/percentile filtering for concordance.** This metric's
  windowing already caps every fixation's contribution at the window length (1.0s): a
  fixation's weight is its *overlap* with the window (`min(end,window_end) -
  max(start,window_start)`), not its raw duration, so nothing can contribute more than
  1000ms to any one window regardless of how long it actually lasted. Percentile-based
  capping would also be unstable here (most individual windows have 0-2 fixations --
  not enough points for a meaningful 95th percentile).
- **Decision: no 0-1 (min-max) rescaling of the tile-value vectors before computing r.**
  Pearson r is invariant to affine rescaling of either vector (`corr(aX+b, Y) ==
  corr(X,Y)` for any `a>0`), so rescaling would be a no-op on the actual metric. Per-panel
  [0,1] normalization is still used for the diagnostic figure's *display* only (already
  in `fixation_density_concordance_diagnostic.py`), which has no bearing on computed r.
- **Decision: no histogram matching (`imhistmatch`-style) either.** Unlike affine
  rescaling, histogram matching is a nonlinear, non-affine transform -- Pearson r is NOT
  invariant to it, so applying it would inject an uncontrolled change into the metric
  rather than remove one. It also doesn't address the actual concern that prompted the
  question (see next entry) -- it would reshape a sparse individual map to superficially
  resemble the group's smoother distribution without adding real information.
- **Open, explicitly deferred (not solved by any of the above):** participants with few
  fixations in a window produce a sparse, "spiky" tile-value vector, which gives a
  noisier/more volatile r estimate against the group's smoother pooled vector -- a
  sample-size/reliability issue, not a scale issue. This is exactly what `MIN_VALID`
  guards against in `isc_analysis.qmd`'s windowed ISC, which this project has already
  decided not to adopt (see "no `MIN_VALID` equivalent" above). **User's call: hold off
  on addressing this for now** rather than revisit the no-minimum decision or add
  per-window data-richness columns (`n_fixations`, total duration) to the output --
  revisit later if needed.

## 2026-10-01 — Switched to closed-form tile values; first full batch run

- **Verified equivalence before switching, per user request:** compared
  raster-then-blur-then-sample vs. closed-form Gaussian evaluated directly at each tile
  center, both on real data (window [10,11)s of `pixar_birds`: correlation 0.99994-0.99997
  between the two methods' 120-tile vectors, max diff <1% of peak value) and on 4
  simulated fixations designed to test cross-fixation amplification (two fixations placed
  close together) -- `outputs/checks/closed_form_vs_raster_blur_verification.png` shows
  the combined/amplified tile value from the two nearby fixations is identical under both
  methods (corr=0.999991), confirming the closed-form sum is not computed "tile by tile in
  isolation" -- it's the same sum over every fixation in the window as a full-image blur,
  evaluated analytically instead of via pixel raster (this follows directly from
  convolution being evaluable pointwise: blurred_density(x) = sum_i weight_i *
  Gaussian(x - fixation_i, sigma)).
- **Decision:** `fixation_density_concordance.py` now computes `closed_form_tile_values`
  directly (no raster image, no `gaussian_filter` call) and gets each participant's
  leave-one-out group vector via `total_vec - individual_vec` (exact, by linearity) rather
  than recomputing a closed-form sum over every other participant's fixations from
  scratch. `fixation_density_concordance_diagnostic.py` was updated to match: it still
  rasterizes+blurs for the DISPLAYED heatmap image (cosmetic only), but computes the
  printed/plotted r via the same `closed_form_tile_values` the batch script uses.
- **Performance:** a single-participant timing run (597 windows, old raster approach) took
  458.4s (767.9ms/window) -- extrapolating to all 41 participants would be ~5.2 hours.
  After the closed-form + total-minus-individual switch, the full 41-participant x
  597-window batch (24,477 rows) ran in **3.0s** (0.12ms per participant-window).
- **First full output:** `outputs/fixation_density_concordance.csv`, all 41 valid adults,
  `pixar_birds`, window=1.0s/step=0.25s. r: mean=0.590, median=0.630, range
  [-0.083, 0.998], NaN rate 1.0% (sparse windows, per the no-minimum-gate decision above).

## 2026-10-01 — Timecourse plots: concordance alone, and overlaid with windowed ISC

- **Built:** `scripts/analysis/plot_fixation_density_concordance.py` (mean concordance r
  per window, across participants, over time) styled to match `isc_analysis.qmd`'s
  windowed-ISC plots (`figures/isc_timecourse.png`) for future side-by-side comparability:
  same wide/short aspect, "Time (s)" x-axis with major ticks every 5s / minor every 1s and
  no vertical gridlines, categorical palette slot 1 (`#2a78d6`) for the line. Currently one
  line (adults only); additional comparison groups later would each get their own fixed
  palette slot + a legend. Output: `outputs/checks/fixation_density_concordance_timecourse.png`.
- **Ported the windowed ISC itself to Python**, since no numeric (CSV) export of
  `isc_analysis.qmd` Part 7's windowed ISC existed -- only in-notebook R figures. New
  script `scripts/analysis/isc_fixation_timecourse.py` reproduces that method exactly
  (same overlap-weighted `fixations_to_binned_series` binning as `isc_fixations.py`,
  `BIN_MS=20`, `WINDOW_S=2.0`, `STEP_S=0.25`, `MIN_VALID=30`), restricted to the same 41
  valid adults used for concordance (for a fair comparison -- not an ISC requirement) and
  `pixar_birds`. This genuinely is ISC (Franchak-style x/y pairwise correlation), so it
  keeps the "isc" name, unlike concordance. Output: `outputs/isc_fixation_timecourse.csv`
  (mean_r=0.260 overall, 0% NaN, 27.6s to compute all 618 windows x 820 pairs).
- **Built the overlay:** `scripts/analysis/plot_concordance_vs_isc.py` ->
  `outputs/checks/concordance_vs_isc_timecourse.png`. Concordance (slot 1 blue) runs
  consistently higher than ISC (slot 2 orange) but the two visually track together at
  several peaks (t~27s, 56s, 90s, 106s) -- expected, since ISC correlates raw, noisier x/y
  position over a 2s window while concordance correlates smoothed spatial density over a
  1s window (less sensitive to microsaccade-level noise).
- **Known grid-misalignment artifact, inherited from the R method, not introduced here:**
  `isc_analysis.qmd`'s `step_bins <- round(STEP_S * 1000 / BIN_MS)` evaluates to
  `round(12.5) = 12` bins, i.e. an ACTUAL step of 240ms, not the intended 250ms. Concordance
  steps in exact 250ms increments. The two grids only coincide every 6s, so an exact-match
  join of the two timecourses only found 25 overlapping points over the whole clip -- not
  enough for a meaningful numeric alignment score (one was printed, -0.054, but flagged to
  the user as not trustworthy). A real quantitative comparison would need interpolation
  onto a shared grid; not done yet, only the visual overlay.
- **Re-ran ISC at window_s=1.0 (matching concordance's window) instead of the qmd's 2.0s**,
  per explicit user request -- `outputs/isc_fixation_timecourse_1s.csv`. Same step-grid
  caveat applies (step is still 0.24s, not 0.25s). mean_r dropped from 0.260 (2.0s window)
  to 0.188 (1.0s window) -- expected, shorter windows are noisier/less averaged-out.

## 2026-10-01 — Pairwise concordance variant (matching ISC's pair structure)

- **Context:** the existing concordance metric compares one individual against a pooled
  leave-one-out group map. The real ISC scripts instead compare every PAIR of
  participants directly and average over pairs. User asked for a concordance variant
  using that same pairwise structure, now that the closed-form computation makes it cheap.
- **Built:** `scripts/analysis/fixation_density_concordance_pairwise.py` -- for every
  window, computes each participant's own 120-tile vector (no pooling), then correlates
  every pair directly. All `C(41,2)=820` pairwise correlations for a window are obtained
  from one `np.corrcoef` call on the stacked (n_participants x 120) matrix, rather than
  looping over pairs -- ran in **0.3s** for all 597 windows. Output schema matches
  `isc_fixation_timecourse.py`'s (`window_idx, window_start_s, window_end_s, mean_r,
  n_pairs`) for direct comparability, confirmed with the user before building (per-window
  aggregate only, not the full per-pair long table, for now).
- **Added alongside, not replacing,** the pooled-group version -- they answer different
  questions ("matches everyone combined" vs. "matches each other person").
- **Result:** mean_r = 0.376 (pairwise) vs. 0.590 (pooled-group) vs. 0.188 (pairwise ISC,
  1.0s window) -- pooled-group concordance > pairwise concordance > pairwise ISC in
  magnitude, consistently, but all three track the same peaks over time (t~27s, 56s, 90s,
  106s). See `scripts/analysis/plot_concordance_comparison.py` ->
  `outputs/checks/concordance_comparison_timecourse.png`.

## 2026-10-01 — Fixed the ISC/concordance step-grid misalignment (Option B)

- **Decision (explicit user choice, "Option B" from the two offered):** changed
  `isc_fixation_timecourse.py`'s `--bin_ms` default from 20 (the qmd's original) to
  **25**, so a 0.25s step divides evenly (10 bins, exactly 250ms) with no rounding --
  20ms made 0.25s/0.02s=12.5 bins, which rounded to 12 (an ACTUAL 0.24s step). The
  alternative (Option A: change concordance's step to 0.24s to match ISC's real
  behavior, leaving ISC's bin_ms untouched) was offered but not chosen -- user prefers
  keeping the "logical" 0.25s step and may revisit this later.
- **Permanent safeguard added, per explicit request:** the script now ALWAYS prints both
  the requested and the ACTUAL resulting window/step on every run (not just when they
  differ), plus an explicit WARNING line if they don't match -- so if `bin_ms` is ever
  changed back to 20 (or to any other value that doesn't evenly divide the requested
  step), the silent 240ms-vs-250ms drift is caught immediately instead of silently
  reintroduced. **User noted they may revert to `bin_ms=20` eventually** since they find
  0.25s step more logical than changing the bin width -- if/when that happens, expect
  the warning to fire and Option A (adjusting concordance's step instead) to be the
  fallback.
- **Re-ran with the fix:** `outputs/isc_fixation_timecourse_1s.csv` (597 rows, matching
  concordance's 597 windows exactly; mean_r=0.192, NaN rate 0.2%). The exact-match join
  with concordance now covers all 597 points (was 25) with a real correlation of
  **0.217** between the two timecourses -- consistent with the peak-alignment seen
  visually before the fix. Regenerated `concordance_vs_isc_timecourse.png` and
  `concordance_comparison_timecourse.png` with the corrected, exactly-aligned ISC data.

## 2026-10-02 — Cut-annotation GUI tool

- **Built:** `scripts/preprocessing/cut_annotation/cut_annotation_tool.html` -- a standalone local HTML/JS tool
  (no build step, no dependencies) for manually marking cut/scene-transition timestamps
  per clip, since those timepoints are relevant to overlay on the concordance/ISC
  timecourse plots. Lets you pick a clip, scrub/play, mark a cut at the current time
  (button or `C` key), frame-step with `,`/`.` using each clip's exact fps (from
  `outputs/stimulus_inventory.csv`), see marked cuts as tick marks on a timeline, and
  export to CSV (per-clip or all-clips) or a JSON session file (for resuming later).
  Not an Artifact -- it needs direct local filesystem access to the stimulus videos,
  which a hosted claude.ai page cannot have; this is a plain file the user runs locally.
- **Bug found and fixed while testing:** recommended serving the tool via `python3 -m
  http.server` for more reliable `localStorage` than `file://`. Testing (via a local
  Chrome instance, with the actual stripped mp4s copied in -- see below) showed this
  advice was wrong and actively broken: Python's built-in `http.server` does not support
  HTTP Range requests (confirmed with `curl -H "Range: ..."` -- it returns a full `200 OK`
  with the entire file regardless of the requested range), which stalls the video element
  (`readyState` stuck at 0) since Chrome's media pipeline expects `206 Partial Content`
  for range-based seeking. **Flipped the tool's own guidance**: `file://` is now the
  primary recommended way to open it (local file reads don't need HTTP Range negotiation
  at all), with `localStorage` reliability noted as the tradeoff (mitigated by the
  CSV/JSON export already built in) -- `python3 -m http.server` is now explicitly called
  out as broken for this use case; a Range-supporting server (e.g. `npx http-server`)
  would be needed to get both reliable auto-save AND working seek.
- **Stimulus videos copied locally for testing:** `stimuli/main_blocks/*_stripped.mp4`
  (12 files, 1.1GB) were copied from the now-mounted `/Volumes/vislearnlab/experiments/
  movie-watching/stimuli/main_blocks/` into the project's local `stimuli/main_blocks/`
  (previously absent from this checkout) -- same "copy only the specific files needed,
  don't point scripts at the live mount" rule as the rest of this project. Confirmed
  git-ignored (`stimuli/**/*.mp4`), won't be committed.

## 2026-10-02 — Extended both concordance metrics to all 12 clips

- **Decision:** ran `fixation_density_concordance.py` and
  `fixation_density_concordance_pairwise.py` for the remaining 11 clips (all of `frank_*`,
  `sesame*`, `slow_*` -- `pixar_birds` was already done), same parameters (sigma=46px,
  window=1.0s, step=0.25s), each video's own exact duration from
  `outputs/stimulus_inventory.csv`. Per-video runs were written to a temporary
  `outputs/concordance_by_video/` directory, concatenated with the existing `pixar_birds`
  rows into the canonical `outputs/fixation_density_concordance.csv` (121,703 rows, 12
  videos) and `outputs/fixation_density_concordance_pairwise.csv` (3,225 rows, 12 videos),
  then the temporary directory was deleted.
- **Windowed ISC also extended to all 12 clips** (same session, user confirmed): ran
  `isc_fixation_timecourse.py` (1.0s window, 0.25s step, bin_ms=25) for the same 11
  remaining clips, combined with the existing `pixar_birds` rows into
  `outputs/isc_fixation_timecourse_1s.csv` (3,227 rows, 12 videos). All three metrics
  (ISC, concordance-group, concordance-pairwise) plus manual cut annotations now cover
  all 12 clips -- the cut-locked 3-metric comparison (`plot_concordance_isc_cut_locked.py`)
  can now be run for any block, not just `pixar_birds`.
  Note: `isc_fixation_timecourse.py`'s mean_r is noticeably lower for most other clips
  (0.02-0.19) than `pixar_birds` (0.19-0.26 depending on window) -- not yet investigated,
  could be genuine (pixar_birds is the longest/most narratively continuous clip) or
  worth a sanity check if it looks surprising once plotted per block.

## 2026-10-02 — Refactor: concordance and ISC computation moved to R

- **Context:** user has an R-using collaborator and wants the computation side of this
  work ("as much as possible") in R, with a new R notebook for the timecourse/cut-locked
  plots we'd been building in Python. Three structural decisions confirmed with the user
  before starting: (1) extend `isc_analysis.qmd`'s existing windowed-ISC code to export a
  CSV, rather than writing a duplicate standalone ISC script; (2) two separate R scripts
  for the two concordance variants (mirroring the Python file structure 1:1); (3) a new,
  separate `.qmd` for plotting rather than appending to the already-large
  `isc_analysis.qmd`.
- **Built `scripts/analysis/fixation_density_concordance.R`** (group/leave-one-out) and
  **`fixation_density_concordance_pairwise.R`**. Faithful ports of the Python closed-form
  method (`outer()`-vectorized Gaussian evaluation at tile centers, `cor()`/`cor(vecs)` for
  the correlations). Verified against the existing Python-generated CSVs for `pixar_birds`:
  max abs diff ~1e-15 / ~3e-16 (floating-point noise only) on every row, identical NaN
  patterns, identical `n_pairs`. Each script loops over all 12 clips in one run (reading
  durations from `outputs/stimulus_inventory.csv`) and writes the same canonical CSV paths
  the Python versions used (`outputs/fixation_density_concordance.csv`,
  `outputs/fixation_density_concordance_pairwise.csv`), now overwritten with R-sourced
  data. ~10s and ~8.5s respectively for all 12 clips.
- **Extended `isc_analysis.qmd`** (new "Part 8b: Export Windowed ISC for Concordance
  Comparison") rather than writing a separate ISC script. `load_binned()` and
  `build_timecourse()` were given new optional parameters (`bin_ms`, `window_s`, `step_s`,
  `min_valid`, `valid_adult_pids`) that all default to the pre-existing globals
  (`BIN_MS=20, WINDOW_S=2, STEP_S=0.25, MIN_VALID=30`, no pid filtering) -- every EXISTING
  call site (`build_timecourse(.x)` in Part 8, `build_timecourse(example_video, ...)` in
  Part 7) is therefore byte-for-byte unaffected; only the new Part 8b export call passes
  different values. Output: `outputs/isc_fixation_timecourse.csv` (9,540 rows: 12 videos x
  3 comparisons x ~235-595 windows), superseding and deleting the earlier Python-written
  `isc_fixation_timecourse_1s.csv` and `isc_fixation_timecourse.csv` (the latter was
  pixar_birds-only at the qmd's original 2.0s window; the new file covers all 12 clips at
  the concordance-matched 1.0s window).
- **Two real bugs found and fixed during this port, both would have silently undermined
  the concordance-vs-ISC comparison if shipped as originally written:**
  1. **Bin-width mismatch:** the qmd's own `BIN_MS=20` makes a 250ms step round to an
     actual 240ms (`round(0.25*1000/20) = round(12.5) = 12` bins), the same misalignment
     bug fixed in the Python port on 2026-10-01 -- fixed here by using `EXPORT_BIN_MS=25`
     (10 bins = exactly 250ms) for the new export call only, leaving the qmd's own
     `BIN_MS=20` completely untouched for its existing Parts 7-9/14.
  2. **Participant population mismatch (found during verification, not anticipated):**
     `isc_raw`'s adult pid list for a video (from the pre-computed, Aug-2026
     `isc_fixation_pairwise.csv`) does NOT equal concordance's `valid_data==TRUE`
     population -- confirmed for `pixar_birds`: 44 vs. 41 adults, a 3/7-person mismatch in
     each direction, because `isc_fixation_pairwise.csv`'s own inclusion rule is just "has
     a fixation file for this code," with no `valid_data` QC step ever applied on the R
     side. **User's explicit call (asked directly, not assumed): filter the new ISC export
     to `valid_data==TRUE` adults, matching concordance exactly** -- implemented via the
     new `valid_adult_pids` parameter, intersected against the video's own adult list.
     Verified this produces exactly 41 adults for `pixar_birds`, matching Python/concordance.
- **Remaining, accepted (not fixed) discrepancy vs. the old Python ISC numbers:** after
  matching bin-width and population, the first 1-2 windows of a clip match the old Python
  output to ~4 decimal places, but later windows drift by ~0.02-0.08 (r scale). Confirmed
  NOT a stale-local-cache issue (checksummed identical fixation files on server vs. local
  cache). Root cause: `build_timecourse` sizes its shared bin matrix from the **median**
  of each participant's own last-fixation-bin length (a deliberate, pre-existing,
  documented robustness choice in the qmd -- guards against one participant's overlong
  file stretching every panel's x-axis), while the Python port used a **fixed** nominal
  duration from `outputs/stimulus_inventory.csv`. Two reasonable, pre-existing-vs-new
  conventions for the same underlying question ("how long is this clip"); kept the qmd's
  existing convention rather than overriding it, since it predates this refactor and is
  already reasoned about in its own code comment.
- **Not yet done:** the new plotting notebook itself (`concordance_isc_timecourses.qmd`),
  replicating the single-metric timecourse, concordance-vs-ISC overlay, 3-way comparison,
  and cut-locked (single video / by block, indexed / raw) plots built in Python this
  session. The Python computation scripts
  (`fixation_density_concordance*.py`, `isc_fixation_timecourse.py`) and plotting scripts
  (`plot_*.py`) are now superseded for ongoing use but left in place, not deleted.
- **Built `scripts/analysis/concordance_isc_timecourses.qmd`** -- the new plotting
  notebook, reading only the three canonical CSVs above plus the cuts CSV (no
  recomputation). Four parts, replicating every timecourse view built in Python this
  session: single-metric timecourse; concordance-vs-ISC overlay; three-way comparison
  with cut lines; cut-locked (single video or `BLOCKS`-pooled, indexed-to-t0 or raw r,
  mean +/- SEM ribbon). Uses fixed hex colors matching the Python-generated figures
  (`#2a78d6` concordance-group, `#eb6834` ISC, `#1baf7a` concordance-pairwise) rather than
  `ggthemes::scale_colour_few()`, which stays reserved for the Adult-Adult/Child-Child/
  Child-Adult comparisons elsewhere in `isc_analysis.qmd`.
- **Validated by extracting and running every chunk via Rscript** (quarto CLI isn't
  installed in this environment, so the notebook couldn't be rendered directly, but every
  chunk's R code ran without error and reproduced the same qualitative pattern already
  seen in Python/the earlier R test: concordance stays near its cut-onset value through
  6s post-cut while ISC decays sharply, replicated independently in both `pixar_birds` and
  `sesame`). **The user should do one real `quarto render` pass on their own machine
  before trusting this as final** -- chunk-by-chunk Rscript execution confirms the R code
  itself is correct, not that the full document renders cleanly end to end (e.g. a
  `patchwork`/`ggthemes` interaction only quarto's rendering path would surface).

## 2026-10-02 — No Python plotting scripts for this analysis, by explicit user rule

- **Decision:** deleted the 5 Python plotting scripts written during this refactor
  (`plot_concordance_comparison.py`, `plot_concordance_isc_cut_locked.py`,
  `plot_concordance_isc_vs_cuts.py`, `plot_concordance_vs_isc.py`,
  `plot_fixation_density_concordance.py`) -- all fully superseded by
  `concordance_isc_timecourses.qmd`. User's stated rule: no plotting scripts in Python for
  this work, only R, going forward.
- **Explicitly asked about, and kept as-is, two adjacent cases** rather than assuming the
  rule applied automatically:
  - `scripts/analysis/plot_results.py` -- a different, pre-existing pipeline (pairs with
    `run_all_clips.py`'s face-gaze/face-looking-proportion analysis, not fixation-position
    concordance/ISC). Out of scope for this refactor; user confirmed leaving it alone.
  - `scripts/analysis/fixation_density_concordance_diagnostic.py` -- not named `plot_*`,
    but is a Python plotting script from this same concordance work (the sample-
    participant-vs-group heatmap sanity check used to validate the closed-form method
    before `fixation_density_concordance.R` was built). User confirmed keeping it as a
    one-off diagnostic tool, not part of the regular (now R) pipeline.

## 2026-10-02 — Renamed scripts for clarity: calc_/plot_/diagnostic_ prefixes

- **Decision:** renamed every script in `scripts/analysis/` touched by this refactor to a
  `calc_` (computation, any language), `plot_` (plotting, R only going forward), or
  `diagnostic_` (one-off sanity-check tool) prefix, confirmed with the user before
  executing (a full rename list, not done piecemeal). Every reference above this entry
  (code mentions, docstrings, METHODS_DECISIONS.md itself above this point) uses the OLD
  names, accurate as of when each entry was written -- not rewritten retroactively, this
  is a changelog, not a wiki page. The mapping, for reading old entries above:
  - `isc_gaze.py` -> `calc_isc_rawgaze.py`
  - `isc_fixations.py` -> `calc_isc_fixations.py`
  - `fixation_density_concordance.py` -> `calc_concordance_group.py`
  - `fixation_density_concordance.R` -> `calc_concordance_group.R`
  - `fixation_density_concordance_pairwise.py` -> `calc_concordance_pairwise.py`
  - `fixation_density_concordance_pairwise.R` -> `calc_concordance_pairwise.R`
  - `isc_fixation_timecourse.py` -> `calc_isc_timecourse.py`
  - `fixation_density_concordance_diagnostic.py` -> `diagnostic_concordance.py`
  - `concordance_isc_timecourses.qmd` -> `plot_concordance_isc_timecourses.qmd`
  (`isc_analysis.qmd` keeps its name -- it's a mixed compute+plot notebook, not a single
  pipeline script, and wasn't part of the rename ask).
- All functional cross-references (Python `from X import Y`, docstring mentions) updated
  and re-verified (`--help` runs cleanly, imports resolve) after the rename. Output CSV
  *data* filenames (e.g. `outputs/fixation_density_concordance.csv`) were deliberately
  left unchanged -- the rename was about script names, not output artifact paths, and
  renaming those would require touching every downstream reader for no requested benefit.

## 2026-10-02 — Renamed the two parameter-decision scripts in stimulus_prep

- **Decision:** in `scripts/preprocessing/stimulus_prep/`, renamed the two scripts that
  were genuinely used to visually decide pipeline parameters (patch diameter/columns,
  ring-exclusion) to a `plot_` prefix, matching the `calc_`/`plot_`/`diagnostic_`
  convention from `scripts/analysis/`: `patch_grid_preview.py` -> `plot_patch_grid.py`,
  `fixation_density_by_patch.py` -> `plot_fixation_density_by_patch.py`. User explicitly
  declined extending `calc_` to `stimulus_inventory.py`/`representative_frames.py` --
  those generate real pipeline data (consumed elsewhere, e.g. by the cut-annotation tool's
  duration lookups), not just decision support, so kept their existing names.
- **Found and fixed 3 actual broken imports** (not just stale docstring mentions) while
  sweeping for cross-references: `scripts/analysis/calc_concordance_group.py`,
  `calc_concordance_pairwise.py`, and `diagnostic_concordance.py` all cross-import
  `compute_patch_grid` from this module via a `sys.path.insert` + bare `from
  patch_grid_preview import ...` -- would have raised `ModuleNotFoundError` after the
  rename if missed. All three re-verified working (`--help` runs cleanly) after fixing.
  Output directory names (e.g. `outputs/checks/patch_grid_preview/`) deliberately left
  unchanged, same reasoning as the `scripts/analysis/` rename.

## 2026-10-02 — Retired `outputs/`; split into `figures/` (checks + real figures) and `data/results/`

- **Context:** `outputs/` was a convention introduced this session, not an existing
  project one. Root `README.md` documents that `data/` and `figures/` are gitignored --
  "generated results/figures stay on the lab server rather than in git, since they're
  large and some contain participant-level information" -- and `figures/` is indeed in
  `.gitignore` with no commits since that rule took effect. `outputs/` had no such
  protection (plain untracked files) and wasn't part of the established convention.
  User's call: stop using `outputs/` entirely; split by content type instead.
- **New convention:**
  - **`figures/checks/`** -- sanity/parameter-decision plots (what used to be
    `outputs/checks/`'s role for one-off verification/exploration images).
  - **`figures/concordance/`** -- real analysis-result figures for the concordance/ISC
    work (mirrors the pre-existing `figures/isc/` pattern from `isc_analysis.qmd`).
  - **`data/results/stimulus_prep/`** and **`data/results/concordance/`** -- result CSVs
    (and the `representative_frames/` PNG directory), mirroring the pre-existing
    `data/results/isc/` pattern. **`data/` is a symlink to the mounted lab server in this
    repo -- writing here writes through to shared storage, not just the local checkout.**
    User explicitly confirmed this (asked directly, given the implication) rather than a
    new local-only `results/` folder.
- **Moved existing content**, sorting `outputs/checks/*` by actual role rather than a
  blanket move: `closed_form_vs_raster_blur_verification.png`, `sigma_preview*.png`,
  `fixation_density_concordance_diagnostic.png`, `fixation_density_by_patch.png`, and
  `patch_grid_preview/` (all genuinely sanity/parameter-decision artifacts) ->
  `figures/checks/`; the eight substantive concordance/ISC timecourse and cut-locked
  result figures -> `figures/concordance/`. CSVs moved to the matching
  `data/results/{stimulus_prep,concordance}/` subfolder. `outputs/` deleted entirely.
- **Updated every script's default paths** (not just moved files) -- `stimulus_inventory.py`,
  `representative_frames.py`, `plot_patch_grid.py`, `plot_fixation_density_by_patch.py`,
  `calc_concordance_group.py`/`.R`, `calc_concordance_pairwise.py`/`.R`,
  `calc_isc_timecourse.py`, `diagnostic_concordance.py`, `isc_analysis.qmd` (Part 8b
  export), and `cut_annotation_tool.html`'s comment. Each script's single `--output_dir`
  became separate `--results_dir`/`--checks_dir` args where it previously wrote both a
  CSV and a check figure, since those now have different homes. Re-verified every
  affected script runs correctly against real data after the change (not just `--help`):
  `calc_concordance_group.R`, `calc_concordance_pairwise.R`,
  `plot_fixation_density_by_patch.py`, `diagnostic_concordance.py`, and the full
  `plot_concordance_isc_timecourses.qmd` notebook (extracted and run via Rscript, as
  before -- quarto CLI still isn't installed in this environment).
- **Added `ggsave()` calls to `plot_concordance_isc_timecourses.qmd`**, which previously
  had none (plots only rendered inline) -- all 8 plots now save to `figures/concordance/`
  with filenames matching the figures they functionally replace. Fixed a stale reference
  in that notebook's Part 4 intro to `plot_concordance_isc_cut_locked.py`, a Python script
  deleted in an earlier entry today.

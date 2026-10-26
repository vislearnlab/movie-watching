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

## 2026-10-14 — Entropy of group fixation density, as a second lens on concordance/ISC

- **New metric, not a replacement for concordance**: Shannon entropy (bits) of the pooled
  group fixation density, normalized across the 120 scene tiles, per analysis window --
  measures how SPREAD OUT the group's collective attention is (low = everyone converges on
  one spot; high = attention genuinely split across several simultaneously salient things).
  Same 120-tile grid (diameter=180px, 15x8) as the concordance work throughout, not a
  separate grid.
- **Checked whether "unique tiles touched" (a cruder, non-entropy proxy used first) agreed
  with true entropy** -- it mostly did, but entropy caught a real block-level difference
  (frank/sesame higher entropy than slow/pixar, ~0.35 bits) that the cruder measure
  understated. User's point: unique-tile-count treats a tile the same whether it got 1% or
  40% of the mass, so it isn't a substitute for entropy and shouldn't be read as one.
- **Fixed-N rarefaction**, to rule out fixation-count as a confound: raw entropy correlates
  with how many fixations happen to land in a window (pooled r=0.33), so a block or clip
  with more fixations could show inflated entropy for a reason that has nothing to do with
  where people actually looked. Fix: for each window, randomly subsample a FIXED number of
  pooled fixations (no replacement), rebuild the density from just those, repeat across many
  draws, average -- the standard rarefaction fix for sample-size bias in diversity/entropy
  indices. Confirmed on the by-block comparison that this barely moves the numbers (every
  block shifted down by a similar amount), i.e. the original block pattern wasn't a
  fixation-count artifact -- but that check doesn't carry over automatically to a
  per-segment breakdown, so the per-segment export (below) gets its own rarefaction pass
  rather than assuming the block-level result generalizes.
- **Rarefaction parameters, chosen deliberately, not just "reasonable round numbers"**:
  `N_FIX_FIXED=75` fixations/window, `N_DRAWS=100` random draws averaged per window
  (`scripts/analysis/calc_entropy_by_segment.py`). An earlier pass used N=60/draws=40 with
  no stated reason to prefer those specific values over other numbers in the same ballpark,
  called out directly by the user as inadequately justified. Revised:
  - `N_FIX_FIXED=75`: the pooled per-window fixation-count distribution across all 12 clips
    has p5=78, p25=94, median=109, p75=124, p95=144. 75 sits just under that p5, so only a
    small minority of windows (2.8%, up from 0.5% at N=60 -- expected, since 75 is closer to
    the actual cutoff) lack enough fixations to rarefy and are left NaN rather than silently
    dropped (tracked per-row via `n_windows_dropped_low_fix`).
  - `N_DRAWS=100`: reduces Monte Carlo noise in the rarefied estimate more than the earlier
    40-draw pass did. Not independently proven sufficient either -- no formal convergence
    check (e.g. 100 vs. 200+ draws) has been run -- but a deliberate, documented choice
    rather than one picked for runtime convenience with no check at all.
- **Segment-level export**: `data/results/entropy_by_segment/entropy_by_segment.csv`, one
  row per (video, cut-to-cut segment) across all 12 clips (184 rows) --
  `video, block, segment_idx, seg_start_s, seg_end_s, seg_dur_s, n_windows,
  n_windows_dropped_low_fix, mean_entropy_rarefied`. Segment boundaries come from
  `annotations/scene-cuts/all_clips_cuts.csv`; exact clip durations via `ffprobe`, not the
  rounded estimates (e.g. 60.3 vs. the true 60.32) used earlier in this exploration, which
  had produced a negative-duration trailing segment for `frank_complex`.
- **Video/montage spot-checks** (`figures/checks/entropy_montage_{lowest,highest}16.png`,
  `entropy_segments_{lowest,highest}/*.mp4`, built via new
  `scripts/preprocessing/viz/visualize_concordance_video.py`-derived tooling) confirmed low-
  and high-entropy windows look exactly like the metric's definition predicts: low-entropy
  montage is tight single-target convergence (faces, a lone bird, hands on a card trick);
  high-entropy montage is genuinely scattered multi-target scenes (group play, a marble run
  with several moving balls). Caught and fixed a real rendering bug in the process: cut
  annotations in `all_clips_cuts.csv` are accurate to roughly a frame, not sub-frame, and
  without a safety margin a segment's rendered video could include a frame from the
  adjacent scene. Fixed via `--start_margin_frames`/`--end_margin_frames` (now 2 frames by
  default, confirmed via `ffmpeg showinfo` on two real cases -- one with a ~5ms annotation
  error, one with a ~33ms/one-frame error -- that 1 frame of margin wasn't always enough but
  2 was, for both).

## 2026-10-15 — Entropy: whole-segment-pooled abandoned, two pipeline bugs fixed, robustness sweeps, headline notebook

- **Tried whole-segment-pooled entropy (pool all fixations across a whole cut-to-cut
  segment, one entropy value per segment) as an alternative to averaging per-window
  entropy across a segment.** Abandoned: it has two confounds the windowed approach
  doesn't. (1) Segment duration itself drives apparent spread -- `slow`'s segments run
  ~15s vs. ~3-4s for the other blocks, so pooling mechanically accumulates more spatial
  coverage regardless of content; block-level entropy "jump size" tracked block-level mean
  segment duration almost exactly. (2) A single fixed N (e.g. 75) represents wildly
  different fractions of each block's available fixation pool when pool size scales with
  duration -- 7.3% of `slow`'s pool vs. 86.0% of `sesame`'s at N=75, so the "same"
  rarefaction isn't doing the same thing across blocks. Per user instruction, all
  whole-segment-pooled work (scripts, CSV, figures) was deleted; nothing from that line of
  work was ever committed.
- **Fixed two real bugs in `calc_entropy_by_segment.py`** (previously committed in
  `109a597` with neither bug present yet):
  - Zero-duration trailing segments: a clip's rounded duration estimate could be
    marginally less than its last cut timestamp (`frank_complex`: estimate 60.3 vs. actual
    last cut 60.32), producing a spurious negative-length final segment. Already partly
    addressed by switching to exact `ffprobe` durations (prior entry), but recurred;
    fixed at the source by skipping any segment with `e - s <= 0`.
  - Window-to-segment assignment required only `start < seg_end`, not full containment.
    That let a window's tail bleed past its segment's boundary into the next segment's
    data -- 100% of windows bleed for any segment shorter than `WINDOW_S`, not just an edge
    case. Fixed to `start >= seg_start AND start + WINDOW_S <= seg_end`. This also
    principles the "should there be a minimum segment length" question raised mid-session:
    yes, exactly `WINDOW_S` -- 15/183 segments (8.2%) are now NaN rather than silently
    computed from leaked data.
  - Canonical `data/results/entropy_by_segment/entropy_by_segment.csv` regenerated after
    both fixes (183 rows, `frank_complex` now has a clean 13 segments).
- **Window-size and grid-size robustness sweeps**, to check whether "pixar is less
  entropic than frank/sesame" is an artifact of the specific 1.0s/120-tile settings:
  - Window-size sweep (0.5/1.0/1.5/2.0s, fixed 120-tile grid): N_FIX_FIXED can't be one
    fixed value across window sizes without penalizing short windows (fewer fixations land
    in a shorter window) or underusing long ones -- at N=75 and WINDOW_S=0.5, 28.5% of
    windows get excluded for pixar alone. Fixed by computing N separately per window size as
    floor(p5) of that window size's own pooled fixation-count distribution: 53/78/105/132
    for 0.5/1.0/1.5/2.0s.
  - Grid-size sweep (6/15/66/120 tiles, fixed 1.0s windows, N=78): tile diameter must be
    chosen so each grid's overlap ratio (diameter/spacing_x) matches the 120-tile grid's
    (~1.4483), not just picked as "smallest diameter that still tiles the screen" -- an
    early pass used mismatched ratios and produced a spuriously null result at 6 tiles
    (F=0.33, p=0.81) because the tiles were too small/sparse relative to the others to catch
    any signal. Corrected diameters: 806/510/243/180px for 6/15/66/120 tiles. 66 (not an
    exact 60) is the closest tile count achievable at the matched ratio on an 11-column
    grid -- 60 tiles forces a different `n_rows` rounding that only reaches ratio 1.28.
  - Both sweeps implemented in `scripts/analysis/calc_entropy_sensitivity_sweep.py` ->
    `data/results/entropy_by_segment/entropy_sensitivity_sweep.csv`. Result: pixar has the
    lowest mean entropy of the three (frank/sesame/pixar; `slow` excluded from this
    comparison for the duration-confound reason above) in every single condition tested
    across both sweeps -- direction never flips. Pixar-sesame (Tukey HSD) is significant at
    every condition in both sweeps; pixar-frank is significant at every window size and at
    the two finest grids (66, 120 tiles), narrowing but not reversing at the two coarsest
    grids (6: p=0.149, 15: p=0.062).
- **New R notebook**: `scripts/analysis/plot_entropy.qmd` -- the step-through writeup of
  the three claims above (lower entropy in pixar; holds across window size; holds across
  grid size), reading both CSVs above, running ANOVA + Tukey HSD live in R (not pasted
  from the Python runs), saving to `figures/entropy/`.

## 2026-10-15 — ISC doesn't survive segmentation; tile-peak ISC built as a fix, promoted to canonical

- **Question**: does ISC (inter-subject correlation of gaze position over time) hold up
  when computed per cut-to-cut segment instead of per whole clip, the same way entropy is?
  pixar_birds only, all checks below in scratch, nothing promoted to a canonical script.
- **No -- confirmed empirically, not just reasoned about.** Fixation-based pairwise ISC
  (`calc_isc_fixations.py`'s `fixations_to_binned_series` + `align_and_isc`, the project's
  existing method): whole-clip mean isc_xy=0.430 vs. segment-mean=0.325 (-24%), and even a
  single uncut 26.87s segment (nearly 1/5 of the clip) only reaches 0.31 -- well short of
  the whole-clip value. This says the gap isn't a short-segment/low-N artifact alone: a
  meaningful share of ISC's signal comes from *repeated* shared-reaction events (joint
  saccades at cuts, shared capture by motion) accumulating across the *whole* clip, which
  by construction can't live inside any single segment, long or short.
- **Tried fixing it with density vectors instead of raw position** (pairwise, each
  person's own closed-form density-vector peak -- argmax tile, not centroid/PC -- tracked
  over WINDOW_S=STEP_S=1.0s non-overlapping windows, correlated the same isc_x/isc_y/mean
  way as the existing pipeline). This did NOT help -- it was worse on every axis: lower
  whole-clip baseline (0.381), worse relative attenuation (-30%), and far fewer scorable
  segments (14/43 vs. 40/43 for fixation ISC). Root cause: WINDOW_S=1.0s is a 50x coarser
  sampling rate than the fixation approach's 20ms bins (1 sample/sec vs. 50/sec), so
  MIN_WINDOWS=3 demands 3.0s of segment just to attempt a correlation, and 67% of
  pixar_birds' 43 segments are under 3.0s (median segment = 2.29s) -- a real duration
  floor, not a fixable windowing artifact (confirmed: every segment >=3.0s scored; no
  near-miss segments lost to grid-alignment phase).
- **Better version, named "tile-peak ISC": GROUP-level, not pairwise, WINDOW_S=STEP_S=0.5s.**
  This is the one structural change that matters most, so it's called out on its own:
  each person is correlated against the LEAVE-ONE-OUT GROUP's peak (everyone else's pooled
  density, same construction as concordance), not against one other person. That's a
  deliberate fix, not a style choice -- the group side, pooled from ~35-41 people, is
  almost never undefined in a given window, unlike a single other person's trajectory,
  which is what let this version recover far more usable data than the pairwise attempt.
  Per person per window: own peak = argmax of their own closed-form density vector (tile
  center x,y, NOT raw gaze/fixation coordinates); group peak = argmax of the group-minus-
  them vector, also tile-center x,y. Track both as time series, Pearson r on x and y
  independently (None if either side's variance is nonzero only due to float noise from
  the `total - own` subtraction -- needs a tolerance check, not exact `==0`; scipy's own
  near-constant detection still returns NaN on ~1e-13-std input that fails an exact-zero
  guard), isc_xy = mean of whichever axis(es) are defined (matching `align_and_isc`'s "at
  least one of isc_x/isc_y" rule). 0.5s chosen because it's just past pixar_birds' own
  median fixation duration (296ms) -- NOT because the entropy window-size sweep validated
  0.5s, which pooled ~40 people's fixations per window and says nothing about whether one
  person's own fixation stream has enough data at that size, a different data-sufficiency
  regime entirely. Result (pixar_birds): whole-clip=0.506, segment-mean=0.434, only -14%
  relative attenuation (vs -24% fixation-pairwise, -30% pairwise-peak-1.0s) -- the only
  variant where segmenting costs meaningfully less. Coverage: 25/43 segments (vs. the 1.0s
  pairwise version's 14/43, still short of fixation ISC's 40/43, since this still needs
  >=1.5s/segment vs. fixation ISC's 1.0s floor).
- **Caveat on comparing the group-level number's magnitude directly to pairwise ISC**:
  correlating an individual against a pooled ~40-person reference is inherently less noisy
  than correlating two individuals against each other, so some of the 0.506 vs. 0.430 gap
  is a statistical property of pooling, not evidence of "more true synchrony." The
  RELATIVE attenuation (whole-clip vs. segment-mean) is the fairer cross-method
  comparison, since both numbers get the same pooling effect.
- **Cross-validated against the existing pairwise ISC, all 12 clips, whole-clip level**:
  the same four clips (slow_people, slow_hands, frank_play, pixar_birds) stand out as the
  highest-ISC clips under BOTH methods, with the same relative ordering of everything else
  -- real agreement at the clip level, not just a coincidence of block averages. The
  block-level ranking does change (pairwise ISC: pixar highest, 0.430; tile-peak ISC: slow
  highest, 0.522, pixar second, 0.506) -- but this traces to pixar being a block of exactly
  one clip (nothing to average down), while slow's block average mixes two very-high clips
  (hands, people) with two middling ones (animals, inanimate). Not a disagreement about
  which moments are synchronized, just an aggregation artifact of block size.
- **Entropy vs. tile-peak ISC**: segment-level, all 12 clips, negative as predicted (low
  entropy = concentrated attention should track high ISC = tight temporal tracking): pooled
  r=-0.354, p<0.0001, n=133; consistent direction in all 4 blocks but only individually
  significant in pixar (r=-0.621, p=0.0009, n=25 -- frank/sesame/slow each r=-0.14 to
  -0.23, ns). Per-CLIP (n=12, whole-clip ISC vs. mean entropy across a clip's segments):
  r=-0.712, p=0.0093 -- the cleanest view of the relationship, with slow_hands anchoring
  the low-entropy/high-ISC end almost exactly on the fit line and slow_people as the one
  real outlier (highest ISC of any clip, only middling entropy).
- **Gaze-overlay spot-checks** (`figures/checks/entropy_isc_spotcheck/`, pixar_birds
  seg07/10/33): confirmed visually. seg7 (entropy=4.00, ISC=0.01): multiple birds on a
  wire plus a stray off-target density blob -- genuinely dispersed. seg10 (entropy=2.66,
  ISC=0.80): one isolated bird, one tight density cloud. seg33 (entropy=2.78, ISC=0.86): a
  single elongated, streaking density cloud -- not a static target but everyone's gaze
  tracking the same *moving* target together, a second, distinct mechanism for the same
  low-entropy/high-ISC combination.
- **Status: promoted to canonical.** `scripts/analysis/calc_isc_tilepeak.py` -- named to
  match `calc_isc_rawgaze.py`/`calc_isc_fixations.py` (by input representation: raw
  samples / fixation events / tile-peak), output `data/results/isc/isc_tilepeak.csv`
  (`isc_tilepeak_x/y/xy` columns, prefixed so they can't get silently confused with
  `isc_fixation_pairwise.csv`'s `isc_x/y/xy`). Whole-clip only, all 12 clips, one row per
  (video, pid) -- matches the table above. The segment-level checks above (pixar only) and
  the entropy-vs-ISC comparison (all 12 clips) are still scratch, not canonical scripts.

## 2026-10-19 — Within-person vs. between-person gaze dispersion: entropy abandoned, tile-vector cosine distance built, four-quadrant framework

- **Motivation**: why does slow_people have high ISC despite only moderate (not lowest)
  entropy? Visual gaze-overlay inspection of its 4 segments found a "coordinated vs.
  concentrated" mechanism -- entropy tracks spatial separation between 2 consistently-
  tracked targets, while ISC stays high as long as the group agrees on which targets to
  split between, regardless of distance. This motivated a new metric, borrowed from
  ecology's diversity-partitioning framework: decompose a segment's gaze pattern into
  within-person spread (how much one person's own attention moves around) and
  between-person spread (how much different people's attention disagrees).
- **First attempt: Shannon entropy of each person's own tile vector (pixar_birds pilot,
  `scripts/analysis/calc_entropy_within_between.py`). Failed at this data's real sample
  sizes.** Unrarefied (all available fixations per person per segment), within-person
  entropy correlated r=0.79 with raw fixation count -- almost pure count confound, not
  signal. Rarefying to a fixed N per person (matching the canonical group pipeline's
  floor(p5) convention) made it worse, not better: at the resulting N=2, entropy on a
  120-tile grid collapses to a narrow, near-ceiling band (1.65-2.03 bits, std=0.08) because
  2 well-separated points almost always touch different tiles regardless of true distance
  -- entropy's log-transform saturates at small N; this holds at every grid coarseness
  tested (6/15/66/120 tiles, confirmed empirically, not just reasoned), since the kernel
  width (sigma=46px, fixed) rather than tile count sets the achievable resolution for a
  2-point sample. Diversity-partitioning's `beta = gamma - alpha` is also not a rigorous
  decomposition for entropy (true additivity needs the Hill-number/multiplicative form),
  unlike variance, which has an exact `law of total variance` identity -- this motivated
  trying variance/distance-based metrics instead of a better entropy estimator.
- **Centroid-based spread (pixel-space, ANOVA-style within/between-of-centroids) rejected
  on conceptual grounds before being built out.** For genuinely bimodal attention (e.g.
  slow_people's two co-moving targets), the arithmetic mean of a person's fixations lands
  in the empty space *between* the two targets, not on either one -- a real, not cosmetic,
  problem: "low between-person spread" would read as "people agree on a target" when the
  true claim is only "people's *average* locations agree," which can be true for bimodal
  splitting with no shared target at all.
- **Built instead: tile-vector cosine distance** (`scripts/analysis/calc_within_between_cosine.py`,
  canonical, all 12 clips). Same closed-form Gaussian/120-tile machinery as concordance and
  `calc_isc_tilepeak.py` (sigma=46px, 15x8 grid, WINDOW_S=STEP_S=0.5s non-overlapping -- no
  new spatial parameters). Two deliberately PARALLEL constructions (both "distance from a
  mean"), NOT an additive decomposition (the quadrant framework below only needs two
  comparable axes, not a literal total=within+between split, and cosine distance doesn't
  decompose additively the way squared Euclidean does -- correlation/cosine was kept for
  consistency with the existing tile-peak ISC/concordance convention instead):
  - `within_person_cosine_dist`: mean cosine distance between a person's own per-window
    tile vectors, over all pairs of windows in the segment -- temporal self-consistency,
    not spatial spread of one snapshot. Captures switching ACROSS windows, not two targets
    fixated simultaneously WITHIN one 0.5s window (blurs into one bimodal vector instead)
    -- accepted limitation.
  - `between_person_cosine_dist`: collapse each person's own windows to one mean vector,
    then cosine distance to the LEAVE-ONE-OUT mean of every other eligible person's mean
    vector. Deliberately NOT time-matched/synchrony-sensitive (unlike ISC) -- two people
    who visited the same targets at different moments score as agreeing. An earlier
    version of this script compared each person's per-WINDOW vector to the per-window
    leave-one-out group (time-matched, ISC-style) instead of mean-to-mean; corrected after
    the user pointed out it wasn't actually parallel to the within-person axis, which
    compares one person's own windows to their own mean, not to a moving group reference.
  - Avoiding entropy's log-transform (comparing raw/cosine vectors directly, never taking
    an entropy of anything) is what fixes the small-N collapse: a vector-distance doesn't
    saturate near a ceiling the way entropy does.
  - Normalization note (unresolved tradeoff, left as cosine): L1-normalize-then-Euclidean
    keeps the additive total=within+between identity and fixes the fixation-count/magnitude
    confound; raw cosine fixes the magnitude confound too but not the additive identity;
    L2-normalize-then-Euclidean is mathematically equivalent to cosine (exact identity:
    squared Euclidean on unit vectors = 2(1-cosine similarity)). Picked cosine for
    consistency with existing tile-peak ISC/concordance work, not because it's uniquely
    correct.
- **MIN_WINDOWS=2** (lowered from an initial 3, which mirrored `calc_isc_tilepeak.py`'s
  floor). That floor's reasoning doesn't transfer here: tile-peak ISC reports one number
  *per person* (where 2-window noise is exposed directly in the final output), but this
  script only ever reports a segment-level mean over ~35-41 people, which should average
  out single-pair noise the way any large average does. Recovers 18/183 segments
  (144->162 valid) with no disruption to the 3 known pixar example segments (already
  >=3 windows, unchanged) or the tile-peak ISC cross-correlation (-0.278 vs -0.269,
  negligible) -- though that check only covers segments already in the old ISC scratch
  file, not the newly-recovered 2-window segments specifically.
- **MAX_FIXATION_MS=2000 fixation-duration filter, added after a concrete data-quality
  find.** Illustrative per-quadrant example segments were picked by ranking participants'
  own within/between values to find the clearest "textbook" cases per segment (most
  extreme = farthest into that quadrant's corner). Several of the resulting "selective"
  examples, on inspection via gaze-overlay GIFs, turned out to be a single absurdly long
  fixation (5-19 seconds vs. a ~296ms corpus median) rather than genuine decisive gaze --
  one participant, MW028, alone accounts for 4 of the 10 longest fixations in the whole
  68,872-fixation corpus. Mechanism: within-person distance rewards near-zero values most
  strongly for exactly this degenerate case (one unmoving fixation spanning many
  consecutive windows gives ~identical tile vectors by construction), so "pick the most
  extreme example" was specifically hunting for this artifact. Fix: drop fixations >2000ms
  before building tile vectors (~98.8% of real fixations are under this; it isn't a tight
  cutoff). Confirmed working on the two flagged cases: frank_play's top-scoring "selective"
  participant changed from MW028 (the repeat offender) to MW049 (normal multi-window
  values); frank_objects' near-zero remaining case (MW002) was checked against raw
  fixation records and is genuine (two real fixations, 1056ms and 146ms, at different
  nearby positions) -- the filter demonstrably removes the artifact without blanket-killing
  every low-within-person-distance case.
- **Four-quadrant framework** (median-split on both axes, pooled across all 183 segments):
  CONCENTRATED (low within, low between -- one shared target), COORDINATED (high within,
  low between -- everyone splits attention the same way, e.g. slow_people's two co-moving
  targets or tracking one moving target), SELECTIVE (low within, high between -- each
  person decisive, but people disagree on WHAT), SCATTERED (high within, high between --
  individually unfocused AND disagreeing). Illustrated with gaze-overlay montages/GIFs in
  `figures/within_between/` -- concentrated reads as a genuinely unified phenomenon across
  every block and content type; coordinated's "two distinct agents" story holds for
  pixar/sesame but not cleanly for frank_play/slow; selective is cleanest for content with
  multiple similar interchangeable options (pixar's row of birds); slow and frank_objects
  blur the selective/scattered distinction.
- **Segment duration is a real confound on the between-person axis, demonstrated not just
  suspected.** Between-person distance is built from each person's segment-long mean
  vector, so longer segments give more time for people's *overall* patterns to converge
  even without real moment-to-moment agreement. Artificially chopping slow's real ~15.2s
  segments into ~3s chunks (matching the rest of the corpus) flipped its quadrant
  distribution from 75% COORDINATED / 0% SELECTIVE to 34% CONCENTRATED / 14% COORDINATED /
  32% SELECTIVE / 20% SCATTERED -- i.e. slow's apparent "coordinated" signature was
  overwhelmingly a duration artifact, not a property of the content. Pixar's one long
  segment (seg0, 26.87s) showed the identical pattern when chopped (mean between-person
  distance 0.155 -> 0.344). **Not fixed in the canonical script** -- `all_clips_by_segment.csv`
  still uses real scene-cut segments with their native (highly unequal: 0.5s-26.9s)
  durations; duration-matched chunking is currently an ad hoc, uncommitted scratch
  technique used only for illustration and the slow/pixar sensitivity check, not applied
  across the whole dataset. Any cross-block or cross-clip comparison using this data should
  account for this before trusting it.
- **Status: canonical for the per-segment metric computation, NOT yet promoted as a
  validated finding.** `scripts/analysis/calc_within_between_cosine.py` outputs
  `data/results/within_between_cosine/all_clips_by_segment.csv`, all 12 clips. Figures in
  `figures/within_between/` (quadrant scatter plots, schematics, gaze-overlay montages and
  GIFs organized by quadrant). `calc_entropy_within_between.py` (the superseded entropy
  pilot) is kept as a historical record of why that approach doesn't work, not deleted.
  Outstanding before this should be treated as a trustworthy result: the duration confound
  above is unresolved in the canonical pipeline, and the quadrant framework's
  interpretability varies noticeably by block/content type rather than being uniform.

## 2026-10-20 — Within/between duration confound: N_CAP=12 adaptive window rarefaction (partial fix)

- **Follow-up to the duration confound documented above.** A long segment gives
  between-person distance more time to converge (people's overall patterns agree more just
  because there's more time to sample the same places) even with no real moment-to-moment
  agreement, and gives within-person distance more time to drift -- same mechanism already
  demonstrated by chopping `slow`'s real ~15.2s segments into ~3s chunks.
- **Fix: adaptive window cap, not a fixed N.** A segment with <=N_CAP windows available
  uses all of them, unchanged, exact. A segment with MORE than N_CAP windows gets N_DRAWS
  random draws of exactly N_CAP windows (no replacement), averaged -- the same Monte Carlo
  rarefaction convention already used in this project for fixation-count confounds in the
  entropy work.
- **Alternatives considered:**
  - **Strict fixed-N for every segment** (same N regardless of segment length) controls the
    confound best, but costs the most coverage: fixed N=6 took coverage from 162/183 valid
    segments down to 89/183 -- most short segments get dropped entirely.
  - **N_CAP=12 (6s), the one chosen**: preserves full coverage (every segment that had
    >=2 windows before still qualifies) at a real but smaller cost to confound control.
  - Tighter caps (N_CAP=8, N_CAP=10) were also checked and control the confound better than
    N_CAP=12, but still not as well as fixed N=6.
- **Why N_CAP=12 over the tighter caps, given it controls the confound the least well of
  the four options tested:** corr(duration, between-person distance) is -0.33 at N_CAP=12,
  worse than -0.23 at N_CAP=8 and -0.14 at fixed N=6 -- caps only intervene on the minority
  of segments above the cap, and most of the corpus (63% at N_CAP=8-10) has fewer than 8
  real windows to begin with, so it's untouched by capping at any value. N_CAP=12 was kept
  anyway for the coverage tradeoff (full coverage vs. 89/183), accepted explicitly as a
  partial fix, not a full solution.
- **Status: not a resolution of the duration confound**, just a smaller version of it than
  doing nothing. Any cross-block or cross-clip comparison should still account for residual
  duration sensitivity. See `scripts/analysis/calc_within_between_cosine.py` for the
  implementation (`N_CAP`, `N_DRAWS`, `RNG_SEED` constants; `within_between_for_windows()`
  is the single-pass helper called once per segment at or under the cap, or once per Monte
  Carlo draw above it).

## 2026-10-21 — Per-tile optical flow motion metric, to test content predictors of quadrant membership

- **Motivation**: characterize what segment content drives concentrated/coordinated/selective
  /scattered quadrant membership. Motion ("how much is moving, and how many separate things")
  was the first candidate feature, computed on the same 120-tile grid as gaze so the two are
  directly comparable (`scripts/analysis/calc_motion_tiles.py`).
- **Dense Farneback optical flow**, not frame-differencing, between every consecutive frame.
- **Global-motion (camera pan) subtraction**: subtract the per-frame median flow vector
  (robust proxy for camera motion) before computing magnitude. Needed: a sanity check on
  pixar_birds found raw flow backwards on camera-tracking shots (background scored as high
  motion, the tracked subject near-zero) in 2 of 6 inspected high-motion scenes. Confirmed
  fixed by inspecting the same windows before/after; non-panning shots were unaffected.
  frank_complex (real handheld footage) showed no comparable inversion — likely pixar-specific
  (deliberate tracking shots), not a general problem.
- **fps normalization**: multiply magnitude by each clip's own fps (px/frame -> px/second).
  The 12 clips span 3 frame rates (25, 23.976, 29.97); resolution is uniform (1280x720) so no
  spatial correction needed, just temporal.
- **Box-pooling, not Gaussian**: motion is pooled into the 15x8 tile grid via
  `cv2.resize(..., INTER_AREA)`, not the Gaussian-at-tile-centers method the gaze pipeline
  uses. Deliberate: gaze needs a kernel to smooth sparse fixation points into a field; flow is
  already a dense field, so box-pooling is the natural analogue. Tile footprints are therefore
  not pixel-identical to the gaze grid's edge-inset Gaussian centers, just the same 15x8
  structure and row-major ordering.
- **Output**: `data/results/motion/<clip>_motion_tiles.csv`, one row per 0.5s window (matches
  `calc_within_between_cosine.py`'s WINDOW_S), columns `tile_000`...`tile_119`.
- **Status**: all 12 clips computed (`data/results/motion/`). Segment-level motion-gaze
  correlation by quadrant built in `calc_motion_gaze_correlation.py` (see entry below).

## 2026-10-21 — Per-tile visual saliency: FineGrained chosen over SpectralResidual

- **Motivation**: a second content feature (alongside motion) to test as a predictor of
  quadrant membership -- low-level bottom-up visual salience (color/contrast/orientation),
  explicitly a hypothesis about what might guide attention, not a prediction of gaze itself.
  True GBVS isn't available in any mainstream Python package; `cv2.saliency` (OpenCV's
  `opencv-contrib-python`, swapped in for the previously-installed `opencv-python`) offers
  two unrelated algorithms: `StaticSaliencySpectralResidual` (Hou & Zhang 2007, frequency-
  domain, grayscale-only) and `StaticSaliencyFineGrained` (different method, not frequency-
  based).
- **Sanity-checked both on pixar_birds**, 10 distinct high-saliency windows each. Found both
  algorithms agree: wheat-field background texture consistently scores higher than the flat-
  shaded bird characters in most scenes (confirmed via raw per-pixel map + tile values for
  one case, w121's falling bird -- a real but weak signal, swamped by a denser 10-bird row
  elsewhere in frame, not a bug). Two independent algorithms agreeing argues this is a
  genuine property of the content (flat cartoon subjects vs. textured backgrounds), not an
  artifact of one method.
- **Decision: FineGrained chosen as the canonical algorithm**, by visual inspection --
  user's read is it resembles GBVS more closely than SpectralResidual does, despite neither
  being literal GBVS. SpectralResidual is much faster (18s vs. ~4min for pixar_birds) but
  that wasn't the deciding factor.
- **Output convention**: `data/results/saliency/<clip>_saliency_tiles.csv` is FineGrained
  (no suffix, the default in `calc_saliency_tiles.py --algo`); SpectralResidual outputs are
  explicitly suffixed `_spectral_residual` where kept, as a comparison artifact.
- **Status**: only pixar_birds computed so far (both algorithms); not yet run on the
  remaining 11 clips.

## 2026-10-22 — DeepMeaning rebuilt to use the project's 120-tile grid, not its native 73%-overlap grid

- **Motivation**: a third content feature (meaning, alongside motion/saliency), via
  Hayes & Henderson (2025) DeepMeaning (https://osf.io/hcnfx/), ported locally as
  `research-projects/DeepMeaning` (CoCa `coca_ViT-L-14`/`mscoco_finetuned_laion2b_s13b_b90k`
  embeddings -> pretrained indoor/outdoor linear regression heads). Its native grid
  (`define_fuzzy_grid`, 128px patches, 73% overlap, 1620 patches/frame at 1920x1080) is a
  different spatial basis than every other feature in this project (120 tiles, 15x8, from
  `compute_patch_grid`). Decision: rebuild the grid-handling to operate natively on the 120
  tile positions rather than compute on the native grid and downsample afterward, so meaning
  is directly comparable/combinable with motion, saliency, center bias, and gaze without a
  resampling step.
- **Patch size matched to 128px** (DeepMeaning's own validated size) at the 120 tile centers,
  rather than inventing a new patch size -- avoids re-validating the CoCa embedding at an
  unvalidated patch scale. The embedding regression itself (weights, BGR/RGB quirk, CoCa
  checkpoint) is untouched.
- **Canvas-space placement, not native-resolution**: all 12 clips are 1280x720 (16:9), which
  `choose_canvas` always maps to the 1920x1080 validated canvas with a clean 1.5x upscale and
  zero letterboxing. Placing the *unscaled* 120 tile-center positions (computed directly at
  1920x1080, same convention used for gaze/motion/saliency/center) directly in that canvas
  space means patches fit cleanly in bounds (x: 26-1894, y: 26-1054) with no edge-clamping
  needed -- scaling centers down to native 1280x720 first (as in an earlier ad hoc check) was
  the wrong space and produced spurious negative-index edge overflow.
- **`deepmeaning/pipeline.py` changes**: `process_video()` gained `patch_centers` (canvas-
  space (x,y) list, overrides `define_fuzzy_grid()` entirely when given) and `patch_size`
  params; `PATCH_SIZE` module constant usage replaced with the local parameter everywhere
  (crop loop, `_assemble_map`). Original paper-faithful fuzzy-grid behavior remains the
  default when `patch_centers` is omitted. `grid.py`/`coca_model.py`/`viz.py` untouched --
  `FrameTransform` and the embedding/classification code are grid-size-agnostic already.
- **Manual indoor/outdoor labeling**: added `manual_label_lookup` param (timestamp -> category,
  confidence reported as 1.0) to bypass the zero-shot classifier per-frame. Built
  `scripts/preprocessing/cut_annotation/indoor_outdoor_annotation_tool.html` (same standalone-
  HTML/file:// pattern as `cut_annotation_tool.html`) to label the existing within/between-
  cosine segments (179 total across 12 clips) as indoor/outdoor/mixed, rather than building a
  new scene segmentation -- keeps labels aligned with every other per-segment analysis.
- **Visually verified on pixar_birds** (16 frames, 0.1 fps, rebuilt 120-tile grid): overlay
  maps and raw per-patch values both show the correct direction on cartoon content -- e.g.
  frame at t=~1.25s, bird-face patches mean predicted_meaning=2.44 vs. sky patches=1.92; a
  later frame (birds on a wire) shows a visually distinct warm band tracking the wire exactly.
  Consistent with the original native-grid check from several weeks prior
  (`figures/deep-meaning/frame003_deepmeaning.png`).
- **Environment**: fresh `.venv` in `research-projects/DeepMeaning` (not reused from any
  other project), `pip install -e .`. Confirmed MPS available and used (`torch 2.14.1`,
  `cuda=False, mps=True`); no Tversky/GPU-cluster dependency at this patch count.
- **Timing**: ~9.2s/frame on MPS for 120 patches (vs. the native grid's 1620 patches/frame,
  which is GPU-cluster-scale). A full corpus run at 1 fps across all 12 clips is estimated at
  roughly 2-2.5 hours on this Mac.
- **Status**: pipeline rebuild done and visually verified; manual-labeling GUI built but not
  yet used to produce real labels; full-corpus run not yet started.

## 2026-10-23 — Motion and saliency re-pooled onto DeepMeaning's patch footprints (tile-center alignment)

- **Problem**: `calc_motion_tiles.py`/`calc_saliency_tiles.py` pooled with `cv2.resize(..., (15, 8), INTER_AREA)`
  = 128x135px boxes tiling the full screen, centers x 64-1856, y 67.5-1012.5 (display px). Gaze and
  DeepMeaning tiles are centered at x 120-1800, y 120-960 (`compute_patch_grid(240, 15, 1920, 1080)`).
  Same index and row-major order, but tile i pointed at a different screen location away from center
  (up to 56px off at the edges). The 2026-10-21 "box-pooling, not Gaussian" note is superseded on this point.
- **Fix**: new `scripts/analysis/tile_pooling.py::tile_box_means` = mean over the exact square DeepMeaning
  crops for each fine patch (240x240 canvas px, centered on the tile center, same `round(c - size/2)`
  rounding), mapped to source frames (160x160px at 1280x720). Both scripts now use it.
- **Verified**: gaze centers x 2/3 == DeepMeaning's recorded `patch_center_x/y` (frank_complex fine240),
  pooling-box centers == the same, all boxes 160x160 in bounds, integral-image means == direct slice means.
- **Footprints still differ by design**: gaze = Gaussian (sigma=46px) evaluated at the center; motion,
  saliency, meaning = 240px square box mean. Centers are identical across all four.
- **Rerun done (all 12 clips)**: new files in `data/results/{motion,saliency}/` (window CSVs + per-frame
  `*_frames.csv`); old resize-grid tile files moved to `data/results/{motion,saliency}/legacy_resize_grid/`. Downstream results built on the
  old files (motion/saliency-gaze correlations, partial correlations, variance partition) are stale until rerun.

## 2026-10-23 — Information-gain analysis (Kümmerer, Wallis & Bethge 2015): setup decisions

- **Code**: `scripts/analysis/information_gain/` (settings in `config.py`). Adults only for now; group label kept in every table.
- **Gaze vectors used as-is**: closed-form duration-weighted Gaussian density (sigma=46px) at the 120 tile centers,
  over the whole segment, normalized to sum to 1 per participant x segment. No per-fixation kernel renormalization,
  even though per-fixation tile mass varies with position (1.14 on a tile center, 0.73 in a gap, 0.10 at 20px from
  the edge, 0.009 in a corner). Revisit later.
- **Averaging (open)**: equal weight per participant x segment cell, then per participant, for now. Looking-time
  weighting across segments (per-cell normalization otherwise erases between-segment looking time) discussed, deferred.
- **Cross-validation unit = clip** (leave-one-clip-out) for floor, epsilon, lambda, and feature parameters:
  segments within a clip share content, so leave-one-segment-out would leak it into the floor.
- **Both segment sets run**: `cuts` (183 cut-to-cut segments) and `slow_chopped` (247; each slow segment split into 5).
- **Feature per segment = mean of frames inside the segment** (motion/saliency at native fps, meaning at 6fps), not
  0.5s windows: windows straddle cuts and mix two shots. Motion/saliency scripts now also write
  `<clip>_<feature>_tiles_frames.csv`. Motion drops the first flow frame of each segment (its frame pair spans the cut). Meaning = averaged fine+coarse at the 120 tile centers.
- **Features plotted at patch centers, smoothed, sampled** (following Haskins et al. 2020 Sci Rep: "average rating for
  each tile was then plotted at its center coordinate", then Gaussian-smoothed). Segment-mean patch values (240px box
  means for motion/saliency, DeepMeaning ratings) placed at patch centers, normalized Gaussian-weighted average evaluated
  at the 120 tile centers: fine sigma=46px (same as gaze) for all three features; coarse meaning sigma=92px (~2x its
  spacing; at sigma=46 the 28 coarse points would be isolated spikes). At fine scale each tile ~ 88% its own patch,
  ~3% per side neighbor. Chosen over overlap-averaging the 240px footprints: at exactly 50% overlap every tile center
  sits on the corner of the 2x2-patch mosaic (unsmoothed sampling is a raster-convention knife-edge; smoothed, each
  tile = 25% own / 75% neighbors). Gaze already = fixations smoothed (sigma=46) and sampled at the same centers; the
  closed form matches raster-blur-then-sample to 0.055% of peak once positions are pixel-rounded (1.3% otherwise, from
  sub-pixel positions the raster can't represent). Meaning fine vs fine+coarse average: median r=0.94 per segment
  (5th pct 0.85); fine vs coarse 0.72; coarse not flat after smoothing + rescaling (tile SD 0.090 vs fine 0.098).
  Check figure: `figures/checks/meaning_point_smoothing_fine46_coarse92.png`.
- **Feature scaling = joint min-max to [0, 1] across all segments** (`FEATURE_SCALING = "corpus"`), following
  Kümmerer et al. 2015 ("jointly rescaled ... over all images at once, not per image"), same operation for all three
  features; meaning's fine and coarse are each rescaled jointly, averaged, rescaled again. Keeps contrast differences
  between segments. Briefly set to within-segment z-scoring earlier the same day; reverted to follow the paper.
  `"segment_z"` kept as an alternative.
- **Why a nonlinearity at all (separate from normalization)**: normalizing alone (q = x / sum(x)) assumes looking
  is *proportional* to the raw feature value, which fixes two things the raw scale has no right to decide:
  (1) the zero point -- adding a constant to a feature changes q (raw meaning sits around 1.9-2.4, so proportional
  q is nearly uniform: a bird-face tile at 2.44 vs. sky at 1.92 gets only 1.27x the predicted looking);
  (2) the sharpness -- how much more looking the top tiles get than the rest. Gaze may be far more (or less)
  concentrated than the feature's raw ratios imply. Normalization only makes the vector sum to 1; it cannot change
  either property. f(x) is fit to gaze to set them, then normalization turns f(x) into a probability vector.
- **Nonlinearity = 20-knot monotone piecewise-linear, following Kümmerer et al. 2015** ("supported in 20 equidistant
  points"), one curve per feature on the jointly [0, 1]-rescaled values, fit by maximum likelihood on training clips.
  Chosen over exp(a*x) (briefly adopted to minimize parameters) because IG should reflect information a feature carries,
  not how well its relationship to gaze fits one curve shape -- a shape misfit would leak into the unique/shared
  partition. exp(a*x) kept as comparison (`NONLINEARITY = "exp"`).
- **Saliency = OpenCV FineGrained** (kept; Haskins et al. 2020 used GBVS).
- **Shuffled-feature control**: each segment paired with a random segment from a different clip.
- **Minimum data**: >= 3 fixations per participant x segment; >= 3 other participants for the ceiling.
- **Pipeline built** (`scripts/analysis/information_gain/`: load_gaze, load_features, baseline, ceiling,
  feature_models, scoring, partition, sanity_checks, plots). Floor eps and ceiling lam chosen by nested
  leave-one-clip-out (inner floors exclude both the held-out and the inner clip); feature curves fit on training
  clips with floors excluding the held-out clip. Ceiling/models scored on identical cells (>= 3 fixations,
  >= 3 others). Variants: knots_floor (headline), exp_floor, knots_nofloor (floor comparison).
- **First pass, 2026-10-23, cuts set, slow_people meaning still missing (177 segments, 45 adults, no CIs)**:
  ceiling 1.24 bits; IG saliency 0.17, motion 0.24, meaning 0.30, all three 0.33 (27% of ceiling); motion+meaning
  0.38 > all three, so unique saliency = -0.04 (equal-weight dilution, anticipated in the plan). Shuffled features
  drop to -0.19/-0.25/-0.37. Motion curve is a step (flat above ~0.1 rescaled): exp(a*x) gets motion only 0.07 bits
  vs 0.24 with knots. Ceiling convergence still rising (k=8: 1.06, 16: 1.18, 32: 1.26 bits), so the ceiling is
  likely underestimated and proportions explained overestimated. Entropy vs ceiling per segment r = -0.44.
- **Full run, all 12 clips (2026-10-23, knots_floor, no CIs yet)**. cuts (181 segments, 45 adults): ceiling 1.24;
  saliency 0.17, motion 0.24, meaning 0.30, all three 0.34 (27% of ceiling); partition shared 0.26, unique meaning
  +0.08, motion +0.04, saliency -0.04. slow_chopped (245 segments): ceiling 1.34; all three 0.38 (28%); shared 0.29,
  unique meaning +0.08, motion +0.06, saliency -0.05. 2 of 183 cut segments drop out (no cells meet >= 3 fixations
  with >= 3 others). All sanity checks pass; shuffled features drop below 0 for every feature in both sets.
- **Knot count sweep** (`knots_sweep.py`, cuts): held-out IG plateaus from 5 to 20 knots (all features within 0.01
  bits of their best); 40 slightly lower; exp loses most of motion (0.07 vs 0.24); 3 knots loses a quarter of
  meaning. 20 kept: fixed a priori by the paper, avoids choosing a hyperparameter on the reported held-out scores.
- **Resolution check** (`explore_dense_grid.py`, same pipeline on full-screen grids, cuts, 20 knots, floor):
  28 locations (7x4): ceiling 0.99 bits, all three 0.27 (26.9% of ceiling); 120 tiles: 1.24, 0.34 (27.1%);
  9,975 locations (133x75): 1.24, 0.32 (26.3%). Bits plateau by 120 (gaze is smoothed with sigma=46 and features
  carry 240px-patch detail, so finer sampling adds no information); proportions of ceiling and the partition are
  stable across all three resolutions (shared ~21%, unique meaning ~6.5%, motion ~3%, saliency ~-3.4%).
- **Trial exclusion: negative ceiling** (`exclusions.py`, `config.EXCLUDE_NEGATIVE_CEILING`). A trial (participant x
  clip) is dropped if its mean ceiling across segments is < 0, i.e. the other viewers predict that person's gaze worse
  than the content-blind floor -- a signature of calibration/tracking failure rather than idiosyncratic viewing. Floor
  and ceiling for the rule are computed once on unfiltered data; the excluded trials are then removed from every
  downstream step (floor, eps, lambda, the "others" in everyone's ceiling, curve fits, scoring). Excludes 2 of 448
  trials in both segment sets: MW006 sesameindia_2 (-0.76 bits) and MW049 sesameus_2 (-0.13). Low-but-positive
  trials (e.g. MW049 overall 0.59 bits, MW012 0.89) are kept.
  Effect (cuts): ceiling 1.24 -> 1.25 bits; all three 27.1% -> 27.3% of ceiling; shared 20.9 -> 21.1%; uniques
  unchanged within 0.1 pt. slow_chopped: all three 28.4%, shared 21.9%. All sanity checks still pass.
- **Kids, adult reference** (`kids_adult_reference.py`, 2026-10-23). Kids' fixations (f329476c) copied from
  `data/preprocessed/fixations/kids` to `data/local_cache/fixations/kids`. All kids with fixation files included
  except MWK005 (valid_data=False): 19 kids. MWK011-021 are not in participant_summary.csv (dated 2026-07-15), so
  have no validity/calibration screen; the negative-ceiling trial rule is their only quality filter. Kids are scored
  against adult-derived models only (adult floor, adult lambda, adult-fit feature curves; ceiling "others" = all adults
  in the segment), so the question is how well adult-derived models predict kids, not a kids-only analysis. Same
  >= 3 fixation cell rule. Result (cuts): ceiling 1.20 bits (adults 1.25); all three 28.1% of ceiling (adults 27.3%);
  shared 22.0%, unique motion +3.7%, meaning +4.8%, saliency -2.3%. 0 of 143 kid trials excluded.
- **Open: single-segment failures not caught by the trial rule** (revisit). The negative-ceiling rule averages over a
  clip, so one bad segment can pass: MWK01 frank_play seg 17 (4 fixations, kid ceiling -3.73 bits; clip mean
  positive). It drives the kids' selective-quadrant value (-1.7% explained; 8.2% without MWK01). A cell-level rule was
  considered and deliberately not adopted for now (7% of cells are negative, mostly noise; cutting them risks
  overfitting to noise).
- **Kids, kid reference** (`IG_GROUP=kids`, 2026-10-23). Whole pipeline rerun on kids only (kid floor, eps, lambda,
  feature curves, ceiling = other kids); outputs in `information_gain/kids/` subfolders (cache, results), figures in `figures/information_gain/kids-kidref/` (adults in `adults/`, adult-reference kids in `kids-adultref/`),
  `figures/checks/information_gain_kids/`). Same 19 kids. Trial rule excludes 1 of 143: MWK02 sesameus_2 (-0.05 bits).
  179 segments (3 lack >= 4 kids). Ceiling 1.17 bits from ~10-13 others (convergence k=8: 1.14, still rising, so
  more underestimated than the adult ceiling). All three 28.2% of ceiling; shared 22.2%, unique motion +4.2%, meaning
  +4.8%, saliency -3.0% -- nearly identical to the adult-reference kids result (28.1%; 22.0/+3.7/+4.8/-2.3).

## 2026-10-23 — Feature–gaze correlation scripts archived

Moved `calc_motion_gaze_correlation.py`, `calc_saliency_gaze_correlation.py`, `calc_motion_saliency_partial_correlation.py`,
`calc_feature_gaze_partial_correlation.py` and `calc_feature_gaze_variance_partition.R` to
`scripts/analysis/archive/feature_gaze_correlation/` (with README). The question they asked -- which content features
explain gaze -- is now addressed by the information-gain analysis. PROJECT_ROOT paths updated so they still run.

## 2026-10-23 — Concordance archived; shared gaze-loading code moved to gaze_tiles.py

Concordance (`calc_concordance_group`/`_pairwise` .py/.R, `diagnostic_concordance.py`, `visualize_concordance_video.py`,
`plot_concordance_isc_timecourses.qmd`, `plot_individual_differences.qmd`, `calc_center_bias.R`) moved to
`scripts/analysis/archive/concordance/` (with README). Before moving, the helpers 17 other scripts imported from
`calc_concordance_group.py` (`load_video_fixations`, `build_participant_arrays`, `closed_form_tile_values`, `DISPLAY_W/H`)
were moved unchanged into `scripts/analysis/gaze_tiles.py`, and the ffmpeg/density-overlay helpers that
`visualize_entropy_segment.py` imported from `visualize_concordance_video.py` into `scripts/preprocessing/viz/video_overlay.py`.
All importers (ISC tile-peak, entropy, within/between, information gain, montages, and the archived scripts) repointed;
import check passes for all 18.

## 2026-10-23 — All plotting moves to R (project-wide rule); information gain converted first

User rule: all plotting is done in R, regardless of analysis; Python computes and writes CSVs. Scope agreed: charts only
for now (frame-overlay maps/montages and videos stay in Python). Information gain converted:
- `plots.py` removed; `plot_data.py` writes `plotdata_groups_<tag>.csv` (group values per facet: overall, block, quadrant,
  clip) and `plotdata_units_<tag>.csv` (per-segment / per-participant values divided by the facet's ceiling), plus
  `per_segment_` and `curves_` CSVs. `kids_adult_reference.py` calls the same export (tag `<set>_kids_adultref`).
- `baseline.py` writes `floor_<set>.csv`; `sanity_checks.py` writes `ceiling_convergence_<set>.csv`; `knots_sweep.py` and
  `explore_dense_grid.py` no longer plot (their CSVs already existed).
- `plot_information_gain.R --group adults|kids-kidref|kids-adultref [--segment-set] [--only]` draws all main figures;
  `plot_information_gain_checks.R [--group adults|kids]` draws floor maps, convergence, knot sweep, dense-grid check.
- Still Python: `check_meaning_point_smoothing.py` (map over video frames, out of current scope).
Remaining Python chart scripts to convert: within/between (`plot_quadrant_scatter`, `plot_quadrant_distribution`,
`plot_within_between_timecourse`), face-gaze `plot_results.py`.

## 2026-10-24 — Scene cuts frame-corrected (`all_clips_cuts_frame_corrected.csv`); fades split at their midpoint

**Problem.** The browser-annotated cuts in `annotations/scene-cuts/all_clips_cuts.csv` are systematically late.
`scripts/preprocessing/cut_annotation/check_cut_frame_accuracy.py` (→ `data/results/stimulus_prep/cut_frame_accuracy.csv`)
finds the true cut frame as the largest frame-to-frame change near each annotation (frame k at t = k/fps, the convention
the feature CSVs use). Of 145 clean hard cuts, 144 are annotated 0–1.5 frames after the first new-scene frame (median
20 ms, max 40 ms), so that frame fell in the previous segment. This is the same thing the 2-frame render margin
(entropy videos, above) was papering over.

**What it affected** (per-segment feature vectors, annotated vs. corrected cuts, 183 segments):
- **Motion: yes.** `load_features.segment_mean` drops a segment's first flow row because it spans the cut, but with late
  cuts the row that really spans the cut (a whole-frame scene change, so very large flow) stayed in the previous segment.
  Median +4% motion; up to 7× in short segments (sesameindia_1 seg 0, 0.8 s: r = 0.31 with its corrected map); 6
  segments r < 0.95.
- Saliency, meaning: negligible (all r ≥ 0.97). Meaning is sampled every 4th frame, so only ~1 cut in 4 put a sample on
  the wrong side.
- Gaze (IG, within/between, entropy): ≤ 40 ms moves between segments, below saccade latency; 4 within/between windows
  held ≤ 40 ms of post-cut time.

**Decision.** New file `annotations/scene-cuts/all_clips_cuts_frame_corrected.csv`, written by
`scripts/preprocessing/cut_annotation/correct_cut_frames.py`. The original annotation file is kept unchanged as the raw
record. Every boundary sits halfway between two frames, (k − 0.5)/fps, where k is the first frame of the new segment:
- **hard (149):** k = the true cut frame. This covers the 145 clean cuts plus 4 pixar cuts that fast motion made look
  ambiguous (verified on contact sheets; frame 1926 at 80.36 s overrides the detector). Shift −19 to −61 ms.
- **fade through black (18; all 12 slow-block cuts, and frank_complex/objects/play at ~20 s and ~40 s): split at the fade
  midpoint** (user's choice). The boundary goes after the middle of the darkest frames (luminance ≤ min + 1). Shift −0.6
  to −163 ms. The fade-out frames stay in the earlier segment and the fade-in frames in the later one.
- **dissolve (3; frank_objects 5.43, 18.74, 35.23 s):** k = the first frame ≥ 50% blended into the next shot, estimated by
  least squares and checked by eye (the blend estimate is noisy where objects move). Shift +76 to +153 ms.
- **unchanged (2):** pixar 64.77 s (a continuous camera tilt with no frame-level transition) and frank_complex 60.32 s
  (the clip end).

Optical flow during fades and dissolves is not removed. Brightness change and cross-blending can register as motion in
the few frames on either side of the boundary. Not quantified.

Scripts repointed to the corrected file: `information_gain/config.py`, `calc_within_between_cosine.py`, and the entropy
scripts (`calc_entropy_by_segment.py`, `calc_entropy_within_between.py`, `calc_entropy_sensitivity_sweep.py`,
`diagnostic_entropy_outlier_sensitivity.py`). Results regenerated: see the next entry when done.

## 2026-10-24 — Results regenerated with frame-corrected cuts

Reran within/between (cuts and slow_chopped), entropy_by_segment, the full IG pipeline for adults and kids
(+ kids_adult_reference, knots sweep, dense-grid checks) and all IG figures. Pre-fix outputs are not kept in the repo.
- **Within/between:** same 162 segments; old vs. new r = 0.994 (within) and 0.997 (between); median change 0, max 0.10
  and 0.06 (fade/dissolve segments whose boundaries moved most).
- **IG (adults, cuts, knots_floor):** ceiling 1.247 → 1.230 bits. All-three model 0.340 → 0.331 bits (27.3% → 26.9% of
  ceiling). Unique motion is unchanged (+0.038); unique meaning 0.083 → 0.080; shared 0.263 → 0.254. slow_chopped moves
  the same way (ceiling 1.346 → 1.331; all three 28.4% → 28.0%). The small, uniform drop includes the gaze-only
  ceiling. Segments now start at the true cut, so their first ~40 ms of gaze still reflects the previous shot (saccade
  latency), rather than the motion fix itself. No conclusion changes.
- Entropy scripts other than `calc_entropy_by_segment.py` (within_between, sensitivity sweep, outlier diagnostic) were
  repointed but not rerun.

## 2026-10-24 — Fixed block colors, project-wide

User choice: pixar yellow; frank, sesame, slow blue, green, red. Exact shades were picked to pass all-pairs CVD
separation (dataviz `validate_palette.js --pairs all`; worst pair slow↔sesame, protan ΔE 9.5). The darker green and
the vermilion red are what make red/green separable. An earlier coral red read as pink.
frank `#2a78d6`, pixar `#eda100`, sesame `#006414`, slow `#e85a32`. Defined once in `scripts/analysis/block_colors.R`,
which `plot_information_gain.R`, `plot_within_between_scatter.R`, `plot_within_between_marginals.R` and
`plot_entropy.qmd` source. Python `plot_quadrant_scatter.py` still has its old per-clip shades (pending R conversion).

## 2026-10-24 — Face feature: human-reviewed boxes, area-weighted

Faces = humans and Muppets only (no Pixar birds or other animals), per the user. YuNet (OpenCV, score ≥ 0.5) proposed
boxes on ~2 Hz review frames per segment (`prepare_face_review.py`); the user reviewed every segment of all 12 clips in
a local GUI (`face_review_server.py`), accepting/rejecting/drawing boxes. Muppets were almost all hand-drawn. The 5 clips
the user listed as face-free (slow_hands, slow_animals, slow_inanimate, frank_objects, pixar_birds) were pre-set to
no faces; detector-confident segments were flagged for review. All flags were false hits (hands, a train front,
birds). Annotations: `annotations/faces/<clip>_faces.json`.
- **Feature value = fraction of each 240px tile footprint covered by face boxes** (`calc_face_tiles.py`, same box-mean
  pooling as motion/saliency → `data/results/face/`), then segment mean, σ=46 smoothing and 120-tile sampling as for the
  other features. User decision: keep it **area-weighted** rather than a size-independent presence map, because a
  larger face is more likely to be looked at, and the probability map should reflect that. So small faces in wide shots
  (e.g. slow_people, frank_complex segments 2/3/6/7) carry little weight.
- Feature is sampled at review frames (~2 Hz), not every frame. Boxes are clipped at the frame edge.
- Visual check: `figures/face_maps/<clip>_face_map.png` (`montage_face_maps.py`).

## 2026-10-24 — Face added to IG as a 4th feature

`config.FEATURES` now saliency, motion, meaning, face (abbrev. F). Face is scaled like the others (corpus min-max).
Every subset is scored as a fixed equal-weight mixture (15 models). The partition is generalized to N features:
unique_k = all-four − the mixture without k, shared = the rest. The commonality formula for any subset (checked
against the old 3-feature formulas, exact match) gives common_<letters> columns, which replace common_all3/common_SM/....
plot_data writes `ig_all`/`prop_all` (= the all-features model) instead of `ig_SMX`/`prop_SMX`. Gaze, floor and ceiling
are untouched; all old 3-feature numbers reproduce exactly. Face color in figures `#b5459b` (passes all-pairs CVD).
- **Segments without faces stay in.** There face = 0 everywhere, so the face-only model = floor (IG 0). But the
  equal-weight all-four mixture is diluted toward the floor there, so unique face can be negative in those segments.
- **Adults, cuts, knots_floor (% of 1.23-bit ceiling):** face alone 15.4% (saliency 13.3, motion 19.1, meaning 24.0);
  shuffled face −18.1%; all three 26.9% → all four 28.4%; unique face +1.5%. Uniques shrink because they are now
  over the other three: meaning 6.5 → 3.9%, motion 3.1 → 1.3%, saliency −3.4 → −1.9%; shared 20.7 → 23.7%.
  slow_chopped: face alone 12.3%, all three 28.0% → all four 28.1%, unique face +0.0%. Dense grid (9975 locations):
  unique face +1.3%; coarse grid (28): +0.8%.
- **Kids (kids reference):** face alone 12.2% (cuts) / 9.0% (slow_chopped); unique face −0.0% / −1.6%.
  Kids vs adult models: unique face −0.4%.
- Fitted face curve: smooth and monotone, ~40× at full coverage, consistent across held-out clips.

## 2026-10-24 — Face analyzed on face clips only; all-clips analysis stays 3-feature

User decision: all 12 clips → saliency, motion, meaning (+ all three); the face analysis uses only the 7 clips with faces
(frank_complex, frank_play, sesameindia_1/2, sesameus_1/2, slow_people) and all four features. This removes the
dilution from face-free clips described in the previous entry. `config.SUBSETS` drives this:
partition/plot_data/kids_adult_reference/explore_dense_grid write `<tag>` (all clips, 3 features) and
`<tag>_faceclips`. `plot_information_gain.R --subset all|faceclips` draws each. The face-clip numbers reuse the
same leave-one-clip-out fits (curves trained on all other clips); only the scored cells are restricted. The mixtures
have fixed equal weights, and face-free segments carry no information for the face curve, so no refit is needed.
Some segments within the face clips still have no faces (they stay in).
- **Adults, cuts (113 segments, ceiling 1.29 bits):** saliency 5.1%, motion 15.3%, meaning 21.5%, **face 25.1%**,
  all three 23.7%, all four 29.5%; unique face +5.9%, meaning +3.9%, motion +0.0%, saliency −4.0%; shuffled face −14.1%.
  slow_chopped: face 22.5%, all four 29.8%, unique face +4.4%. Dense grid 9975: unique face +5.5%; coarse 28: +4.7%.
- **Kids (kids reference), cuts:** face 19.6% (meaning 18.3, motion 18.1); all three 23.2% → all four 27.0%; unique
  face +3.8%. slow_chopped +1.9%. Kids vs adult models: unique face +3.3%.
- All-clips (3-feature) numbers are unchanged from before face was added.
- Added facet "faces" (clips with vs. without faces) to plot_data; `supplementary_by_faces_<tag>.png` shows the
  3-feature bars per group, each against its own ceiling. Adults, cuts: with faces 1.29-bit ceiling, saliency 5% /
  motion 15% / meaning 22% / all three 24%; without faces 1.14-bit ceiling, 26% / 25% / 29% / 32%.
- Added subset `_allclips4` (all 12 clips, 4 features) for the faces facet only: `overall_ig[_people]_by_faces_<tag>_allclips4.png`.
  Adults, cuts: in face-free clips, unique face −5.5% (pure dilution), all four 26.5% vs all three 32.0%. In face clips
  the numbers equal `_faceclips`.

## 2026-10-25 — Center-bias check: uniform baseline, spatial prior as a feature

Prompted by a reviewer-style concern: every feature model is feature × floor (β=1), so all models share the floor,
and "shared" may partly be that shared ingredient rather than content overlap. The knots_nofloor variant (β=0,
still scored against the floor) can't settle this: every model then underperforms the floor, all IGs go negative,
and shared (−42.6%) mostly reflects the floor that all three models lack (three-way overlap +8.9% → −61%).
Pairwise overlaps don't shrink without the floor (SM 3.2 → 4.6, SX 4.6 → 4.6, MX 4.0 → 9.6%).
User decision: make uniform the baseline ("floor") and enter the old floor as a 4th feature, "spatial" (P).
`uniform_baseline.py` scores all 15 equal-weight mixtures of P + the no-floor S/M/X models over uniform. The ceiling is
the same model as before, re-expressed over uniform (ig_ceiling + ll_floor − ll_uniform). Figures:
`overall_ig[_people]_<set>_uniform.png`, `supplementary_<set>_uniform.png` (`plot_information_gain.R --variant uniform`).
Spatial color `#5a3e9e` (passes all-pairs CVD). partition.py generalized to any feature letters (existing outputs
byte-identical).
- **Adults, cuts (ceiling 2.30 bits over uniform):** spatial 46.6%, saliency 14.7%, motion 26.3%, meaning 26.6%;
  S+M+X 29.9%; all four 41.1% (below spatial alone, since equal weighting dilutes the strongest component). Shared
  34.5%: **+35.5% involves spatial, −0.9% is overlap among saliency/motion/meaning alone**. Unique: spatial +11.2%,
  meaning +0.5%, motion −0.2%, saliency −5.0%. slow_chopped is the same pattern (shared 34.3% = +33.9% with spatial,
  +0.5% content-only).
- Interpretation: under this decomposition, the shared bucket in the floor-based analysis is almost entirely
  center-bias overlap. Caveat: fixed equal-weight mixtures; when one component dominates, adding weaker ones lowers IG,
  which pushes their unique shares negative. Fitted mixture weights would be needed for a fairer partition.

## 2026-10-25 — Within/between quadrants: kids vs. adults

- `calc_within_between_cosine.py --group kids` (same kid set as IG: all kid files minus MWK005; 7-14 kids per segment,
  median 11, vs. adult median 35) -> `all_clips_by_segment_kids.csv`. 162 segments valid in both groups.
- `compare_quadrants_kids_adults.py`: each group classified on its own pooled thresholds. Across segments, adult-kid
  r = 0.86 (within) and 0.79 (between). Same quadrant: 68/75 (91%) of segments classified in both at q=0.4 (the
  convention); 115/162 (71%) with a median split (chance ~25%). Kids on adult thresholds: 88% / 66%.
- Caveat: between-person distance uses a leave-one-out mean of ~10 kids vs ~34 adults, so kid values carry more
  sampling noise (kids' means slightly higher: within 0.686 vs 0.630, between 0.357 vs 0.338). Adults not subsampled to kid N.

## 2026-10-25 — IG confidence intervals: segment bootstrap; CI + waterfall figure

- `information_gain/bootstrap_ci.py --unit participant|segment|clip`: 2,000 resamples; each draw recomputes model group
  averages (mean of participant means), the ceiling, % of ceiling and the partition; percentile 95% CIs. Models not refit.
- Adults, unchopped, 3 features: segment-level CIs are ~3-4x wider than participant-level (stimulus variability
  dominates); all uniques exclude 0 at every level (unique motion at clip level 0.5-6.0%). Segment level chosen for
  figures (clip level = 12 units, too few for precise percentiles; kept as a robustness check).
- `plot_ig_ci_waterfall.R` (default `--unit segment`): (A) single models + all three with CIs, (B) waterfall shared ->
  unique salience -> motion -> semantics -> all three. Slide colors/names (salience #1e4c7c, motion #357a76, semantics
  #f09937); proposed replacement for the violin "overall" figure.
- Grouped CIs: `bootstrap_ci.py --by block|quadrant|face` bootstraps segments within each group; `--q 0.5` classifies
  quadrants on a median split (default 0.4 leaves the middle band unclassified). User chose 50/50 for quadrant figures
  (162 segments classified: 36 selective, 45 scattered, 45 concentrated, 36 coordinated). Adults, % of ceiling, all three
  (shared): selective 15.2 (6.8), scattered 26.5 (19.6), concentrated 30.5 (24.6), coordinated 31.8 (26.9). Face clips
  (`config.FACE_CLIPS`, 3-feature models) 23.7 (16.1, unique semantics +9.0) vs no-face 32.0 (28.0, +3.0); confounded
  with block. Figures: `ig_waterfall_by_{block,quadrant,quadrant_q50,face}_cuts_knots_floor.png`.
- "Shared" = all 2-way + 3-way commonalities (unique_k = loss from dropping only k). Adults: SM 3.2 + SX 4.6 + MX 4.0
  + SMX 8.9 = 20.7%. Plotted as one bar; split not shown.

## 2026-10-26 — IG kids vs. adults (kid reference), all / by block / by adult quadrant

- Kids scored on the kids-only pipeline (`IG_GROUP=kids`: own floor, curves, ceiling), segment bootstrap as adults.
  Quadrants for both groups = adult 50/50 classification. 19 kids total, 9-14 per clip (adults ~35-41).
- All blocks, adults vs kids, % of ceiling: all three 26.9 vs 28.1; shared 20.7 vs 22.1; unique motion +3.1 vs +4.1,
  unique semantics +6.5 vs +4.7. Per block, every component within ~2 points; blocks differ far more than groups.
  Only adult-selective segments show kids lower (10.1 vs 15.2; kid CIs very wide). Not a finding at this N.
- `plot_ig_waterfall_kids_adults.R` (`--block`) -> `figures/information_gain/kids_vs_adults/`.
- Least-adultlike segments (`montage_kids_vs_adults_mismatch.py`): mismatch = kid-to-adult minus kid-kid cosine distance
  (kids agreeing with each other but not with adults). Kid-adult distance tracks kid-kid r = 0.96, so kids' distance from
  adults is mostly their own scatter. One outlier: frank_play seg17 (~4 SD; adults on the face, kids on the pencil/hand).
  No null distribution yet.
- Kid ages at test (from the user's DOB sheet, not stored in the repo): 17 of 19 with DOB, mean 7.1 y (SD 1.0), median
  7.7, range 5.3-8.2; 12 DOBs month-only (15th assumed); MWK008/MWK009 missing. Narrow, older range: a within-sample
  age slope (development-slope prediction) is not estimable; would need younger children.

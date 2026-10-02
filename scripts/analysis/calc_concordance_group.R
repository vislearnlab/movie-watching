## calc_concordance_group.R
##
## R port of calc_concordance_group.py -- individual-to-group fixation density
## similarity metric ("concordance", NEVER "ISC" -- ISC is reserved for the Franchak-style
## x/y time-series analysis in calc_isc_rawgaze.py/calc_isc_fixations.py/isc_analysis.qmd).
##
## - For every participant and sliding window, computes a duration-weighted Gaussian
##   density value at each of 120 scene-tile centers (same patch grid as
##   scripts/preprocessing/stimulus_prep/plot_patch_grid.py: 180px diameter, 15x8,
##   edge-touching), for that participant and for the leave-one-out group, then correlates
##   the two 120-value vectors (Pearson r).
## - Computed in CLOSED FORM (no raster image, no Gaussian blur of a full frame): each
##   tile's value is sum over overlapping fixations of
##   overlap_weight_ms * Gaussian(distance_to_center, sigma) -- verified mathematically
##   and empirically identical to rasterize-then-blur-then-sample (see
##   figures/checks/closed_form_vs_raster_blur_verification.png, METHODS_DECISIONS.md).
## - Leave-one-out group vector = total_vec - individual_vec (exact, by linearity of the
##   closed-form sum), not a resummation over every other participant per participant.
## - Confirmed parameters (METHODS_DECISIONS.md): sigma=46px fixed, window=1.0s,
##   step=0.25s, no minimum-data gate, no bound/percentile filtering, no 0-1 rescaling.
## - Reads fixation data from a LOCAL COPY only -- never the live server mount.
##
## Loops over every clip in VIDEOS (default: all 12) in one run and writes ONE combined
## CSV (unlike the Python version, which wrote one CSV per invocation and needed an outer
## shell loop + a separate concat step to cover all clips).
##
## Output: data/results/concordance/fixation_density_concordance.csv -- one row per (pid, video, window):
##   pid, video_name, window_idx, window_start_s, window_end_s, r

library(dplyr)
library(readr)
library(purrr)
library(here)

## ---- Config -----------------------------------------------------------------
FIXATION_DIR <- here("data", "local_cache", "fixations", "adults")
PARTICIPANT_SUMMARY_PATH <- here("data", "local_cache", "participant_summary.csv")
PARAM_CODE <- "f329476c"
## NOTE: data/ is a symlink to the mounted lab server in this repo, so this writes
## through to shared storage, not just the local checkout.
OUTPUT_PATH <- here("data", "results", "concordance", "fixation_density_concordance.csv")

DISPLAY_W <- 1920
DISPLAY_H <- 1080
SIGMA_PX <- 46.0
WINDOW_S <- 1.0
STEP_S <- 0.25
PATCH_DIAMETER_PX <- 180
PATCH_N_COLS <- 15

## All 12 clips, with exact durations from data/results/stimulus_prep/stimulus_inventory.csv.
inventory <- read_csv(here("data", "results", "stimulus_prep", "stimulus_inventory.csv"), show_col_types = FALSE)
VIDEOS <- setNames(inventory$duration_sec, inventory$video_name)

## ---- Patch grid (port of patch_grid_preview.compute_patch_grid) -------------
compute_patch_grid <- function(diameter_px, n_cols, display_width, display_height) {
  stopifnot(n_cols >= 2)
  radius <- diameter_px / 2
  usable_w <- display_width - diameter_px
  usable_h <- display_height - diameter_px
  stopifnot(usable_w > 0, usable_h > 0)

  spacing_x <- usable_w / (n_cols - 1)
  n_rows <- max(2, round(usable_h / spacing_x) + 1)
  spacing_y <- usable_h / (n_rows - 1)

  centers <- expand.grid(row = 0:(n_rows - 1), col = 0:(n_cols - 1))
  centers$center_x <- radius + centers$col * spacing_x
  centers$center_y <- radius + centers$row * spacing_y

  list(
    centers = centers,
    info = list(n_cols = n_cols, n_rows = n_rows, spacing_x = spacing_x, spacing_y = spacing_y,
                ratio_x = diameter_px / spacing_x, ratio_y = diameter_px / spacing_y,
                diameter_px = diameter_px, radius = radius)
  )
}

## ---- Data loading -------------------------------------------------------------
load_video_fixations <- function(fixation_dir, video, param_code, participant_summary_path) {
  pattern <- paste0("_", video, "_", param_code, "\\.csv$")
  fix_paths <- list.files(fixation_dir, pattern = pattern, full.names = TRUE)
  stopifnot(length(fix_paths) > 0)

  summary <- read_csv(participant_summary_path, show_col_types = FALSE)
  valid_ids <- summary %>%
    filter(participant_type == "adults", valid_data == TRUE) %>%
    pull(participant_id)

  fix_df <- map_dfr(fix_paths, read_csv, show_col_types = FALSE) %>%
    filter(pid %in% valid_ids)

  cat(sprintf("Loaded %d fixations from %d valid adults for %s.\n",
              nrow(fix_df), n_distinct(fix_df$pid), video))
  fix_df
}

## {pid: list(startT, endT, img_x, img_y)} -- img_y flipped to top-left origin once, up
## front, matching the Tobii-coordinate convention used throughout this project.
build_participant_arrays <- function(fix_df, display_height) {
  split_df <- split(fix_df, fix_df$pid)
  lapply(split_df, function(g) {
    list(startT = g$startT, endT = g$endT, img_x = g$xpos, img_y = display_height - g$ypos)
  })
}

## ---- Closed-form tile values ---------------------------------------------------
## Duration-weighted Gaussian density evaluated directly at each tile center -- no raster
## image is ever built. Vectorized via outer() over (fixation x tile) pairs; `w * gauss`
## relies on R recycling a length-n_fixation vector down the ROWS of an
## (n_fixation x n_tile) matrix, which scales each fixation's row by its own weight.
closed_form_tile_values <- function(startT, endT, img_x, img_y, window_start_ms, window_end_ms,
                                     sigma_px, centers_x, centers_y) {
  overlap_ms <- pmin(endT, window_end_ms) - pmax(startT, window_start_ms)
  mask <- overlap_ms > 0
  if (!any(mask)) return(rep(0, length(centers_x)))

  x <- img_x[mask]; y <- img_y[mask]; w <- overlap_ms[mask]
  dist_sq <- outer(x, centers_x, "-")^2 + outer(y, centers_y, "-")^2
  gauss <- exp(-dist_sq / (2 * sigma_px^2)) / (2 * pi * sigma_px^2)
  colSums(w * gauss)
}

## ---- Per-video batch ------------------------------------------------------------
run_one_video <- function(video, duration_s) {
  fix_df <- load_video_fixations(FIXATION_DIR, video, PARAM_CODE, PARTICIPANT_SUMMARY_PATH)

  grid <- compute_patch_grid(PATCH_DIAMETER_PX, PATCH_N_COLS, DISPLAY_W, DISPLAY_H)
  cx <- grid$centers$center_x
  cy <- grid$centers$center_y
  n_tiles <- nrow(grid$centers)

  pid_arrays <- build_participant_arrays(fix_df, DISPLAY_H)
  pids <- names(pid_arrays)

  n_windows <- max(0, as.integer((duration_s - WINDOW_S) %/% STEP_S) + 1)
  cat(sprintf("%s: %d participants, %d windows (window=%.2fs, step=%.2fs)\n",
              video, length(pids), n_windows, WINDOW_S, STEP_S))

  rows <- vector("list", n_windows)
  for (w in 0:(n_windows - 1)) {
    start_s <- w * STEP_S
    end_s <- start_s + WINDOW_S
    start_ms <- start_s * 1000
    end_ms <- end_s * 1000

    per_pid_vec <- vector("list", length(pids))
    names(per_pid_vec) <- pids
    total_vec <- numeric(n_tiles)
    for (pid in pids) {
      a <- pid_arrays[[pid]]
      vec <- closed_form_tile_values(a$startT, a$endT, a$img_x, a$img_y,
                                      start_ms, end_ms, SIGMA_PX, cx, cy)
      per_pid_vec[[pid]] <- vec
      total_vec <- total_vec + vec
    }

    r_vals <- numeric(length(pids))
    for (i in seq_along(pids)) {
      vec <- per_pid_vec[[i]]
      group_vec <- total_vec - vec
      if (sd(vec) == 0 || sd(group_vec) == 0) {
        r_vals[i] <- NA_real_
      } else {
        r_vals[i] <- cor(vec, group_vec)
      }
    }

    rows[[w + 1]] <- tibble(pid = pids, video_name = video, window_idx = w,
                             window_start_s = round(start_s, 3), window_end_s = round(end_s, 3),
                             r = r_vals)
  }

  out_df <- bind_rows(rows)
  cat(sprintf("  -> %d rows, NaN rate %.1f%%\n\n", nrow(out_df), 100 * mean(is.na(out_df$r))))
  out_df
}

## ---- Run all videos, write combined CSV -----------------------------------------
main <- function() {
  t0 <- Sys.time()
  all_results <- imap(VIDEOS, ~ run_one_video(.y, .x))  # imap passes (value, name); run_one_video wants (video, duration)
  combined <- bind_rows(all_results)
  write_csv(combined, OUTPUT_PATH)
  cat(sprintf("Wrote %d rows (%d videos) to %s (%.1fs total)\n",
              nrow(combined), length(VIDEOS), OUTPUT_PATH,
              as.numeric(Sys.time() - t0, units = "secs")))
}

if (sys.nframe() == 0) main()

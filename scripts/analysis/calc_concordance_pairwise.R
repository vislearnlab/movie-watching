## calc_concordance_pairwise.R
##
## R port of calc_concordance_pairwise.py -- same "concordance" metric as
## calc_concordance_group.R, but structured like the real ISC scripts (every pair of
## participants correlated directly, not individual-vs-leave-one-out-pooled-group).
##
## Still "concordance", never "ISC" -- see calc_concordance_group.R's header for the
## naming rule and the closed-form methodology this builds on (reused verbatim here).
##
## - For every window, every pair of participants' own 120-tile closed-form density
##   vectors are correlated directly (Pearson r), then averaged across all pairs.
## - All pairwise correlations for a window come from ONE cor() call on the
##   (n_tiles x n_participants) matrix (R's cor() correlates COLUMNS, unlike numpy's
##   corrcoef which correlates ROWS by default -- so we build the matrix tile-by-participant,
##   the transpose of the Python version's participant-by-tile layout, to get the same
##   n_participants x n_participants result from one cor() call without an extra transpose).
## - Same parameters as calc_concordance_group.R: sigma=46px, window=1.0s,
##   step=0.25s, no minimum-data gate.
##
## Loops over every clip in VIDEOS (default: all 12) in one run.
##
## Output: data/results/concordance/fixation_density_concordance_pairwise.csv -- one row per (video, window):
##   video_name, window_idx, window_start_s, window_end_s, mean_r, n_pairs

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
OUTPUT_PATH <- here("data", "results", "concordance", "fixation_density_concordance_pairwise.csv")

DISPLAY_W <- 1920
DISPLAY_H <- 1080
SIGMA_PX <- 46.0
WINDOW_S <- 1.0
STEP_S <- 0.25
PATCH_DIAMETER_PX <- 180
PATCH_N_COLS <- 15

inventory <- read_csv(here("data", "results", "stimulus_prep", "stimulus_inventory.csv"), show_col_types = FALSE)
VIDEOS <- setNames(inventory$duration_sec, inventory$video_name)

## ---- Shared helpers (identical to calc_concordance_group.R) --------------
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
  list(centers = centers, info = list(n_cols = n_cols, n_rows = n_rows))
}

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

build_participant_arrays <- function(fix_df, display_height) {
  split_df <- split(fix_df, fix_df$pid)
  lapply(split_df, function(g) {
    list(startT = g$startT, endT = g$endT, img_x = g$xpos, img_y = display_height - g$ypos)
  })
}

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
  pid_arrays <- build_participant_arrays(fix_df, DISPLAY_H)
  pids <- names(pid_arrays)
  n_pids <- length(pids)
  n_possible_pairs <- n_pids * (n_pids - 1) / 2

  grid <- compute_patch_grid(PATCH_DIAMETER_PX, PATCH_N_COLS, DISPLAY_W, DISPLAY_H)
  cx <- grid$centers$center_x
  cy <- grid$centers$center_y
  n_tiles <- nrow(grid$centers)

  n_windows <- max(0, as.integer((duration_s - WINDOW_S) %/% STEP_S) + 1)
  cat(sprintf("%s: %d participants -> %d pairs/window, %d windows\n",
              video, n_pids, n_possible_pairs, n_windows))

  upper_idx <- which(upper.tri(matrix(0, n_pids, n_pids)))

  rows <- vector("list", n_windows)
  for (w in 0:(n_windows - 1)) {
    start_s <- w * STEP_S
    end_s <- start_s + WINDOW_S
    start_ms <- start_s * 1000
    end_ms <- end_s * 1000

    ## tile x participant matrix, so cor() (which correlates COLUMNS) directly gives the
    ## n_participants x n_participants pairwise correlation matrix we want.
    vecs <- vapply(pids, function(pid) {
      a <- pid_arrays[[pid]]
      closed_form_tile_values(a$startT, a$endT, a$img_x, a$img_y, start_ms, end_ms, SIGMA_PX, cx, cy)
    }, numeric(n_tiles))

    corr_matrix <- suppressWarnings(cor(vecs))  # n_pids x n_pids, NA for zero-variance columns
    pair_r <- corr_matrix[upper_idx]
    valid <- !is.na(pair_r)
    mean_r <- if (any(valid)) mean(pair_r[valid]) else NA_real_

    rows[[w + 1]] <- tibble(video_name = video, window_idx = w,
                             window_start_s = round(start_s, 3), window_end_s = round(end_s, 3),
                             mean_r = mean_r, n_pairs = sum(valid))
  }

  out_df <- bind_rows(rows)
  cat(sprintf("  -> %d rows, mean_r=%.3f, NaN rate %.1f%%\n\n",
              nrow(out_df), mean(out_df$mean_r, na.rm = TRUE), 100 * mean(is.na(out_df$mean_r))))
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

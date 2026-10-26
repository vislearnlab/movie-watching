## calc_center_bias.R
##
## Windowed "center bias" indicator, to check whether a metric's cut-locked behavior
## (see plot_concordance_isc_timecourses.qmd Part 4/6) is driven by viewers reflexively
## looking toward screen center right after a cut, rather than genuine content-driven
## synchrony.
##
## For each participant and sliding window, computes a duration-weighted mean Euclidean
## distance from screen center (960, 540) across that participant's overlapping fixations,
## then averages across participants for a single group-level number per window -- lower
## distance = more central clustering = stronger center bias. Same window=1.0s, step=0.25s
## grid as calc_concordance_group.R/calc_concordance_pairwise.R, so it aligns on
## window_start_s with those outputs and with isc_fixation_timecourse.csv for direct
## comparison and cut-locked realignment.
##
## Reads fixation data from a LOCAL COPY only -- never the live server mount.
##
## Output: data/results/concordance/center_bias_timecourse.csv -- one row per (video, window):
##   video_name, window_idx, window_start_s, window_end_s, mean_dist_px, n_participants

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
OUTPUT_PATH <- here("data", "results", "concordance", "center_bias_timecourse.csv")

DISPLAY_W <- 1920
DISPLAY_H <- 1080
CENTER_X <- DISPLAY_W / 2
CENTER_Y <- DISPLAY_H / 2
WINDOW_S <- 1.0
STEP_S <- 0.25

inventory <- read_csv(here("data", "results", "stimulus_prep", "stimulus_inventory.csv"), show_col_types = FALSE)
VIDEOS <- setNames(inventory$duration_sec, inventory$video_name)

## ---- Data loading (identical to calc_concordance_group.R) --------------------
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

## Duration-weighted mean distance from screen center, over fixations overlapping the
## window, for ONE participant. NA if the participant has no fixation data in this window.
weighted_center_distance <- function(startT, endT, img_x, img_y, window_start_ms, window_end_ms,
                                      center_x, center_y) {
  overlap_ms <- pmin(endT, window_end_ms) - pmax(startT, window_start_ms)
  mask <- overlap_ms > 0
  if (!any(mask)) return(NA_real_)

  w <- overlap_ms[mask]
  dist <- sqrt((img_x[mask] - center_x)^2 + (img_y[mask] - center_y)^2)
  sum(w * dist) / sum(w)
}

## ---- Per-video batch ------------------------------------------------------------
run_one_video <- function(video, duration_s) {
  fix_df <- load_video_fixations(FIXATION_DIR, video, PARAM_CODE, PARTICIPANT_SUMMARY_PATH)
  pid_arrays <- build_participant_arrays(fix_df, DISPLAY_H)
  pids <- names(pid_arrays)

  n_windows <- max(0, as.integer((duration_s - WINDOW_S) %/% STEP_S) + 1)
  cat(sprintf("%s: %d participants, %d windows\n", video, length(pids), n_windows))

  rows <- vector("list", n_windows)
  for (w in 0:(n_windows - 1)) {
    start_s <- w * STEP_S
    end_s <- start_s + WINDOW_S
    start_ms <- start_s * 1000
    end_ms <- end_s * 1000

    per_pid_dist <- vapply(pids, function(pid) {
      a <- pid_arrays[[pid]]
      weighted_center_distance(a$startT, a$endT, a$img_x, a$img_y, start_ms, end_ms, CENTER_X, CENTER_Y)
    }, numeric(1))

    rows[[w + 1]] <- tibble(video_name = video, window_idx = w,
                             window_start_s = round(start_s, 3), window_end_s = round(end_s, 3),
                             mean_dist_px = mean(per_pid_dist, na.rm = TRUE),
                             n_participants = sum(!is.na(per_pid_dist)))
  }

  out_df <- bind_rows(rows)
  cat(sprintf("  -> %d rows, mean_dist_px=%.1f\n\n", nrow(out_df), mean(out_df$mean_dist_px, na.rm = TRUE)))
  out_df
}

## ---- Run all videos, write combined CSV -----------------------------------------
main <- function() {
  t0 <- Sys.time()
  all_results <- imap(VIDEOS, ~ run_one_video(.y, .x))
  combined <- bind_rows(all_results)
  write_csv(combined, OUTPUT_PATH)
  cat(sprintf("Wrote %d rows (%d videos) to %s (%.1fs total)\n",
              nrow(combined), length(VIDEOS), OUTPUT_PATH,
              as.numeric(Sys.time() - t0, units = "secs")))
}

if (sys.nframe() == 0) main()

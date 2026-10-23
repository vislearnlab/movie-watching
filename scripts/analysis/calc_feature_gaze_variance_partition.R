## calc_feature_gaze_variance_partition.R
##
## Per-segment UNIQUE variance in the group gaze tile-vector contributed by each content
## feature (motion, saliency, meaning, center), controlling for all other active features
## jointly -- an R, paper-track extension of calc_feature_gaze_partial_correlation.py's
## partial-correlation framework (chat discussion 2026-10-22). Instead of pairwise partial r,
## this fits gaze ~ all features jointly (R2_full) vs. gaze ~ all features except one
## (R2_reduced), and reports:
##   unique_R2(X) = R2_full - R2_reduced(X)  -- X's unique contribution. Always >= 0
##                   in-sample (nested OLS models: dropping a predictor can't raise R2).
##   shared       = R2_full - sum(unique_R2) -- variance explainable but not uniquely
##                   attributable to any one feature. CAN go negative under strong
##                   multicollinearity (a known property of commonality analysis, not a
##                   bug). Reported as ONE number, not decomposed into per-pair/triple
##                   terms -- avoids the higher-order suppression-effect interpretability
##                   problem that gives full variance-partitioning decompositions their bad
##                   reputation (see chat discussion 2026-10-22).
##
## NO null/floor computation yet, deliberately (as of 2026-10-22): an earlier version ported
## the Python partial-correlation script's shuffled-segment-pairing null directly, but that
## null is degenerate for `center` -- center bias is a fixed constant vector (identical
## across every segment), so "shuffling" it swaps in the exact same values, making its
## "null" reflect real signal rather than chance (shuffling motion/saliency, which DO vary
## by segment, is fine). Pulled out while the right floor/ceiling definition gets more
## thought -- see chat discussion 2026-10-22.
##
## IMPORTANT tile-ordering fix vs. calc_concordance_group.R: that script's
## expand.grid(row=, col=) varies row fastest (column-major) -- harmless there since it only
## self-correlates gaze against gaze, but motion/saliency/meaning's tile_XXX CSV columns are
## Python row-major (col varies fastest within a row). This script's compute_patch_grid()
## builds centers in that same col-fastest order, verified against Python's traversal
## directly, so tile index i means the same physical tile on both sides of every merge here.
## calc_concordance_group.R's fixation-loading functions are duplicated below rather than
## sourced for the same reason: that file also defines its own WINDOW_S (1.0s, a different
## quantity -- its sliding-window step, not this script's segment-aggregation window), which
## would silently collide with this script's WINDOW_S (0.5s) if sourced wholesale.
##
## Usage:
##   Rscript scripts/analysis/calc_feature_gaze_variance_partition.R
##   Rscript scripts/analysis/calc_feature_gaze_variance_partition.R --features motion,saliency,center
##
## Output: data/results/variance_partition/feature_gaze_variance_partition_<tag>[_slowunchopped].csv
##   one row per segment: video, block, seg_idx, r2_full, shared, unique_<feature>...
## (no null/floor columns yet -- see note above).

library(dplyr)
library(readr)
library(purrr)
library(here)

## ---- Config --------------------------------------------------------------------------------
WINDOW_S <- 0.5
MIN_WINDOWS <- 2
SIGMA <- 46.0
MAX_FIXATION_MS <- 2000.0
DISPLAY_W <- 1920
DISPLAY_H <- 1080
PARAM_CODE <- "f329476c"

CANONICAL_PATH <- here("data", "results", "within_between_cosine", "all_clips_by_segment.csv")
SLOW_CHOPPED_PATH <- here("data", "results", "within_between_cosine", "slow_chopped_by_subsegment.csv")
MOTION_DIR <- here("data", "results", "motion")
SALIENCY_DIR <- here("data", "results", "saliency")
MEANING_DIR <- here("data", "results", "meaning")
FIXATION_DIR <- here("data", "local_cache", "fixations", "adults")
PARTICIPANT_SUMMARY_PATH <- here("data", "local_cache", "participant_summary.csv")
OUT_DIR <- here("data", "results", "variance_partition")

ALL_CLIPS <- c("frank_complex", "frank_objects", "frank_play", "pixar_birds",
               "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2",
               "slow_animals", "slow_hands", "slow_inanimate", "slow_people")
CHOPPED_CLIPS <- c("slow_animals", "slow_hands", "slow_inanimate", "slow_people")
BLOCKS <- c(frank_complex = "frank", frank_objects = "frank", frank_play = "frank",
            pixar_birds = "pixar",
            sesameindia_1 = "sesame", sesameindia_2 = "sesame", sesameus_1 = "sesame", sesameus_2 = "sesame",
            slow_animals = "slow", slow_hands = "slow", slow_inanimate = "slow", slow_people = "slow")

## ---- Patch grid, col-fastest (see header note) --------------------------------------------
compute_patch_grid <- function(diameter_px, n_cols, display_width, display_height) {
  radius <- diameter_px / 2
  usable_w <- display_width - diameter_px
  usable_h <- display_height - diameter_px
  spacing_x <- usable_w / (n_cols - 1)
  n_rows <- max(2, round(usable_h / spacing_x) + 1)
  spacing_y <- usable_h / (n_rows - 1)

  centers <- expand.grid(col = 0:(n_cols - 1), row = 0:(n_rows - 1))[, c("row", "col")]
  centers$center_x <- radius + centers$col * spacing_x
  centers$center_y <- radius + centers$row * spacing_y
  list(centers = centers, n_rows = n_rows, n_cols = n_cols)
}

.grid <- compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)  ## 2026-10-22: moved from 180px to 240px (50% overlap)
cx <- .grid$centers$center_x
cy <- .grid$centers$center_y
N_TILES <- length(cx)
stopifnot(N_TILES == 120)

## ---- Center bias: fixed anisotropic Gaussian, same params as the Python pipeline
## (sigma_x=384, sigma_y=216 -- approved 2026-10-22, METHODS_DECISIONS.md) --------------------
.cx0 <- DISPLAY_W / 2; .cy0 <- DISPLAY_H / 2
.sigma_x <- DISPLAY_W / 4 * 0.8; .sigma_y <- DISPLAY_H / 4 * 0.8
CENTER_BIAS_VEC <- exp(-((cx - .cx0)^2 / (2 * .sigma_x^2) + (cy - .cy0)^2 / (2 * .sigma_y^2)))

## ---- Fixation loading (duplicated from calc_concordance_group.R -- see header note) --------
load_video_fixations <- function(fixation_dir, video, param_code, participant_summary_path) {
  pattern <- paste0("_", video, "_", param_code, "\\.csv$")
  fix_paths <- list.files(fixation_dir, pattern = pattern, full.names = TRUE)
  stopifnot(length(fix_paths) > 0)

  summary <- read_csv(participant_summary_path, show_col_types = FALSE)
  valid_ids <- summary %>%
    filter(participant_type == "adults", valid_data == TRUE) %>%
    pull(participant_id)

  map_dfr(fix_paths, read_csv, show_col_types = FALSE) %>%
    filter(pid %in% valid_ids)
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

## ---- Gaze: per-pid, per-0.5s-window 120-tile density (port of
## montage_best_examples.build_clip_vectors) --------------------------------------------------
build_clip_vectors <- function(video) {
  fix <- load_video_fixations(FIXATION_DIR, video, PARAM_CODE, PARTICIPANT_SUMMARY_PATH) %>%
    filter((endT - startT) <= MAX_FIXATION_MS)
  arrays <- build_participant_arrays(fix, DISPLAY_H)
  pids <- sort(names(arrays))

  canonical <- read_csv(CANONICAL_PATH, show_col_types = FALSE)
  real_segs <- canonical %>% filter(video == !!video) %>% arrange(seg_idx)
  dur <- max(real_segs$seg_end_s)
  n_windows_total <- as.integer(dur %/% WINDOW_S)

  all_vecs <- setNames(vector("list", length(pids)), pids)
  for (pid in pids) {
    a <- arrays[[pid]]
    mat <- matrix(0, nrow = n_windows_total, ncol = N_TILES)
    for (w in 0:(n_windows_total - 1)) {
      s_ms <- w * WINDOW_S * 1000; e_ms <- (w + 1) * WINDOW_S * 1000
      mat[w + 1, ] <- closed_form_tile_values(a$startT, a$endT, a$img_x, a$img_y, s_ms, e_ms, SIGMA, cx, cy)
    }
    all_vecs[[pid]] <- mat
  }
  list(pids = pids, all_vecs = all_vecs, n_windows_total = n_windows_total, real_segs = real_segs)
}

group_mean_vector <- function(pids, seg_mats) {
  own_means <- list()
  for (pid in pids) {
    m <- seg_mats[[pid]]
    valid_w <- which(rowSums(m) > 0)
    if (length(valid_w) < MIN_WINDOWS) next
    own_means[[length(own_means) + 1]] <- colMeans(m[valid_w, , drop = FALSE])
  }
  if (length(own_means) < 3) return(NULL)
  colMeans(do.call(rbind, own_means))
}

segment_list_for <- function(video, slow_chopped) {
  path <- if (!(video %in% CHOPPED_CLIPS) || !slow_chopped) CANONICAL_PATH else SLOW_CHOPPED_PATH
  read_csv(path, show_col_types = FALSE) %>% filter(video == !!video)
}

## ---- Content feature loaders ---------------------------------------------------------------
load_motion <- function(video) read_csv(file.path(MOTION_DIR, paste0(video, "_motion_tiles.csv")), show_col_types = FALSE)
load_saliency <- function(video) read_csv(file.path(SALIENCY_DIR, paste0(video, "_saliency_tiles.csv")), show_col_types = FALSE)
load_meaning <- function(video) {
  path <- file.path(MEANING_DIR, paste0(video, ".csv"))
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}
MEANING_CLIPS <- ALL_CLIPS[map_lgl(ALL_CLIPS, ~ file.exists(file.path(MEANING_DIR, paste0(.x, ".csv"))))]

## Mean over windows in [s, e) for motion/saliency (window_idx-indexed, tile_000..tile_119).
window_feature_vector <- function(df, w_idx) {
  tile_cols <- grep("^tile_", names(df), value = TRUE)
  rows <- df %>% filter(window_idx %in% w_idx)
  if (nrow(rows) == 0) return(NULL)
  colMeans(rows[, tile_cols])
}

## Mean predicted_meaning over 1fps samples with frame_time_sec in [s, e), by patch_index --
## aggregated directly against segment bounds (not window_idx) since meaning is sampled
## sparser than WINDOW_S=0.5s; short segments can legitimately catch zero 1fps samples.
segment_meaning_vector <- function(df, s, e) {
  rows <- df %>% filter(frame_time_sec >= s, frame_time_sec < e)
  if (nrow(rows) == 0) return(NULL)
  agg <- rows %>% group_by(patch_index) %>% summarise(v = mean(predicted_meaning), .groups = "drop") %>% arrange(patch_index)
  if (nrow(agg) != N_TILES) return(NULL)
  agg$v
}

## ---- R2 / unique-variance machinery ----------------------------------------------------------
r_squared <- function(y, X) {
  Z <- cbind(1, X)
  beta <- tryCatch(qr.solve(Z, y), error = function(e) NULL)
  if (is.null(beta)) return(NA_real_)
  ss_res <- sum((y - Z %*% beta)^2)
  ss_tot <- sum((y - mean(y))^2)
  if (ss_tot == 0) return(NA_real_)
  1 - ss_res / ss_tot
}

unique_variance <- function(y, feat_list) {
  feats <- names(feat_list)
  r2_full <- r_squared(y, do.call(cbind, feat_list))
  out <- list(r2_full = r2_full)
  for (f in feats) {
    others <- feat_list[setdiff(feats, f)]
    r2_red <- if (length(others) == 0) 0 else r_squared(y, do.call(cbind, others))
    out[[paste0("unique_", f)]] <- r2_full - r2_red
  }
  out$shared <- r2_full - sum(unlist(out[paste0("unique_", feats)]))
  out
}

## ---- Main per-segment loop --------------------------------------------------------------------
run_analysis <- function(features, slow_chopped, clips) {
  rows <- list()
  motion_cache <- list(); saliency_cache <- list(); meaning_cache <- list()

  for (video in clips) {
    cat(sprintf("Processing %s (slow_chopped=%s)...\n", video, slow_chopped))
    cv <- build_clip_vectors(video)
    pids <- cv$pids; n_windows_total <- cv$n_windows_total

    if ("motion" %in% features) motion_cache[[video]] <- load_motion(video)
    if ("saliency" %in% features) saliency_cache[[video]] <- load_saliency(video)
    if ("meaning" %in% features) meaning_cache[[video]] <- load_meaning(video)

    segs <- segment_list_for(video, slow_chopped)
    w_all <- 0:(n_windows_total - 1)
    for (r in seq_len(nrow(segs))) {
      seg <- segs[r, ]
      s <- seg$seg_start_s; e <- seg$seg_end_s
      w_idx <- w_all[w_all * WINDOW_S >= s & (w_all + 1) * WINDOW_S <= e]
      if (length(w_idx) < MIN_WINDOWS) next

      seg_mats <- setNames(lapply(pids, function(pid) cv$all_vecs[[pid]][w_idx + 1, , drop = FALSE]), pids)
      gaze_vec <- group_mean_vector(pids, seg_mats)
      if (is.null(gaze_vec) || sd(gaze_vec) == 0) next

      this_feats <- list()
      ok <- TRUE
      for (f in features) {
        v <- switch(f,
          motion = window_feature_vector(motion_cache[[video]], w_idx),
          saliency = window_feature_vector(saliency_cache[[video]], w_idx),
          meaning = segment_meaning_vector(meaning_cache[[video]], s, e),
          center = CENTER_BIAS_VEC
        )
        if (is.null(v) || sd(v) == 0) { ok <- FALSE; break }
        this_feats[[f]] <- v
      }
      if (!ok) next
      if (is.na(seg$within_person_cosine_dist)) next

      uv <- unique_variance(gaze_vec, this_feats)
      rows[[length(rows) + 1]] <- c(list(video = video, block = BLOCKS[[video]], seg_idx = seg$seg_idx), uv)
    }
  }

  bind_rows(rows)
}

## ---- CLI + run -----------------------------------------------------------------------------
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  features <- c("motion", "saliency", "meaning", "center")
  idx <- which(args == "--features")
  if (length(idx) > 0) features <- strsplit(args[idx + 1], ",")[[1]]

  clips <- if ("meaning" %in% features) MEANING_CLIPS else ALL_CLIPS
  stopifnot(length(clips) > 0)
  cat(sprintf("Features: %s | Clips (%d): %s\n", paste(features, collapse = ","),
              length(clips), paste(clips, collapse = ",")))

  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  tag <- paste(features, collapse = "_")

  for (chopped in c(TRUE, FALSE)) {
    df <- run_analysis(features, slow_chopped = chopped, clips = clips)
    suffix <- if (chopped) "" else "_slowunchopped"
    out_csv <- file.path(OUT_DIR, sprintf("feature_gaze_variance_partition_%s%s.csv", tag, suffix))
    write_csv(df, out_csv)
    cat(sprintf("Saved %s (%d segments)\n", out_csv, nrow(df)))
  }
}

if (sys.nframe() == 0) main()

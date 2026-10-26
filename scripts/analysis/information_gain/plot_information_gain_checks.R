## plot_information_gain_checks.R
## Information-gain check figures (figures/checks[/information_gain_kids]): floor per held-out clip, ceiling
## convergence, knot-count sweep and grid-resolution (dense) check. Data written by baseline.py, sanity_checks.py,
## knots_sweep.py and explore_dense_grid.py.
##
## Usage: Rscript scripts/analysis/information_gain/plot_information_gain_checks.R [--group adults|kids]

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(here)
})

a <- commandArgs(trailingOnly = TRUE)
GROUP <- if ("--group" %in% a) a[match("--group", a) + 1] else "adults"
RESULTS <- here("data", "results", "information_gain")
DATA_DIR <- if (GROUP == "adults") RESULTS else file.path(RESULTS, GROUP)
CHECK_DIR <- if (GROUP == "adults") here("figures", "checks") else here("figures", "checks", paste0("information_gain_", GROUP))
dir.create(CHECK_DIR, recursive = TRUE, showWarnings = FALSE)

COLORS <- c(saliency = "#2a78d6", motion = "#eb6834", meaning = "#1baf7a", face = "#b5459b", shared = "#8a8984")  # face passes all-pairs CVD check
FEATURES <- c("saliency", "motion", "meaning")  # all-clips checks; face is analyzed on the face clips only
INK <- "#0b0b0b"; INK2 <- "#52514e"
theme_ig <- theme_classic(base_size = 11) +
  theme(axis.text = element_text(color = INK2), axis.title = element_text(color = INK2), strip.background = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 11))
save <- function(p, name, w, h) {
  ggsave(file.path(CHECK_DIR, paste0(name, ".png")), p, width = w, height = h, dpi = 150, bg = "white")
  cat("Saved", file.path(CHECK_DIR, paste0(name, ".png")), "\n")
}

## ---- Floor per held-out clip (15 x 8 tiles) ------------------------------------
for (ss in c("cuts", "slow_chopped")) {
  path <- file.path(DATA_DIR, paste0("floor_", ss, ".csv"))
  if (!file.exists(path)) next
  fl <- read_csv(path, show_col_types = FALSE) |>
    pivot_longer(starts_with("tile_"), names_to = "tile", values_to = "p") |>
    mutate(i = as.integer(sub("tile_", "", tile)), col = i %% 15, row = i %/% 15,
           panel = factor(sprintf("held out: %s\neps=%.1e", video, eps), unique(sprintf("held out: %s\neps=%.1e", video, eps))))
  p <- ggplot(fl, aes(col, row, fill = p)) +
    geom_tile() +
    facet_wrap(~panel, ncol = 4) +
    scale_y_reverse() + coord_equal() +
    scale_fill_viridis_c(option = "magma", limits = c(0, NA), name = "floor probability\nper tile") +
    labs(x = NULL, y = NULL, title = sprintf("Floor per held-out clip (pooled gaze from the other 11 clips), 15x8 tiles, %s, %s", ss, GROUP)) +
    theme_ig + theme(axis.text = element_blank(), axis.ticks = element_blank(), axis.line = element_blank(),
                     strip.text = element_text(size = 8))
  save(p, paste0("information_gain_floor_", ss), 14, 6.4)
}

## ---- Ceiling convergence ----------------------------------------------------------
path <- file.path(DATA_DIR, "ceiling_convergence_cuts.csv")
if (file.exists(path)) {
  cv <- read_csv(path, show_col_types = FALSE)
  full <- cv$ig[cv$k == -1]; cv <- filter(cv, k > 0)
  p <- ggplot(cv, aes(k, ig)) +
    geom_hline(yintercept = full, color = INK2, linetype = "dashed") +
    annotate("text", x = max(cv$k), y = full, label = "all others", hjust = 1, vjust = -0.4, size = 3, color = INK2) +
    geom_line(color = COLORS[["saliency"]], linewidth = 0.8) + geom_point(color = COLORS[["saliency"]], size = 2.5) +
    scale_x_continuous(trans = "log2", breaks = cv$k) +
    labs(x = "number of other participants in the ceiling", y = "group ceiling (bits)", title = paste("Ceiling convergence, cuts,", GROUP)) +
    theme_ig
  save(p, "information_gain_ceiling_convergence_cuts", 5.5, 4)
}

## ---- Knot-count sweep (adults) -----------------------------------------------------
path <- file.path(DATA_DIR, "knots_sweep_cuts.csv")
if (file.exists(path)) {
  ks <- read_csv(path, show_col_types = FALSE) |>
    pivot_longer(any_of(FEATURES), names_to = "feature", values_to = "ig") |>
    mutate(n_knots = ifelse(label == "exp", 1, n_params + 1))
  p <- ggplot(ks, aes(n_knots, ig, color = feature)) +
    geom_line(data = filter(ks, label != "exp"), linewidth = 0.8) +
    geom_point(data = filter(ks, label != "exp"), size = 2.2) +
    geom_point(data = filter(ks, label == "exp"), shape = 15, size = 2.5) +
    scale_x_continuous(trans = "log2", breaks = c(1, 3, 5, 10, 20, 40), labels = c("exp", 3, 5, 10, 20, 40)) +
    scale_color_manual(values = COLORS, breaks = FEATURES) +
    labs(x = "number of knots (exp = 1 parameter)", y = "held-out information gain (bits)", color = NULL,
         title = "Single-feature models vs. nonlinearity flexibility, cuts") +
    theme_ig + theme(legend.position = "right")
  save(p, "information_gain_knots_sweep_cuts", 6, 4.2)
}

## ---- Grid-resolution check (adults): partition at each dense resolution --------------
for (path in list.files(DATA_DIR, pattern = "^summary_dense_[0-9]+(_faceclips)?\\.csv$", full.names = TRUE)) {
  n <- sub("summary_dense_([0-9]+).*", "\\1", basename(path))
  face <- grepl("_faceclips", basename(path))
  s <- read_csv(path, show_col_types = FALSE, col_names = c("name", "value"), skip = 1) |> (\(d) setNames(d$value, d$name))()
  parts <- c(shared = "shared", unique_motion = "motion", unique_saliency = "saliency", unique_meaning = "meaning", unique_face = "face")
  parts <- parts[names(parts) %in% names(s)]
  all_label <- if (face) "all four" else "all three"
  d <- tibble(component = factor(names(parts), names(parts)), key = parts, prop = s[names(parts)] / s[["ceiling"]])
  p <- ggplot(d, aes(component, prop, fill = key)) +
    geom_col(width = 0.62) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.3) +
    geom_hline(yintercept = s[["ig_all"]] / s[["ceiling"]], color = INK2, linetype = "dashed") +
    annotate("text", x = length(parts) + 0.4, y = s[["ig_all"]] / s[["ceiling"]], label = sprintf("%s %.1f%%", all_label, 100 * s[["ig_all"]] / s[["ceiling"]]),
             hjust = 1, vjust = -0.4, size = 3, color = INK2) +
    scale_fill_manual(values = COLORS, guide = "none") +
    scale_x_discrete(labels = sprintf("%s\n%+.1f%%", ifelse(parts == "shared", "shared", paste0("unique\n", parts)), 100 * d$prop)) +
    scale_y_continuous(labels = scales::label_percent()) +
    labs(x = NULL, y = sprintf("proportion of ceiling (ceiling = %.2f bits)", s[["ceiling"]]),
         title = sprintf("Overall IG on a %s-location grid (%sunchopped, 20 knots, floor)", n, if (face) "7 clips with faces, " else "")) +
    theme_ig
  save(p, paste0("overall_ig_dense_", n, if (face) "_faceclips" else ""), 6.5, 4.6)
}

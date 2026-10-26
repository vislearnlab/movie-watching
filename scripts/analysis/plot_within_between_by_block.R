## plot_within_between_by_block.R
## Presentation breakout of plot_within_between_scatter.R: one PNG per block, clips in shades of the block color, other
## blocks as faint gray context. Axes, canvas and dot-size scale are identical across PNGs so slides line up.
## Crosshair = medians over all segments (the 50/50 quadrant split), the same on every PNG.
##
## --scaled: each axis min-max rescaled over all segments, 0 = most similar segment, 1 = least similar.
##
## Output: figures/presentations/within_between_by_block/<block>[_scaled].png
## Usage: Rscript scripts/analysis/plot_within_between_by_block.R [--scaled]

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(readr)
  library(ggplot2)
  library(here)
})

source(here("scripts", "analysis", "block_colors.R"))  # BLOCK_COLORS
INK2 <- "#52514e"; CONTEXT <- "#d9d8d4"
SCALED <- "--scaled" %in% commandArgs(trailingOnly = TRUE)
to_unit <- function(d) (d - min(d)) / (max(d) - min(d))
mix <- function(col, with, f) rgb(t((1 - f) * col2rgb(col) + f * col2rgb(with)), maxColorValue = 255)
shades <- function(base, n) if (n == 1) base else colorRampPalette(c(mix(base, "black", 0.4), mix(base, "white", 0.42)))(n)  # even dark-to-light steps

w <- read_csv(here("data", "results", "within_between_cosine", "all_clips_by_segment.csv"), show_col_types = FALSE) |>
  filter(!is.na(within_person_cosine_dist), !is.na(between_person_cosine_dist)) |>
  mutate(x = if (SCALED) to_unit(within_person_cosine_dist) else within_person_cosine_dist,
         y = if (SCALED) to_unit(between_person_cosine_dist) else between_person_cosine_dist) |>
  arrange(desc(seg_dur_s))  # big dots underneath
X_MIN <- 0.3  # display cut on within (unscaled only); the few segments below it fall off the plot
pad <- function(r, lo = TRUE) r + diff(r) * c(if (lo) -0.04 else 0, 0.04)
lims <- list(x = if (SCALED) pad(range(w$x)) else pad(c(X_MIN, max(w$x)), lo = FALSE), y = pad(range(w$y)))  # X_MIN is the exact edge
size_lim <- range(w$seg_dur_s)
med <- c(x = median(w$x), y = median(w$y))
inset <- c(diff(lims$x), diff(lims$y)) * 0.015
quad_labels <- tibble(x = c(lims$x[1] + inset[1], lims$x[2] - inset[1])[c(1, 2, 1, 2)],  # in the plot corners, where points are sparse
                      y = c(lims$y[2] - inset[2], lims$y[1] + inset[2])[c(1, 1, 2, 2)],
                      hjust = c(0, 1, 0, 1), vjust = c(1, 1, 0, 0),
                      label = c("SELECTIVE", "SCATTERED", "CONCENTRATED", "COORDINATED"))
out_dir <- here("figures", "presentations", "within_between_by_block")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (b in names(BLOCK_COLORS)) {
  d <- filter(w, block == b)
  clips <- sort(unique(d$video))
  cols <- setNames(shades(BLOCK_COLORS[[b]], length(clips)), clips)
  labs_n <- count(d, video) |> with(setNames(sprintf("%s (n=%d)", gsub("_", " ", sub(paste0("^", b, "_?"), "", video)), n), video))
  p <- ggplot(mapping = aes(x, y, size = seg_dur_s)) +
    geom_vline(xintercept = med[["x"]], linetype = "dashed", color = "#8c8b87", linewidth = 0.6) +
    geom_hline(yintercept = med[["y"]], linetype = "dashed", color = "#8c8b87", linewidth = 0.6) +
    geom_point(data = filter(w, block != b), fill = CONTEXT, shape = 21, color = "white", stroke = 0.3) +
    geom_point(data = d, aes(fill = video), shape = 21, color = "white", stroke = 0.5, alpha = 0.9) +
    geom_text(data = quad_labels, aes(x, y, label = label, hjust = hjust, vjust = vjust), inherit.aes = FALSE,
              fontface = "italic", color = "#6f6e6a", size = 6) +
    scale_fill_manual(values = cols, labels = labs_n, name = NULL) +
    scale_size_area(max_size = 12, limits = size_lim, guide = "none") +
    guides(fill = guide_legend(override.aes = list(size = 6), nrow = 1)) +
    coord_cartesian(xlim = lims$x, ylim = lims$y, expand = FALSE) +
    labs(title = b,
         x = if (SCALED) "within-person distance (scaled, 1 = least similar)" else "within-person cosine distance",
         y = if (SCALED) "between-person distance (scaled, 1 = least similar)" else "between-person cosine distance") +
    theme_classic(base_size = 18) +
    theme(plot.title = element_text(face = "bold", color = BLOCK_COLORS[[b]], size = 24),
          aspect.ratio = 0.85, axis.title = element_text(color = INK2), legend.position = "bottom", legend.box = "vertical",
          legend.title = element_text(color = INK2, size = 14))
  out <- file.path(out_dir, paste0(b, if (SCALED) "_scaled", ".png"))
  ggsave(out, p, width = 9.5, height = 9.5, dpi = 200, bg = "white")
  cat("Saved", out, "\n")
}

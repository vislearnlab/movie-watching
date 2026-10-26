## plot_within_between_scatter.R
## Within- vs. between-person cosine distance, one point per (unchopped) segment, colored by block, area = segment duration.
##
## --scaled: each axis min-max rescaled, 0 = most similar segment, 1 = least similar.
##
## Output: figures/within_between/within_vs_between_scatter[_scaled].png
## Usage: Rscript scripts/analysis/plot_within_between_scatter.R [--scaled]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(here)
})

source(here("scripts", "analysis", "block_colors.R"))  # BLOCK_COLORS
INK2 <- "#52514e"
SCALED <- "--scaled" %in% commandArgs(trailingOnly = TRUE)
to_unit <- function(d) (d - min(d)) / (max(d) - min(d))

w <- read_csv(here("data", "results", "within_between_cosine", "all_clips_by_segment.csv"), show_col_types = FALSE) |>
  filter(!is.na(within_person_cosine_dist), !is.na(between_person_cosine_dist)) |>
  mutate(block = factor(block, names(BLOCK_COLORS))) |>
  arrange(desc(seg_dur_s)) |>  # big dots underneath
  mutate(x = if (SCALED) to_unit(within_person_cosine_dist) else within_person_cosine_dist,
         y = if (SCALED) to_unit(between_person_cosine_dist) else between_person_cosine_dist)
block_n <- count(w, block) |> with(setNames(sprintf("%s (n=%d)", block, n), block))

p <- ggplot(w, aes(x, y)) +
  geom_point(aes(fill = block, size = seg_dur_s), shape = 21, color = "white", stroke = 0.4, alpha = 0.8) +
  scale_fill_manual(values = BLOCK_COLORS, labels = block_n, name = NULL) +
  scale_size_area(max_size = 12, breaks = c(1, 5, 10, 20), name = "segment\nduration (s)") +
  guides(fill = guide_legend(override.aes = list(size = 5), order = 1),
         size = guide_legend(override.aes = list(fill = "#b5b4b0"), order = 2)) +
  labs(x = if (SCALED) "within-person distance (scaled, 1 = least similar)" else "within-person cosine distance",
       y = if (SCALED) "between-person distance (scaled, 1 = least similar)" else "between-person cosine distance") +
  theme_classic(base_size = 18) +
  theme(axis.title = element_text(color = INK2), legend.title = element_text(color = INK2, size = 14))

out <- here("figures", "within_between", paste0("within_vs_between_scatter", if (SCALED) "_scaled", ".png"))
ggsave(out, p, width = 10, height = 7.5, dpi = 200, bg = "white")
cat("Saved", out, "\n")

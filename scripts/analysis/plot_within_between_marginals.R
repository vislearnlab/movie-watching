## plot_within_between_marginals.R
## Per-block histograms of within- and between-person cosine distance, one value per (unchopped) segment.
## Rows = block, columns = metric; densities, so blocks with more segments don't dominate.
##
## Output: figures/within_between/within_between_hist_by_block.png
## Usage: Rscript scripts/analysis/plot_within_between_marginals.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(here)
})

source(here("scripts", "analysis", "block_colors.R"))  # BLOCK_COLORS
BINS <- 15
INK2 <- "#52514e"

w <- read_csv(here("data", "results", "within_between_cosine", "all_clips_by_segment.csv"), show_col_types = FALSE) |>
  filter(!is.na(within_person_cosine_dist), !is.na(between_person_cosine_dist)) |>
  mutate(block = factor(block, names(BLOCK_COLORS)))
block_n <- count(w, block) |> with(setNames(sprintf("%s\n(n=%d)", block, n), block))
long <- w |>
  pivot_longer(c(within_person_cosine_dist, between_person_cosine_dist), names_to = "metric") |>
  mutate(metric = factor(metric, c("within_person_cosine_dist", "between_person_cosine_dist"),
                         c("within-person cosine distance", "between-person cosine distance")))
# shared breaks per metric, so bins line up across blocks
br <- long |> group_by(metric) |> summarise(lo = min(value), hi = max(value))
long <- long |> left_join(br, by = "metric") |>
  mutate(bin = pmin(floor((value - lo) / (hi - lo) * BINS), BINS - 1), x = lo + (bin + 0.5) * (hi - lo) / BINS,
         width = (hi - lo) / BINS)
binned <- long |> count(block, metric, x, width) |> group_by(block, metric) |> mutate(density = n / sum(n) / width)

p <- ggplot(binned, aes(x, density, fill = block)) +
  geom_col(aes(width = width), color = "white", linewidth = 0.4) +
  facet_grid(block ~ metric, scales = "free_x", switch = "x", labeller = labeller(block = block_n)) +
  scale_fill_manual(values = BLOCK_COLORS, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "density") +
  theme_classic(base_size = 18) +
  theme(strip.background = element_blank(), strip.placement = "outside", strip.text.y = element_text(angle = 0, hjust = 0),
        strip.text.x = element_text(color = INK2, size = 18), axis.title = element_text(color = INK2),
        panel.spacing.x = unit(1.5, "lines"), panel.spacing.y = unit(1, "lines"))
out <- here("figures", "within_between", "within_between_hist_by_block.png")
ggsave(out, p, width = 12, height = 9, dpi = 200, bg = "white")
cat("Saved", out, "\n")

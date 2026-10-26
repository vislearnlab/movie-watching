## plot_ceiling_by_segment.R
## Ceiling (bits over the floor, mean over participants) for every scene-cut segment on one axis, sorted descending
## and colored by block. Data from plot_data.py (per_segment_<set>_knots_floor.csv).
##
## Output: figures/information_gain/adults/ceiling_by_segment_<set>.png
## Usage:  Rscript scripts/analysis/information_gain/plot_ceiling_by_segment.R [--set cuts|slow_chopped]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(here)
})
source(here("scripts", "analysis", "block_colors.R"))

a <- commandArgs(trailingOnly = TRUE)
SET <- if ("--set" %in% a) a[match("--set", a) + 1] else "cuts"
OUT_DIR <- here("figures", "information_gain", "adults")
INK2 <- "#52514e"

d <- read_csv(here("data", "results", "information_gain", paste0("per_segment_", SET, "_knots_floor.csv")), show_col_types = FALSE) |>
  arrange(desc(ceiling)) |> mutate(rank = row_number())
overall <- mean(d$ceiling)

p <- ggplot(d, aes(rank, ceiling, fill = block)) +
  geom_hline(yintercept = overall, color = INK2, linetype = "dashed", linewidth = 0.4) +
  geom_col(width = 0.85) +
  scale_fill_manual(values = BLOCK_COLORS, name = NULL) +
  scale_x_continuous(expand = expansion(mult = 0.01)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "segments, sorted by ceiling", y = "ceiling (bits over floor)",
       title = sprintf("Ceiling per segment: how well everyone else's gaze predicts each viewer (%s, %d segments)",
                       if (SET == "cuts") "unchopped" else "slow chopped", nrow(d)),
       subtitle = sprintf("Mean over participants per segment; dashed line = mean across segments (%.2f bits)", overall)) +
  theme_classic(base_size = 11) +
  theme(axis.text = element_text(color = INK2), axis.title = element_text(color = INK2), strip.background = element_blank(),
        plot.subtitle = element_text(color = INK2), legend.position = "top", legend.justification = "left",
        axis.text.x = element_blank(), axis.ticks.x = element_blank())
out <- file.path(OUT_DIR, paste0("ceiling_by_segment_", SET, ".png"))
ggsave(out, p, width = 12, height = 5, dpi = 200, bg = "white")
cat("Saved", out, "\n")
print(summary(d$ceiling))

## ---- Block means (bar = mean over the block's segments), segments as points ---------------------
blocks <- d |> group_by(block) |> summarise(ceiling = mean(ceiling), n = n(), .groups = "drop") |>
  mutate(x = as.numeric(factor(block, names(BLOCK_COLORS))))
set.seed(1)
segs <- d |> mutate(x = as.numeric(factor(block, names(BLOCK_COLORS))) + runif(n(), -0.28, 0.28))
p2 <- ggplot() +
  geom_col(data = blocks, aes(x, ceiling, fill = block), width = 0.75) +
  geom_point(data = segs, aes(x, ceiling), shape = 21, size = 1.8, fill = "white", color = "#0b0b0b", stroke = 0.4, alpha = 0.85) +
  scale_fill_manual(values = BLOCK_COLORS, guide = "none") +
  scale_x_continuous(breaks = blocks$x, labels = blocks$block) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = "ceiling (bits over floor)") +
  theme_classic(base_size = 12) +
  theme(axis.text = element_text(color = INK2), axis.text.x = element_text(size = 12, color = "#0b0b0b"),
        axis.title = element_text(color = INK2))
out2 <- file.path(OUT_DIR, paste0("ceiling_by_block_", SET, ".png"))
ggsave(out2, p2, width = 7, height = 5, dpi = 200, bg = "white")
cat("Saved", out2, "\n")
print(blocks)

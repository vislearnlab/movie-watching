## plot_waterfall_schematic.R
## Presentation schematic of the IG waterfall with made-up values (shared 20%, each unique +10%, all three 50%).
## White bars outlined in the slide feature colors (all three in black), no CIs or value labels.
##
## Output: figures/presentations/analysis schematics/waterfall_schematic.png
## Usage:  Rscript scripts/analysis/information_gain/plot_waterfall_schematic.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(here)
})

COLORS <- c(saliency = "#1e4c7c", motion = "#357a76", meaning = "#f09937", shared = "#8a8984", all = "#0b0b0b")  # slide colors, used as outlines
INK <- "#0b0b0b"; INK2 <- "#52514e"

wf <- tibble(label = c("shared", "+ unique\nsalience", "+ unique\nmotion", "+ unique\nsemantics", "= all\nthree"),
             key = c("shared", "saliency", "motion", "meaning", "all"),
             estimate = c(20, 10, 10, 10, 50), i = 1:5) |>
  mutate(end = if_else(key == "all", estimate, cumsum(estimate)), start = if_else(key == "all", 0, lag(end, default = 0)))

p <- ggplot(wf) +
  geom_segment(data = filter(wf, i < 5), aes(x = i + 0.3, xend = i + 0.7, y = end, yend = end),
               color = INK2, linewidth = 0.5, linetype = "dotted") +
  geom_rect(aes(xmin = i - 0.3, xmax = i + 0.3, ymin = start, ymax = end, color = key), fill = "white", linewidth = 1.6) +
  scale_color_manual(values = COLORS, guide = "none") +
  scale_x_continuous(breaks = wf$i, labels = wf$label, expand = expansion(add = 0.4)) +
  scale_y_continuous(limits = c(0, 60), breaks = seq(0, 60, 10), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  labs(x = NULL, y = "information gain explained\n(% of ceiling)") +
  theme_classic(base_size = 20) +
  theme(axis.text = element_text(color = INK2), axis.text.x = element_text(color = INK, size = 19),
        axis.title = element_text(color = INK2))
out_dir <- here("figures", "presentations", "analysis schematics")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out <- file.path(out_dir, "waterfall_schematic.png")
ggsave(out, p, width = 10, height = 6.5, dpi = 200, bg = "white")
cat("Saved", out, "\n")

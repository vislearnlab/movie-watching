## plot_quadrant_schematic.R
## Presentation schematic: empty within/between quadrant space (x = within-subject, y = between-subject distance, as in
## plot_within_between_scatter.R) with the four types named in their quadrants. No data.
##
## Output: figures/presentations/analysis schematics/quadrants_empty.png
## Usage:  Rscript scripts/analysis/plot_quadrant_schematic.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(here)
})

OUT <- here("figures", "presentations", "analysis schematics", "quadrants_empty.png")
INK <- "#0b0b0b"; INK2 <- "#52514e"; RULE <- "#bdbcb7"

quads <- data.frame(
  x = c(0.25, 0.75, 0.25, 0.75), y = c(0.25, 0.25, 0.75, 0.75),
  name = c("Concentrated", "Coordinated", "Selective", "Scattered"),
  desc = c("one shared target", "everyone shifts\nthe same way", "each person focused,\non different things", "unfocused and\ndisagreeing")
)

p <- ggplot(quads) +
  annotate("segment", x = 0.5, xend = 0.5, y = 0, yend = 1, color = RULE, linewidth = 1) +
  annotate("segment", x = 0, xend = 1, y = 0.5, yend = 0.5, color = RULE, linewidth = 1) +
  geom_text(aes(x, y + 0.04, label = name), size = 12, fontface = "bold", color = INK) +
  geom_text(aes(x, y - 0.05, label = desc), size = 7, color = INK2, lineheight = 0.9) +
  scale_x_continuous(limits = c(0, 1), breaks = c(0.04, 0.96), labels = c("low", "high"), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0.04, 0.96), labels = c("low", "high"), expand = c(0, 0)) +
  coord_fixed(clip = "off") +
  labs(x = "Within-subject distance\n(how much each person's gaze moves around)",
       y = "Between-subject distance\n(how much people disagree with each other)") +
  theme_classic(base_size = 26) +
  theme(axis.title = element_text(color = INK, face = "bold", lineheight = 0.9),
        axis.title.x = element_text(margin = margin(t = 14)), axis.title.y = element_text(margin = margin(r = 14)),
        axis.text = element_text(color = INK2, size = 20), axis.ticks = element_blank(),
        axis.line = element_line(color = INK, linewidth = 1.2, arrow = grid::arrow(length = unit(0.35, "cm"), type = "closed")),
        plot.margin = margin(20, 30, 20, 20))
ggsave(OUT, p, width = 11, height = 11, dpi = 200, bg = "white")
cat("Saved", OUT, "\n")

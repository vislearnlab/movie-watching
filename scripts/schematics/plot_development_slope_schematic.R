## plot_development_slope_schematic.R
## Presentation schematic of the prediction: adult-likeness of gaze rises with age, with a shallow slope for stimuli
## where salience and semantics overlap (e.g. Pixar; kids adult-like early) and a steep slope where they diverge.
## Illustrative lines only, no data.
##
## Output: figures/presentations/analysis schematics/development_slope_prediction.png
## Usage:  Rscript scripts/schematics/plot_development_slope_schematic.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(here)
})
source(here("scripts", "analysis", "block_colors.R"))

INK <- "#0b0b0b"; INK2 <- "#52514e"
HIGH <- BLOCK_COLORS[["pixar"]]; LOW <- BLOCK_COLORS[["frank"]]
Y0_HIGH <- 0.82; Y0_LOW <- 0.18  # adult-likeness at the youngest age (illustrative)

lines <- data.frame(x = 0, xend = 1, y = c(Y0_HIGH, Y0_LOW), yend = 1, col = c(HIGH, LOW))
p <- ggplot() +
  geom_hline(yintercept = 1, linetype = "dashed", color = INK2, linewidth = 0.6) +
  annotate("text", x = 1, y = 1.035, label = "adult-aligned gaze", hjust = 1, size = 6, color = INK2) +
  geom_segment(data = lines, aes(x, y, xend = xend, yend = yend, color = col), linewidth = 2.4, lineend = "round") +
  geom_point(data = lines, aes(x, y, color = col), size = 5) +
  scale_color_identity() +
  annotate("text", x = 0.01, y = 0.76, hjust = 0, vjust = 1, size = 7, fontface = "bold", color = HIGH, lineheight = 0.95,
           label = "Salience & semantics overlap\n(e.g., Pixar): shallow slope") +
  annotate("text", x = 0.01, y = 0.625, hjust = 0, vjust = 1, size = 5.4, color = INK, lineheight = 0.95,
           label = "salience alone lands on the adult\ntarget, so kids look adult-like early") +
  annotate("text", x = 1, y = 0.24, hjust = 1, size = 7, fontface = "bold", color = LOW,
           label = "Salience & semantics diverge: steep slope") +
  annotate("text", x = 1, y = 0.165, hjust = 1, size = 5.4, color = INK,
           label = "the adult target needs semantic knowledge, so kids catch up later") +
  scale_x_continuous(limits = c(0, 1.04), breaks = c(0, 1), labels = c("young\nchildren", "adults"), expand = c(0.02, 0)) +
  scale_y_continuous(limits = c(0, 1.08), breaks = NULL, expand = c(0, 0)) +
  labs(x = "age", y = "adult-likeness of gaze") +
  theme_classic(base_size = 22) +
  theme(axis.text.x = element_text(color = INK, size = 18), axis.title = element_text(color = INK),
        axis.line = element_line(linewidth = 0.8, arrow = arrow(length = unit(0.25, "cm"), type = "closed")),
        axis.ticks = element_blank())
out_dir <- here("figures", "presentations", "analysis schematics")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out <- file.path(out_dir, "development_slope_prediction.png")
ggsave(out, p, width = 12, height = 6.5, dpi = 200, bg = "white")
cat("Saved", out, "\n")

## plot_ig_schematic.R
## Presentation schematic of information gain: one viewer's fixations scored under the floor, a feature model and the
## ceiling (everyone else's gaze), with the 0% (floor) to 100% (ceiling) scale underneath. Data from schematic_data.py.
##
## Output: figures/presentations/ig_schematic/<clip>_seg<idx>_<pid>.png
## Usage:  Rscript scripts/analysis/information_gain/plot_ig_schematic.R [--tag sesameus_1_seg15_MW012]

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(png)
  library(here)
})

a <- commandArgs(trailingOnly = TRUE)
TAG <- if ("--tag" %in% a) a[match("--tag", a) + 1] else "sesameus_1_seg15_MW012"
DATA <- here("data", "results", "information_gain", "schematic")
OUT_DIR <- here("figures", "presentations", "ig_schematic")
INK <- "#0b0b0b"; INK2 <- "#52514e"; RULE <- "#bdbcb7"
FIX_FILL <- "#00e6ff"  # cyan reads on inferno
W <- 1920; H <- 1080

maps <- read_csv(file.path(DATA, paste0(TAG, "_maps.csv")), show_col_types = FALSE)
fix <- read_csv(file.path(DATA, paste0(TAG, "_fixations.csv")), show_col_types = FALSE)
sc <- read_csv(file.path(DATA, paste0(TAG, "_scores.csv")), show_col_types = FALSE)
frame <- readPNG(file.path(OUT_DIR, "frames", paste0(sub("_MW.*", "", TAG), ".png")))
feature <- sc$feature[1]; pid <- sc$participant[1]
ig <- setNames(sc$ig_bits, sc$map); ll <- setNames(sc$ll_bits, sc$map)
pct <- 100 * ig[["model"]] / ig[["ceiling"]]

long <- maps |> pivot_longer(c(floor, model, ceiling), names_to = "map", values_to = "p") |> mutate(p = 100 * p)
pmax_all <- max(long$p)

screen <- function(show_frame = TRUE) list(
  if (show_frame) annotation_raster(frame, 0, W, -H, 0) else annotate("rect", xmin = 0, xmax = W, ymin = -H, ymax = 0, fill = "black"),
  scale_x_continuous(limits = c(0, W), expand = c(0, 0)), scale_y_continuous(limits = c(-H, 0), expand = c(0, 0)),
  coord_fixed(), theme_void(),
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0.5, color = INK),
        plot.subtitle = element_text(size = 11, hjust = 0.5, color = INK2, lineheight = 1.05),
        plot.caption = element_text(size = 12.5, hjust = 0.5, color = INK, lineheight = 1.15, margin = margin(t = 6))))

map_panel <- function(m, title) {
  d <- filter(long, map == m)
  cap <- if (m == "floor") sprintf("avg log2 p = %.2f bits\ngain over floor: 0 (reference)", ll[[m]])
         else sprintf("avg log2 p = %.2f bits\ngain over floor: +%.2f bits", ll[[m]], ig[[m]])
  ggplot() + screen(show_frame = m != "floor") +  # floor comes from other clips: no frame behind it
    geom_point(data = d, aes(x, -y, fill = p), shape = 21, size = 6.2, color = "white", stroke = 0.3, alpha = 0.85) +
    geom_point(data = fix, aes(x, -y), shape = 21, size = 3.2, fill = FIX_FILL, color = INK, stroke = 0.7) +
    scale_fill_viridis_c(option = "inferno", limits = c(0, pmax_all), trans = "sqrt", breaks = c(0, pmax_all),
                         labels = c("low probability", "high probability"), name = NULL) +
    labs(title = title, caption = cap) +
    theme(plot.margin = margin(0, 10, 0, 10))
}

top <- ggplot() + screen() +
  geom_point(data = fix, aes(x, -y, size = dur_ms), shape = 21, fill = FIX_FILL, color = INK, stroke = 0.9) +
  scale_size_area(max_size = 9, guide = "none") +
  labs(title = sprintf("One viewer's fixations (%s, %d fixations)", pid, nrow(fix)))

p_floor <- map_panel("floor", "Floor: spatial prior")
p_model <- map_panel("model", sprintf("Model: %s x floor", feature))
p_ceil <- map_panel("ceiling", "Ceiling: everyone else, this segment")

bar <- ggplot() +
  annotate("segment", x = 0, xend = ig[["ceiling"]], y = 0, yend = 0, linewidth = 6, color = "#e6e5e1", lineend = "round") +
  annotate("segment", x = 0, xend = ig[["model"]], y = 0, yend = 0, linewidth = 6, color = "#eb6834", lineend = "round") +
  annotate("point", x = c(0, ig[["ceiling"]]), y = 0, size = 5, color = INK) +
  annotate("text", x = 0, y = -0.6, label = "floor\n0 bits = 0%", size = 4.6, color = INK, lineheight = 0.95) +
  annotate("text", x = ig[["ceiling"]], y = -0.6, label = sprintf("ceiling\n%.2f bits = 100%%", ig[["ceiling"]]), size = 4.6, color = INK, lineheight = 0.95) +
  annotate("text", x = ig[["model"]], y = 0.62, label = sprintf("%s: +%.2f bits = %.0f%% of ceiling", feature, ig[["model"]], pct),
           size = 5.2, fontface = "bold", color = INK) +
  annotate("text", x = ig[["ceiling"]] / 2, y = 1.35, size = 4.4, color = INK2,
           label = "information gain = avg log2 p(model) - avg log2 p(floor);  % of ceiling = model gain / ceiling gain") +
  scale_x_continuous(expand = expansion(mult = 0.12)) + scale_y_continuous(limits = c(-1.1, 1.6)) +
  theme_void()

p <- (plot_spacer() | top | plot_spacer()) + plot_layout(widths = c(0.5, 2, 0.5))
p <- p / (p_floor | p_model | p_ceil) / bar +
  plot_layout(heights = c(1.25, 1.05, 0.5), guides = "collect") &
  theme(legend.title = element_text(size = 10, color = INK2), legend.text = element_text(size = 9, color = INK2))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
out <- file.path(OUT_DIR, paste0(TAG, ".png"))
ggsave(out, p, width = 15, height = 11, dpi = 200, bg = "white")
cat("Saved", out, "\n")

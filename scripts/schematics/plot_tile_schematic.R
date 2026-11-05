## plot_tile_schematic.R
## Schematic: (a) frame, (b) group gaze as 120-tile overlay, (c) the same tiles flattened row by row into a 120 x 1 vector. One row per time point.
## Data from export_tile_schematic.py.
##
## Usage: Rscript scripts/schematics/plot_tile_schematic.R [--video pixar_birds]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(png)
  library(here)
})

a <- commandArgs(trailingOnly = TRUE)
VIDEO <- if ("--video" %in% a) a[match("--video", a) + 1] else "pixar_birds"
W <- 1920; H <- 1080; RADIUS <- 40  # circles at tile centers (tile spacing 120px)
INK2 <- "#52514e"

d <- read_csv(here("data", "results", "presentations", paste0("tile_schematic_", VIDEO, ".csv")), show_col_types = FALSE)
d <- d |> group_by(t) |> mutate(value = value / max(value)) |> ungroup()  # relative density, each frame scaled to its peak
vmax <- 1
frame_raster <- function(t) readPNG(here("figures", "presentations", "tile_schematic", "frames", sprintf("%s_%gs.png", VIDEO, t)))

theme_blank <- theme_void() + theme(plot.title = element_text(size = 12, face = "bold", hjust = 0, margin = margin(b = 4)),
                                    plot.margin = margin(4, 6, 4, 6))
frame_base <- function(t) {
  ggplot() + annotation_raster(frame_raster(t), 0, W, -H, 0) +
    coord_fixed(xlim = c(0, W), ylim = c(-H, 0), expand = FALSE) + theme_blank
}

circles <- function(g, n = 40) {
  th <- seq(0, 2 * pi, length.out = n)
  bind_rows(lapply(seq_len(nrow(g)), \(i) tibble(tile = g$tile[i], value = g$value[i],
                                                 x = g$center_x[i] + RADIUS * cos(th), y = -g$center_y[i] + RADIUS * sin(th))))
}

row_plots <- function(t, first) {
  g <- filter(d, t == !!t)
  pa <- frame_base(t)
  pb <- frame_base(t) +
    geom_polygon(data = circles(g), aes(x, y, group = tile, fill = value), alpha = 0.75, color = NA) +
    scale_fill_viridis_c(option = "inferno", limits = c(0, vmax), trans = "sqrt", guide = "none")
  g <- mutate(g, x = tile)
  pc <- ggplot(g, aes(x, 0, fill = value)) +
    geom_tile(width = 1, height = 10, color = NA) +
    annotate("text", x = g$x[c(1, 120)], y = -7.5, label = c("1", "120"), size = 3, color = INK2) +
    scale_fill_viridis_c(option = "inferno", limits = c(0, vmax), trans = "sqrt", name = "gaze\ndensity", breaks = c(0, 1), labels = c("low", "high"),
                        guide = if (first) guide_colorbar(barheight = unit(3, "cm")) else "none") +
    coord_fixed(ratio = 1, xlim = c(0, max(g$x) + 1), ylim = c(-H / W * (max(g$x) + 1) / 2, H / W * (max(g$x) + 1) / 2),
                clip = "off", expand = FALSE) + theme_blank
  if (first) {
    pa <- pa + labs(title = "a  frame")
    pb <- pb + labs(title = "b  group gaze on 120 tiles")
    pc <- pc + labs(title = "c  tile vector (120 x 1)")
  }
  list(pa + labs(subtitle = sprintf("t = %gs", t)) + theme(plot.subtitle = element_text(size = 9, color = INK2)), pb, pc)
}

times <- sort(unique(d$t))
plots <- unlist(lapply(seq_along(times), \(i) row_plots(times[i], i == 1)), recursive = FALSE)
p <- wrap_plots(plots, ncol = 3, widths = c(1, 1, 1)) + plot_layout(guides = "collect")
out <- here("figures", "presentations", "tile_schematic", paste0("tile_schematic_", VIDEO, ".png"))
ggsave(out, p, width = 15, height = 2.35 * length(times) + 0.5, dpi = 200, bg = "white")
cat("Saved", out, "\n")

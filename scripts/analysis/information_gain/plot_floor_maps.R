## plot_floor_maps.R
## Presentation figure: the IG floor ("spatial") for each of the 12 clips = pooled gaze from the other 11 clips, on the
## 120-tile grid drawn at screen geometry. Shared color scale; screen outline in the block color. Data from baseline.py.
##
## Output: figures/presentations/floor_maps/floor_maps_<set>[_<group>].png
## Usage:  Rscript scripts/analysis/information_gain/plot_floor_maps.R [--group adults|kids]

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(here)
})
source(here("scripts", "analysis", "block_colors.R"))

a <- commandArgs(trailingOnly = TRUE)
GROUP <- if ("--group" %in% a) a[match("--group", a) + 1] else "adults"
RESULTS <- here("data", "results", "information_gain")
DATA_DIR <- if (GROUP == "adults") RESULTS else file.path(RESULTS, GROUP)
OUT_DIR <- here("figures", "presentations", "floor_maps")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
INK2 <- "#52514e"
SCREEN_W <- 1920; SCREEN_H <- 1080; SPACING <- 120  # tile centers 120..1800 x 120..960 px, 15 x 8

for (ss in c("cuts", "slow_chopped")) {
  path <- file.path(DATA_DIR, paste0("floor_", ss, ".csv"))
  if (!file.exists(path)) next
  fl <- read_csv(path, show_col_types = FALSE) |>
    pivot_longer(starts_with("tile_"), names_to = "tile", values_to = "p") |>
    mutate(i = as.integer(sub("tile_", "", tile)),
           x = SPACING * (i %% 15 + 1), y = SPACING * (i %/% 15 + 1),
           xmin = if_else(i %% 15 == 0, 0, x - SPACING / 2), xmax = if_else(i %% 15 == 14, SCREEN_W, x + SPACING / 2),  # edge cells reach the screen edge
           ymin = if_else(i %/% 15 == 0, 0, y - SPACING / 2), ymax = if_else(i %/% 15 == 7, SCREEN_H, y + SPACING / 2),
           pct = 100 * p,
           block = sub("sesame(india|us)", "sesame", sub("_.*", "", video)),
           video = factor(video, unique(video)))
  screens <- distinct(fl, video, block)
  p <- ggplot(fl) +
    geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = pct)) +
    geom_rect(data = screens, aes(xmin = 0, xmax = SCREEN_W, ymin = 0, ymax = SCREEN_H, color = block),
              fill = NA, linewidth = 1.1) +
    facet_wrap(~video, ncol = 4) +
    scale_y_reverse() + coord_equal(xlim = c(0, SCREEN_W), ylim = c(SCREEN_H, 0), expand = FALSE) +
    scale_fill_viridis_c(option = "inferno", limits = c(0, NA), name = "% of fixation\nprobability\nper tile") +
    scale_color_manual(values = BLOCK_COLORS, name = "block") +
    labs(title = "Spatial prior (IG floor) for each clip",
         subtitle = sprintf("Pooled gaze from the other 11 clips (clip never informs its own map) - %s, %s%s",
                            GROUP, if (ss == "cuts") "unchopped" else "slow chopped",
                            sprintf("; uniform = %.2f%% per tile", 100 / 120))) +
    theme_void(base_size = 12) +
    theme(strip.text = element_text(color = INK2, size = 10, margin = margin(b = 3)),
          plot.title = element_text(hjust = 0.5, size = 14, margin = margin(b = 4)),
          plot.subtitle = element_text(hjust = 0.5, color = INK2, size = 10, margin = margin(b = 10)),
          legend.title = element_text(color = INK2, size = 9), legend.text = element_text(color = INK2, size = 9),
          panel.spacing = unit(8, "pt"), plot.margin = margin(10, 10, 10, 10))
  out <- file.path(OUT_DIR, paste0("floor_maps_", ss, if (GROUP != "adults") paste0("_", GROUP), ".png"))
  ggsave(out, p, width = 12, height = 5.6, dpi = 200, bg = "white")
  cat("Saved", out, "\n")
}

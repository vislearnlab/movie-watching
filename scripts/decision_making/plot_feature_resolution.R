# Frame | motion | saliency | meaning | face (current: sigma=46 blur) | face (240px box mean + sigma=46), 10 frames per block.
# Each dot = one of the 120 tile values as the IG models see them; colour scaled 0 to the panel's max.
# Input: data/local_cache/feature_resolution/ (compare_feature_resolution.py)
# Output: figures/checks/feature_resolution_<block>.png
# Usage: Rscript scripts/decision_making/plot_feature_resolution.R

suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(data.table)})

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))
in_dir <- file.path(root, "data", "local_cache", "feature_resolution")
out_dir <- file.path(root, "figures", "checks")
W <- 1920; H <- 1080

frames <- fread(file.path(in_dir, "frames.csv"))
tiles <- fread(file.path(in_dir, "tiles.csv"))[, y := H - y]
tiles[, scaled := if (max(value) > 0) value / max(value) else 0, by = .(frame_id, feature)]
feat_labels <- c(motion = "motion", saliency = "saliency", meaning = "meaning",
                 face_current = "face: blur sigma=46 (current)", face_box = "face: 240px box mean + sigma=46")

theme_panel <- theme_void() + theme(plot.title = element_text(size = 8), legend.position = "none",
                                    panel.background = element_rect(fill = "black", colour = NA),
                                    plot.margin = margin(2, 2, 2, 2))
xy <- coord_fixed(xlim = c(0, W), ylim = c(0, H), expand = FALSE)

row_plots <- function(fid, first) {
  img <- png::readPNG(file.path(in_dir, paste0(fid, ".png")))
  p0 <- ggplot() + annotation_raster(img, 0, W, 0, H) + xy + theme_void() +
    theme(plot.title = element_text(size = 8), plot.margin = margin(2, 2, 2, 2)) + labs(title = fid)
  c(list(p0), lapply(names(feat_labels), function(f) {
    d <- tiles[frame_id == fid & feature == f]
    ggplot(d, aes(x, y, colour = scaled)) + geom_point(size = 2.6, shape = 16) +
      scale_colour_viridis_c(option = "inferno", limits = c(0, 1)) + xy + theme_panel +
      labs(title = if (first) feat_labels[[f]] else NULL)
  }))
}

for (b in unique(frames$block)) {
  fids <- frames[block == b, frame_id]
  ps <- unlist(lapply(seq_along(fids), function(i) row_plots(fids[i], i == 1)), recursive = FALSE)
  p <- wrap_plots(ps, ncol = 6) + plot_annotation(
    title = sprintf("Feature spatial resolution: %s block (10 frames)", b),
    subtitle = "Dots = the 120 tile values the IG models use, colour scaled 0 to each panel's max (inferno). Motion/saliency = 240px box means + sigma=46; meaning = fine + coarse ratings, smoothed.")
  ggsave(file.path(out_dir, sprintf("feature_resolution_%s.png", b)), p, width = 15, height = 15, dpi = 130, bg = "white")
}

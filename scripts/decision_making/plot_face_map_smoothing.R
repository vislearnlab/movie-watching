# Side-by-side face maps for 20 frames: frame + SAM3 mask outline | (A) tile circle touches a face -> 1 at center, Gaussian bumps summed | (B) smoothed pixel mask.
# Input: data/local_cache/face_masks/smoothing_comparison/sigma<N>/ (compare_face_map_smoothing.py --sigma N)
# Output: figures/checks/face_map_smoothing_sigma<N>_p1.png, _p2.png
# Usage: Rscript scripts/decision_making/plot_face_map_smoothing.R [sigma, default 46]

suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(data.table)})

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))
args <- commandArgs(trailingOnly = TRUE)
sigma <- if (length(args)) args[1] else "46"
in_dir <- file.path(root, "data", "local_cache", "face_masks", "smoothing_comparison", paste0("sigma", sigma))
out_dir <- file.path(root, "figures", "checks")
W <- 1280; H <- 720

frames <- fread(file.path(in_dir, "frames.csv"))
maps <- fread(file.path(in_dir, "maps.csv"))[, y := H - y]
tiles <- fread(file.path(in_dir, "tiles.csv"))[, y := H - y]

theme_panel <- theme_void() + theme(plot.title = element_text(size = 9), legend.position = "none",
                                    plot.margin = margin(2, 2, 2, 2))
xy <- list(coord_fixed(xlim = c(0, W), ylim = c(0, H), expand = FALSE))
fill_scale <- scale_fill_viridis_c(option = "inferno", limits = c(0, 1), oob = scales::squish)  # float values a hair over 1 would otherwise drop out

row_plots <- function(fid, label) {
  m <- maps[frame_id == fid]; t <- tiles[frame_id == fid]
  img <- png::readPNG(file.path(in_dir, paste0(fid, ".png")))
  p0 <- ggplot() + annotation_raster(img, 0, W, 0, H) +
    geom_contour(data = m, aes(x, y, z = mask), breaks = 0.5, colour = "cyan", linewidth = 0.4) +
    geom_point(data = t, aes(x, y, colour = factor(a_center)), size = 0.5) +
    scale_colour_manual(values = c("0" = "grey70", "1" = "red")) + xy + theme_panel +
    labs(title = sprintf("%s  (%s)", fid, label))
  pa <- ggplot(m, aes(x, y, fill = a)) + geom_raster() + fill_scale + xy + theme_panel +
    labs(title = sprintf("A: face in tile circle, bumps (peak %.2f)", max(m$a)))
  pb <- ggplot(m, aes(x, y, fill = b)) + geom_raster() + fill_scale + xy + theme_panel +
    labs(title = sprintf("B: pixel mask, blurred (peak %.2f)", max(m$b)))
  list(p0, pa, pb)
}

pages <- split(seq_len(nrow(frames)), ceiling(seq_len(nrow(frames)) / 10))
for (i in seq_along(pages)) {
  ps <- unlist(lapply(pages[[i]], function(k) row_plots(frames$frame_id[k], frames$kind[k])), recursive = FALSE)
  key <- ggplot(data.frame(x = 0:1, y = 0, z = c(0, 1)), aes(x, y, fill = z)) + geom_raster() + fill_scale +
    theme_void() + theme(legend.position = "bottom") + labs(fill = "face map value")
  p <- wrap_plots(ps, ncol = 3) + plot_annotation(
    title = sprintf("Face map smoothing, A vs B (page %d of %d)", i, length(pages)),
    subtitle = sprintf("Left: cyan = SAM3 face mask (human + Muppet + Pixar bird), red dots = tile circles (240px) touching a face.\nA = Gaussian bumps at red centers, summed, 1 = all neighbours on. Both maps: sigma = %s display px, shared 0-1 colour scale.", sigma))
  ggsave(file.path(out_dir, sprintf("face_map_smoothing_sigma%s_p%d.png", sigma, i)), p, width = 12, height = 21, dpi = 150, bg = "white")
}

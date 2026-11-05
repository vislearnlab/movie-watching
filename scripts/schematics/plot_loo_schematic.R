## plot_loo_schematic.R
## Leave-one-out schematic: row 1 = one participant, row 2 = group without them. (a) frame, (b) gaze circles at the first
## window, (c) illustrative 15-block tile vectors for consecutive 0.5s windows (made up). Data from export_loo_schematic.py.
##
## Usage: Rscript scripts/schematics/plot_loo_schematic.R [--video sesameus_1]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(png)
  library(here)
})

a <- commandArgs(trailingOnly = TRUE)
VIDEO <- if ("--video" %in% a) a[match("--video", a) + 1] else "sesameus_1"
W <- 1920; H <- 1080; RADIUS <- 40
INK2 <- "#52514e"
ROW_LABELS <- c(individual = "one participant", group_loo = "group, leaving them out")

d <- read_csv(here("data", "results", "presentations", paste0("loo_schematic_", VIDEO, ".csv")), show_col_types = FALSE) |>
  group_by(who, t) |> mutate(value = value / max(value)) |> ungroup()  # relative density, each vector scaled to its peak
times <- sort(unique(d$t))
SHIFT <- 1.2  # blocks each later vector is offset right
N_SHOW <- 15  # panel c vectors are illustrative (made up), not data
bump <- function(mu, sd, h = 1) h * exp(-((1:N_SHOW) - mu)^2 / (2 * sd^2))
spike <- function(at, h) replace(numeric(N_SHOW), at, h)
fake <- list(
  individual = list(spike(c(3, 4, 5, 12), c(0.3, 1, 0.35, 0.2)), spike(c(8, 9, 10, 4), c(0.4, 1, 0.5, 0.15)),
                    spike(c(9, 10, 11), c(0.3, 1, 0.45))),
  group_loo = list(bump(8.5, 1.3) + bump(4, 0.8, 0.3), bump(9.5, 1.3) + bump(3.5, 0.8, 0.2), bump(10.5, 1.3) + bump(5, 0.8, 0.15))
)
fake_df <- bind_rows(lapply(names(fake), \(w) bind_rows(lapply(seq_along(times), \(k)
  tibble(who = w, t = times[k], x = 1:N_SHOW, value = pmin(1, fake[[w]][[k]] / max(fake[[w]][[k]])))))))
frame <- readPNG(here("figures", "presentations", "tile_schematic", "frames", sprintf("%s_%gs.png", VIDEO, times[1])))

fill_scale <- function(show) scale_fill_viridis_c(option = "inferno", limits = c(0, 1), trans = "sqrt", name = "gaze\ndensity",
                                                  breaks = c(0, 1), labels = c("low", "high"),
                                                  guide = if (show) guide_colorbar(barheight = unit(3, "cm")) else "none")
theme_blank <- theme_void() + theme(plot.title = element_text(size = 12, face = "bold", hjust = 0, margin = margin(b = 4)),
                                    plot.subtitle = element_text(size = 14, color = INK2), plot.margin = margin(4, 6, 4, 6))
frame_base <- function() {
  ggplot() + annotation_raster(frame, 0, W, -H, 0) + coord_fixed(xlim = c(0, W), ylim = c(-H, 0), expand = FALSE) + theme_blank
}
circles <- function(g, n = 40) {
  th <- seq(0, 2 * pi, length.out = n)
  bind_rows(lapply(seq_len(nrow(g)), \(i) tibble(tile = g$tile[i], value = g$value[i],
                                                 x = g$center_x[i] + RADIUS * cos(th), y = -g$center_y[i] + RADIUS * sin(th))))
}

row_plots <- function(who, first) {
  g <- filter(d, who == !!who)
  pa <- frame_base() + labs(subtitle = ROW_LABELS[[who]])
  pb <- frame_base() +
    geom_polygon(data = circles(filter(g, t == times[1])), aes(x, y, group = tile, fill = value), alpha = 0.75, color = NA) +
    fill_scale(FALSE)
  shift <- (seq_along(times) - 1) * SHIFT  # stagger each later window to the right to convey time
  strips <- filter(fake_df, who == !!who) |> mutate(k = match(t, times), x = x + shift[k], y = -k * 1.5)
  pc <- ggplot(strips, aes(x, y, fill = value)) +
    geom_tile(width = 0.92, height = 1, color = NA) +
    annotate("text", x = rep(c(-0.1, N_SHOW + 1.1), length(times)) + rep(shift, each = 2), y = rep(-seq_along(times) * 1.5, each = 2), label = "\u2026", size = 5, color = INK2) +
    annotate("text", x = N_SHOW + 2.2 + max(shift), y = -seq_along(times) * 1.5, label = sprintf("t = %.1fs", times), hjust = 0, size = 4.6, color = INK2) +
    fill_scale(FALSE) +
    coord_fixed(ratio = 1, xlim = c(0.5, N_SHOW + 0.5 + max(shift)), ylim = c(-length(times) * 1.5 - 1.5, 0), clip = "off", expand = FALSE) +
    theme_blank + theme(plot.margin = margin(4, 150, 4, 30))
  if (first) {
    pa <- pa + labs(title = "a  frame")
    pb <- pb + labs(title = sprintf("b  gaze on 120 tiles (t = %.1fs)", times[1]))
    pc <- pc + labs(title = "c  tile vectors over time (illustrative)")
  }
  list(pa, pb, pc)
}

p <- wrap_plots(c(row_plots("individual", TRUE), row_plots("group_loo", FALSE)), ncol = 3) + plot_layout(guides = "collect")
out <- here("figures", "presentations", "tile_schematic", paste0("loo_schematic_", VIDEO, ".png"))
ggsave(out, p, width = 16, height = 5.4, dpi = 200, bg = "white")
cat("Saved", out, "\n")

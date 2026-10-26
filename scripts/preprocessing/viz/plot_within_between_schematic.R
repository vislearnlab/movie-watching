## plot_within_between_schematic.R
## Generic schematic of the within/between cosine metric (calc_within_between_cosine.py), with illustrative (made-up)
## 15-block tile vectors; distances shown are computed from those vectors, and e/f place them in 2D by classical MDS.
##   within  = one person's windows in a segment, mean pairwise cosine distance
##   between = each person's segment-mean vector vs. the leave-one-out mean of everyone else's, averaged over people
##
## Output: figures/presentations/{within,between}_schematic.png
## Usage: Rscript scripts/preprocessing/viz/plot_within_between_schematic.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(here)
})

N <- 15; N_WIN <- 4; N_PEOPLE <- 4
SHIFT <- 1.2  # blocks each later window is offset right
TS <- 2  # text scale
INK <- "#0b0b0b"; INK2 <- "#52514e"; ACCENT <- "#2a78d6"

bump <- function(mu, sd = 0.8, h = 1) h * exp(-((1:N) - mu)^2 / (2 * sd^2))
cos_dist <- function(a, b) 1 - sum(a * b) / sqrt(sum(a^2) * sum(b^2))
# person x window peaks: people move between targets, and differ in which targets and how often
peaks <- list(c(4, 4.5, 11, 11.5), c(4.5, 5, 4, 10.5), c(7.5, 8, 11, 7), c(11, 11.5, 4, 11.5))
V <- lapply(peaks, \(pk) lapply(pk, \(m) bump(m) + bump(ifelse(m < 8, m + 7, m - 7), 0.9, 0.12)))

within_pairs <- combn(N_WIN, 2)
D1 <- matrix(NA, N_WIN, N_WIN)
for (k in seq_len(ncol(within_pairs))) {
  i <- within_pairs[1, k]; j <- within_pairs[2, k]
  D1[i, j] <- D1[j, i] <- cos_dist(V[[1]][[i]], V[[1]][[j]])
}
within_p1 <- mean(D1[upper.tri(D1)])
means <- lapply(V, \(w) Reduce(`+`, w) / N_WIN)
loo <- lapply(seq_len(N_PEOPLE), \(p) Reduce(`+`, means[-p]) / (N_PEOPLE - 1))
d_between <- sapply(seq_len(N_PEOPLE), \(p) cos_dist(means[[p]], loo[[p]]))

fill_scale <- scale_fill_viridis_c(option = "inferno", limits = c(0, 1), trans = "sqrt", guide = "none")
theme_s <- theme_void(base_size = 12 * TS) +
  theme(plot.title = element_text(size = 13 * TS, face = "bold", hjust = 0, margin = margin(b = 10)),
        plot.subtitle = element_text(size = 11 * TS, color = INK2, margin = margin(b = 14)), plot.margin = margin(6, 18, 6, 10))
strip_df <- function(vecs, labels, shift = 0) {
  bind_rows(lapply(seq_along(vecs), \(k) tibble(row = k, label = labels[k], x = 1:N + (k - 1) * shift, value = vecs[[k]] / max(vecs[[k]]))))
}
strips <- function(df, gap = 1.5, label_size = 4 * TS, extra_right = 0) {
  y_of <- function(r) -r * gap
  lab <- distinct(df, row, label) |> mutate(x = max(df$x) + 1.2 + extra_right, y = y_of(row))
  ggplot(df, aes(x, y_of(row), fill = value)) +
    geom_tile(width = 0.92, height = 1) +
    geom_text(data = lab, aes(x, y, label = label), inherit.aes = FALSE, hjust = 0, size = label_size, color = INK2) +
    fill_scale + coord_fixed(clip = "off") + theme_s
}

## a: one person's windows across a segment
win_labels <- sprintf("window %d", 1:N_WIN)
pa <- strips(strip_df(V[[1]], win_labels, SHIFT)) +
  annotate("segment", x = 0.5, xend = N + (N_WIN - 1) * SHIFT + 0.5, y = -(N_WIN + 1) * 1.5, yend = -(N_WIN + 1) * 1.5,
           arrow = arrow(length = unit(2, "mm")), color = INK2) +
  annotate("text", x = 0.5, y = -(N_WIN + 1) * 1.5 - 0.9, label = "time in the segment (0.5s windows)", hjust = 0, size = 3.6 * TS, color = INK2) +
  labs(title = "a  Within-person: one participant", subtitle = "one tile vector per window (15 of 120 tiles shown)") +
  theme(plot.margin = margin(6, 162, 6, 10))

## b: pairwise distances between that person's windows
dm <- expand.grid(i = 1:N_WIN, j = 1:N_WIN) |> mutate(d = D1[cbind(i, j)]) |> filter(i < j)
pb <- ggplot(dm, aes(j, -i)) +
  geom_tile(aes(fill = d), width = 0.94, height = 0.94) +
  geom_text(aes(label = sprintf("%.2f", d), color = d > 0.5), size = 4.2 * TS) +
  scale_fill_gradient(low = "#e8f1fb", high = ACCENT, limits = c(0, 1), guide = "none") +
  scale_color_manual(values = c(`FALSE` = INK, `TRUE` = "white"), guide = "none") +
  annotate("text", x = 2:N_WIN, y = -0.25, label = 2:N_WIN, size = 3.6 * TS, color = INK2) +
  annotate("text", x = 0.6, y = -(1:(N_WIN - 1)), label = 1:(N_WIN - 1), size = 3.6 * TS, color = INK2) +
  annotate("text", x = (N_WIN + 2) / 2, y = -N_WIN - 0.1, size = 4.4 * TS, color = INK, fontface = "bold",
           label = sprintf("within-person = mean of %d pairs = %.2f", ncol(within_pairs), within_p1)) +
  coord_fixed(xlim = c(0.4, N_WIN + 0.6), ylim = c(-N_WIN - 0.5, 0), clip = "off") +
  labs(title = "b  Pairwise cosine distance", subtitle = "window × window\nlow = looking at the same thing") +
  theme_s

## c: each person's segment-mean vector
pc <- strips(strip_df(means, sprintf("participant %d", 1:N_PEOPLE))) +
  annotate("rect", xmin = 0.45, xmax = N + 0.55, ymin = -1.5 - 0.6, ymax = -1.5 + 0.6, fill = NA, color = ACCENT, linewidth = 0.8) +
  labs(title = "c  Between-person: mean vectors", subtitle = "average of each person's windows") +
  theme(plot.margin = margin(6, 198, 6, 10))

## d: one person vs. the leave-one-out mean of the others
d_df <- strip_df(list(means[[1]], loo[[1]]), c("participant 1", "mean of everyone else"))
pd <- strips(d_df, gap = 2) +
  annotate("rect", xmin = 0.45, xmax = N + 0.55, ymin = -2 - 0.6, ymax = -2 + 0.6, fill = NA, color = ACCENT, linewidth = 0.8) +
  annotate("text", x = 0.5, y = -5.6, hjust = 0, vjust = 1, size = 4.2 * TS, color = INK2,
           label = sprintf("cosine distance = %.2f\nrepeat for each participant: %s", d_between[1],
                           paste(sprintf("%.2f", d_between), collapse = ", "))) +
  annotate("text", x = 0.5, y = -9.8, hjust = 0, size = 4.4 * TS, color = INK, fontface = "bold",
           label = sprintf("between-person = mean over participants = %.2f", mean(d_between))) +
  labs(title = "d  Each participant vs. everyone else", subtitle = "leave one out: their mean vs. the mean of the others") +
  theme(plot.margin = margin(6, 270, 90, 10))

## e, f: the same vectors as points in vector space (2D classical MDS of their cosine distances)
PCOL <- c("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
all_win <- unlist(V, recursive = FALSE)
Dall <- outer(seq_along(all_win), seq_along(all_win), Vectorize(\(i, j) cos_dist(all_win[[i]], all_win[[j]])))
xy <- cmdscale(Dall, k = 2)
pts <- tibble(person = factor(rep(1:N_PEOPLE, each = N_WIN)), win = rep(1:N_WIN, N_PEOPLE), x = xy[, 1], y = xy[, 2])
cent <- pts |> group_by(person) |> summarise(x = mean(x), y = mean(y))
grand <- summarise(cent, x = mean(x), y = mean(y))
lims <- list(x = range(pts$x) + c(-0.12, 0.12), y = range(pts$y) + c(-0.12, 0.12))
space_theme <- theme_s + theme(panel.border = element_rect(fill = NA, color = "#c3c2be"), legend.position = "none",
                               plot.margin = margin(6, 54, 6, 30))
p1_pairs <- filter(pts, person == 1) |> (\(q) bind_rows(lapply(seq_len(ncol(within_pairs)), \(k)
  tibble(x = q$x[within_pairs[1, k]], y = q$y[within_pairs[1, k]], xend = q$x[within_pairs[2, k]], yend = q$y[within_pairs[2, k]]))))()
pe <- ggplot(pts, aes(x, y)) +
  geom_point(data = filter(pts, person != 1), aes(color = person), size = 4, alpha = 0.25) +
  geom_segment(data = p1_pairs, aes(x = x, y = y, xend = xend, yend = yend), color = PCOL[1], linewidth = 1, alpha = 0.7) +
  geom_point(data = filter(pts, person == 1), color = PCOL[1], size = 5.5) +
  geom_text(data = filter(pts, person == 1), aes(label = paste0("w", win)), color = PCOL[1], size = 3.8 * TS,
            nudge_x = c(-0.07, 0.07, -0.07, 0.07), nudge_y = c(0.04, -0.04, 0.04, -0.04)) +
  scale_color_manual(values = PCOL) + coord_equal(xlim = lims$x, ylim = lims$y) +
  labs(title = "e  Within, vector space", subtitle = "dot = one window\nlines = pairs in b") + space_theme
pf <- ggplot(pts, aes(x, y)) +
  geom_point(aes(color = person), size = 4, alpha = 0.3) +
  geom_segment(data = cent, aes(x = x, y = y, xend = grand$x, yend = grand$y, color = person), linewidth = 1.2) +
  geom_text(data = cent, aes(color = person), label = "\u2605", size = 14) +
  geom_point(data = grand, shape = 4, size = 9, stroke = 3, color = INK) +
  scale_color_manual(values = PCOL) + coord_equal(xlim = lims$x, ylim = lims$y) +
  labs(title = "f  Between, vector space", subtitle = "\u2605 = participant mean\n\u2716 = group mean") + space_theme

CAPTION <- "Illustrative vectors (made up); distances are computed from them. Real vectors have 120 tiles."
save_fig <- function(p, name, w, h) {
  p <- p + plot_annotation(caption = CAPTION) & theme(plot.caption = element_text(color = INK2, size = 10 * TS, hjust = 0))
  out <- here("figures", "presentations", name)
  withCallingHandlers(  # patchwork 1.3.1 + ggplot2 4.0 emit a spurious annotation-theme warning
    ggsave(out, p, width = w, height = h, dpi = 200, bg = "white"),
    warning = function(w) if (grepl("annotation\\$theme", conditionMessage(w))) invokeRestart("muffleWarning"))
  cat("Saved", out, "\n")
}
save_fig(pa + pb + pe + plot_layout(widths = c(1.3, 0.75, 0.8)), "within_schematic.png", 24, 7)
save_fig(pc + pd + pf + plot_layout(widths = c(1, 1.05, 0.8)), "between_schematic.png", 24, 6.8)
cat(sprintf("within (participant 1) = %.3f; between per person = %s; mean = %.3f\n", within_p1,
            paste(sprintf("%.3f", d_between), collapse = ", "), mean(d_between)))

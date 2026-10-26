## plot_ig_waterfall_kids_adults.R
## Adults vs kids, all blocks combined: (A) each model's information gain explained (% of ceiling) and (B) the
## shared -> unique -> all-three waterfall, adults and kids side by side. Each group scored on its own pipeline
## (own floor, curves and ceiling). Whiskers = segment-bootstrap 95% CIs from bootstrap_ci.py --unit segment.
##
## Output: figures/information_gain/kids_vs_adults/ig_waterfall_kids_adults_<set>_<variant>[_<block>].png
## Usage:  Rscript scripts/analysis/information_gain/plot_ig_waterfall_kids_adults.R [--set cuts --variant knots_floor --block pixar]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(here)
})

a <- commandArgs(trailingOnly = TRUE)
SET <- if ("--set" %in% a) a[match("--set", a) + 1] else "cuts"
VARIANT <- if ("--variant" %in% a) a[match("--variant", a) + 1] else "knots_floor"
BLOCK <- if ("--block" %in% a) a[match("--block", a) + 1] else NULL  # one block, from bootstrap_ci.py --by block
OUT_DIR <- here("figures", "information_gain", "kids_vs_adults")
COLORS <- c(saliency = "#1e4c7c", motion = "#357a76", meaning = "#f09937", shared = "#8a8984", all = "white")  # slide colors
INK <- "#0b0b0b"; INK2 <- "#52514e"
GROUPS <- c(adults = "", kids = "kids")
ALPHA <- c(adults = 1, kids = 0.45)  # kids = lighter tint of the same feature color
DX <- 0.19  # half-gap between the adult and kid bar

b <- bind_rows(lapply(names(GROUPS), function(g)
  read_csv(here("data", "results", "information_gain", GROUPS[[g]], sprintf("bootstrap_%s_%s_segment%s.csv", SET, VARIANT, if (is.null(BLOCK)) "" else "_by_block")),
           show_col_types = FALSE) |> filter(if (is.null(BLOCK)) TRUE else group == BLOCK) |> mutate(who = g))) |>
  mutate(across(c(estimate, lo, hi), ~ 100 * .x), who = factor(who, names(GROUPS)), off = if_else(who == "adults", -DX, DX))

YMAX <- max(40, 5 * ceiling((max(b$hi) + 4) / 5))
theme_ig <- theme_classic(base_size = 13) +
  theme(axis.text = element_text(color = INK2), axis.text.x = element_text(color = INK, size = 12),
        axis.title = element_text(color = INK2), plot.title = element_text(size = 14, face = "bold"),
        legend.position = "top", legend.justification = "left", legend.text = element_text(size = 12, color = INK))
alpha_scale <- scale_alpha_manual(values = ALPHA, name = NULL,
                                  guide = guide_legend(override.aes = list(fill = "#52514e", color = NA)))
outline <- scale_color_manual(values = c(`TRUE` = INK, `FALSE` = NA), guide = "none")
pct <- function(x) paste0(x, "%")

## ---- A: single models + all three ---------------------------------------------------------------------
models <- tibble(quantity = c("prop_S", "prop_M", "prop_X", "prop_SMX"),
                 label = c("salience", "motion", "semantics", "all three"),
                 key = c("saliency", "motion", "meaning", "all")) |>
  mutate(i = row_number()) |> left_join(b, by = "quantity")
pA <- ggplot(models, aes(i + off)) +
  geom_rect(aes(xmin = i + off - 0.17, xmax = i + off + 0.17, ymin = 0, ymax = estimate, fill = key, alpha = who,
                color = key == "all"), linewidth = 0.7) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.1, linewidth = 0.6, color = INK) +
  geom_text(aes(y = hi, label = sprintf("%.1f", estimate)), vjust = -0.6, size = 3.6, color = INK) +
  scale_fill_manual(values = COLORS, guide = "none") + alpha_scale + outline +
  scale_x_continuous(breaks = 1:4, labels = c("salience", "motion", "semantics", "all\nthree"), expand = expansion(add = 0.4)) +
  scale_y_continuous(limits = c(0, YMAX), expand = c(0, 0), labels = pct) +
  labs(title = "A. Each model", x = NULL, y = "information gain explained (% of ceiling)") +
  theme_ig

## ---- B: waterfall, adults and kids side by side ---------------------------------------------------------
steps <- tibble(quantity = c("shared", "unique_saliency", "unique_motion", "unique_meaning"),
                key = c("shared", "saliency", "motion", "meaning"), i = 1:4)
waterfall_pair <- function(b, title, ymax, ylab = NULL, ymin = NULL) {
  wf <- b |> inner_join(steps, by = "quantity") |> arrange(who, i) |> group_by(who) |>
    mutate(end = cumsum(estimate), start = lag(end, default = 0)) |> ungroup() |>
    mutate(ci_lo = start + lo, ci_hi = start + hi)
  tot <- b |> filter(quantity == "prop_SMX") |>
    mutate(key = "all", i = 5, start = 0, end = estimate, ci_lo = lo, ci_hi = hi)
  wf <- bind_rows(wf, tot) |>
    mutate(txt = if_else(key %in% c("shared", "all"), sprintf("%.1f", estimate), sprintf("%+.1f", estimate)))
  ggplot(wf) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.4) +
    geom_rect(aes(xmin = i + off - 0.17, xmax = i + off + 0.17, ymin = start, ymax = end, fill = key, alpha = who,
                  color = key == "all"), linewidth = 0.7) +
    geom_errorbar(aes(x = i + off, ymin = ci_lo, ymax = ci_hi), width = 0.1, linewidth = 0.6, color = INK) +
    geom_text(aes(x = i + off, y = pmax(start, end, ci_hi), label = txt), vjust = -0.6, size = 3.4, color = INK) +
    scale_fill_manual(values = COLORS, guide = "none") + alpha_scale + outline +
    scale_x_continuous(breaks = 1:5, labels = c("shared", "+ unique\nsalience", "+ unique\nmotion", "+ unique\nsemantics", "= all\nthree"),
                       expand = expansion(add = 0.4)) +
    scale_y_continuous(expand = c(0, 0), labels = pct) +
    coord_cartesian(ylim = c(if (is.null(ymin)) min(0, 5 * floor(min(wf$ci_lo) / 5)) else ymin, ymax)) +
    labs(title = title, x = NULL, y = ylab) +
    theme_ig
}
pB <- waterfall_pair(b, "B. How the all-three model breaks down", YMAX)

n_kids <- 19  # participants in kids scores_<set>_<variant>.parquet
p <- pA + pB + plot_layout(widths = c(1, 1.35), guides = "collect") &
  theme(legend.position = "top", legend.justification = "left")
p <- p + plot_annotation(caption = sprintf(paste0(
  "Left bar = adults (45), right bar = kids (%d); each group scored on its own pipeline (own floor, curves and ceiling), %s.\n",
  "%% of ceiling = group's average information gain / group's average ceiling. Whiskers = 95%% CI, bootstrap over segments (2,000 draws). Unchopped segments."), n_kids,
  if (is.null(BLOCK)) "all blocks combined" else sprintf("%s block only (%d segments)", BLOCK, b$n_segments[b$who == "kids"][1])),
  theme = theme(plot.caption = element_text(color = INK2, size = 10, hjust = 0)))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
out <- file.path(OUT_DIR, sprintf("ig_waterfall_kids_adults_%s_%s%s.png", SET, VARIANT, if (is.null(BLOCK)) "" else paste0("_", BLOCK)))
ggsave(out, p, width = 13, height = 5.8, dpi = 200, bg = "white")
cat("Saved", out, "\n")

## ---- Waterfall per block / per adult quadrant (50/50 split), 2 x 2 (bootstrap_ci.py --unit segment --by ...) ---
GROUPINGS <- list(block = list(levs = c("frank", "pixar", "sesame", "slow"), tag = "_by_block", note = "block"),
                  quadrant = list(levs = c("selective", "scattered", "concentrated", "coordinated"), tag = "_by_quadrant_q50",
                                  note = "adult quadrant (median split on adult within/between; same segments for kids)"))
for (by in names(GROUPINGS)) {
  G <- GROUPINGS[[by]]
  bb <- bind_rows(lapply(names(GROUPS), function(g)
    read_csv(here("data", "results", "information_gain", GROUPS[[g]], sprintf("bootstrap_%s_%s_segment%s.csv", SET, VARIANT, G$tag)),
             show_col_types = FALSE) |> mutate(who = g))) |>
    mutate(across(c(estimate, lo, hi), ~ 100 * .x), who = factor(who, names(GROUPS)), off = if_else(who == "adults", -DX, DX))
  ymin4 <- min(0, 5 * floor(min(bb$lo[bb$quantity %in% c("prop_SMX", "shared")]) / 5))  # shared axis across panels
  ymax4 <- 5 * ceiling((max(bb$hi[bb$quantity %in% c("prop_SMX", "shared")]) + 5) / 5)
  panels <- lapply(G$levs, function(k) {
    d <- filter(bb, group == k)
    waterfall_pair(d, sprintf("%s (%d segments)", tools::toTitleCase(k), d$n_segments[d$who == "kids"][1]), ymax4,
                   if (k %in% G$levs[c(1, 3)]) "information gain explained (% of ceiling)" else NULL, ymin4)
  })
  p4 <- wrap_plots(panels, ncol = 2) + plot_layout(guides = "collect") &
    theme(legend.position = "top", legend.justification = "left")
  p4 <- p4 + plot_annotation(caption = sprintf(paste0(
    "Left bar = adults (45), right bar = kids (%d); each group scored on its own pipeline (own floor, curves and ceiling). Grouped by %s.\n",
    "%% of ceiling = group's average information gain / group's average ceiling. Whiskers = 95%% CI, bootstrap over segments within the group (2,000 draws). Unchopped segments."),
    n_kids, G$note), theme = theme(plot.caption = element_text(color = INK2, size = 10, hjust = 0)))
  out4 <- file.path(OUT_DIR, sprintf("ig_waterfall_kids_adults%s_%s_%s.png", G$tag, SET, VARIANT))
  ggsave(out4, p4, width = 14, height = 10.5, dpi = 200, bg = "white")
  cat("Saved", out4, "\n")
}

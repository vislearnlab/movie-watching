## plot_ig_ci_waterfall.R
## Alternative to the violin "overall" figure: (A) each model's information gain explained (% of ceiling) with
## bootstrap 95% CIs (default: resampling segments); (B) waterfall from 0 through shared and each unique component to the all-three model.
## Data from bootstrap_ci.py.
##
## Output: figures/information_gain/adults/ig_ci_waterfall_<set>_<variant>.png
## Usage:  Rscript scripts/analysis/information_gain/plot_ig_ci_waterfall.R [--set cuts --variant knots_floor --unit segment|participant|clip --q 0.4|0.5]

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
UNIT <- if ("--unit" %in% a) a[match("--unit", a) + 1] else "segment"
Q <- if ("--q" %in% a) as.numeric(a[match("--q", a) + 1]) else 0.4  # quadrant split quantile
QTAG <- if (Q == 0.4) "" else sprintf("_q%d", round(100 * Q))
OUT_DIR <- here("figures", "information_gain", "adults")
COLORS <- c(saliency = "#1e4c7c", motion = "#357a76", meaning = "#f09937", shared = "#8a8984", all = "white")  # slide colors; all three = outlined
INK <- "#0b0b0b"; INK2 <- "#52514e"
FACE_CLIPS <- c("frank_complex", "frank_play", "sesameindia_1", "sesameindia_2", "sesameus_1", "sesameus_2", "slow_people")  # config.FACE_CLIPS

b <- read_csv(here("data", "results", "information_gain", sprintf("bootstrap_%s_%s%s.csv", SET, VARIANT, if (UNIT == "participant") "" else paste0("_", UNIT))), show_col_types = FALSE) |>
  mutate(across(c(estimate, lo, hi), ~ 100 * .x))
get <- function(q) filter(b, quantity == q)

theme_ig <- theme_classic(base_size = 13) +
  theme(axis.text = element_text(color = INK2), axis.text.x = element_text(color = INK, size = 12),
        axis.title = element_text(color = INK2), plot.title = element_text(size = 14, face = "bold"),
        legend.position = "none")

## ---- A: single models + all three ---------------------------------------------------------------------
models <- tibble(quantity = c("prop_S", "prop_M", "prop_X", "prop_SMX"),
                 label = c("salience", "motion", "semantics", "all three"),
                 key = c("saliency", "motion", "meaning", "all")) |>
  left_join(b, by = "quantity") |> mutate(label = factor(label, label))
pA <- ggplot(models, aes(label, estimate, fill = key)) +
  geom_col(aes(color = key == "all"), width = 0.6, linewidth = 0.7) +
  scale_color_manual(values = c(`TRUE` = INK, `FALSE` = NA), guide = "none") +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, linewidth = 0.6, color = INK) +
  geom_text(aes(y = hi, label = sprintf("%.1f%%", estimate)), vjust = -0.7, size = 4.2, color = INK) +
  scale_fill_manual(values = COLORS) +
  scale_y_continuous(limits = c(0, 35), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  labs(title = "A. Each model", x = NULL, y = "information gain explained (% of ceiling)") +
  theme_ig

## ---- B: waterfall 0 -> shared -> uniques -> all three --------------------------------------------------
waterfall <- function(bb, title, ymax = 35, ylab = NULL, ymin = NULL) {
  total <- filter(bb, quantity == "prop_SMX")
  steps <- tibble(quantity = c("shared", "unique_saliency", "unique_motion", "unique_meaning"),
                  label = c("shared", "+ unique\nsalience", "+ unique\nmotion", "+ unique\nsemantics"),
                  key = c("shared", "saliency", "motion", "meaning")) |>
    left_join(bb, by = "quantity") |>
    mutate(end = cumsum(estimate), start = lag(end, default = 0), i = row_number())
  wf <- bind_rows(steps, tibble(label = "= all\nthree", key = "all", start = 0, end = total$estimate,
                                estimate = total$estimate, lo = total$lo, hi = total$hi, i = nrow(steps) + 1)) |>
    mutate(label = factor(label, label), ci_lo = start + lo, ci_hi = start + hi,
           ci_lo = if_else(key == "all", lo, ci_lo), ci_hi = if_else(key == "all", hi, ci_hi),
           txt = if_else(key %in% c("shared", "all"), sprintf("%.1f%%", estimate), sprintf("%+.1f%%", estimate)))
  ggplot(wf) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.4) +
    geom_segment(data = filter(wf, i < max(i)), aes(x = i + 0.3, xend = i + 0.7, y = end, yend = end),
                 color = INK2, linewidth = 0.4, linetype = "dotted") +
    geom_rect(aes(xmin = i - 0.3, xmax = i + 0.3, ymin = start, ymax = end, fill = key, color = key == "all"), linewidth = 0.7) +
    scale_color_manual(values = c(`TRUE` = INK, `FALSE` = NA), guide = "none") +
    geom_errorbar(aes(x = i, ymin = ci_lo, ymax = ci_hi), width = 0.14, linewidth = 0.6, color = INK) +
    geom_text(aes(x = i, y = pmax(start, end, ci_hi), label = txt), vjust = -0.7, size = 4.2, color = INK) +
    scale_fill_manual(values = COLORS) +
    scale_x_continuous(breaks = wf$i, labels = levels(wf$label), expand = expansion(add = 0.4)) +
    scale_y_continuous(expand = c(0, 0), labels = function(x) paste0(x, "%")) +
    coord_cartesian(ylim = c(if (is.null(ymin)) min(0, wf$ci_lo) else ymin, ymax)) +
    labs(title = title, x = NULL, y = ylab) +
    theme_ig
}
pB <- waterfall(b, "B. How the all-three model breaks down")

p <- pA + pB + plot_layout(widths = c(1, 1.3)) +
  plot_annotation(caption = sprintf("Bars = group value (participants weighted equally); whiskers = 95%% CI, bootstrap over %ss (2,000 draws). Unchopped segments, 45 adults.", UNIT),
                  theme = theme(plot.caption = element_text(color = INK2, size = 10, hjust = 0)))
out <- file.path(OUT_DIR, sprintf("ig_ci_waterfall_%s_%s%s.png", SET, VARIANT, if (UNIT == "segment") "" else paste0("_", UNIT)))
ggsave(out, p, width = 12, height = 5.2, dpi = 200, bg = "white")
cat("Saved", out, "\n")

## ---- Waterfall per block / per quadrant, 2 x 2 (bootstrap_ci.py --by block|quadrant) -----------------------
GROUPINGS <- list(block = c("frank", "pixar", "sesame", "slow"),
                  quadrant = c("selective", "scattered", "concentrated", "coordinated"),  # quadrant layout as the schematic
                  face = c("face", "no_face"))
TITLES <- c(face = "Face clips", no_face = "No-face clips")
counts <- read_csv(here("data", "results", "information_gain", sprintf("plotdata_groups_%s_%s.csv", SET, VARIANT)),
                   show_col_types = FALSE)
for (by in names(GROUPINGS)) {
  path <- here("data", "results", "information_gain", sprintf("bootstrap_%s_%s%s_by_%s%s.csv", SET, VARIANT,
                                                             if (UNIT == "participant") "" else paste0("_", UNIT), by, if (by == "quadrant") QTAG else ""))
  if (!file.exists(path)) next
  bb <- read_csv(path, show_col_types = FALSE) |> mutate(across(c(estimate, lo, hi), ~ 100 * .x))
  n_seg <- if ("n_segments" %in% names(bb)) distinct(bb, facet = group, n_segments) else counts |> filter(facet_type == by)
  ymax <- 5 * ceiling((max(bb$hi[bb$quantity == "prop_SMX"], bb$estimate[bb$quantity == "shared"] + 12) + 4) / 5)
  ymin <- min(0, 5 * floor(min(bb$lo[bb$quantity == "shared"]) / 5))
  levs <- GROUPINGS[[by]]
  panels <- lapply(levs, function(g)
    waterfall(filter(bb, group == g), sprintf("%s (%d segments)", if (g %in% names(TITLES)) TITLES[[g]] else tools::toTitleCase(g), n_seg$n_segments[n_seg$facet == g]),
              ymax, if (g %in% levs[c(1, 3)]) "information gain explained (% of ceiling)" else NULL, ymin))
  cap <- sprintf("%% of ceiling = group's average information gain / group's average ceiling (ceiling measured per segment).\nWhiskers = 95%% CI, bootstrap over %ss within the group (2,000 draws). Unchopped segments, 45 adults.", UNIT)
  if (by == "quadrant") cap <- paste0(cap, if (Q == 0.5) "\nQuadrants: pooled median (50/50) split on within/between distance; every segment with a within/between value classified."
                                       else sprintf("\nQuadrants: pooled %.0f/%.0f split on within/between distance; middle-band segments unclassified and left out.", 100 * Q, 100 * (1 - Q)))
  p4 <- wrap_plots(panels, ncol = 2) +
    plot_annotation(caption = cap, theme = theme(plot.caption = element_text(color = INK2, size = 10, hjust = 0)))
  out4 <- file.path(OUT_DIR, sprintf("ig_waterfall_by_%s_%s_%s%s%s.png", by, SET, VARIANT, if (UNIT == "segment") "" else paste0("_", UNIT),
                                     if (by == "quadrant") QTAG else ""))
  if (by == "face") p4 <- p4 + plot_annotation(caption = paste0(cap, sprintf("\nFace clips: %s. Same three-feature (SMX) models in both.", paste(FACE_CLIPS, collapse = ", "))))
  ggsave(out4, p4, width = 13, height = if (length(levs) == 2) 5.6 else 10, dpi = 200, bg = "white")
  cat("Saved", out4, "\n")
}

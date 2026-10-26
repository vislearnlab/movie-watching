## plot_information_gain.R
## All information-gain figures (plotting is in R; the Python pipeline writes the CSVs via plot_data.py).
## Overall IG (violins of segments or participants + stacked partition), faceted by block / quadrant / clip,
## supplementary model bars, per-segment scatters, and the fitted-curves check.
##
## Usage:
##   Rscript scripts/analysis/information_gain/plot_information_gain.R --group adults --segment-set cuts
##   --group: adults | kids-kidref | kids-adultref;  --segment-set: cuts | slow_chopped;  --only <regex>
##   --subset: all (12 clips; saliency, motion, meaning) | faceclips (7 clips with faces; + face) | allclips4 (12 clips; + face)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(here)
})

## ---- Args -------------------------------------------------------------------
arg <- function(name, default) {
  a <- commandArgs(trailingOnly = TRUE)
  i <- match(paste0("--", name), a)
  if (is.na(i)) default else a[i + 1]
}
GROUP <- arg("group", "adults")
SEGMENT_SET <- arg("segment-set", "cuts")
VARIANT <- arg("variant", "knots_floor")
ONLY <- arg("only", ".")
SUBSET <- arg("subset", "all")
SUFFIX <- if (SUBSET == "all") "" else paste0("_", SUBSET)

RESULTS <- here("data", "results", "information_gain")
DATA_DIR <- if (GROUP == "kids-kidref") file.path(RESULTS, "kids") else RESULTS
DATA_TAG <- paste0(if (GROUP == "kids-adultref") paste0(SEGMENT_SET, "_kids_adultref") else paste0(SEGMENT_SET, "_", VARIANT), SUFFIX)
CURVES_TAG <- paste0(SEGMENT_SET, "_", VARIANT)  # curves are fit once on all clips
FIG_TAG <- paste0(SEGMENT_SET, "_", VARIANT, SUFFIX)
FIG_DIR <- here("figures", "information_gain", GROUP)
CHECK_DIR <- if (GROUP == "adults") here("figures", "checks") else here("figures", "checks", "information_gain_kids")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

## ---- Style ------------------------------------------------------------------
COLORS <- c(spatial = "#5a3e9e", saliency = "#2a78d6", motion = "#eb6834", meaning = "#1baf7a", face = "#b5459b",
            shared = "#8a8984")  # face and spatial each pass all-pairs CVD against saliency/motion/meaning
FEATURES <- c(if (VARIANT == "uniform") "spatial", "saliency", "motion", "meaning", if (SUBSET != "all") "face")
ABBR <- c(spatial = "P", saliency = "S", motion = "M", meaning = "X", face = "F")
ALL_LABEL <- if (length(FEATURES) == 4) "all four" else "all three"
source(here("scripts", "analysis", "block_colors.R"))  # BLOCK_COLORS
INK <- "#0b0b0b"; INK2 <- "#52514e"
COMP_KEY <- c(shared = "shared", unique_spatial = "spatial", unique_motion = "motion", unique_saliency = "saliency", unique_meaning = "meaning", unique_face = "face")
COMP_KEY <- COMP_KEY[COMP_KEY %in% c("shared", FEATURES)]
COMPONENTS <- setNames(ifelse(COMP_KEY == "shared", "shared", paste0("unique\n", COMP_KEY)), names(COMP_KEY))
FACET_ORDER <- list(
  block = names(BLOCK_COLORS),
  quadrant = c("coordinated", "concentrated", "selective", "scattered"),
  faces = c("clips with faces", "clips without faces"),
  clip = c("frank_complex", "frank_objects", "frank_play", "pixar_birds", "sesameindia_1", "sesameindia_2",
           "sesameus_1", "sesameus_2", "slow_animals", "slow_hands", "slow_inanimate", "slow_people")
)
UNIT_NOUN <- c(segment = "segments", participant = "participants")
pct <- function(x, acc = 0.1) scales::percent(x, accuracy = acc)
signed_pct <- function(x) sprintf("%+.1f%%", 100 * x)

theme_ig <- theme_classic(base_size = 11) +
  theme(axis.text = element_text(color = INK2), axis.title = element_text(color = INK2),
        axis.ticks = element_line(color = INK2), strip.background = element_blank(),
        strip.text = element_text(size = 9, color = INK), plot.title = element_text(hjust = 0.5, size = 11),
        plot.caption = element_text(hjust = 0, size = 7.5, color = INK2), plot.caption.position = "plot")

save <- function(p, name, w, h, dir = FIG_DIR) {
  if (!grepl(ONLY, name)) return(invisible())
  withCallingHandlers(  # patchwork 1.3.1 + ggplot2 4.0 emit a spurious annotation-theme warning
    ggsave(file.path(dir, paste0(name, ".png")), p, width = w, height = h, dpi = 150, bg = "white"),
    warning = function(w) if (grepl("annotation\\$theme", conditionMessage(w))) invokeRestart("muffleWarning"))
  cat("Saved", file.path(dir, paste0(name, ".png")), "\n")
}

## ---- Data -------------------------------------------------------------------
groups <- read_csv(file.path(DATA_DIR, paste0("plotdata_groups_", DATA_TAG, ".csv")), show_col_types = FALSE)
units <- read_csv(file.path(DATA_DIR, paste0("plotdata_units_", DATA_TAG, ".csv")), show_col_types = FALSE)
overall <- filter(groups, facet_type == "overall")
who <- if (GROUP == "adults") "adults" else "kids"

subtitle <- paste0(c(all = "", faceclips = "7 clips with faces, ", allclips4 = "all 12 clips, face as 4th feature, ")[[SUBSET]], c(cuts = "unchopped", slow_chopped = "slow chopped")[[SEGMENT_SET]], ", ",
                   c(knots_floor = "20 knots, floor", exp_floor = "exp, floor", knots_nofloor = "20 knots, no floor",
                     uniform = "uniform baseline, spatial prior as a feature")[[VARIANT]],
                   c(adults = "", `kids-kidref` = "; KIDS, kids reference", `kids-adultref` = "; KIDS vs adult models")[[GROUP]])
floor_txt <- if (VARIANT == "knots_nofloor") "feature models do NOT include the floor (beta=0); scored against the floor" else
  if (VARIANT == "uniform") paste0("baseline = uniform; spatial = the old floor (leave-one-clip-out pooled gaze); saliency/motion/meaning ",
                                   "are the no-floor models (beta=0); ceiling = same model as before, scored over uniform") else
  "each feature model includes the floor (beta=1)"
ref_txt <- if (GROUP == "kids-adultref") paste0("Kids scored against adult-derived models: floor, lambda, feature curves and the ",
                                               "ceiling's other viewers are all adults. ") else ""
base_caption <- paste0(ref_txt, "Combined models are fixed equal-weight mixtures of the single-feature models; ", floor_txt,
                       ". Unique = ", ALL_LABEL, " minus the mixture without that feature; shared = ", ALL_LABEL, " minus the uniques. ",
                       overall$n_participants, " ", who, ", ", overall$n_segments, " segments, leave-one-clip-out. No confidence intervals yet.")
wrap <- function(x, width = 190) paste(strwrap(x, width), collapse = "\n")

## ---- Overall IG: violins + stacked partition ---------------------------------
y_limits <- function(lo, hi) { pad <- 0.05 * (hi - lo); c(lo - pad, hi + pad) }

stacked_panel <- function(g, ylim) {
  v <- unlist(g[names(COMPONENTS)])
  pos <- cumsum(pmax(v, 0)); neg <- cumsum(pmin(v, 0))
  bars <- tibble(component = names(v), ymin = ifelse(v >= 0, pos - v, neg - v), ymax = ifelse(v >= 0, pos, neg))
  total <- g$ig_all / g$ceiling
  ggplot(bars) +
    geom_rect(aes(xmin = -0.25, xmax = 0.25, ymin = ymin, ymax = ymax, fill = component), color = "white", linewidth = 0.6) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.3) +
    { if (ylim[2] >= 1) geom_hline(yintercept = 1, color = INK2, linetype = "dashed", linewidth = 0.4) } +
    annotate("segment", x = -0.3, xend = 0.3, y = total, yend = total, color = INK, linewidth = 0.7) +
    annotate("text", x = 0.33, y = total, label = paste0(ALL_LABEL, "\n", pct(total)), hjust = 0, size = 2.9, color = INK) +
    scale_fill_manual(values = setNames(COLORS[COMP_KEY], names(COMP_KEY)), guide = "none") +
    scale_x_continuous(limits = c(-0.6, 1.0), breaks = 0, labels = "stacked") +
    scale_y_continuous(labels = scales::label_percent(), limits = ylim) +
    labs(x = NULL, y = NULL) + theme_ig
}

headline <- function(unit_type, title_extra, caption_extra) {
  g <- overall
  u <- units |> filter(facet_type == "overall", unit_type == !!unit_type) |>
    mutate(component = factor(component, names(COMPONENTS)))
  gv <- tibble(component = factor(names(COMPONENTS), names(COMPONENTS)), value = unlist(g[names(COMPONENTS)]))
  v <- gv$value; total <- g$ig_all / g$ceiling
  pos <- sum(pmax(v, 0)); neg <- sum(pmin(v, 0))
  ylim <- y_limits(min(0, u$prop, neg), max(total, pos, u$prop))
  set.seed(0)
  p1 <- ggplot(u, aes(component, prop)) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.3) +
    { if (ylim[2] >= 1) geom_hline(yintercept = 1, color = INK2, linetype = "dashed", linewidth = 0.4) } +
    geom_violin(aes(fill = component, color = component), alpha = 0.55, scale = "width", width = 0.7, trim = TRUE) +
    geom_jitter(width = 0.06, height = 0, size = 0.9, alpha = 0.35, color = INK, stroke = 0) +
    geom_segment(data = gv, aes(x = as.numeric(component) - 0.25, xend = as.numeric(component) + 0.25, y = value, yend = value),
                 color = INK, linewidth = 0.8) +
    scale_fill_manual(values = setNames(COLORS[COMP_KEY], names(COMP_KEY)), guide = "none") +
    scale_color_manual(values = setNames(COLORS[COMP_KEY], names(COMP_KEY)), guide = "none") +
    scale_x_discrete(labels = paste0(COMPONENTS, "\n", signed_pct(v))) +
    scale_y_continuous(labels = scales::label_percent(), limits = ylim) +
    labs(x = NULL, y = sprintf("information gain, proportion of ceiling\n(100%% = ceiling = %.2f bits)", g$ceiling)) + theme_ig
  (p1 + stacked_panel(g, ylim) + plot_layout(widths = c(4, 1.4))) +
    plot_annotation(title = paste0("Overall IG\n", subtitle, title_extra), caption = wrap(paste0(base_caption, caption_extra))) &
    theme(plot.title = element_text(hjust = 0.5, size = 11), plot.caption = element_text(hjust = 0, size = 7.5, color = INK2))
}

## ---- Faceted violins (block / quadrant / clip) --------------------------------
faceted <- function(facet_type, unit_type, ncol = NULL, title_extra = "", caption_extra = "") {
  noun <- if (facet_type == "faces") "group" else facet_type
  g <- groups |> filter(facet_type == !!facet_type)
  lv <- intersect(FACET_ORDER[[facet_type]], g$facet)
  g <- g |> mutate(facet = factor(facet, lv), strip = sprintf("%s: %d %s\nceiling %.2f bits, %s %s", facet,
                   if (unit_type == "segment") n_segments else n_participants, UNIT_NOUN[[unit_type]], ceiling, ALL_LABEL, pct(ig_all / ceiling))) |>
    arrange(facet) |> mutate(strip = factor(strip, strip))
  gv <- g |> select(facet, strip, all_of(names(COMPONENTS))) |> pivot_longer(all_of(names(COMPONENTS)), names_to = "component") |>
    mutate(component = factor(component, names(COMPONENTS)), x = interaction(facet, component, lex.order = TRUE),
           label = paste0(COMPONENTS[as.character(component)], "\n", signed_pct(value)))
  u <- units |> filter(facet_type == !!facet_type, unit_type == !!unit_type, facet %in% lv) |>
    mutate(facet = factor(facet, lv), component = factor(component, names(COMPONENTS)),
           x = interaction(facet, component, lex.order = TRUE)) |> left_join(distinct(g, facet, strip), by = "facet")
  set.seed(0)
  ggplot(u, aes(x, prop)) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.3) +
    geom_violin(aes(fill = component, color = component), alpha = 0.55, scale = "width", width = 0.7, trim = TRUE) +
    geom_jitter(width = 0.06, height = 0, size = 0.9, alpha = 0.35, color = INK, stroke = 0) +
    geom_errorbar(data = gv, aes(x = x, ymin = value, ymax = value), inherit.aes = FALSE, width = 0.5, color = INK, linewidth = 0.8) +
    facet_wrap(~strip, ncol = ncol %||% length(lv), scales = "free_x") +
    scale_fill_manual(values = setNames(COLORS[COMP_KEY], names(COMP_KEY)), guide = "none") +
    scale_color_manual(values = setNames(COLORS[COMP_KEY], names(COMP_KEY)), guide = "none") +
    scale_x_discrete(labels = setNames(gv$label, as.character(gv$x))) +
    scale_y_continuous(labels = scales::label_percent()) +
    labs(x = NULL, y = sprintf("information gain, proportion of %s's ceiling", noun),
         title = paste0("Overall IG by ", facet_type, "\n", subtitle, sub("FACET", noun, title_extra)),
         caption = wrap(paste0(base_caption, " Each panel is scaled to its own ", noun, "'s ceiling (participants weighted ",
                               "equally within ", noun, "); black line = ", noun, " value.", caption_extra),
                        if (is.null(ncol)) 230 else 150)) +
    theme_ig + theme(axis.text.x = element_text(size = 7.5))
}

seg_cap <- " Violins/dots: one per segment (participant-mean within segment, divided by the overall ceiling). Black line = group value (participants weighted equally), so it can differ slightly from the segment mean."
ppl_cap <- " Violins/dots: one per participant (mean over that participant's segments, divided by the overall ceiling). Black line = group value = mean of the dots."
save(headline("segment", " (dots = segments)", seg_cap), paste0("overall_ig_", FIG_TAG), 9.5, 5.2)
save(headline("participant", " (dots = participants)", ppl_cap), paste0("overall_ig_people_", FIG_TAG), 9.5, 5.2)
q_cap <- " Quadrants from the within/between cosine metric (adult gaze), pooled 40%/60% split on each axis; middle-band segments excluded."
for (ut in c("segment", "participant")) {
  stem <- if (ut == "segment") "overall_ig_by_" else "overall_ig_people_by_"
  extra <- if (ut == "segment") " (dots = segments)" else " (dots = participants, mean within FACET)"
  save(faceted("block", ut, title_extra = extra), paste0(stem, "block_", FIG_TAG), 13.5, 5.2)
  save(faceted("quadrant", ut, title_extra = extra, caption_extra = q_cap), paste0(stem, "quadrant_", FIG_TAG), 13.5, 5.2)
  if (SUBSET != "faceclips") save(faceted("faces", ut, ncol = 2, title_extra = extra, caption_extra = " Clips with faces = the 7 clips with human or Muppet faces; without = slow_hands, slow_animals, slow_inanimate, frank_objects, pixar_birds."),
                           paste0(stem, "faces_", FIG_TAG), 9.5, 5.2)
  if (GROUP == "adults") save(faceted("clip", ut, ncol = 3, title_extra = extra), paste0(stem, "clip_", FIG_TAG), 10.5,
                                 4.75 * ceiling(sum(groups$facet_type == "clip") / 3))
}

## ---- Adults only: supplementary bars, per-segment scatters, curves check -------
if (GROUP == "adults") {
  sup <- tibble(model = factor(c(FEATURES, ALL_LABEL), c(FEATURES, ALL_LABEL)),
                key = c(FEATURES, "shared"), ig = c(unlist(overall[paste0("ig_", ABBR[FEATURES])]), overall$ig_all))
  p <- ggplot(sup, aes(model, ig, fill = key)) +
    geom_col(width = 0.62) +
    geom_text(aes(label = sprintf("%.3f\n(%s)", ig, pct(ig / overall$ceiling, 1))), vjust = -0.2, size = 3, color = INK) +
    geom_hline(yintercept = 0, color = INK, linewidth = 0.3) +
    geom_hline(yintercept = overall$ceiling, color = INK2, linetype = "dashed") +
    annotate("text", x = nrow(sup) + 0.4, y = overall$ceiling, label = sprintf("ceiling %.2f", overall$ceiling), hjust = 1, vjust = -0.4, size = 2.8, color = INK2) +
    scale_fill_manual(values = COLORS, guide = "none") +
    labs(x = NULL, y = paste0("information gain over ", if (VARIANT == "uniform") "uniform" else "floor", " (bits)"),
         title = paste0("Single-feature and ", ALL_LABEL, " models (% of ceiling)\n", subtitle)) +
    theme_ig
  save(p, paste0("supplementary_", FIG_TAG), 7, 4.4)

  fg <- groups |> filter(facet_type == "faces")
  if (nrow(fg) > 1) {  # same bars, split by clips with vs. without faces, each against its own ceiling
    fg <- fg |> mutate(strip = factor(sprintf("%s (%d clips, %d segments)", facet, ifelse(facet == "clips with faces", 7, 5), n_segments)))
    supf <- fg |> select(strip, ceiling, all_of(paste0("ig_", ABBR[FEATURES])), ig_all) |>
      pivot_longer(-c(strip, ceiling), names_to = "col", values_to = "ig") |>
      mutate(key = c(setNames(FEATURES, paste0("ig_", ABBR[FEATURES])), ig_all = "shared")[col],
             model = factor(ifelse(key == "shared", ALL_LABEL, key), c(FEATURES, ALL_LABEL)))
    p <- ggplot(supf, aes(model, ig, fill = key)) +
      geom_col(width = 0.62) +
      geom_text(aes(label = sprintf("%.3f\n(%s)", ig, pct(ig / ceiling, 1))), vjust = -0.2, size = 3, color = INK) +
      geom_hline(yintercept = 0, color = INK, linewidth = 0.3) +
      geom_hline(aes(yintercept = ceiling), color = INK2, linetype = "dashed") +
      geom_text(data = distinct(supf, strip, ceiling), aes(x = length(FEATURES) + 1.4, y = ceiling, label = sprintf("ceiling %.2f", ceiling)),
                inherit.aes = FALSE, hjust = 1, vjust = -0.4, size = 2.8, color = INK2) +
      facet_wrap(~strip) +
      scale_fill_manual(values = COLORS, guide = "none") +
      labs(x = NULL, y = "information gain over floor (bits)",
           title = paste0("Single-feature and ", ALL_LABEL, " models by clips with/without faces (% of that group's ceiling), ", subtitle)) +
      theme_ig
    save(p, paste0("supplementary_by_faces_", FIG_TAG), 11, 4.6)
  }

  seg <- read_csv(file.path(DATA_DIR, paste0("per_segment_", DATA_TAG, ".csv")), show_col_types = FALSE)
  scatter <- function(d, x, y, xlab, ylab, title) {
    ggplot(d, aes(.data[[x]], .data[[y]], color = block)) +
      geom_point(size = 1.8, alpha = 0.8) +
      scale_color_manual(values = BLOCK_COLORS) +
      labs(x = xlab, y = ylab, title = title, color = NULL) + theme_ig + theme(legend.position = c(0.9, 0.85))
  }
  save(scatter(seg, "ceiling", "prop_all", "ceiling (bits)", paste("proportion explained by", ALL_LABEL),
               paste0("Per segment: ceiling vs. proportion explained, ", subtitle)), paste0("ceiling_vs_prop_explained_", FIG_TAG), 6, 5)
  if ("mean_entropy_rarefied" %in% names(seg)) {
    m <- filter(seg, !is.na(mean_entropy_rarefied))
    save(scatter(m, "mean_entropy_rarefied", "ceiling", "group entropy (bits, rarefied, smoothed maps)", "ceiling (bits)",
                 sprintf("Per segment: entropy vs. ceiling (n=%d), r=%.2f", nrow(m), cor(m$mean_entropy_rarefied, m$ceiling))),
         paste0("ceiling_vs_entropy_", FIG_TAG), 6, 5)
  }

  curves <- read_csv(file.path(DATA_DIR, paste0("curves_", CURVES_TAG, ".csv")), show_col_types = FALSE) |>
    filter(feature %in% FEATURES) |>
    pivot_longer(starts_with("k"), names_to = "knot", values_to = "f") |>
    group_by(held_out, feature) |> mutate(x = (row_number() - 1) / (n() - 1)) |> ungroup() |>
    mutate(feature = factor(feature, FEATURES))
  p <- ggplot(curves, aes(x, f, group = held_out, color = feature)) +
    geom_line(alpha = 0.5, linewidth = 0.5) +
    facet_wrap(~feature, nrow = 1, labeller = as_labeller(function(x) paste0(x, " (one line per held-out clip)"))) +
    scale_y_log10() + scale_color_manual(values = COLORS, guide = "none") +
    labs(x = "feature value (rescaled 0-1 across segments)", y = "f(x) relative to f(0), log scale",
         title = paste0("Fitted nonlinearities, ", subtitle)) + theme_ig
  save(p, paste0("information_gain_curves_", FIG_TAG), 3.75 * length(FEATURES), 3.8, dir = CHECK_DIR)
}

## plot_participant_qc.R
## Draws each block's validation gaze on the 1920x1080 validation screen, with the 2 deg inclusion ring around each
## target, above each block's per-movie % valid samples (tracking quality is separate from calibration accuracy), next to
## reference rings (2-20 deg) so the error sizes can be read as real screen extents.
##
## Input:  data/results/qc/<PID>_block_qc.csv, _trial_qc.csv, _validation_samples.csv, validation_rings.csv
##         (scripts/preprocessing/qc/participant_block_qc.py)
## Output: figures/QC/<PID>_validation_screen.png
## Usage:  Rscript scripts/preprocessing/qc/plot_participant_qc.R MW056 [MW057 ...]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(here)
})

PIDS <- commandArgs(trailingOnly = TRUE)
W <- 1920; H <- 1080; PAD <- 120
SCREEN <- "#808080"  # PsychoPy default window gray
GAZE <- "#ffd84d"
INK2 <- "#52514e"
STATUS_FILL <- c("Recorded and usable" = "#2a78d6", "Recorded but NOT usable (calibration > 2°)" = "#eb6834")
QC_DIR <- here("data", "results", "qc")
rings <- read_csv(file.path(QC_DIR, "validation_rings.csv"), show_col_types = FALSE)

screen_base <- function() {
  list(
    annotate("rect", xmin = 0, xmax = W, ymin = 0, ymax = H, fill = SCREEN),
    scale_y_reverse(), coord_fixed(xlim = c(-PAD, W + PAD), ylim = c(H + PAD, -PAD), expand = FALSE),
    theme_void(base_size = 11),
    theme(plot.background = element_rect(fill = "white", color = NA), strip.text = element_text(hjust = 0, size = 10.5, margin = margin(4, 0, 4, 0)),
          plot.title = element_text(size = 12, face = "bold"), plot.subtitle = element_text(size = 9.5, color = INK2, lineheight = 1.15),
          panel.spacing = unit(10, "pt"))
  )
}

ref <- filter(rings, kind == "reference", radius_deg >= 2)
ref_labels <- ref |> group_by(radius_deg) |> slice_max(x, n = 1, with_ties = FALSE) |> ungroup() |>
  mutate(label = paste0(radius_deg, "°"))
p_ref <- ggplot() + screen_base() +
  geom_path(data = ref, aes(x, y, group = radius_deg), color = "white", linewidth = 0.5) +
  annotate("point", x = 960, y = 540, shape = 3, size = 2.5, color = "black") +
  geom_text(data = ref_labels, aes(x - 10, y, label = label), size = 3, color = "white", hjust = 1) +
  labs(title = "How big is a degree on this screen?",
       subtitle = "Rings 2, 5, 10 and 20° from screen center (70 cm viewing distance).\n1° ≈ 44 px ≈ 1.2 cm on screen, about a thumbnail at arm's length.\nThe 20° ring runs off the top and bottom of the screen.")

for (pid in PIDS) {
  qc_path <- file.path(QC_DIR, paste0(pid, "_block_qc.csv"))
  if (!file.exists(qc_path)) { message("No QC output for ", pid, ", skipping"); next }
  s <- read_csv(file.path(QC_DIR, paste0(pid, "_validation_samples.csv")), show_col_types = FALSE)
  n_pts <- s |> distinct(block_index, target_x, target_y) |> count(block_index, name = "n_targets")
  qc <- read_csv(qc_path, show_col_types = FALSE) |>
    left_join(n_pts, by = "block_index") |>
    mutate(strip = sprintf("Block %d · %s\nCalibration error %.2f° · %s%s", block_index, block_id, deg_analyzed,
                           ifelse(included, "analyzed", "EXCLUDED (> 2°)"),
                           ifelse(coalesce(n_targets, 0L) < 5, sprintf(" · only %d/5 targets", coalesce(n_targets, 0L)), "")),
           strip = factor(strip, strip))
  s <- left_join(s, select(qc, block_index, strip), by = "block_index")
  trials <- read_csv(file.path(QC_DIR, paste0(pid, "_trial_qc.csv")), show_col_types = FALSE) |>
    left_join(select(qc, block_index, strip), by = "block_index") |>
    group_by(block_index) |> mutate(pos = row_number()) |> ungroup() |>
    mutate(status = factor(ifelse(included, "Recorded and usable", "Recorded but NOT usable (calibration > 2°)"),
                           c("Recorded and usable", "Recorded but NOT usable (calibration > 2°)")))
  targets <- distinct(s, strip, target_x, target_y, point_deg, presentation_order) |>
    mutate(label = ifelse(is.na(presentation_order), sprintf("%.1f°", point_deg),
                          sprintf("#%d · %.1f°", as.integer(presentation_order), point_deg)), lx = target_x + ifelse(target_x > W / 2, -70, 70),
           ly = target_y + ifelse(target_y > H / 2, 75, -75))
  thr <- filter(rings, kind == "threshold") |> tidyr::crossing(distinct(qc, strip)) |> arrange(strip, target_x, target_y, order)
  sequence <- targets |> filter(!is.na(presentation_order)) |> arrange(strip, presentation_order)  # target order on screen
  landed <- s |> group_by(strip, target_x, target_y) |> summarise(gx = median(x), gy = median(y), .groups = "drop")

  p_blocks <- ggplot() + screen_base() +
    geom_path(data = thr, aes(x, y, group = interaction(target_x, target_y)), color = "white", linewidth = 0.4, linetype = "22") +
    geom_path(data = sequence, aes(target_x, target_y, group = strip), color = "white", linewidth = 0.3, linetype = "11", alpha = 0.6) +
    geom_point(data = s, aes(x, y), color = GAZE, size = 0.6, alpha = 0.7) +
    geom_segment(data = landed, aes(target_x, target_y, xend = gx, yend = gy), color = GAZE, linewidth = 0.6,
                 arrow = arrow(length = unit(5, "pt"), type = "closed")) +
    geom_point(data = targets, aes(target_x, target_y), shape = 3, size = 2, stroke = 0.9, color = "black") +
    geom_label(data = targets, aes(lx, ly, label = label), size = 2.7, linewidth = 0, fill = "white", color = "black",
               label.padding = unit(1.2, "pt")) +
    facet_wrap(~strip, nrow = 1) +
    labs(title = sprintf("%s — where is the participant looking, and is the tracker recording?", pid),
         subtitle = paste0("Top: the validation check before each block. + = target · yellow dots = gaze samples · arrow = target → where gaze landed · ",
                           "dashed ring = 2°, the cutoff for a block to be analyzed · labels = #presentation order · error at that target.\nDotted line = order the targets were shown in. ",
                           "Bottom: % of samples with valid gaze in each movie of that block."))

  p_valid <- ggplot(trials, aes(pos, pct_valid_samples, fill = status)) +
    geom_col(width = 0.7, show.legend = TRUE) +  # keep both keys even when one status is absent
    geom_text(aes(label = sprintf("%.0f%%", pct_valid_samples)), vjust = -0.4, size = 2.8, color = "#0b0b0b") +
    geom_text(aes(y = -6, label = video), size = 2.5, color = INK2, vjust = 1) +
    facet_wrap(~strip, nrow = 1) +
    scale_fill_manual(values = STATUS_FILL, name = NULL, drop = FALSE) +
    scale_x_continuous(limits = c(0.4, 4.6)) +
    scale_y_continuous(limits = c(-16, 112), breaks = c(0, 50, 100), labels = function(b) paste0(b, "%")) +
    labs(x = NULL, y = "Valid samples",
         caption = paste0("Valid samples = the tracker found the eyes (data were recorded). Calibration error = whether we know WHERE on the screen they were looking. ",
                          "A block needs both: high % valid with calibration > 2° is recorded but not usable.")) +
    theme_minimal(base_size = 10) +
    theme(strip.text = element_blank(), axis.text.x = element_blank(), panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(), panel.grid.major.y = element_line(color = "#e6e5e0", linewidth = 0.3),
          legend.position = "bottom", legend.justification = "left", plot.caption = element_text(hjust = 0, color = INK2, size = 9.5),
          panel.spacing = unit(10, "pt"))

  p <- ((p_blocks / p_valid) + plot_layout(heights = c(1, 0.85))) | (plot_spacer() / p_ref / plot_spacer() + plot_layout(heights = c(0.15, 1, 0.6)))
  p <- p + plot_layout(widths = c(4, 1.15))
  dir.create(here("figures", "QC"), showWarnings = FALSE)
  ggsave(here("figures", "QC", paste0(pid, "_validation_screen.png")), p, width = 19, height = 7.6, dpi = 160, bg = "white")
}

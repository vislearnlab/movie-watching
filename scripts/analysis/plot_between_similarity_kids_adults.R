## plot_between_similarity_kids_adults.R
## Between-subject gaze similarity (1 - between-person cosine distance, segment mean over participants) across one
## clip's timeline: adult-adult and kid-kid (each vs. their own group's leave-one-out mean) and kid-adult (each kid vs.
## all adults' mean). Each scene-cut segment is a flat step. Data from calc_within_between_cosine.py, calc_between_kid_to_adult.py.
##
## Output: figures/within_between/kids_vs_adults/<clip>_between_similarity_timecourse.png
## Usage:  Rscript scripts/analysis/plot_between_similarity_kids_adults.R [--clip pixar_birds]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(here)
})

a <- commandArgs(trailingOnly = TRUE)
CLIP <- if ("--clip" %in% a) a[match("--clip", a) + 1] else "pixar_birds"
WB_DIR <- here("data", "results", "within_between_cosine")
OUT_DIR <- here("figures", "within_between", "kids_vs_adults")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
GROUP_COLORS <- c("adult-adult" = "#3a3a38", "kid-kid" = "#e85a32", "kid-adult" = "#2a78d6")
INK2 <- "#52514e"

d <- bind_rows(
  read_csv(file.path(WB_DIR, "all_clips_by_segment.csv"), show_col_types = FALSE) |> mutate(group = "adult-adult"),
  read_csv(file.path(WB_DIR, "all_clips_by_segment_kids.csv"), show_col_types = FALSE) |> mutate(group = "kid-kid"),
  read_csv(file.path(WB_DIR, "all_clips_by_segment_kid_to_adult.csv"), show_col_types = FALSE) |>
    mutate(group = "kid-adult", between_person_cosine_dist = kid_to_adult_cosine_dist)
) |>
  filter(video == CLIP, !is.na(between_person_cosine_dist)) |>
  mutate(sim = 1 - between_person_cosine_dist, mid = (seg_start_s + seg_end_s) / 2,
         group = factor(group, names(GROUP_COLORS)))
cuts <- d |> distinct(seg_start_s) |> filter(seg_start_s > 0)
means <- d |> group_by(group) |> summarise(sim = mean(sim), .groups = "drop")
w <- tidyr::pivot_wider(d, id_cols = seg_idx, names_from = group, values_from = sim) |> na.omit()

p <- ggplot(d, aes(color = group)) +
  geom_vline(data = cuts, aes(xintercept = seg_start_s), color = "#d9d8d4", linewidth = 0.3) +
  geom_hline(data = means, aes(yintercept = sim, color = group), linetype = "dashed", linewidth = 0.4, show.legend = FALSE) +
  geom_segment(aes(x = seg_start_s, xend = seg_end_s, y = sim, yend = sim), linewidth = 1.1, lineend = "round") +
  geom_line(aes(mid, sim), linewidth = 0.3, alpha = 0.5) +
  scale_color_manual(values = GROUP_COLORS, name = NULL) +
  scale_x_continuous(expand = expansion(mult = 0.01), breaks = seq(0, 150, 15)) +
  labs(x = "time in clip (s)", y = "between-subject similarity\n(1 - cosine distance to reference mean)",
       title = sprintf("%s: between-subject gaze similarity per scene-cut segment", CLIP),
       subtitle = sprintf(paste0("Segment means over participants; gray lines = cuts; dashed = clip mean. Reference: adult-adult and kid-kid = rest of own group, ",
                                 "kid-adult = all adults.\nAcross segments, adult-adult vs. kid-kid r = %.2f, adult-adult vs. kid-adult r = %.2f. Kids n ~ 13, adults ~ 40."),
                          cor(w$`adult-adult`, w$`kid-kid`), cor(w$`adult-adult`, w$`kid-adult`))) +
  theme_classic(base_size = 11) +
  theme(axis.text = element_text(color = INK2), axis.title = element_text(color = INK2),
        plot.title = element_text(size = 12), plot.subtitle = element_text(color = INK2, size = 9),
        legend.position = "top", legend.justification = "left")
out <- file.path(OUT_DIR, paste0(CLIP, "_between_similarity_timecourse.png"))
ggsave(out, p, width = 11, height = 4.6, dpi = 200, bg = "white")
cat("Saved", out, "\n")
print(means)

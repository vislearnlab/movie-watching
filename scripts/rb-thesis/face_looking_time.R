# Looking time to faces (human, Muppet, animal) per adult participant and Sesame segment (sum segments for clip totals).
# Run from the movie-watching folder: Rscript scripts/rb-thesis/face_looking_time.R

library(dplyr)
library(readr)

# folder with one fixation file per participant per clip
fix_dir <- "data/local_cache/fixations/adults"
# segment cuts (first frame of each new segment)
cuts <- read_csv("annotations/scene-cuts/all_clips_cuts_frame_corrected.csv", show_col_types = FALSE)
# clip lengths in seconds (end of the last segment)
inventory <- read_csv("data/results/stimulus_prep/stimulus_inventory.csv", show_col_types = FALSE)
# folder with the face masks (one file per clip)
mask_dir <- "data/results/rb-thesis/mask_cells"
# the four Sesame Street clips
clips <- c("sesameus_1", "sesameus_2", "sesameindia_1", "sesameindia_2")
# face types to measure
face_types <- c("human", "muppet", "animal")
# size of one mask cell, in video pixels
cell_size <- 8
# screen is 1920 px wide, video is 1280 px wide, so screen px x 2/3 = video px
screen_to_video <- 1280 / 1920

# list the fixation files for the Sesame clips
fix_files <- list.files(fix_dir, pattern = "sesame.*\\.csv$", full.names = TRUE)
# read them all and stack them into one table
fixations <- bind_rows(lapply(fix_files, read_csv, show_col_types = FALSE))

# empty list to hold one result table per clip
results <- list()
# empty list to hold the segment numbers of each clip
segment_lists <- list()

# loop over the clips
for (clip in clips) {
  # US clips play at 29.97 frames per second, India clips at 25
  frame_ms <- ifelse(startsWith(clip, "sesameus"), 1001 / 30, 40)
  # read this clip's mask cells (frame, face_type, cell_x, cell_y)
  masks <- read_csv(file.path(mask_dir, paste0(clip, ".csv.gz")), show_col_types = FALSE)
  # keep only the face types we measure
  masks <- filter(masks, face_type %in% face_types)

  # keep only this clip's fixations
  clip_fix <- filter(fixations, video_name == clip)
  # start time (ms) of every segment after the first, from this clip's cuts
  cut_ms <- filter(cuts, video_name == clip)$first_new_frame * frame_ms
  # segment each fixation starts in (0 = before the first cut)
  clip_fix$segment <- findInterval(clip_fix$startT, cut_ms)
  # clip length in ms
  clip_ms <- inventory$duration_sec[inventory$video_name == clip] * 1000
  # all segment numbers in this clip (one more segment than cuts)
  segment <- 0:length(cut_ms)
  # start time (ms) of each segment
  segment_start_ms <- round(c(0, cut_ms))
  # duration (ms) of each segment: next start (or clip end) minus this start
  segment_dur <- round(c(cut_ms, clip_ms) - c(0, cut_ms))
  # store this clip's segments
  segment_lists[[clip]] <- data.frame(video_name = clip, segment, segment_start_ms, segment_dur)
  # end time (ms) of the segment each fixation starts in
  clip_fix$segment_end_ms <- c(cut_ms, clip_ms)[clip_fix$segment + 1]
  # only count the part of each fixation before its segment ends (so looking_dur can't exceed segment_dur)
  clip_fix$dur_in_segment <- pmin(clip_fix$endT, clip_fix$segment_end_ms) - clip_fix$startT
  # time halfway through the counted part of each fixation (ms from clip start)
  clip_fix$mid_ms <- clip_fix$startT + clip_fix$dur_in_segment / 2
  # video frame showing at that time
  clip_fix$frame <- floor(clip_fix$mid_ms / frame_ms)
  # flip y: eye tracker counts from the bottom of the screen, masks count from the top
  clip_fix$ypos_top <- 1080 - clip_fix$ypos
  # fixation position in video pixels, then in mask cells
  clip_fix$cell_x <- floor(clip_fix$xpos * screen_to_video / cell_size)
  clip_fix$cell_y <- floor(clip_fix$ypos_top * screen_to_video / cell_size)

  # match each fixation to the masks: a match means it landed on a face on that frame
  # (many-to-many: where two face types overlap, a fixation counts for both)
  on_face <- inner_join(clip_fix, masks, by = c("frame", "cell_x", "cell_y"), relationship = "many-to-many")

  # add up fixation durations (ms) per participant, segment and face type
  summed <- on_face %>%
    group_by(pid, block_id, video_name, segment, face_type) %>%
    summarise(looking_ms = sum(dur_in_segment), .groups = "drop")

  # store this clip's result
  results[[clip]] <- summed
}

# stack the clips into one table
looking <- bind_rows(results)

# every segment of every clip
all_segments <- bind_rows(segment_lists)
# each participant x clip they watched
all_combos <- distinct(fixations, pid, block_id, video_name)
# pair each participant with every segment of that clip
all_combos <- inner_join(all_combos, all_segments, by = "video_name", relationship = "many-to-many")
# pair each participant x segment with every face type
all_combos <- cross_join(all_combos, tibble(face_type = face_types))
# attach the looking times (rows with no looking get NA)
looking <- left_join(all_combos, looking, by = c("pid", "block_id", "video_name", "segment", "face_type"))
# no looking = 0 ms
looking$looking_ms[is.na(looking$looking_ms)] <- 0

# final columns, with looking time in ms (same units as segment_dur)
looking <- looking %>%
  transmute(participant = pid, block = block_id, clip = video_name, segment, segment_start_ms, segment_dur, face_type,
            looking_dur = round(looking_ms))

# save the result
write_csv(looking, "data/results/rb-thesis/face_looking_time_adults.csv")

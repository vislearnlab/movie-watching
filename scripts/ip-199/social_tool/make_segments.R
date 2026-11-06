# Builds segments.js (one entry per segment of all 12 clips) for social_interaction_tool.html.
# Run from the movie-watching folder: Rscript scripts/ip-199/social_tool/make_segments.R

library(dplyr)
library(readr)
library(jsonlite)

# segment cuts (first frame of each new segment)
cuts <- read_csv("annotations/scene-cuts/all_clips_cuts_frame_corrected.csv", show_col_types = FALSE)
# clip lengths in seconds and which block each clip is in
inventory <- read_csv("data/results/stimulus_prep/stimulus_inventory.csv", show_col_types = FALSE)

# empty table to fill with segments
segments <- data.frame()

# loop over the 12 clips
for (clip in inventory$video_name) {
  # this clip's cuts
  clip_cuts <- filter(cuts, video_name == clip)
  # frames per second
  fps <- clip_cuts$fps[1]
  # length of one frame in ms (29.97 and 23.976 fps are exactly 30000/1001 and 24000/1001)
  frame_ms <- ifelse(round(fps) != fps, 1001 / round(fps), 1000 / fps)
  # clip length in ms
  clip_ms <- inventory$duration_sec[inventory$video_name == clip] * 1000
  # number of frames in the clip
  n_frames <- round(clip_ms / 1000 * fps)
  # first frame of each segment
  first_frame <- c(0, clip_cuts$first_new_frame)
  # last frame of each segment (the frame before the next cut)
  last_frame <- c(clip_cuts$first_new_frame - 1, n_frames - 1)
  # start time (ms) of each segment after the first
  cut_ms <- clip_cuts$first_new_frame * frame_ms
  # add this clip's segments to the table
  segments <- bind_rows(segments, data.frame(
    clip = clip,
    block = inventory$block_id[inventory$video_name == clip],
    segment = 0:length(cut_ms),
    fps = 1000 / frame_ms,
    first_frame = first_frame,
    last_frame = last_frame,
    # start time (ms), same rounding as the rb-thesis CSVs
    segment_start_ms = round(c(0, cut_ms)),
    # duration (ms): next start (or clip end) minus this start
    segment_dur = round(c(cut_ms, clip_ms) - c(0, cut_ms))
  ))
}

# write the table as a JavaScript file the tool can load
writeLines(paste0("const SEGMENTS = ", toJSON(segments, digits = NA), ";"), "scripts/ip-199/social_tool/segments.js")
# report how many segments per clip
print(count(segments, clip))

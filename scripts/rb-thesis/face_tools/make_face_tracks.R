# Makes face_tracks.js, the faces and shots that face_label_tool.html and missing_faces_tool.html show.
# One "track" = one face that SAM3 followed through one shot (same prompt, shot, and object id).
# Run from the movie-watching folder:  Rscript scripts/rb-thesis/face_tools/make_face_tracks.R

library(dplyr)
library(readr)
library(jsonlite)

clips <- c("sesameus_1", "sesameus_2", "sesameindia_1", "sesameindia_2")

# Face type implied by each SAM3 prompt; the generic prompts ("faces", "face (unlabelled)") get none
human_prompts <- c("child face", "children faces", "child faces", "adult face", "adult faces",
                   "child and adult faces", "person face", "people faces")
muppet_prompts <- c("muppet face", "muppet Face", "sheep muppet face")
animal_prompts <- c("chicken faces", "sheep face", "goat face", "turkey face", "cow face", "llama face")

# Read the SAM3 frames.csv for each clip and keep only the face rows
all_frames <- data.frame()
for (clip in clips) {
  frames <- read_csv(paste0("data/local_cache/face_masks/", clip, "/frames.csv"), show_col_types = FALSE)
  frames <- filter(frames, category == "face")
  all_frames <- bind_rows(all_frames, frames)
}

all_frames <- all_frames %>%
  mutate(track_id = paste(video_name, prompt_idx, shot_idx, obj_id, sep = "_")) %>%
  arrange(video_name, time_ms)

# One row per track; best_ms is the frame where the face is biggest (easiest to recognize)
tracks <- all_frames %>%
  group_by(track_id, video_name, prompt_idx, prompt, shot_idx, obj_id) %>%
  summarise(start_ms = min(time_ms),
            end_ms = max(time_ms),
            n_frames = n(),
            best_ms = time_ms[which.max(mask_area_px)],
            .groups = "drop") %>%
  mutate(sam3_type = case_when(prompt %in% human_prompts ~ "human",
                               prompt %in% muppet_prompts ~ "muppet",
                               prompt %in% animal_prompts ~ "animal",
                               TRUE ~ "")) %>%
  arrange(video_name, start_ms, prompt_idx, obj_id)

# Attach the box for every frame of each track: [time_ms, x, y, width, height]
track_list <- list()
for (i in 1:nrow(tracks)) {
  rows <- filter(all_frames, track_id == tracks$track_id[i])
  track <- as.list(tracks[i, ])
  track$boxes <- as.matrix(rows[, c("time_ms", "bbox_x", "bbox_y", "bbox_w", "bbox_h")])
  track_list[[i]] <- track
}

# One row per shot, from the frame-corrected cuts (same shot numbers as SAM3's shot_idx)
cuts <- read_csv("annotations/scene-cuts/all_clips_cuts_frame_corrected.csv", show_col_types = FALSE)
inventory <- read_csv("data/results/stimulus_prep/stimulus_inventory.csv", show_col_types = FALSE)

shots <- data.frame()
for (clip in clips) {
  clip_cuts <- filter(cuts, video_name == clip)
  fps <- clip_cuts$fps[1]
  n_frames <- round(inventory$duration_sec[inventory$video_name == clip] * fps)
  first_frame <- c(0, clip_cuts$first_new_frame)
  last_frame <- c(clip_cuts$first_new_frame - 1, n_frames - 1)
  shots <- bind_rows(shots, data.frame(video_name = clip,
                                       shot_idx = 0:(length(first_frame) - 1),
                                       start_ms = round(first_frame / fps * 1000, 2),
                                       end_ms = round(last_frame / fps * 1000, 2)))
}

# How many face tracks SAM3 found in each shot
faces_per_shot <- count(tracks, video_name, shot_idx, name = "n_faces")
shots <- shots %>%
  left_join(faces_per_shot, by = c("video_name", "shot_idx")) %>%
  mutate(n_faces = ifelse(is.na(n_faces), 0, n_faces))

tracks_json <- toJSON(track_list, auto_unbox = TRUE, digits = NA)
shots_json <- toJSON(shots, digits = NA)
writeLines(c(paste0("const TRACKS = ", tracks_json, ";"),
             paste0("const SHOTS = ", shots_json, ";")),
           "scripts/rb-thesis/face_tools/face_tracks.js")

cat("Wrote", nrow(tracks), "face tracks and", nrow(shots), "shots\n")
print(count(tracks, sam3_type))

# =============================================================================
# create_datavyu_files.rb
#
# Reads stimuli_prompts.csv and creates one Datavyu (.opf) file per unique
# video_block. Each file gets two columns:
#
#   prompts  — one cell per prompt_idx
#              codes: prompt_idx (integer), prompt_name (string)
#              onset/offset from start_time_s / end_time_s in the CSV
#
#   coords   — one cell per prompt_idx (same cells, empty until filled in)
#              codes: prompt_idx (integer), x (integer), y (integer),
#                     include (string: "true" | "false")
#
# PARAMETERS — edit these before running
# =============================================================================

# Paths are relative to this script's location (scripts/preprocessing/segmentation/)
REPO_ROOT   = File.expand_path("../../../..", __FILE__)
CSV_PATH    = File.join(REPO_ROOT, "annotations", "stimuli_prompts_round0csv")
OUTPUT_DIR  = File.join(REPO_ROOT, "scripts", "preprocessing", "segmentation", "datavyu_segmentation")

require 'csv'
 
# ---------------------------------------------------------------------------
# Helper: convert seconds (float) to Datavyu milliseconds (integer)
# ---------------------------------------------------------------------------
def s_to_ms(seconds)
  (seconds.to_f * 1000).round
end
 
# ---------------------------------------------------------------------------
# Load and group the CSV by video_block
# ---------------------------------------------------------------------------
rows_by_video = Hash.new { |h, k| h[k] = [] }
 
CSV.foreach(CSV_PATH, headers: true) do |row|
  video_block = row["video_block"].strip
  rows_by_video[video_block] << {
    prompt_idx:   row["prompt_idx"].to_i,
    prompt_name:  row["prompt"].to_s.strip,
    start_ms:     s_to_ms(row["start_time_s"]),
    end_ms:       s_to_ms(row["end_time_s"])
  }
end
 
# ---------------------------------------------------------------------------
# Create one Datavyu file per video_block
# ---------------------------------------------------------------------------
rows_by_video.each do |video_block, rows|
 
  # -- 1. Build the "prompts" column -----------------------------------------
  prompts_col = createNewColumn("prompts", "prompt_idx", "prompt_name")
 
  rows.each do |r|
    cell = prompts_col.make_new_cell()
    cell.change_code("prompt_idx",  r[:prompt_idx].to_s)
    cell.change_code("prompt_name", r[:prompt_name])
    cell.onset  = r[:start_ms]
    cell.offset = r[:end_ms]
  end
 
  setColumn(prompts_col)
 
  # -- 2. Build the "coords" column ------------------------------------------
  coords_col = createNewColumn("coords", "prompt_idx", "x", "y", "include")

 
  setColumn(coords_col)
 
  # -- 3. Save the spreadsheet -----------------------------------------------
  # Sanitize video_block name for use as a filename
  safe_name = video_block.gsub(/[\/\\:*?"<>|]/, "_").gsub(/\s+/, "_")
  out_path  = File.join(OUTPUT_DIR, "#{safe_name}.opf")
 
  save_db(out_path)
  puts "Saved: #{out_path}  (#{rows.length} prompts)"
 
  # Clear columns from the spreadsheet before the next iteration
  # (Datavyu keeps a single active spreadsheet; we save then wipe for the next)
  deleteColumn("prompts")
  deleteColumn("coords")
 
end
 
puts "\nDone. Created #{rows_by_video.size} Datavyu file(s) in #{OUTPUT_DIR}"
 

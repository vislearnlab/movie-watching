# Annotation pipeline

Face and hand masks for every stimulus video, made with SAM3 from the Datavyu annotations plus review corrections. The output is one `frames.csv` and one `masks.npz` per video in `output/current/`.

## Layout

| path | what |
|---|---|
| `pipeline/` | all code (see below) |
| `inputs/datavyu_exports/round1/`, `round2/` | Datavyu CSV exports, one per video. The newest round wins. |
| `inputs/datavyu_projects/` | Datavyu `.opf` project files (`legacy/create_datavyu_files.rb` pre-fills them) |
| `inputs/spreadsheets/` | spreadsheet prompts; `stimuli_prompts_round0.csv` is used for the 3 object-only videos with no Datavyu export |
| `inputs/corrections/` | fixes found while reviewing outputs, one CSV per video (format in `prepare_prompts.py`; reviewer instructions in `REVIEW_BRIEF.md`) |
| `scene-cuts/all_clips_cuts.csv` | reference cut times (also read by `scripts/analysis/`) |
| `scene-cuts/cut_overrides.csv` | cuts fixed by eye |
| `scene-cuts/all_clips_cuts_frames.csv` | cuts as frame indices, from `refine_cuts.py`; every SAM3 session stays inside one shot |
| `shot_scale/` | shot-scale coding (not used by this pipeline) |
| `sam3/` | SAM3 source (Meta); `.env` holds the HuggingFace token |
| `output/current/` | **the** current output: per-video results, `prompts.csv`, `qc/`, `logs/` |
| `output/archive/` | every earlier run, read-only. `pass1_20261021` is the first run of this pipeline. |
| `legacy/` | the previous SAM script and `.ipynb` notebooks |
| `tests/` | `pytest` suite |

## Installation

Connect to Tversky and use the `moviewatching` conda env:

```
conda activate moviewatching          # /Scratch/<user>/miniconda3/envs/moviewatching
pip install -r annotations/requirements.txt
```

SAM3 is imported from `sam3/`, so there's no separate install. The model needs a HuggingFace token with access to [facebook/sam3](https://huggingface.co/facebook/sam3), stored as `HF_TOKEN=…` in `annotations/.env` (see `.env_template`). `ffmpeg` must be on the path for rendering.

## Running: the review cycle

Everything runs in the tmux session `annotations` (`tmux attach -t annotations`):

| window | what |
|---|---|
| `status` | the accuracy table, redrawn every 30 s; the plot is `output/current/qc/status.png` |
| `cycle` | `pipeline/cycle.sh`: one pass after another |
| `preview` | writes `qc/review/{video}_shot{k}.jpg` the moment a shot finishes (`latest.jpg` = most recent) |
| `workers` | which shot each worker is on |
| `gpu` | `nvidia-smi` for GPUs 2 and 3 |

To start it from scratch:

```
cd /labs/vislearnlab/experiments/movie-watching/annotations
tmux new-session -d -s annotations -n cycle "bash pipeline/cycle.sh"
tmux new-window -t annotations -n status "python -u pipeline/status.py --watch 30"
tmux new-window -t annotations -n preview "python pipeline/live_preview.py"
```

**One pass** (`cycle.sh`):
1. `prepare_prompts.py` applies `inputs/corrections/`.
2. `annotate_videos.py run` workers (2 per GPU on GPUs 2 and 3; override with `GPUS="0 1 2 3" WORKERS_PER_GPU=2`) segment every shot whose tasks changed since its last run.
3. `merge` and `evaluate.py` run.
4. The cycle then waits for a change in `inputs/corrections/` and starts the next pass. Only the shots a correction touches rerun.

**Reviewing.** After a pass, each shot's sheet in `qc/review/` is scored and fixed by a reviewer (a person or a Claude subagent following `inputs/corrections/REVIEW_BRIEF.md`):
- **Scores:** one row per shot and pass goes in `qc/reviews/*.csv` (`video, shot, iteration, score, issues`). The score is the estimated fraction of face and hand instances that are masked with the right extent and label.
- **Fixes:** go in `inputs/corrections/*.csv`.
- **Coordinates:** `python pipeline/review.py frame --video X --frame N` renders a full-size frame with a 100 px grid for reading off box coordinates.
- **Validation:** `python pipeline/prepare_prompts.py --check` validates corrections without writing anything.

**Accuracy plot** (`status.py` → `qc/status.png`):
- **Left panel:** each video's frame-weighted review score, pass 1 (hollow) against the latest review (filled).
- **Right panel:** every shot's estimated error. The number in each cell is how many times that shot has been segmented.

Individual steps, from `annotations/`:

```
python pipeline/refine_cuts.py                        # cut times → cut frames (only when cuts change)
python pipeline/prepare_prompts.py                    # Datavyu CSVs + corrections → output/current/prompts.csv
python pipeline/qc_clicks.py                          # check click positions → qc/clicks/
python pipeline/annotate_videos.py run --gpu 2 --video sesameus_1 --shot 3   # one shot
python pipeline/annotate_videos.py merge              # shards → frames.csv / masks.npz / clicks.csv
python pipeline/render.py video --video sesameus_1    # annotated.mp4
python pipeline/review.py sheets --video sesameus_1   # review sheets from the merged output
python pipeline/evaluate.py                           # qc/report.csv
python pipeline/status.py                             # qc/status.png
python -m pytest -q tests
```

Each worker is capped at 16 CPU threads (`OMP_NUM_THREADS`). Without the cap, torch starts 128 threads per worker and four workers saturate the machine. Session frames go to `/Scratch/$USER/annotation_tmp` (local NVMe), not the network share.

## How it works

**1. Prompts (`prepare_prompts.py`).** Each Datavyu prompt becomes one row with its time window and its clicks.
- **Click timing:** every click keeps the time it was made (`coords.onset`), because a click is only valid on that frame.
- **Coordinates:** screen positions are mapped to the 1280×720 video. Round 1 on the Remora Mac Mini used a letterboxed display (see `config.VIDEOS`). Check the mapping by eye with `qc_clicks.py`.
- **Text:** the label is singularised for SAM3 ("children faces" → "child face").
- **Broken windows:** windows with end ≤ start (8 of them) become "the shot containing the start".
- **Orphan clicks:** two clicks point at prompt ids that don't exist. They were reassigned by eye in `config.CLICK_REMAP`.
- **Corrections:** rows in `inputs/corrections/` add prompts, change a prompt's SAM3 text or window, or add boxes and clicks. Correction prompts get ids from 1,000,000 up, so Datavyu ids never shift.
- **Auto prompts:** every video gets `face (unlabelled)` and `hand (unlabelled)` (except where `config.VIDEOS[...]["auto"]` says otherwise: no hands in the animal and object videos, no faces in frank_objects), which run the generic text on every shot. They catch faces and hands no annotator prompted, and they lose every dedup, so they never relabel a prompted object. On frames a generic real prompt covers (`hands`, `faces`), what they find takes that prompt's label. They're skipped in a shot where such a prompt already covers the whole shot, because that would be the same SAM3 session twice. They also inherit the shot's exclude clicks. Any mask still labelled `(unlabelled)` after review is a missing prompt.

**2. Shots and tasks (`refine_cuts.py`, `plan.py`).**
- **Cuts:** 74 of the 172 reference cuts mapped one frame late. `refine_cuts.py` moves a cut to the frame with the largest frame-to-frame change within ±3 frames, but only when that change is at least twice every other one. For a fade through black, the cut is the first frame after the darkest one (18 cuts). `scene-cuts/cut_overrides.csv` fixes two pixar_birds cuts by eye.
- **Frames:** a click at t ms is on frame ⌊t·fps⌋, the frame Datavyu shows.
- **Clicks at cuts:** a click up to 2 frames outside a shot that its prompt covers is moved into that shot. With the refined cuts this affects 10 of 328 face/hand clicks.
- **Shot assignment:** a prompt covers a shot if it overlaps it by at least min(1 s, half the shot). Window edges within 1 s of a cut snap to the cut.
- **Tasks:** prompts with the same text in the same shot share one SAM3 session (a "task"), so the same face is never segmented twice under one label.
- **Reruns:** each shot's tasks are hashed (`annotate_videos.signature`). A shot reruns only when its hash changes (prompts, windows, clicks, boxes or `PIPELINE_VERSION`). Its `.json` stamp counts the iterations.

**3. Segmentation (`segment.py`).**
- **Text detection:** SAM3 runs the text prompt over the shot plus up to 1 s of context, detecting and tracking every instance.
- **Generic fallback:** if the specific text ("adult face") covers fewer than half the output frames, the generic noun ("face") runs too. If the prompt is the only real prompt of its category in the shot, every generic track keeps its label. Otherwise a generic track keeps the label only if it was clicked or overlaps one of the specific text's own tracks, so "face" can't label the adult next to the muppet as "muppet face".
- **Late-start backfill:** SAM3 outputs a new detection only after confirming it over several frames, so tracks would start ~10 frames into a shot. A track first seen within the first 20 session frames is re-seeded from its own box on that frame and propagated back to the start.
- **Exclude clicks** remove the closest track within 30 px, on the nearest frame within ±2.
- **Include clicks** that already sit on a track mark it as clicked. SAM3 starts new tracks a few frames late, so a track under the click within ±15 frames also counts (`tracked_nearby`). Otherwise, the image detector proposes a box under the click. The box plus the click seed a new tracker object on that frame, propagated in both directions.
- **Review boxes** replace every track they overlap on their frame with a tracker object seeded from the box, propagated both ways. They fix partial masks (Grover's sleep mask only) and anchor missed objects. A second box on an object an earlier box seeded adds a conditioning frame to that object. Two separate objects on one head would split its pixels between them.
- **No bare clicks:** a click alone is never used as a prompt, because SAM3 then picks the whole body or the background.
- **Ears:** SAM3's "face" stops before the ears. Each shot with a face task runs one extra "ear" pass, shared by its face tasks. Each ear joins the nearest face within 15 px, provided it's at most half that face's area.

**4. Dedup and output (`annotate_videos.py`).**
- **Dedup:** two masks that overlap by more than 60% of the smaller one are the same object. The winner is a real prompt over an auto prompt, then clicked > specific text > generic noun > higher score. The loser goes on every frame where both cover the object. Across different prompts, if that's most shared frames, it goes entirely, so each object keeps one label.
- **Black frames:** frames with mean intensity under 12 (fade troughs) get no masks.
- **Assignment:** each frame belongs to one prompt of a task.

## Outputs (`output/current/{video}/`)

- **`frames.csv`:** one row per object per frame. Columns: `prompt_idx`, `prompt`, `sam_text`, `text_used`, `category`, `shot_idx`, `frame_idx_global`, `time_ms`, `obj_id`, `source` (`text` / `generic` / `click_box`), `clicked`, `prob`, bbox, mask area, mask centroid.
- **`masks.npz`:** one boolean 720×1280 mask per prompt per frame, key `r{prompt_idx}_{prompt}|{frame}`.
- **`clicks.csv`:** what happened to each click: `already_tracked`, `tracked_nearby`, `added`, `no_candidate`, `removed`, or `no_track`.
- **`annotated.mp4`:** full length, original size, and exact source frame rate. Masks are labelled `prompt_idx_raw. name` (the id corrections use), and clicks are drawn on their matched frame.

## Checking the results

**Comparison with `output/archive/output_round1_fixed`** on the same 179 face include clicks (Oct 2026 run):

| output | include hit | shuffle control | gap |
|---|---|---|---|
| `output_round1_fixed` | 0.83 | 0.31 | 0.52 |
| pass 1 (`output/archive/pass1_20261021`) | 0.92 | 0.21 | 0.71 |

Hands in pass 1: 0.89 include hit, 0.11 control, over 127 clicks. No exclude click is violated. Per-video numbers are in `qc/report.csv`.

**After the review cycle** (passes 2–7, Oct 22 2026; reviewer estimates from 8 frames per shot, `qc/status.png`):
- **Estimated accuracy:** every video is at 86–100% (pass 1: 0–100%).
- **Faces:** include hit 0.94 (194 clicks), control 0.20. Frames with a face mask rose 14%, because unprompted faces now get masks.
- **Hands:** include hit 0.87 (126 clicks), control 0.09. Frames with a hand mask rose 22%.
- **Exclude violations:** one, a sesameus_1 shot 9 rope exclude that a later hand track covers.

**Limits that corrections don't fix** (they need code changes):
- **Forearms:** generic "hand" spreads up the forearm in close-ups and blurry shots.
- **Hair:** face masks take same-coloured hair.
- **Cartoon birds:** round birds are masked as one blob, and the big bird's face splits into beak and eye pieces.
- **Muppet faces:** a box widens a fur face only near its own frame.
- **Stray pieces:** a separate piece of a mask can't be removed without removing the whole track.

**Known limitations:**
- SAM3's "hand" sometimes includes the forearm.
- About 12 clicks have no detector candidate within 30 px, even at a 0.05 score threshold. Those objects stay unsegmented on that frame.
- Round cartoon birds: in pixar_birds at ~86 s (a view of the birds' bellies), "small birds faces" covers bodies with no face visible.

`qc/report.csv`, per video and category:
- **`include_hit`:** fraction of include clicks on a mask with their text. It should be close to 1.
- **`control_hit`:** the same for random points on the same frames, as a shuffle control. It should be low. Oversized masks raise both numbers, so a large gap between `include_hit` and `control_hit` is the evidence that masks are correct.
- **`exclude_viol`:** exclude clicks still covered by a mask. It should be 0.
- **`text_recall`:** fraction of include clicks SAM3's text prompt found without help (`already_tracked`, or `tracked_nearby` within 15 frames). This is the held-out accuracy of the detector.
- **`big_face_frames`:** face masks over 15% of the frame. These are whole-body suspects; check them in the video.

Because clicks are also inputs, `include_hit` only shows that they were honoured. `text_recall` and the contact sheets are the independent checks.

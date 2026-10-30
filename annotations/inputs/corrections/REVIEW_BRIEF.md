# Shot review brief

You review SAM3 face and hand masks for movie clips shown to infants and adults in an eye-tracking study. The analysis asks how often people look at faces and hands, so every visible face and hand needs a mask with the right label and the right extent.

Work only inside `/labs/vislearnlab/experiments/movie-watching/annotations/`. Python: `/Scratch/<user>/miniconda3/envs/moviewatching/bin/python` (run from `annotations/`). Do not use GPUs or start SAM3. Do not edit code.

## Inputs

- **Review sheets:** `output/current/qc/review/{video}_shot{k:03d}.jpg`. These show 8 frames spread over shot k, with every mask outlined in white, filled in the prompt's colour and labelled `{prompt_idx_raw}. {prompt_name}`, where the id is the one corrections use. Sheets drawn before Oct 22 19:30 show the internal `prompt_idx` instead. Fills are light (25%) so the object under a mask stays visible; check the image under a mask before calling it background. Clicks are crosses (green = include, red = exclude) with `prompt_idx:status`. The header gives the shot's global frame range; each tile shows its frame number (`f1234`).
- **Grid frames:** for exact coordinates, render a full-size 1280×720 frame with masks and a 100 px grid:
  `python pipeline/review.py frame --video {video} --frame 1234 1300` → `output/current/qc/review/frames/{video}_f1234.jpg`. Use these whenever you write a box or click. Coordinates are video pixels (x right, y down, 1280×720).
- **Prompts:** `output/current/prompts.csv` (filter `video_stem`). The useful columns are `prompt_idx_raw` (the id you reference in corrections), `prompt_idx` (the number shown on the sheet labels), `prompt_name`, `sam_text`, `category`, `start_ms`, `end_ms`.

## What correct looks like

- **Faces:** every visible face of a person, muppet, animal or cartoon character that's larger than about 15 px. The mask covers the whole face, including the ears, eyes, nose, mouth and a muppet's fur face. It excludes hair, hats and the neck. Back-of-head views with no face visible need no mask. A round cartoon bird whose body is mostly head (the small birds in `pixar_birds`) may be masked whole; that counts as correct.
- **Hands:** each visible hand (or glove or paw used as a hand) up to the wrist. It never covers forearms, whole bodies or the background.
- **Labels:** use the Datavyu vocabulary, e.g. `child face`, `adult face`, `person face`, `muppet face`, `dog face`, `bird face`, `big bird face`, `small birds faces`, `hands`, `muppet hand`. A new face prompt must contain the word `face`; a new hand prompt must contain the word `hand`.
- **Timing:** a prompt's mask should only appear in shots where its object is visible. If the first or last tile of a sheet shows a different scene from the rest, the cut is misplaced; record it in `issues` with the prefix `CUT:`.

## The next pass

The next pass reruns every shot with your corrections applied. It also adds automatic generic `face` and `hand` prompts on every shot, labelled `face (unlabelled)` and `hand (unlabelled)` (ids c999 and c998, reserved). They pick up faces and hands no other prompt covers, but their labels are wrong for analysis, so still add a properly labelled prompt for any face or hand that has none.

SAM3 behaviour to plan around:
- Specific text like `adult face` or `child face` often finds nothing, while the generic `face` finds every face. A generic track keeps a prompt's label only if a click or box anchors it.
- If a shot shows only one kind of face, the cheapest fix is a `text` correction to `face` (or `hand`). Note that `text` changes the prompt in every shot it covers, so check the prompt's other shots first.
- If several kinds of face share the shot (e.g. a muppet and an adult), anchor each object with a `box` on a frame where it's clearly visible instead.
- A `box` replaces any track it overlaps on that frame and propagates both ways through the shot. Use it for partial masks (only a hat or nose masked), wrong extent, and missing objects. Draw it tight around the face including the ears. Add a second box on another frame if the object changes a lot.
- If a prompt's text grabs look-alike objects (e.g. `sheep face` masking patches of wool), set its `sam_text` to `none`. That prompt then keeps only its boxes and clicks, with no text detection. You can do this with a `text` row, or with the `sam_text` field of a new `prompt` row.
- An `exclude` click inside a wrong mask (e.g. a hand mask covering a whole cyclist) removes that track for the whole shot.

## Write two files

**1. Corrections** (`inputs/corrections/{file}.csv`). Use this exact header, one row per correction, and leave unused fields empty. The example rows only show the format: their coordinates are made up, so never copy them.

```
video,kind,prompt_idx,prompt_name,sam_text,shot,start_s,end_s,frame,x,y,x0,y0,x1,y1,note
slow_people,prompt,c1,dog face,,0,,,,,,,,,,dog walking toward camera
slow_people,box,c1,,,,,,193,,,780,560,860,640,
sesameindia_1,text,381,,face,,,,,,,,,,,only the 3 girls' faces in shot 23; child face finds none
sesameindia_2,box,293,,,,,,1400,,,140,230,420,520,muppet face mask only covers hat+nose
slow_people,exclude,103,,,,,,258,690,360,,,,,hand mask covers whole background cyclist
pixar_birds,window,305,,,3,,,,,,,,,,
```

- `prompt`: new prompt with `prompt_idx` `c1`, `c2`, …, unique within the video. Give `shot` (index) or `start_s`/`end_s`. Leave `sam_text` empty to derive it from the name. A new prompt with no box or click relies on SAM3's text alone, so add a `box` row that references it.
- `label`: new `prompt_name` for an existing prompt, in every shot it covers (same category: a face label stays a face label). Use it when the object is right but the label is wrong, e.g. `muppet face` → `grover face`. For a wrong label in only some shots, add a new `prompt` with a `box` instead.
- `text`, `label`, `window`, `box`, `include` and `exclude` reference an existing `prompt_idx_raw` (as typed in `prompts.csv`, e.g. `381`, `14a`) or one of your `c<n>` ids.
- `box` uses frame plus x0,y0 (top-left) and x1,y1 (bottom-right). Clicks use frame plus x,y.
- After writing, run `python pipeline/prepare_prompts.py --check` and fix any error. Only `--check`: never run it without the flag.

**2. Review scores** (`output/current/qc/reviews/{file}.csv`), one row per shot:

```
video,shot,iteration,score,issues
```

- `iteration` is the number in the sheet header (`iteration N`). Pass-1 sheets have no number; use 1.
- `score` is your estimate, 0–1, of the fraction of face and hand instances in the 8 tiles that are correct: masked, full extent, and the right label. A wrong mask (on a body, the background, or the wrong object) counts as an error too. A shot with no faces or hands and no masks scores 1.0.
- `issues` gives the specific problems, short, separated by `;`. Leave it empty if the shot is fine. Never put a comma in `issues` or `note` unless you wrap the whole field in double quotes. Otherwise the row splits into extra columns and the dashboard crashes.

Look at every shot sheet you're assigned, not only ones with prompts. When you're unsure whether something is a face or a hand, render a grid frame and look.

## Later passes (iteration 2 and up)

- **Corrections file:** append to the existing file. Never delete rows that worked. Fix or delete a row only when the sheet shows it did harm (a box that grabbed the wrong object, an exclude that removed a real face).
- **Did each correction work?** Check every earlier correction for your shots against the new sheet, and note in `issues` any that didn't. Masks seeded from a box, or from a click with no track under it, have no outline difference, so compare against the correction's frame.
- **`(unlabelled)` masks:** a mask labelled `face (unlabelled)` or `hand (unlabelled)` is an object with no prompt. If it's a real face or hand, give it a proper label with a new `prompt` plus `box` (the labelled box track then wins over the auto one). If it isn't a face or hand, add an `exclude` click with `prompt_idx` `c999` (face) or `c998` (hand) on a frame where it shows.
- **Scores:** add a new row per shot with the new iteration. Keep the old rows.

## Final report

Reply with:
- the number of shots reviewed;
- mean score per video;
- the 5 worst shots, with their problems;
- any CUT problems;
- any pattern you saw that a code change should fix rather than per-shot corrections.

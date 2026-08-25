# Segmentation pipeline

## Installation
In a new Terminal window, connect to the server (for example, Tversky). Set up a `miniconda` environment. To set one up in Tversky, follow [these instructions](https://app.notion.com/p/Tversky-and-SSRDE-12d24d0b4b29808fbfa6fb0730683772?source=copy_link). Then, navigate to this project directory (for example, `cd /labs/vislearnlab/experiments/movie-watching`) and run the following commands. 

```
cd scripts/preprocessing/segmentation
conda create -n moviewatching python=3.10
conda activate moviewatching
pip install -r requirements.txt
cd sam3
pip install -e .
cd ..
```

If there is no `.env` file present with a `HF_TOKEN` set, you need to get access to `SAM3`. (Create a HuggingFace account and sign up for access to the model)[https://huggingface.co/facebook/sam3]. Then create a (https://huggingface.co/docs/hub/en/security-tokens)[token] and create a `.env` file, similar to `.env_template`

## Current pipeline

- Step 1: Open your file in DataVyu, fix timestamps, and add or remove coordinates. This is round 1 of annotations. More specific instructions on the annotation process are [here](https://docs.google.com/document/d/1fhUzfY8VWFMOn10aSMx2T_v5cItVB_VacRSHnlNzwlA/edit?usp=sharing).

- Step 2: Export the Round 1 annotations as a CSV file and add to the `annotations/round1` folder here.

- Step 3: Run `prepare_stimuli_videos_datavyu.ipynb` in this directory. Update the `CURRENT_ROUND` variable to `round1` and update the `CSV_FILES` and `BLOCK_TO_STEM` mappings as needed. This will create a new file `stimuli_prompts_round1.csv` that is ready to be inputted into `SAM`.

- Step 3: Run `python annotate_videos.py --csv annotations/stimuli_prompts_round1.csv --output scripts/preprocessing/segmentation/output_round1 --render_video --skip_existing --debug_points` in a [tmux](https://www.redhat.com/en/blog/introduction-tmux-linux) window. This selects all GPUs by default. You can specify certain GPUs by using `--gpu 0 1` for example. Check available GPUs in Tversky by running `nvidia-smi` This step tries to annotate SAM on all of the videos with the raw annotations. If you run into an authorization error, check that the `.env` is set or refer to Installation above. If the `.env` is set, run `export HF_TOKEN=xxx` (using `HF_TOKEN` value from the `.env` file) and then try running the command above.
- Step 3: Run `python annotate_videos.py --csv annotations/stimuli_prompts_round1.csv --output scripts/preprocessing/segmentation/output_round1 --render_video --skip_existing --debug_points` in a [tmux](https://www.redhat.com/en/blog/introduction-tmux-linux) window. This selects all GPUs by default. You can specify certain GPUs by using `--gpu 0 1` for example. Check available GPUs in Tversky by running `nvidia-smi` This step tries to annotate SAM on all of the videos with the raw annotations. If you run into an authorization error, check that the `.env` is set or refer to Installation above. If the `.env` is set, run `export HF_TOKEN=xxx` (using `HF_TOKEN` value from the `.env` file) and then try running the command above.

- Step 4: New videos will be created in `output_round1`. Download these videos and add them to the corresponding folder on the shared Drive (`experiments/movie-watching/segmented_round1_annotations`). Make a copy of your DataVyu file, add it to the `round2` folder, and replace the old video from `round0` with the new annotated video from `round1` and begin annotating again. You will probably have to add or remove more coordinates so the model can find the videos and make sure the start and end frames are set correctly.

- Step 5: Export the Round 2 annotations and add to the `annotations/round2` folder here. Then, set `CURRENT_ROUND` to be `round2` at the top of `prepare_stimuli_videos_datavyu.ipynb` before running it and update the `CSV_FILES` and `BLOCK_TO_STEM` mappings as needed. This will create a new file `stimuli_prompts_round2.csv` that is ready to be inputted into `SAM` again. 

- Step 6: Run `annotate_videos.py --csv annotations/stimuli_prompts_round2.csv --output scripts/preprocessing/segmentation/output_round2 --render_video --skip_existing --debug_points` in a `tmux` window, similar to Step 3.

## Older pipeline
- Step 1: Annotate each clip with the list of salient objects. For each salient object, provide the time that the object appears and the time that it disappears or changes position noticeably (for example, face is turned). Each object can be annotated separate from other objects. This file is stored in `annotations/raw_stimuli_prompts.csv` In the future this could be done directly in DataVyu.

- Step 2: Run `prepare_stimuli_videos.ipynb`. This formats the raw sheet and adds it to `annotations/stimuli_prompts_round0.csv`. 

- Step 3: Run `annotate_videos.py --csv annotations/stimuli_prompts_round0.csv --output scripts/preprocessing/segmentation/output_raw --render_video` in a `tmux` window. This selects all GPUs by default. You can specify certain GPUs by using `--gpu 0 1` for example. Check available GPUs in Tversky by running `nvidia-smi` This step tries to annotate SAM on all of the videos with the raw annotations.

- Step 4: Run `create_datavyu_files.rb` directly in DataVyu (Script -> Run Script). This creates DataVyu files that is pre-filled based on the stimuli prompts for SAM. For each file, add the rendered video as the data source. After this, follow the steps in `Current pipeline` above.


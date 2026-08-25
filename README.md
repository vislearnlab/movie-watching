# Movie watching
Stimuli, data, and analysis for a study measuring the kinds of information that children and adults look at while watching movies.

## Repository structure

- `experiment/` — the PsychoPy/Tobii experiment that collects raw gaze data
- `scripts/analysis/` — ISC and gaze/face analysis code
- `scripts/preprocessing/` — QC checks, segmentation/annotation pipeline, gaze/heatmap visualization, fixation detection
- `annotations/` — Datavyu annotation CSVs and the stimulus-prompt files derived from them (output of `scripts/preprocessing/segmentation/`)
- `reports/` — QC reports (output of `scripts/preprocessing/qc/`)
- `documentation/` — data/event/experiment key dictionaries
- `stimuli/` — experiment stimuli and calibration assets
- `data/` and `figures/` are gitignored — raw per-participant data and generated results/figures stay on the lab server rather than in git, since they're large and some contain participant-level information

## Experiment
The experiment folder contains the PsychoPy experiment, integrated with a Tobii eye tracker, to run the movie watching experiment. This code been tested with Python=3.10 and with a Tobii Fusion. Run the code below to test. Before running this code, replace the `stimuli` folder with the files within [this Google Drive folder](https://drive.google.com/drive/folders/1R2pX_tSUkch8JeJ90WEl-de1RUH0tiMu?usp=drive_link). More information about the key bindings you can use during the experiment is in `documentation/exp_dictionary.csv`

```
conda create -n moviewatching python=3.10
conda activate moviewatching
pip install -r requirements.txt
cd experiment
python run_movie.py --subject <SUBJECT_ID>
```

If you do not have an eye tracker, alternatively run `python run_movie.py --subject <SUBJECT_ID> --mock` in place of the last line.

Output data is saved to `data/raw`, logs are saved to `data/logs`, and more information about the data variables is in `documentation/data_dictionary.csv`

## Switching Conda from Rosetta to ARM in MacOS
Here are instructions to switch your Conda environment from Rosetta to ARM in MacOS. First right-click on the Terminal icon in Applications > Utilities. Select Get Info > Uncheck the tick box that says 'Open using Rosetta'
```
rm -rf ~/miniconda3
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/Miniconda3-latest-MacOSX-arm64.sh
```

1. Press Return to review Miniconda’s End User License Agreement (EULA). You can view Anaconda’s Terms of Service (TOS) at https://www.anaconda.com/legal.
2. Enter yes to agree to the EULA.
3. Return to accept the default install location (PREFIX=/Users/<USER>/miniconda3), or enter another file path to specify an alternate installation directory. The installation might take a few minutes to complete. 
4. Choose an initialization options: Yes (Recommended) - conda modifies your shell configuration to initialize conda whenever you open a new shell and to recognize conda commands automatically.

Run `source ~/miniconda3/bin/activate` to apply the changes automatically in the current Terminal window.

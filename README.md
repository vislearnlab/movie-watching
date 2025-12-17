# Movie watching
Stimuli, data, and analysis for a study measuring the kinds of information that children and adults look at while watching movies.


## For updating conda for switching to this branch from old branch
(instructions for Urvi!)

To remove old miniconda install, cd ```/Users/visuallearninglab``` and then run ```rm -rf ~/miniconda3```
Or navigate to this folder in Finder and just manually delete the folder -- that works too!

Now, we want to reinstall miniconda.


## Experiment
The experiment folder contains the PsychoPy experiment, integrated with a Tobii eye tracker, to run the movie watching experiment. This code been tested with Python=3.10 and with a Tobii Fusion. Run the code below to test. More information about the key bindings you can use during the experiment is in `documentation/exp_dictionary.csv`

```
conda create -n moviewatching python=3.10
conda activate moviewatching
pip install -r requirements.txt
cd experiment
python moviewatching.py --subject <SUBJECT_ID>
```

Output data is saved to `data/raw` and more information about the data variables is in `documentation/data_dictionary.csv`

## Switching from Rosetta to ARM in MacOS
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

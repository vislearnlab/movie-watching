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
pip install -r requirements.txt
cd experiment
python moviewatching.py --subject <SUBJECT_ID>
```

Output data is saved to `data/raw` and more information about the data variables is in `documentation/data_dictionary.csv`

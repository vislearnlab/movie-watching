"""
Eyetracking Data Quality Control Script

Processes Tobii eyetracking data from multiple participants and generates
a QC report with validity metrics and calibration accuracy per trial.

This script does NOT exclude any data — it only reports metrics so you can
make exclusion decisions in downstream analyses.

HOW TO USE:
1. Open Terminal
2. Navigate to the folder containing this script
3. Run: python dataQC_check.py /path/to/data
4. Find your output in data/qc_checks/qc_report_YYYYMMDD.csv

You can also specify a custom output path:
    python dataQC_check.py /path/to/data /path/to/output.csv

INPUT STRUCTURE:
    data/raw/
        adults/
            MW001/
                MW001_YYYYMMDD_HHMMSS.csv                 <- main gaze data
                MW001_YYYYMMDD_HHMMSS_validation_summary.csv
                MW001_YYYYMMDD_HHMMSS_trial_order.csv
            MW002/
                ...
        kids/
            MWK01/
                ...
        infants/
"""

import argparse
import os                   # for working with file paths and directories
import sys                  # for reading command line arguments
import glob                 # for finding files that match a pattern
import re                   # for pattern matching in strings (regular expressions)
import pandas as pd         # for working with tabular data (like Excel in Python)
import numpy as np          # for numerical operations
from pathlib import Path    # another way to work with file paths

# Project root: two levels up from this script (preprocessing/qc/ -> project root)
MAIN_DIR = Path(__file__).resolve().parents[2]


def resolve_path(p):
    """Resolve p relative to MAIN_DIR if it is not already absolute."""
    p = Path(p)
    return p if p.is_absolute() else MAIN_DIR / p


def find_latest_qc_report(qc_dir):
    """Return the most recent qc_report_*.csv in qc_dir, or None if none exist."""
    reports = sorted(Path(qc_dir).glob("qc_report_*.csv"))
    return reports[-1] if reports else None


# =============================================================================
# FILE DISCOVERY
# =============================================================================
# These functions help locate the right data files for each participant.
# Each participant has a folder containing 4 CSV files with a shared naming
# pattern, and we need to identify which file is which based on the suffix.
# =============================================================================

def find_participant_files(participant_dir):
    """
    Find the set of data files for a participant.
    
    Returns dict with keys: 'gaze', 'validation_summary', 'trial_order'
    """
    csv_files = glob.glob(os.path.join(participant_dir, '**', '*.csv'), recursive=True)  # find all CSVs in folder and in subfolders
    
    files = {                           # initialize empty dict to store file paths
        'gaze': None,
        'validation_summary': None,
        'validation': None,
        'trial_order': None,
    }
    
    for f in csv_files:                                     # loop through each CSV file
        basename = os.path.basename(f)                      # get just the filename, not full path
        if '_validation_summary.csv' in basename:           # check which type of file this is
            files['validation_summary'] = f
        elif '_validation.csv' in basename:
            files['validation'] = f
        elif '_trial_order.csv' in basename:
            files['trial_order'] = f
        elif basename.count('_') == 2 and basename.endswith('.csv'):
            files['gaze'] = f                               # main gaze file has exactly 2 underscores
    
    return files


def get_participant_id(participant_dir):
    """Extract participant ID from folder name."""
    return os.path.basename(participant_dir)                # e.g., '/data/MW005' -> 'MW005'


# =============================================================================
# DATA PARSING
# =============================================================================
# These functions extract meaningful information from the raw data files.
# The main tasks are:
#   1. Find where each trial starts and ends in the gaze data
#   2. Calculate how much valid data we got for each trial
#   3. Pull out calibration accuracy from the validation summary file
#   4. Figure out which calibration goes with which block of trials
# =============================================================================

def parse_trial_boundaries(gaze_df):
    """
    Parse trial start/end events from the events column.
    
    The events column contains markers like 'Trial_Start_0|Video_frank_play'
    when a trial begins and 'Trial_End_0' when it ends. This function finds
    all these markers and records which rows of data belong to each trial.
    
    Returns list of dicts with keys:
        - trial_index: int
        - trial_name: str
        - start_idx: row index in gaze_df
        - end_idx: row index in gaze_df
    """
    trials = []                         # will hold info about each trial we find
    current_trial = None                # keeps track of trial we're currently inside
    
    for idx, row in gaze_df.iterrows():                     # loop through each row of data
        event = row.get('events', '')                       # get the event column value
        if pd.isna(event) or event == '':                   # skip if no event on this row
            continue
        
        # Check for trial start: "Trial_Start_0|Video_frank_play"
        # The pattern below looks for: 'Trial_Start_' + a number + '|Video_' + the video name
        start_match = re.match(r'Trial_Start_(\d+)\|Video_(.+)', str(event))
        if start_match:
            trial_index = int(start_match.group(1))         # extract trial number
            trial_name = start_match.group(2)               # extract video name
            current_trial = {
                'trial_index': trial_index,
                'trial_name': trial_name,
                'start_idx': idx,                           # remember which row this started on
                'end_idx': None,
            }
            continue
        
        # Check for trial end: "Trial_End_0"
        end_match = re.match(r'Trial_End_(\d+)', str(event))
        if end_match and current_trial is not None:
            trial_index = int(end_match.group(1))
            if trial_index == current_trial['trial_index']: # make sure end matches start
                current_trial['end_idx'] = idx              # record where trial ended
                trials.append(current_trial)                # save this trial's info
                current_trial = None                        # reset for next trial
    
    return trials


def compute_trial_validity(gaze_df, start_idx, end_idx):
    """
    Compute validity metrics for a trial.
    
    Validity tells us what percentage of gaze samples were successfully
    tracked by the eyetracker. Low validity means the participant looked
    away, blinked a lot, or the tracker lost the eye for some reason.
    
    Returns dict with:
        - n_samples
        - left_n_valid, left_percent_valid
        - right_n_valid, right_percent_valid
        - mean_percent_valid
        - trial_duration_sec, valid_duration_sec
    """
    trial_data = gaze_df.loc[start_idx:end_idx]             # slice out just this trial's rows
    
    n_samples = len(trial_data)                             # total number of gaze samples
    
    if n_samples == 0:                                      # handle edge case of empty trial
        return {
            'n_samples': 0,
            'left_n_valid': 0,
            'left_percent_valid': 0.0,
            'right_n_valid': 0,
            'right_percent_valid': 0.0,
            'mean_percent_valid': 0.0,
            'trial_duration_sec': 0.0,
            'valid_duration_sec': 0.0,
        }
    
    left_valid = trial_data['left_valid'].sum()             # count valid samples for left eye
    right_valid = trial_data['right_valid'].sum()           # count valid samples for right eye
    
    left_pct = (left_valid / n_samples) * 100               # convert to percentage
    right_pct = (right_valid / n_samples) * 100
    mean_pct = (left_pct + right_pct) / 2                   # average of both eyes
    
    # Calculate durations (timestamps are in microseconds, sampling rate is 250 Hz)
    first_ts = trial_data['system_time_stamp'].iloc[0]      # first timestamp in trial
    last_ts = trial_data['system_time_stamp'].iloc[-1]      # last timestamp in trial
    trial_duration_sec = (last_ts - first_ts) / 1_000_000   # convert microseconds to seconds
    
    mean_valid_samples = (left_valid + right_valid) / 2     # average valid samples across eyes
    valid_duration_sec = mean_valid_samples / 250           # convert samples to seconds at 250 Hz
    
    return {
        'n_samples': n_samples,
        'left_n_valid': int(left_valid),
        'left_percent_valid': round(left_pct, 2),
        'right_n_valid': int(right_valid),
        'right_percent_valid': round(right_pct, 2),
        'mean_percent_valid': round(mean_pct, 2),
        'trial_duration_sec': round(trial_duration_sec, 2),
        'valid_duration_sec': round(valid_duration_sec, 2),
    }


def parse_validation_summary(validation_summary_df):
    """
    Parse the validation summary file to extract mean accuracy per validation step.
    
    The validation summary file contains accuracy measurements for each
    calibration point, plus a 'mean' row summarizing overall accuracy.
    We only care about the mean rows here.
    
    Returns dict keyed by validation_step with values:
        - left_deg, right_deg, mean_deg
        - left_px, right_px, mean_px
    """
    validations = {}
    
    mean_rows = validation_summary_df[validation_summary_df['point'] == 'mean']  # filter to mean rows only
    
    for _, row in mean_rows.iterrows():                     # loop through each validation step
        step = row['validation_step']                       # e.g., 'pre_validation_1'
        left_deg = row['Mean_accuracy_degrees_left']
        right_deg = row['Mean_accuracy_degrees_right']
        left_px = row['Mean_accuracy_pixels_left']
        right_px = row['Mean_accuracy_pixels_right']
        
        validations[step] = {                               # store all accuracy metrics
            'val_left_deg': round(left_deg, 4),
            'val_right_deg': round(right_deg, 4),
            'val_mean_deg': round((left_deg + right_deg) / 2, 4),
            'val_left_px': round(left_px, 4),
            'val_right_px': round(right_px, 4),
            'val_mean_px': round((left_px + right_px) / 2, 4),
        }
    
    return validations


def map_blocks_to_validations(trial_order_df, validations):
    """
    Map each block_index to its corresponding validation step.
    
    Calibration happens before block 1, and validation checks happen before
    each subsequent block. The validation is named after the first trial of
    the block it precedes (e.g., 'block_validation_trial3_1' happens before
    trial 3 starts, which is the first trial of that block).
    
    If a validation failed and recalibration was needed, there may be multiple
    attempts (e.g., '_1', '_2'). We take the last attempt since that's what
    was used for the actual data collection.
    
    Returns dict: block_index -> validation metrics dict
    """
    
    # =========================================================================
    # TODO FOR TARUN: MID-TRIAL RECALIBRATION HANDLING
    # =========================================================================
    # Currently, this function assigns ONE validation per block — all trials
    # in a block share the same calibration accuracy values.
    #
    # If we add mid-trial recalibration, we'll need to:
    #   1. Change the return structure from block_index -> validation
    #      to trial_index -> validation (so each trial can have its own)
    #   2. Parse any new validation events that occur mid-block
    #      (need to know what these will be named in the events column)
    #   3. For each trial, find the most recent validation that occurred
    #      BEFORE that trial started
    #
    # The logic would look something like:
    #   - Build a list of (timestamp, validation_metrics) for ALL validations
    #   - For each trial, find the validation with the largest timestamp
    #     that is still less than the trial's start timestamp
    #
    # This will also require changes in process_participant() where we
    # currently do: val_metrics = block_validations.get(block_idx, {})
    # Instead it would be: val_metrics = trial_validations.get(trial_idx, {})
    # =========================================================================
    
    block_validations = {}
    
    # Group trials by block and find first/last trial in each block
    blocks = trial_order_df.groupby('block_index').agg({
        'total_trial_index': ['min', 'max']
    }).reset_index()
    blocks.columns = ['block_index', 'first_trial', 'last_trial']  # rename columns for clarity
    blocks = blocks.sort_values('block_index')
    
    validation_steps = list(validations.keys())             # get all validation step names
    
    for _, block_row in blocks.iterrows():                  # loop through each block
        block_idx = block_row['block_index']
        first_trial = block_row['first_trial']
        
        if block_idx == 0:
            # First block uses pre_validation (the initial calibration)
            pre_vals = [v for v in validation_steps if v.startswith('pre_validation')]
            if pre_vals:
                pre_vals_sorted = sorted(pre_vals)          # sort to get attempts in order
                block_validations[block_idx] = validations[pre_vals_sorted[-1]]  # take last attempt
            else:
                block_validations[block_idx] = None
        else:
            # Other blocks: find validation named after this block's first trial
            # Pattern: block_validation_trial{N}_{attempt}
            matching_vals = []
            for v in validation_steps:
                match = re.match(r'block_validation_trial(\d+)_(\d+)', v)
                if match:
                    val_trial = int(match.group(1))         # which trial this validation precedes
                    val_attempt = int(match.group(2))       # attempt number (1, 2, etc.)
                    if val_trial == first_trial:            # does it match our block's first trial?
                        matching_vals.append((val_attempt, v))
            
            if matching_vals:
                matching_vals.sort()                        # sort by attempt number
                block_validations[block_idx] = validations[matching_vals[-1][1]]  # take last attempt
            else:
                block_validations[block_idx] = None
    
    return block_validations


# =============================================================================
# MAIN PROCESSING
# =============================================================================
# These functions tie everything together:
#   - process_participant: handles one participant's data
#   - process_all_participants: loops through all participant folders
# =============================================================================

def process_participant(participant_dir):
    """
    Process a single participant and return list of trial QC rows.
    
    This function loads all the data files for one participant, extracts
    trial boundaries, computes validity metrics, and matches up calibration
    accuracy with each trial based on which block it belongs to.
    """
    participant_id = get_participant_id(participant_dir)
    files = find_participant_files(participant_dir)
    
    # Check that we found all required files
    missing = [k for k, v in files.items() if v is None and k != 'validation']
    if missing:
        print(f"  WARNING: Missing files for {participant_id}: {missing}")
        return []
    
    # Load all data files into pandas DataFrames
    print(f"  Loading gaze data...")
    gaze_df = pd.read_csv(files['gaze'])
    
    print(f"  Loading validation summary...")
    validation_summary_df = pd.read_csv(files['validation_summary'])
    
    print(f"  Loading trial order...")
    trial_order_df = pd.read_csv(files['trial_order'])
    
    # Find where each trial starts and ends
    print(f"  Parsing trial boundaries...")
    trials = parse_trial_boundaries(gaze_df)
    print(f"    Found {len(trials)} trials")
    
    # Extract calibration accuracy from validation summary
    print(f"  Parsing validation data...")
    validations = parse_validation_summary(validation_summary_df)
    print(f"    Found {len(validations)} validation steps")
    
    # Figure out which validation goes with which block
    block_validations = map_blocks_to_validations(trial_order_df, validations)
    
    # Build a lookup table: trial_index -> block info
    trial_to_block = {}
    for _, row in trial_order_df.iterrows():
        trial_to_block[row['total_trial_index']] = {
            'block_id': row['block_id'],
            'block_index': row['block_index'],
        }
    
    # Process each trial and build output rows
    results = []
    for trial in trials:
        trial_idx = trial['trial_index']
        trial_name = trial['trial_name']
        
        # Look up which block this trial belongs to
        block_info = trial_to_block.get(trial_idx, {'block_id': None, 'block_index': None})
        block_idx = block_info['block_index']
        
        # Calculate validity metrics for this trial
        validity = compute_trial_validity(gaze_df, trial['start_idx'], trial['end_idx'])
        
        # Get the calibration accuracy for this trial's block
        val_metrics = block_validations.get(block_idx, {})
        if val_metrics is None:                             # handle missing validation data
            val_metrics = {
                'val_left_deg': None,
                'val_right_deg': None,
                'val_mean_deg': None,
                'val_left_px': None,
                'val_right_px': None,
                'val_mean_px': None,
            }
        
        # Combine everything into one row
        row = {
            'participant_id': participant_id,
            'trial_index': trial_idx,
            'trial_name': trial_name,
            'block_id': block_info['block_id'],
            'block_index': block_idx,
            **validity,                                     # unpack validity metrics into row
            **val_metrics,                                  # unpack validation metrics into row
        }
        results.append(row)
    
    return results


def process_all_participants(data_dir, output_path, existing_df=None):
    """
    Process all participants in the data directory and save aggregated QC report.

    Loops through each subfolder in the data directory, processes that
    participant's data, and combines all results into one CSV file.

    If existing_df is provided, participants already present in it are skipped
    and their rows are carried forward into the output unchanged.
    """
    # Find all participant folders (exclude qc_checks folder)
    participant_dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, '**', '*'), recursive=True)
        if os.path.isdir(d) and not os.path.basename(d) in ['qc_checks', 'adults', 'infants', 'kids']
    ])

    already_done = set()
    if existing_df is not None and not existing_df.empty:
        already_done = set(existing_df['participant_id'].unique())
        print(f"Loaded {len(already_done)} already-processed participant(s) from existing report.")

    print(f"Found {len(participant_dirs)} participant directories")

    new_results = []                                        # will hold results from new participants

    for participant_dir in participant_dirs:
        participant_id = get_participant_id(participant_dir)

        if participant_id in already_done:
            print(f"\nSkipping {participant_id} (already in existing report)")
            continue

        print(f"\nProcessing {participant_id}...")

        try:
            results = process_participant(participant_dir)
            new_results.extend(results)                     # add this participant's trials to list
            print(f"  Completed: {len(results)} trials")
        except Exception as e:
            print(f"  ERROR processing {participant_id}: {e}")
            continue

    # Merge new results with any existing data
    existing_rows = existing_df if existing_df is not None else pd.DataFrame()
    if new_results:
        new_df = pd.DataFrame(new_results)
        all_df = pd.concat([existing_rows, new_df], ignore_index=True)
    else:
        all_df = existing_rows

    # Convert results to DataFrame and save
    if not all_df.empty:
        output_df = all_df
        
        # Put columns in a sensible order
        column_order = [
            'participant_id', 'trial_index', 'trial_name', 'block_id', 'block_index',
            'n_samples', 'trial_duration_sec', 'valid_duration_sec',
            'left_n_valid', 'left_percent_valid', 
            'right_n_valid', 'right_percent_valid', 'mean_percent_valid',
            'val_left_deg', 'val_right_deg', 'val_mean_deg',
            'val_left_px', 'val_right_px', 'val_mean_px',
        ]
        output_df = output_df[column_order]
        
        output_df.to_csv(output_path, index=False)          # save to CSV (no row numbers)
        print(f"\n{'='*60}")
        print(f"QC report saved to: {output_path}")
        print(f"Total trials processed: {len(output_df)}")
        print(f"Participants: {output_df['participant_id'].nunique()}")
    else:
        print("\nNo data processed!")
    
    return output_df if not all_df.empty else None


# =============================================================================
# ENTRY POINT
# =============================================================================
# This section runs when you execute the script from the command line.
# It reads the command line arguments, sets up the output path, and
# kicks off the processing.
# =============================================================================

if __name__ == '__main__':
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Eyetracking QC script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/raw",
        help="Path to data directory (absolute or relative to project root).",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Output CSV path. Defaults to data/qc_checks/qc_report_YYYYMMDD.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process all participants even if a prior report exists.",
    )
    args = parser.parse_args()

    data_dir = str(resolve_path(args.data_dir))

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Default output goes to data/qc_checks/ relative to project root
    qc_dir = str(MAIN_DIR / "data" / "qc_checks")
    os.makedirs(qc_dir, exist_ok=True)

    if args.output_path:
        output_path = str(resolve_path(args.output_path))
    else:
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = os.path.join(qc_dir, f'qc_report_{timestamp}.csv')

    # Load existing report unless --overwrite
    existing_df = None
    if not args.overwrite:
        latest = find_latest_qc_report(qc_dir)
        if latest:
            print(f"Found existing report: {latest}")
            existing_df = pd.read_csv(latest)

    process_all_participants(data_dir, output_path, existing_df=existing_df)

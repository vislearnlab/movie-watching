from argparse import ArgumentParser
import os
import numpy as np
import random
from pathlib import Path
from psychopy import core, visual, event, monitors, prefs, logging
from psychopy.visual.movies import MovieStim
from psychopy_tobii_infant import TobiiInfantController
logging.console.setLevel(logging.ERROR)
import tobii_research as tr
import time
import pandas as pd
import yaml
import sounddevice as sd
import datetime
from glob import glob

# known issues with directly integrating sounddevice that we are circumventing (https://github.com/psychopy/psychopy-sounddevice/issues/5)
from psychopy_sounddevice import SoundDeviceSound

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  
os.environ["FFREPORT"] = "file=/dev/null"  
os.environ['FFMPEG_LOG_LEVEL'] = 'quiet'
# Get list of all audio devices
devices = sd.query_devices()
candidates = []
for idx, device in enumerate(devices):
    device_name = device['name']
    # Check if it's an output device and matches monitor keywords
    if device['max_output_channels'] > 0:
        if any(keyword.lower() in device_name.lower() 
                for keyword in ['hdmi', 'display', 'monitor', 'nvidia', 'amd', 'PA24']):
            candidates.append((idx, device))
print(candidates)
if len(candidates) > 0:
    best_idx, best_device = candidates[-1]
    # Set the default output device for sounddevice
    sd.default.device = best_idx
    prefs.hardware['audioLib'] = ['sounddevice']
    prefs.hardware['audioDevice'] = best_device['name']
    print("Using monitor audio device:", best_device['name'])
else:
    print("No monitor audio found, using default.")

DIR = Path("../")
trial_types = ["sesame", "slow", "frank", "pixar"]
with open("config.yaml", 'r') as stream:
    config_data = yaml.safe_load(stream)

def main():
    DIR = Path("../")
    parser = ArgumentParser(description="Movie Watching Experiment")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., S001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    run_experiment(args.subject, args.debug)

def trial_order(dir, debug=False):
    blocks = {}
    for file in os.listdir(dir):
        if file.endswith('.mp4') and not file.startswith("intertrial_calibration"):
            block_num = file.split('_')[0]
            block_num = block_num.replace('india', '').replace('us', '')
            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(file.removesuffix('.mp4'))
    
    for block in blocks:
        random.shuffle(blocks[block])

    Trials = []
    block_keys = list(blocks.keys())
    random.shuffle(block_keys)
    if debug:
        for block in blocks:
            blocks[block] = blocks[block][:2]
        block_keys = block_keys[:2]

    trial_idx = 0
    
    non_pixar_block_keys = [block for block in block_keys if block != 'pixar']
    for iblock, block in enumerate(non_pixar_block_keys):
        # Add trials from this block
        for within_trial_idx, video_name in enumerate(blocks[block]):
            Trials.append({
                'total_trial_index': trial_idx,
                'video_path': os.path.join(dir, video_name + '.mp4'),
                'video_name': video_name,
                'block_id': block,
                'block_index': iblock,
                'within_block_trial_index': within_trial_idx
            })
            trial_idx += 1
    
    # Add pixar trials at the end
    if 'pixar' in blocks and not debug:
        for within_trial_idx, video_name in enumerate(blocks['pixar']):
            Trials.append({
                'total_trial_index': trial_idx,
                'video_path': os.path.join(dir, video_name + '.mp4'),
                'video_name': video_name,
                'block_id': 'pixar',
                'block_index': len(block_keys) - 1,
                'within_block_trial_index': within_trial_idx
            })
            trial_idx += 1
    
    return Trials

def calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event, mode="initial", skip_first_calibration=False):
    # Run validation loop
    global config_data
    max_retries = config_data[f"{mode}_validation"]["max_retries"]
    good_threshold = config_data[f"{mode}_validation"]["good_threshold"]
    bad_threshold = config_data[f"{mode}_validation"]["bad_threshold"]
    i = 0
    while i < max_retries:

        # Run calibration
        if not skip_first_calibration or i > 0:
            calibration_sound = SoundDeviceSound(CALIB_SOUND)
            controller.run_calibration(CALIPOINTS, CALISTIMS, audio=calibration_sound)
            calibration_sound.stop()

        # Run validation
        validation_sound = SoundDeviceSound(VALID_SOUND)
        result = controller.run_validation(
            validation_points=CALIPOINTS,
            infant_stims=CALISTIMS,
            show_results=True,
            event=f"{calib_event}_{i+1}",
            audio=validation_sound
        )
        validation_sound.stop()

        # Extract the 5 accuracy values for each eye 
        left = [
            result.get(f"Point_{idx}_accuracy_degrees_left", bad_threshold+.01)
            for idx in range(1, 6)
        ]
        right = [
            result.get(f"Point_{idx}_accuracy_degrees_right", bad_threshold+.01)
            for idx in range(1, 6)
        ]

        # Check: does a single eye individually pass these thresholds?
        def eye_passes(targets):
            good = sum(t < good_threshold for t in targets)   
            bad  = sum(t > bad_threshold for t in targets)   
            return (good >= 4) and (bad == 0)

        # Average both eyes per target
        avg = [(l + r) / 2 for l, r in zip(left, right)]

        avg_good = sum(t < good_threshold for t in avg)
        avg_bad  = sum(t > bad_threshold for t in avg)

        avg_ok = (avg_good >= 4) and (avg_bad == 0)

        # If average fails, check each eye individually
        if avg_ok:
            break
        else:
            left_ok  = eye_passes(left)
            right_ok = eye_passes(right)

            if left_ok or right_ok:
                break

        # If nothing passed then recalibrate
        i += 1
        if i < max_retries:
            controller.display_text(
                "Validation failed. Recalibrating...",
                duration=2
            )

def check_and_resume_session(subject_dir, Sub, TIMESTAMP):
    """
    Check for existing records within the last hour and prompt user to resume.
    
    Returns:
        tuple: (filename, should_calibrate, start_trial_index, timestamp_to_use, existing_trial_order)
    """
    global config_data
    existing_files = glob(str(subject_dir / f'{Sub}_*.csv'))
    recent_file = None
    last_timestamp = None

    if existing_files:
        current_time = datetime.datetime.now()
        for file in existing_files:
            try:
                # Extract timestamp from filename (assumes format: Sub_YYYYMMDD_HHMMSS.csv)
                file_timestamp_str = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
                file_timestamp = datetime.datetime.strptime(file_timestamp_str, '%Y%m%d_%H%M%S')
                
                if current_time - file_timestamp < datetime.timedelta(hours=config_data['reuse_session']['time_delta_hours']):
                    recent_file = file
                    last_timestamp = file_timestamp_str
                    break
            except (ValueError, IndexError):
                continue

    # No recent file found - start fresh
    if not recent_file:
        return subject_dir / f'{Sub}_{TIMESTAMP}.csv', True, 0, TIMESTAMP, None

    # Recent file found - ask user
    response = input(f"Found existing record from {last_timestamp}. Use existing records? (y/n): ").strip().lower()
    
    if response != 'y':
        return subject_dir / f'{Sub}_{TIMESTAMP}.csv', True, 0, TIMESTAMP, None
    
    # User wants to resume - load existing trial order and find last trial
    print(f"Resuming from existing file: {recent_file}")
    
    # Load the existing trial order
    trial_order_file = subject_dir / f"{Sub}_{last_timestamp}_trial_order.csv"
    existing_trial_order = None
    
    try:
        if trial_order_file.exists():
            existing_trial_order = pd.read_csv(trial_order_file).to_dict('records')
            print(f"Loaded existing trial order from {trial_order_file}")
        else:
            print(f"Warning: Trial order file not found at {trial_order_file}")
            return subject_dir / f'{Sub}_{TIMESTAMP}.csv', True, 0, TIMESTAMP, None
    except Exception as e:
        print(f"Error reading trial order: {e}. Starting fresh session.")
        return subject_dir / f'{Sub}_{TIMESTAMP}.csv', True, 0, TIMESTAMP, None
    
    # Find last completed trial
    try:
        existing_data = pd.read_csv(recent_file)
        if not existing_data.empty and 'trial' in existing_data.columns:
            last_trial_idx = existing_data['trial'].max()
            # Skip to the trial after the last completed one, plus one more as specified
            start_trial_idx = last_trial_idx + 2
            print(f"Resuming from trial {start_trial_idx} (skipping trial {last_trial_idx + 1})")
            return recent_file, False, start_trial_idx, last_timestamp, existing_trial_order
    except Exception as e:
        print(f"Error reading existing data: {e}. Starting from the beginning of existing trial order.")
    
    return recent_file, False, 0, last_timestamp, existing_trial_order

def run_experiment(Sub, debug=False):
    # Constants
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  
    os.environ["FFREPORT"] = "file=/dev/null"  
    os.environ['FFMPEG_LOG_LEVEL'] = 'quiet'
    import gc
    gc.collect()
    
    global DIR, config_data
    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    DISPSIZE = (1920, 1080)
    CALINORMP = [(-0.4, 0.4 ), (-0.4, -0.4), (0.0, 0.0), (0.4, 0.4), (0.4, -0.4)]
    CALIPOINTS = [(x * DISPSIZE[0], y * DISPSIZE[1]) for x, y in CALINORMP]
    STIM_DIR = DIR / os.path.join('stimuli')
    CALIB_DIR = STIM_DIR / 'calibration'
    CALIB_SOUND = os.path.join(CALIB_DIR, 'hothothot3.wav')
    #calibration_sound = SoundDeviceSound(CALIB_SOUND)
    VALID_SOUND = os.path.join(CALIB_DIR, 'upchime.wav')
    #validation_sound = SoundDeviceSound(VALID_SOUND)
    CALISTIMS = [
        f"{CALIB_DIR}/{x}" for x in os.listdir(CALIB_DIR)
        if x.endswith('.png') and not x.startswith('.')
    ]

    # Create data directory
    data_dir = Path("../data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    ###############################################################################
    # Setup Eye Tracker and Monitor
    eye_tracker = tr.find_all_eyetrackers()[0]

    # Get display area information
    display_area = eye_tracker.get_display_area()
    screen_width_cm = display_area.width / 10
    screen_height_cm = display_area.height / 10
    viewing_distance_mm = (
        display_area.top_left[2] + 
        display_area.top_right[2] + 
        display_area.bottom_left[2] + 
        display_area.bottom_right[2]
    ) / 4
    viewing_distance_cm = viewing_distance_mm / 10

    print(f"Screen width: {screen_width_cm} cm")
    print(f"Screen height: {screen_height_cm} cm")
    print(f"Viewing distance: {viewing_distance_cm} cm")

    # Create monitor
    mon = monitors.Monitor('TobiiFusion')
    mon.setWidth(screen_width_cm)
    mon.setDistance(viewing_distance_cm)
    mon.setSizePix([DISPSIZE[0], DISPSIZE[1]])
    mon.saveMon()

    # Create window
    win = visual.Window(size=[DISPSIZE[0], DISPSIZE[1]],
                        units='pix',
                        monitor=mon,
                        screen=1,
                        fullscr=True,
                        allowGUI=False,
                        checkTiming=False)
    # Initialize TobiiController
    subject_dir = data_dir / f"{Sub}"
    os.makedirs(subject_dir, exist_ok=True)

    # Check for existing session and get parameters
    filename, should_calibrate, start_trial_idx, timestamp_to_use, existing_trial_order = check_and_resume_session(
        subject_dir, Sub, TIMESTAMP
    )

    controller = TobiiInfantController(win, calibration_disc_size=200, filename=str(filename))

    ###############################################################################
    # Show Status and Calibration.
    win.flip()
    core.wait(0.1)
    grabber = MovieStim(win, f"{STIM_DIR}/ag/Attentiongrabber.mp4", size=[600, 600], units='pix')
    grabber.setAutoDraw(True)
    grabber.play()
    controller.show_status()
    controller.eyetracker.set_gaze_output_frequency(250)
    grabber.setAutoDraw(False)
    grabber.stop()
    calibration_sound = SoundDeviceSound(CALIB_SOUND)
    VALID_SOUND = os.path.join(CALIB_DIR, 'upchime.wav')
    validation_sound = SoundDeviceSound(VALID_SOUND)
    if should_calibrate:
        calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event="pre_validation", mode="initial")
    if existing_trial_order is not None:
        Trials = existing_trial_order
        print("Using existing trial order") 
    else:
        Trials = trial_order(f"{DIR}/stimuli/main_blocks", debug=debug)
        pd.DataFrame(Trials).to_csv(subject_dir / f"{Sub}_{timestamp_to_use}_trial_order.csv", index=False)
    # Skip to the appropriate trial if resuming
    if start_trial_idx > 0:
        Trials = Trials[start_trial_idx:]
    print(Trials)
    
    pd.DataFrame(Trials).to_csv(subject_dir / f"{Sub}_{TIMESTAMP}_trial_order.csv", index=False)
    # Start Recording
    controller.start_recording()

    for trial in Trials:
        recalibrate = 1
        if trial['within_block_trial_index'] == 0 and trial['block_index'] != 0:
            calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event=f"block_validation_trial{trial['total_trial_index']}", mode="later", skip_first_calibration=True)
        t1 = time.time()
        trial_id = trial['total_trial_index']
        video_path = trial['video_path']
        video_name = trial['video_name']
        
        print(f"Starting Trial {trial_id}: {video_name}")
        
        # Record trial start event
        controller.record_event(f"Loop_Start_{trial_id}")
        
        # Create movie stimulus
        movie = MovieStim(
            win,
            video_path,
            size=[1920, 1080],
            units='pix',
            loop=False,
            name=video_name,
            movieLib="ffpyplayer"
        )
        
        movie.play()
        movie.setAutoDraw(True)
        win.flip()  # Ensure movie is on screen
        t2 = time.time()
        controller.record_event(f"Trial_Start_{trial_id}|Video_{video_name}")  
        event_type = "play"

        # Collect looking time with pause handling
        remaining_time = config_data['trial_config']['max_time']
        total_lt = 0
        print(config_data)
        while remaining_time > 1:
            lt, event_type = controller.collect_lt_with_calibration(remaining_time, config_data['trial_config']['away_time'])
            total_lt += lt
            print(f'Trial {trial_id} Looking time: %.3fs (total: %.3fs)' % (lt, total_lt))
            if event_type == "pause":
                controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Paused")
                movie.pause()         
                event.waitKeys(keyList=['space'])
                movie.play()  # resume playback
                controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Resumed")
                # Update remaining time and continue loop
                remaining_time = 60 - total_lt
                
            else:
                # Trial ended for another reason (looking_away, calibration, escape, or normal)
                break
        
        # Record final event based on how trial ended
        if event_type == "calibration":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Forced_Recalibration_Key_Press")
        elif event_type == "looking_away":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Looked_Away")
        elif event_type == "next_trial":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Ended_Trial")
        elif event_type == "escape":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Ended_Experiment")
        else:
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Normal")
        controller.record_event(f"Trial_End_{trial_id}")
        controller._flush_data_csv()
        core.wait(0.05)
        movie.setAutoDraw(False)
        movie.stop()
        # delete the movie object
        del movie
        import gc; gc.collect()
        core.wait(0.05)
        if event_type == "escape":
            break
        t3 = time.time()
        if event_type != "normal":
            if event_type == "looking_away":
                controller.display_text("Press 'c' to recalibrate, or press 'p' (or wait 5s) to proceed.")
                start_wait = time.time()
                key = None
                
                while time.time() - start_wait < 5:
                    keys = event.getKeys(keyList=['c', 'p'])
                    if keys:
                        key = keys[0]
                        break
                    core.wait(0.01)
                if key == 'c':
                    controller.record_event(f"Trial_{trial_id}_Forced_Recalibation_Looked_Away")
                    calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event=f"lb_forced_validation_{trial['total_trial_index']}", mode="later")
                    recalibrate += 1
            elif event_type == "calibration":
                calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event=f"key_forced_validation_trial{trial['total_trial_index']}", mode="later")
        win.flip()
        controller.record_event(f"Loop_End_{trial_id}")
        t4 = time.time()
        # Check for escape key
        keys = event.getKeys()
        if 'escape' in keys:
            break
        print(f"Trial {trial_id} duration: {t4 - t1:.2f} seconds, First frame delay (0.5s): {t2 - t1:.2f} seconds, Movie duration (120s): {t3 - t2:.2f} seconds, ISI (1s): {t4 - t3:.2f} seconds")

    ###############################################################################
    # Stop recording and cleanup
    controller.stop_recording()
    controller.close()
    win.close()
    core.quit()

if __name__ == '__main__':
    main()


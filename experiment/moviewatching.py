from argparse import ArgumentParser
import os
import numpy as np
import random
from pathlib import Path
from psychopy import core, visual, event, monitors, sound
from psychopy_tobii_infant import TobiiInfantController
from psychopy import logging
logging.console.setLevel(logging.ERROR)
import tobii_research as tr
import time
import pandas as pd
from gooey import Gooey

DIR = Path("../")
trial_types = ["sesame", "slow", "frank", "pixar"]

#@Gooey(program_name="Movie watching")
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
    
    for iblock, block in enumerate(block_keys):
        if block == 'pixar':
            continue
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

def run_experiment(Sub, debug=False):
    # Constants
    global DIR
    DISPSIZE = (1920, 1200)
    CALINORMP = [(-0.4, 0.4 ), (-0.4, -0.4), (0.0, 0.0), (0.4, 0.4), (0.4, -0.4)]
    CALIPOINTS = [(x * DISPSIZE[0], y * DISPSIZE[1]) for x, y in CALINORMP]
    STIM_DIR = DIR / os.path.join('stimuli')
    CALIB_DIR = STIM_DIR / 'calibration'
    CALIB_SOUND = os.path.join(CALIB_DIR, 'hothothot3.wav')
    calibration_sound = sound.Sound(CALIB_SOUND)
    VALID_SOUND = os.path.join(CALIB_DIR, 'upchime.wav')
    validation_sound = sound.Sound(VALID_SOUND)
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
                        allowGUI=False)

    # Initialize TobiiController
    filename = data_dir / f'{Sub}.csv'
    controller = TobiiInfantController(win, calibration_disc_size=200, filename=str(filename))

    ###############################################################################
    # Show Status and Calibrationj.
    grabber = visual.MovieStim(win, f"{STIM_DIR}/ag/Attentiongrabber.mp4", size=[600, 600], units='pix')
    grabber.setAutoDraw(True)
    grabber.play()
    controller.show_status()
    controller.eyetracker.set_gaze_output_frequency(250)
    grabber.setAutoDraw(False)
    grabber.stop()

    # Run validation loop
    max_retries = 2
    good_threshold = 1.5
    bad_threshold = 4
    i = 0

    while i <= max_retries:

        # Run calibration
        controller.run_calibration(CALIPOINTS, CALISTIMS, audio=calibration_sound)

        # Run validation
        result = controller.run_validation(
            validation_points=CALIPOINTS,
            infant_stims=CALISTIMS,
            show_results=True,
            event=f"pre_validation_{i+1}",
            audio=validation_sound
        )

        # Extract the 5 accuracy values for each eye 
        left  = [result[f"Point_{idx}_accuracy_degrees_left"]
                for idx in range(1, 6)]
        right = [result[f"Point_{idx}_accuracy_degrees_right"]
                for idx in range(1, 6)]

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
        controller.display_text(
            "Validation failed. Recalibrating...",
            duration=2
        )

        i += 1

    
    Trials = trial_order(f"{DIR}/stimuli/main_blocks", debug=debug)
    pd.DataFrame(Trials).to_csv(data_dir / f"{Sub}_trial_order.csv", index=False)
    # Start Recording
    controller.start_recording()

    for trial in Trials:
        if trial['within_block_trial_index'] == 0 and trial['block_index'] != 0:
            controller.run_validation(validation_points=CALIPOINTS, 
                                infant_stims=CALISTIMS, 
                                show_results=True, event=f"validation_{trial['block_index']}", audio=validation_sound)
        t1 = time.time()
        trial_id = trial['total_trial_index']
        video_path = trial['video_path']
        video_name = trial['video_name']
        
        print(f"Starting Trial {trial_id}: {video_name}")
        
        # Record trial start event
        controller.record_event(f"Trial_{trial_id}_Start|Video_{video_name}")
        
        # Create movie stimulus
        movie = visual.MovieStim(
            win,
            video_path,
            size=[1920, 1080],
            units='pix',
            loop=False,
            name=video_name
        )
        
        # Present fixation / attention getter?
        win.flip()
        core.wait(0.5)
        # Start movie
        movie.setAutoDraw(True)
        win.flip()  # Ensure movie is on screen
        t2 = time.time()
        controller.record_event(f"Trial_{trial_id}_Video_Start")
        
        # Collect looking time (60 seconds max, 20 seconds away minimum)
        # todo: if keeping habituation maybe need a seperate mp3 stream?
        lt = controller.collect_lt(60, 20)
        print(f'Trial {trial_id} Looking time: %.3fs' % lt)
        
        # Stop movie
        movie.setAutoDraw(False)
        controller.record_event(f"Trial_{trial_id}_Video_End")
        controller.record_event(f"Trial_{trial_id}_LookingTime_{lt}")
        t3 = time.time()
        # ISI
        win.flip()
        core.wait(1)
        controller.record_event(f"Trial_{trial_id}_End")
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
    args = main()
    run_experiment(args.subject)
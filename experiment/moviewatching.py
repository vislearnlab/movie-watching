from argparse import ArgumentParser
import os
import numpy as np
import random
from pathlib import Path
from psychopy import core, visual, event, monitors
from psychopy_tobii_infant import TobiiInfantController
from psychopy import logging
logging.console.setLevel(logging.ERROR)
import tobii_research as tr
import time
from gooey import Gooey

#@Gooey(program_name="Movie watching")
def main():
    parser = ArgumentParser(description="Movie Watching Experiment")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., S001)')
    args = parser.parse_args()
    run_experiment(args.subject)

def run_experiment(Sub):
    # Constants
    DIR = os.path.dirname(__file__)
    DISPSIZE = (1920, 1200)
    CALINORMP = [(-0.4, 0.4 ), (-0.4, -0.4), (0.0, 0.0), (0.4, 0.4), (0.4, -0.4)]
    CALIPOINTS = [(x * DISPSIZE[0], y * DISPSIZE[1]) for x, y in CALINORMP]
    STIM_DIR = os.path.join(DIR, 'exp', 'stimuli', 'infant')
    CALISTIMS = [
        'exp/stimuli/infant/{}'.format(x) for x in os.listdir(os.path.join(STIM_DIR))
        if x.endswith('.png') and not x.startswith('.')
    ]
    

    # Create data directory
    data_dir = Path('data') / 'raw'
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
    controller = TobiiInfantController(win, calibration_disc_size=200)

    ###############################################################################
    # Show Status and Calibration
    grabber = visual.MovieStim(win, f"{STIM_DIR}/Sea.mp4", size=[600, 600], units='pix')
    grabber.setAutoDraw(True)
    grabber.play()
    #controller.show_status()
    controller.eyetracker.set_gaze_output_frequency(250)
    print(controller.eyetracker.get_display_area()) 
    #grabber.setAutoDraw(False)
    #grabber.stop()

    # Run validation loop
    i = 0
    #controller.run_calibration(CALIPOINTS, CALISTIMS)
    '''
    while (i <= 3):
        result = controller.run_validation(validation_points=CALIPOINTS, 
                                        infant_stims=CALISTIMS, 
                                        show_results=True)
        if result['Mean_accuracy_degrees_left'] > 15.0 or result['Mean_accuracy_degrees_right'] > 15.0:
            controller.display_text("Validation failed. Recalibrating...", duration=2)
        else:
            break
        i += 1
    print(result)
    '''
    ###############################################################################
    # Prepare Stimulus List
    list_of_videos = [f"{STIM_DIR}/Sea.mp4", f"{STIM_DIR}/Sea.mp4"]  # Add more videos as needed
    random.shuffle(list_of_videos)

    # Create trial list with video information
    Trials = []
    for trial_idx, video_path in enumerate(list_of_videos):
        Trials.append({
            'trial_id': trial_idx,
            'video_path': video_path,
            'video_name': os.path.basename(video_path)
        })

    ###############################################################################
    # Start Recording
    filename = data_dir / f'{Sub}.csv'
    controller.start_recording(str(filename))

    for trial in Trials:
        t1 = time.time()
        trial_id = trial['trial_id']
        video_path = trial['video_path']
        video_name = trial['video_name']
        
        print(f"Starting Trial {trial_id}: {video_name}")
        
        # Record trial start event
        controller.record_event(f"Trial_{trial_id}_Start|Video_{video_name}")
        
        # Create movie stimulus
        movie = visual.MovieStim(
            win,
            video_path,
            size=[600, 600],
            units='pix',
            loop=False,
            name=video_name
        )
        
        # Present fixation / attention getter?
        win.flip()
        core.wait(0.5)
        t1_5 = time.time()
        # Start movie
        movie.setAutoDraw(True)
        win.flip()  # Ensure movie is on screen
        t2 = time.time()
        print(f"Render video took {t2 - t1:.2f} seconds")
        controller.record_event(f"Trial_{trial_id}_Video_Start")
        
        # Collect looking time (10 seconds max, 2 seconds away minimum)
        lt = controller.collect_lt(10, 2)
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
        print(f"Trial {trial_id} duration: {t4 - t1:.2f} seconds, First frame delay (0.5s): {t2 - t1:.2f} seconds, Movie duration (10s): {t3 - t2:.2f} seconds, ISI (1s): {t4 - t3:.2f} seconds")

    ###############################################################################
    # Stop recording and cleanup
    controller.stop_recording()
    controller.close()
    win.close()
    core.quit()

if __name__ == '__main__':
    args = main()
    run_experiment(args.subject)
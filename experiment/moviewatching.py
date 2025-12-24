import os
import sys
from argparse import ArgumentParser
import numpy as np
import random
from pathlib import Path
from logging_config import setup_logging, log_exception, safe_execute
from psychopy import logging as psychopy_logging
psychopy_logging.console.setLevel(psychopy_logging.ERROR)
from psychopy import core, visual, event, monitors, prefs
from psychopy.visual.movies import MovieStim
from psychopy_tobii_infant import TobiiInfantController, MockTobiiInfantController
import tobii_research as tr
from mock_tobii_research import MockTobiiResearch
import time
import pandas as pd
import yaml
import datetime
from glob import glob
from psychopy_sounddevice import SoundDeviceSound

DIR = Path("../")
trial_types = ["sesame", "slow", "frank", "pixar"]
with open("config.yaml", 'r') as stream:
    config_data = yaml.safe_load(stream)

# Global logger - will be initialized in run_experiment
logger = None

def _configure_audio_device():
    global logger
    try:
        import sounddevice as sd
    except Exception as e:
        if logger:
            logger.warning(f"Audio setup skipped (sounddevice import failed): {e}")
        else:
            print(f"Audio setup skipped (sounddevice import failed): {e}")
        return

    # known issues with directly integrating sounddevice that we are circumventing (https://github.com/psychopy/psychopy-sounddevice/issues/5)
    # Get list of all audio devices (best-effort)
    try:
        devices = sd.query_devices()
    except Exception as e:
        if logger:
            logger.warning(f"Audio device query failed; using default audio. Error: {e}")
        else:
            print(f"Audio device query failed; using default audio. Error: {e}")
        return

    candidates = []
    for idx, device in enumerate(devices):
        device_name = device['name']
        # Check if it's an output device and matches monitor keywords
        if device['max_output_channels'] > 0:
            if any(keyword.lower() in device_name.lower() 
                    for keyword in ['hdmi', 'display', 'monitor', 'nvidia', 'amd', 'PA24']):
                candidates.append((idx, device))

    if len(candidates) > 0:
        best_idx, best_device = candidates[-1]
        # Set the default output device for sounddevice
        sd.default.device = best_idx
        prefs.hardware['audioLib'] = ['sounddevice']
        prefs.hardware['audioDevice'] = best_device['name']
        if logger:
            logger.info(f"Using monitor audio device: {best_device['name']}")
        else:
            print(f"Using monitor audio device: {best_device['name']}")
    else:
        if logger:
            logger.info("No monitor audio found, using default.")
        else:
            print("No monitor audio found, using default.")

def main():
    DIR = Path("../")
    parser = ArgumentParser(description="Movie Watching Experiment")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., S001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mock', action='store_true', help='Enable mock eye tracker mode')
    args = parser.parse_args()
    run_experiment(args.subject, args.debug, args.mock)

def trial_order(dir, debug=False):
    global logger
    blocks = {}
    for file in os.listdir(dir):
        if file.endswith('.mp4') and not file.startswith("intertrial_calibration") and not "stripped" in file:
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
        logger.info("Debug mode: Limited to 2 blocks with 2 trials each")

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
    
    logger.info(f"Generated trial order with {len(Trials)} trials across {len(block_keys)} blocks")
    return Trials

def calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event, mode="initial", skip_first_calibration=False):
    global logger, config_data
    import gc
    import sounddevice as sd
    
    max_retries = config_data[f"{mode}_validation"]["max_retries"]
    good_threshold = config_data[f"{mode}_validation"]["good_threshold"]
    bad_threshold = config_data[f"{mode}_validation"]["bad_threshold"]
    
    logger.info(f"Starting {mode} calibration routine (event: {calib_event})")
    logger.debug(f"Max retries: {max_retries}, Good threshold: {good_threshold}, Bad threshold: {bad_threshold}")
    
    i = 0
    while i < max_retries:
        # reset stimli before each validation/calibration
        if i > 0:
            logger.debug(f"Resetting stimuli for retry {i}")
            controller.targets.reset_stims()
            controller.win.flip()
            core.wait(0.1)
            event.clearEvents()
        
        if not skip_first_calibration or i > 0:
            sd.stop()
            gc.collect()
            core.wait(0.1)
            
            logger.debug(f"Creating calibration sound for iteration {i}")
            calibration_sound = SoundDeviceSound(CALIB_SOUND)
            try:
                logger.info(f"Running calibration iteration {i+1}/{max_retries}")
                controller.run_calibration(CALIPOINTS, CALISTIMS, audio=calibration_sound)
            except Exception as e:
                log_exception(logger, e, f"calibration iteration {i}")
            finally:
                try:
                    calibration_sound.stop()
                except:
                    pass
                del calibration_sound
                sd.stop()
                gc.collect()
                core.wait(0.1)

        # Cleanup before validation
        sd.stop()
        gc.collect()
        core.wait(0.1)
        
        logger.debug(f"Creating validation sound for iteration {i}")
        validation_sound = SoundDeviceSound(VALID_SOUND)
        try:
            logger.info(f"Running validation iteration {i+1}/{max_retries}")
            result = controller.run_validation(
                validation_points=CALIPOINTS,
                infant_stims=CALISTIMS,
                show_results=True,
                event=f"{calib_event}_{i+1}",
                audio=validation_sound
            )
        except Exception as e:
            log_exception(logger, e, f"validation iteration {i}")
            result = {}
        finally:
            try:
                validation_sound.stop()
            except:
                pass
            del validation_sound
            sd.stop()
            gc.collect()
            core.wait(0.2)

        # validation checking logic
        left = [result.get(f"Point_{idx}_accuracy_degrees_left", bad_threshold+.01) for idx in range(1, 6)]
        right = [result.get(f"Point_{idx}_accuracy_degrees_right", bad_threshold+.01) for idx in range(1, 6)]

        def eye_passes(targets):
            good = sum(t < good_threshold for t in targets)   
            bad  = sum(t > bad_threshold for t in targets)   
            return (good >= 4) and (bad == 0)

        avg = [(l + r) / 2 for l, r in zip(left, right)]
        avg_good = sum(t < good_threshold for t in avg)
        avg_bad  = sum(t > bad_threshold for t in avg)
        avg_ok = (avg_good >= 4) and (avg_bad == 0)

        logger.debug(f"Validation results - Avg good: {avg_good}/5, Avg bad: {avg_bad}/5")
        
        if avg_ok:
            logger.info(f"Validation passed on iteration {i+1} (average accuracy)")
            break
        else:
            left_ok  = eye_passes(left)
            right_ok = eye_passes(right)
            if left_ok or right_ok:
                logger.info(f"Validation passed on iteration {i+1} (single eye acceptable)")
                break
            else:
                logger.warning(f"Validation failed on iteration {i+1}")

        i += 1
        if i < max_retries:
            controller.display_text("Validation failed. Recalibrating...", duration=2)
    
    if i >= max_retries:
        logger.warning(f"Calibration routine completed with max retries ({max_retries})")
    
    # Final cleanup
    controller.targets.reset_stims()
    sd.stop()
    gc.collect()
    core.wait(0.1)

def check_and_resume_session(subject_dir, Sub, TIMESTAMP):
    """
    Check for existing records within the last hour and prompt user to resume.
    
    Returns:
        tuple: (filename, start_trial_index, timestamp_to_use, existing_trial_order)
    """
    global logger, config_data
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
                    logger.info(f"Found recent session file: {file}")
                    break
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse timestamp from file {file}: {e}")
                continue

    # No recent file found - start fresh
    if not recent_file:
        logger.info("No recent session found, starting fresh")
        return subject_dir / f'{Sub}_{TIMESTAMP}.csv', 0, TIMESTAMP, None

    # Recent file found - ask user
    response = input(f"Found existing record from {last_timestamp}. Use existing records? (y/n): ").strip().lower()
    
    if response != 'y':
        logger.info("User chose not to resume, starting fresh")
        return subject_dir / f'{Sub}_{TIMESTAMP}.csv', 0, TIMESTAMP, None
    
    # User wants to resume - load existing trial order and find last trial
    logger.info(f"Resuming from existing file: {recent_file}")
    
    # Load the existing trial order
    trial_order_file = subject_dir / f"{Sub}_{last_timestamp}_trial_order.csv"
    existing_trial_order = None
    
    try:
        if trial_order_file.exists():
            existing_trial_order = pd.read_csv(trial_order_file).to_dict('records')
            logger.info(f"Loaded existing trial order from {trial_order_file}")
        else:
            logger.warning(f"Trial order file not found at {trial_order_file}")
            return subject_dir / f'{Sub}_{TIMESTAMP}.csv', 0, TIMESTAMP, None
    except Exception as e:
        log_exception(logger, e, "loading trial order file")
        return subject_dir / f'{Sub}_{TIMESTAMP}.csv', 0, TIMESTAMP, None
    
    try:
        existing_data = pd.read_csv(recent_file)
        if not existing_data.empty and 'events' in existing_data.columns:
            trial_end_events = existing_data[existing_data['events'].str.startswith('Trial_End_', na=False)]
            if not trial_end_events.empty:
                last_completed = trial_end_events['events'].str.extract(r'Trial_End_(\d+)')[0].astype(int).max()
                start_trial_idx = last_completed + 1
                logger.info(f"Resuming from trial {start_trial_idx} (last completed: {last_completed})")
                return recent_file, start_trial_idx, last_timestamp, existing_trial_order
    except Exception as e:
        log_exception(logger, e, "reading existing data file")
    
    return recent_file, 0, last_timestamp, existing_trial_order

def run_experiment(Sub, debug=False, mock=False):
    global logger, DIR, config_data
    
    # Initialize logging first
    logger = setup_logging(Sub, log_dir="../data/logs", debug=True)
    logger.info("="*80)
    logger.info(f"Starting experiment for subject {Sub}")
    logger.info(f"Debug mode: {debug}, Mock mode: {mock}")
    logger.info("="*80)
    import gc
    gc.collect()
    
    _configure_audio_device()

    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    DISPSIZE = (1920, 1080)
    CALINORMP = [(-0.4, 0.4 ), (-0.4, -0.4), (0.0, 0.0), (0.4, 0.4), (0.4, -0.4)]
    CALIPOINTS = [(x * DISPSIZE[0], y * DISPSIZE[1]) for x, y in CALINORMP]
    STIM_DIR = DIR / os.path.join('stimuli')
    CALIB_DIR = STIM_DIR / 'calibration'
    CALIB_SOUND = os.path.join(CALIB_DIR, 'hothothot3.wav')
    VALID_SOUND = os.path.join(CALIB_DIR, 'upchime.wav')
    CALISTIMS = [
        f"{CALIB_DIR}/{x}" for x in os.listdir(CALIB_DIR)
        if x.endswith('.png') and not x.startswith('.')
    ]

    # Create data directory
    data_dir = Path("../data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {data_dir}")

    ###############################################################################
    # Setup Eye Tracker and Monitor
    try:
        if not mock:
            logger.info("Connecting to Tobii eye tracker...")
            eye_tracker = tr.find_all_eyetrackers()[0]
        else:
            logger.info("Using mock eye tracker")
            eye_tracker = MockTobiiResearch.find_all_eyetrackers()[0]
    except Exception as e:
        log_exception(logger, e, "finding eye tracker")
        raise

    # Get display area information
    try:
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

        logger.info(f"Screen width: {screen_width_cm} cm")
        logger.info(f"Screen height: {screen_height_cm} cm")
        logger.info(f"Viewing distance: {viewing_distance_cm} cm")
    except Exception as e:
        log_exception(logger, e, "getting display area information")
        raise

    # Create monitor
    mon = monitors.Monitor('TobiiFusion')
    mon.setWidth(screen_width_cm)
    mon.setDistance(viewing_distance_cm)
    mon.setSizePix([DISPSIZE[0], DISPSIZE[1]])
    mon.saveMon()
    
    # Initialize TobiiController
    subject_dir = data_dir / f"{Sub}"
    os.makedirs(subject_dir, exist_ok=True)

    # Check for existing session and get parameters
    filename, start_trial_idx, timestamp_to_use, existing_trial_order = check_and_resume_session(
        subject_dir, Sub, TIMESTAMP
    )

    # Create window
    try:
        logger.info("Creating PsychoPy window...")
        win = visual.Window(size=[DISPSIZE[0], DISPSIZE[1]],
                            units='pix',
                            monitor=mon,
                            screen=1,
                            fullscr=True,
                            allowGUI=False,
                            checkTiming=False)
    except Exception as e:
        log_exception(logger, e, "creating PsychoPy window")
        raise

    try:
        if not mock:
            controller = TobiiInfantController(win, calibration_disc_size=200, filename=str(filename))
        else:
            controller = MockTobiiInfantController(win, calibration_disc_size=200, filename=str(filename))
        logger.info("Controller initialized successfully")
    except Exception as e:
        log_exception(logger, e, "initializing controller")
        raise

    ###############################################################################
    # Show Status and Calibration.
    try:
        win.flip()
        core.wait(0.1)
        logger.info("Loading attention grabber...")
        grabber = MovieStim(win, f"{STIM_DIR}/ag/Attentiongrabber.mp4", size=[600, 600], units='pix')
        grabber.setAutoDraw(True)
        grabber.play()
        controller.show_status()
        controller.eyetracker.set_gaze_output_frequency(250)
        grabber.setAutoDraw(False)
        grabber.stop()
    except Exception as e:
        log_exception(logger, e, "showing status and attention grabber")
    
    try:
        calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event="pre_validation", mode="initial")
    except Exception as e:
        log_exception(logger, e, "initial calibration routine")
    
    if existing_trial_order is not None:
        Trials = existing_trial_order
        logger.info("Using existing trial order") 
    else:
        try:
            Trials = trial_order(f"{DIR}/stimuli/main_blocks", debug=debug)
            pd.DataFrame(Trials).to_csv(subject_dir / f"{Sub}_{timestamp_to_use}_trial_order.csv", index=False)
            logger.info(f"Trial order saved to {Sub}_{timestamp_to_use}_trial_order.csv")
        except Exception as e:
            log_exception(logger, e, "generating trial order")
            raise
    
    if start_trial_idx > 0:
        Trials = [t for t in Trials if t['total_trial_index'] >= start_trial_idx]
        logger.info(f"Skipping to trial {start_trial_idx}, {len(Trials)} trials remaining")
    
    try:
        controller.start_recording()
        logger.info("Eye tracking recording started")
    except Exception as e:
        log_exception(logger, e, "starting recording")
        raise
    
    first_trial = True
    for trial in Trials:
        t1 = time.time()
        trial_id = trial['total_trial_index']
        video_path = trial['video_path']
        video_name = trial['video_name']
        
        logger.info(f"Starting trial {trial_id}: {video_name} (block: {trial['block_id']})")
        
        if not first_trial:  # Skip on first trial
            try:
                controller._flush_data_csv()
            except Exception as e:
                log_exception(logger, e, f"flushing data before trial {trial_id}")
                
        if trial['within_block_trial_index'] == 0 and not first_trial:
            logger.info(f"Block validation at trial {trial_id}")
            try:
                calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event=f"block_validation_trial{trial['total_trial_index']}", mode="later", skip_first_calibration=True)
                if mock:
                    controller.eyetracker.subscribe_to(controller.tr.EYETRACKER_GAZE_DATA, controller._on_gaze_data, as_dictionary=True)
            except Exception as e:
                log_exception(logger, e, f"block validation at trial {trial_id}")

        stripped_video_path = video_path.replace('.mp4', '_stripped.mp4')
        audio_path = video_path.replace('.mp4', '.mp3')
        
        controller.record_event(f"Loop_Start_{trial_id}")
        
        try:
            logger.debug(f"Loading movie: {stripped_video_path}")
            movie = MovieStim(
                win,
                stripped_video_path,
                size=[1920, 1080],
                units='pix',
                name=video_name,
                noAudio=True
            )
            logger.debug(f"Movie loaded successfully")
        except Exception as e:
            log_exception(logger, e, f"loading movie for trial {trial_id}")
            continue
        
        try:
            import soundfile as sf
            import sounddevice as sd
            import threading
            
            audio_data, samplerate = sf.read(audio_path)
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)
            logger.debug(f"Audio loaded: {audio_path}, sample rate: {samplerate}")
        except Exception as e:
            log_exception(logger, e, f"loading audio for trial {trial_id}")
            continue
        
        class SeekableAudio:
            def __init__(self, data, samplerate):
                self.data = data
                self.samplerate = samplerate
                self.stream = None
                self.current_frame = 0
                self.lock = threading.Lock()
                self.playing = False
                
            def _callback(self, outdata, frames, time_info, status):
                with self.lock:
                    if not self.playing or self.current_frame >= len(self.data):
                        outdata.fill(0)
                        if self.current_frame >= len(self.data):
                            raise sd.CallbackStop()
                        return
                    
                    end_frame = min(self.current_frame + frames, len(self.data))
                    chunk = self.data[self.current_frame:end_frame]
                    
                    if len(chunk) < frames:
                        outdata[:len(chunk)] = chunk
                        outdata[len(chunk):].fill(0)
                    else:
                        outdata[:] = chunk
                        
                    self.current_frame = end_frame
                
            def play(self, start_time=0):
                with self.lock:
                    self.current_frame = int(start_time * self.samplerate)
                    self.playing = True
                    
                if self.stream is not None:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    
                self.stream = sd.OutputStream(
                    samplerate=self.samplerate,
                    channels=self.data.shape[1],
                    callback=self._callback
                )
                self.stream.start()
                
            def stop(self):
                with self.lock:
                    self.playing = False
                    
                if self.stream is not None:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    self.stream = None
                    
            def get_current_time(self):
                with self.lock:
                    return self.current_frame / self.samplerate
        
        audio = SeekableAudio(audio_data, samplerate)
        
        core.wait(0.1)
        
        try:
            movie.play()
            audio.play()
            logger.debug(f"Trial {trial_id} playback started")
        except Exception as e:
            log_exception(logger, e, f"starting playback for trial {trial_id}")
        
        playback_start_time = time.time()
        accumulated_pause_time = 0
        
        t2 = time.time()
        controller.record_event(f"Trial_Start_{trial_id}|Video_{video_name}")  
        event_type = "play"

        if trial['block_id'] != "pixar":
            total_time = config_data['trial_config']['max_time']
        else:
            total_time = 150
        
        END_EARLY_SEC = 0.25
        total_time = max(1.0, float(total_time) - END_EARLY_SEC)
        remaining_time = total_time
        total_lt = 0
        playback_start_time = time.time()
        accumulated_pause_time = 0
        total_lt = 0
        absent_lt = 0
        while True:
            # Calculate actual remaining time
            current_time = time.time()
            elapsed_playback = current_time - playback_start_time - accumulated_pause_time
            remaining_time = total_time - elapsed_playback
            
            if remaining_time <= 0.1:
                break
            
            movie.draw()
            win.flip()
            
            check_interval = min(0.033, remaining_time)
            try:
                lt, event_type, curr_absent_lt = controller.collect_lt_with_calibration(check_interval, config_data['trial_config']['away_time'])
            except Exception as e:
                log_exception(logger, e, f"collecting looking time for trial {trial_id}")
                break
            
            # subtracting 0.01 to allow for some noisiness/latency in window flipping
            if curr_absent_lt >= check_interval - 0.01:
                absent_lt += curr_absent_lt
                if absent_lt >= config_data['trial_config']['away_time']:
                    event_type = "looking_away"
            else:
                absent_lt = 0
            total_lt += lt        
            if event_type == "pause":
                current_playback_time = time.time()
                elapsed_playback = current_playback_time - playback_start_time - accumulated_pause_time
                paused_audio_time = audio.get_current_time()
                controller.record_event(f"Trial_{trial_id}_LookingTime_{round(elapsed_playback,3)}_Paused")
                logger.info(f"Trial {trial_id} paused at {round(elapsed_playback,3)}s")
                
                controller.eyetracker.unsubscribe_from(controller.tr.EYETRACKER_GAZE_DATA, controller._on_gaze_data)
                controller.recording = False

                movie.pause()
                audio.stop()
                
                pause_start = time.time()
                while True:
                    movie.draw()
                    win.flip()
                    keys = event.getKeys(keyList=['space'])
                    if 'space' in keys:
                        break
                    core.wait(0.01)
                pause_duration = time.time() - pause_start
                accumulated_pause_time += pause_duration
                
                movie.play()
                resume_time = paused_audio_time
                audio.play(start_time=resume_time)
                # start recording again
                controller.eyetracker.subscribe_to(controller.tr.EYETRACKER_GAZE_DATA, controller._on_gaze_data, as_dictionary=True)
                controller.recording = True
                controller.record_event(f"Trial_{trial_id}_LookingTime_{round(elapsed_playback,3)}_Resumed")
                logger.info(f"Trial {trial_id} resumed after {round(pause_duration,3)}s pause")
                
            elif event_type != "normal":
                # Trial ended for another reason (looking_away, calibration, escape, or normal)
                break
                
        total_lt = round(elapsed_playback, 3)
        # Record final event based on how trial ended
        if event_type == "calibration":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Forced_Recalibration_Key_Press")
            logger.info(f"Trial {trial_id} ended: forced recalibration (LT: {total_lt}s)")
        elif event_type == "looking_away":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Looked_Away")
            logger.info(f"Trial {trial_id} ended: looked away (LT: {total_lt}s)")
        elif event_type == "next_trial":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Ended_Trial")
            logger.info(f"Trial {trial_id} ended: next trial key (LT: {total_lt}s)")
        elif event_type == "escape":
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Ended_Experiment")
            logger.warning(f"Trial {trial_id} ended: escape pressed (LT: {total_lt}s)")
        else:
            controller.record_event(f"Trial_{trial_id}_LookingTime_{total_lt}_Normal")
            logger.info(f"Trial {trial_id} completed normally (LT: {total_lt}s)")

        controller.record_event(f"Trial_End_{trial_id}")
        
        movie.pause()
        win.flip()
        audio.stop()
        core.wait(0.1)
        controller._flush_data_csv()
        core.wait(0.1)
        movie.stop()
        del audio
        del audio_data
        core.wait(0.05)
        del movie
        gc.collect()
        core.wait(0.05)
                    
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
                    logger.info(f"User requested recalibration after looking away in trial {trial_id}")
                    try:
                        calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event=f"lb_forced_validation_{trial['total_trial_index']}", mode="later")
                    except Exception as e:
                        log_exception(logger, e, f"forced calibration after trial {trial_id}")
            elif event_type == "calibration":
                try:
                    calibration_routine(controller, CALIPOINTS, CALISTIMS, CALIB_SOUND, VALID_SOUND, calib_event=f"key_forced_validation_trial{trial['total_trial_index']}", mode="later")
                except Exception as e:
                    log_exception(logger, e, f"key-forced calibration at trial {trial_id}")
                    
        controller.record_event(f"Loop_End_{trial_id}")
        if event_type == "escape":
            logger.warning("Experiment terminated by user (escape key)")
            break
        t4 = time.time()
        
        keys = event.getKeys()
        if 'escape' in keys:
            logger.warning("Experiment terminated by user (escape key)")
            break
        first_trial = False
    
    logger.info("Stopping recording and closing experiment")
    try:
        controller.stop_recording()
        controller.close()
        win.close()
        core.quit()
    except Exception as e:
        log_exception(logger, e, "closing experiment")
    
    logger.info("="*80)
    logger.info("Experiment completed successfully")
    logger.info("="*80)

if __name__ == '__main__':
    main()

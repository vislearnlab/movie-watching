import atexit
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tobii_research as tr
from mock_tobii_research import MockTobiiResearch
from PIL import Image, ImageDraw
from psychopy import logging
logging.console.setLevel(logging.CRITICAL)
from psychopy import core, event, visual
from psychopy.tools.monitorunittools import cm2pix, deg2pix, pix2cm, pix2deg

_has_addons = True
# yapf: disable
try:
    from tobii_research_addons import (
        ScreenBasedCalibrationValidation, Point2)
    from math import ceil
except ModuleNotFoundError:
    try:
        from tobii_research_addons import (
            ScreenBasedCalibrationValidation, Point2)
        from math import ceil
    except ModuleNotFoundError:
        _has_addons = False
# yapf: enable
__version__ = "0.8.0"


class InfantStimuli:
    """Stimuli for infant-friendly calibration and validation.

    Args:
        win: psychopy.visual.Window object.
        infant_stims: list of image files.
        shuffle: whether to shuffle the presentation order of the stimuli.
            Default is True.
        *kwargs: other arguments to pass into psychopy.visual.ImageStim.

    Attributes:
        present_order: the presentation order of the stimuli.
    """
    def __init__(self, win, infant_stims, shuffle=True, *kwargs):
        self.win = win
        self.stims = dict((i, visual.ImageStim(self.win, image=stim, *kwargs))
                          for i, stim in enumerate(infant_stims))
        self.stim_size = dict(
            (i, image_stim.size) for i, image_stim in self.stims.items())
        self.present_order = [*self.stims]
        if shuffle:
            np.random.shuffle(self.present_order)

    def get_stim(self, idx):
        """Get the stimulus by presentation order.

        Args:
        idx: index of the presentation order. If it is larger than the number
            of provided image files, it will re-iterate.

        Returns:
            psychopy.visual.ImageStim
        """
        return self.stims[self.present_order[idx % len(self.present_order)]]

    def get_stim_original_size(self, idx):
        """Get the original size of the stimulus by presentation order.

        Args:
        idx: index of the presentation order. If it is larger than the number
            of provided image files, it will re-iterate.

        Returns:
            The size (width, height) of the stimulus in the stimulus units.
        """
        return self.stim_size[self.present_order[idx %
                                                 len(self.present_order)]]
    
    def reset_stims(self):
        """Reset all stimuli to their original state"""
        for stim_obj in self.stims.values():  # Use .values() to get ImageStim objects
            try:
                # Reset to original properties
                stim_obj.setOri(0)
                # Force texture reload
                if hasattr(stim_obj, '_needTextureUpdate'):
                    stim_obj._needTextureUpdate = True
                # Reset status
                if hasattr(stim_obj, 'status'):
                    stim_obj.status = 0  # NOT_STARTED
            except Exception as e:
                print(f"Warning resetting stim: {e}")


class TobiiController:
    """Tobii controller for PsychoPy.

        tobii_research are required for this module.

    Args:
        win: psychopy.visual.Window object.
        id: the id of eyetracker. Default is 0 (use the first found eye
            tracker).
        filename: the name of the data file.

    Attributes:
        shrink_speed: the shrinking speed of target in calibration.
            Default is 1.5.
        calibration_dot_size: the size of the central dot in the
            calibration target. Default is _default_calibration_dot_size
            according to the units of self.win.
        calibration_dot_color: the color of the central dot in the
            calibration target. Default is grey.
        calibration_disc_size: the size of the disc in the
            calibration target. Default is _default_calibration_disc_size
            according to the units of self.win.
        calibration_disc_color: the color of the disc in the
            calibration target. Default is deep blue.
        calibration_target_min: the minimum size of the calibration target.
            Default is 0.2.
        numkey_dict: keys used for calibration. Default is the number pad.
            If it is changed, the keys in calibration result will not
            update accordingly (my bad), be cautious!
        update_calibration: the presentation of calibration target.
            Default is auto calibration.
    """
    _default_numkey_dict = {
        "0": -1,
        "num_0": -1,
        "1": 0,
        "num_1": 0,
        "2": 1,
        "num_2": 1,
        "3": 2,
        "num_3": 2,
        "4": 3,
        "num_4": 3,
        "5": 4,
        "num_5": 4,
        "6": 5,
        "num_6": 5,
        "7": 6,
        "num_7": 6,
        "8": 7,
        "num_8": 7,
        "9": 8,
        "num_9": 8,
    }
    _default_calibration_dot_size = {
        "norm": 0.02,
        "height": 0.01,
        "pix": 10.0,
        "degFlatPos": 0.25,
        "deg": 0.25,
        "degFlat": 0.25,
        "cm": 0.25,
    }
    _default_calibration_disc_size = {
        "norm": 0.08,
        "height": 0.04,
        "pix": 40.0,
        "degFlatPos": 1.0,
        "deg": 1.0,
        "degFlat": 1.0,
        "cm": 1.0,
    }
    _shrink_speed = 1.5
    _shrink_sec = 3 / _shrink_speed
    calibration_dot_color = (0, 0, 0)
    calibration_disc_color = (-1, -1, 0)
    calibration_target_min = 0.2
    update_calibration = None
    update_validation = None
    recording = False
    datafile = None
    validation_result_buffers = None
    validation_summary_buffers = list()

    def __init__(self, win, id=0, filename="gaze_TOBII_output.txt", mock=False):
        self.eyetracker_id = id
        self.win = win
        self.filename = filename
        self.validation_filename = filename.replace(".csv", "_validation.csv")
        self.validation_summary_filename = filename.replace(".csv", "_validation_summary.csv")
        if os.path.isfile(self.filename):
            self.existing_gaze_df = pd.read_csv(self.filename)
        if os.path.isfile(self.validation_filename):
            self.existing_validation_df = pd.read_csv(self.validation_filename)
        if os.path.isfile(self.validation_summary_filename):
            self.existing_validation_summary_df = pd.read_csv(self.validation_summary_filename)
        # FIXME: self.numkey_dict is not updated accordingly
        self.numkey_dict = self._default_numkey_dict
        self.calibration_dot_size = self._default_calibration_dot_size[
            self.win.units]
        self.calibration_disc_size = self._default_calibration_disc_size[
            self.win.units]
        self.trial_start_time = None
        if not mock:
            self.tr = tr
            eyetrackers = self.tr.find_all_eyetrackers()
        else:
            self.tr = MockTobiiResearch
            eyetrackers = MockTobiiResearch.find_all_eyetrackers()

        if len(eyetrackers) == 0:
            raise RuntimeError("No Tobii eyetrackers detected.")

        try:
            self.eyetracker = eyetrackers[self.eyetracker_id]
        except IndexError:
            raise ValueError(
                "Invalid eyetracker ID {}\n({} eyetrackers found)".format(
                    self.eyetracker_id, len(eyetrackers)))

        self.calibration = self.tr.ScreenBasedCalibration(self.eyetracker)

        self.update_calibration = self._update_calibration_auto
        if _has_addons:
            self.update_validation = self._update_validation_auto
        self.gaze_data = []
        atexit.register(self.close)

    def _tobii_to_pixels(self, left_x, left_y, right_x, right_y):
        """Convert Tobii normalized position to pixel position"""
        left_x = left_x * self.win.size[0]
        left_y = self.win.size[1] - (left_y * self.win.size[1])  # Flip y-axis
        right_x = right_x * self.win.size[0]
        right_y = self.win.size[1] - (right_y * self.win.size[1]) 
        ave_x = left_x if np.isnan(right_x) else right_x if np.isnan(left_x) else (left_x + right_x) / 2.0
        ave_y = left_y if np.isnan(right_y) else right_y if np.isnan(left_y) else (left_y + right_y) / 2.0
        return { "left_x": round(left_x, 4), "left_y": round(left_y, 4), "right_x": round(right_x, 4), "right_y": round(right_y, 4), "ave_x": round(ave_x, 4), "ave_y": round(ave_y, 4) }

    def _on_gaze_data(self, gaze_data):
        """Callback function used by Tobii SDK.

        Args:
            gaze_data: gaze data provided by the eye tracker.

        Returns:
            None
        """
        # Split coordinate tuples into separate columns
        self.gaze_data.append(gaze_data)

    def _get_psychopy_pos(self, p, units=None):
        """Convert Tobii ADCS coordinates to PsychoPy coordinates.

        Args:
            p: Gaze position (x, y) in Tobii ADCS.
            units: The PsychoPy coordinate system to use.

        Returns:
            Gaze position in PsychoPy coordinate systems. For example: (0,0).
        """
        if units is None:
            units = self.win.units

        if units == "norm":
            return (2 * p[0] - 1, -2 * p[1] + 1)
        elif units == "height":
            return ((p[0] - 0.5) * (self.win.size[0] / self.win.size[1]),
                    -p[1] + 0.5)
        elif units in ["pix", "cm", "deg", "degFlat", "degFlatPos"]:
            p_pix = self._tobii2pix(p)
            if units == "pix":
                return p_pix
            elif units == "cm":
                return tuple(pix2cm(pos, self.win.monitor) for pos in p_pix)
            elif units == "deg":
                tuple(pix2deg(pos, self.win.monitor) for pos in p_pix)
            else:
                return tuple(
                    pix2deg(np.array(p_pix),
                            self.win.monitor,
                            correctFlat=True))
        else:
            raise ValueError("unit ({}) is not supported.".format(units))

    def _get_tobii_pos(self, p, units=None):
        """Convert PsychoPy coordinates to Tobii ADCS coordinates.

        Args:
            p: Gaze position (x, y) in PsychoPy coordinate systems.
            units: The PsychoPy coordinate system of p.

        Returns:
            Gaze position in Tobii ADCS. For example: (0,0).
        """
        if units is None:
            units = self.win.units

        if units == "norm":
            return (p[0] / 2 + 0.5, p[1] / -2 + 0.5)
        elif units == "height":
            return (p[0] * (self.win.size[1] / self.win.size[0]) + 0.5,
                    -p[1] + 0.5)
        elif units == "pix":
            return self._pix2tobii(p)
        elif units in ["cm", "deg", "degFlat", "degFlatPos"]:
            if units == "cm":
                p_pix = (cm2pix(p[0], self.win.monitor),
                         cm2pix(p[1], self.win.monitor))
            elif units == "deg":
                p_pix = (
                    deg2pix(p[0], self.win.monitor),
                    deg2pix(p[1], self.win.monitor),
                )
            elif units in ["degFlat", "degFlatPos"]:
                p_pix = deg2pix(np.array(p),
                                self.win.monitor,
                                correctFlat=True)
            p_pix = tuple(round(pos, 0) for pos in p_pix)
            return self._pix2tobii(p_pix)
        else:
            raise ValueError("unit ({}) is not supported".format(units))

    def _pix2tobii(self, p):
        """Convert PsychoPy pixel coordinates to Tobii ADCS.

            Called by _get_tobii_pos.

        Args:
            p: Gaze position (x, y) in pixels.

        Returns:
            Gaze position in Tobii ADCS. For example: (0,0).
        """
        return (p[0] / self.win.size[0] + 0.5, -p[1] / self.win.size[1] + 0.5)

    def _tobii2pix(self, p):
        """Convert Tobii ADCS to PsychoPy pixel coordinates.

            Called by _get_psychopy_pos.

        Args:
            p: Gaze position (x, y) in Tobii ADCS.

        Returns:
            Gaze position in PsychoPy pixels coordinate system. For example:
            (0, 0).
        """
        return (round(self.win.size[0] * (p[0] - 0.5),
                      0), round(-self.win.size[1] * (p[1] - 0.5), 0))

    def _get_psychopy_pos_from_trackbox(self, p, units=None):
        """Convert Tobii TBCS coordinates to PsychoPy coordinates.

            Called by show_status.

        Args:
            p: Gaze position (x, y) in Tobii TBCS.
            units: The PsychoPy coordinate system to use.

        Returns:
            Gaze position in PsychoPy coordinate systems. For example: (0,0).
        """
        if units is None:
            units = self.win.units

        if units == "norm":
            return (-2 * p[0] + 1, -2 * p[1] + 1)
        elif units == "height":
            return ((-p[0] + 0.5) * (self.win.size[0] / self.win.size[1]),
                    -p[1] + 0.5)
        elif units in ["pix", "cm", "deg", "degFlat", "degFlatPos"]:
            p_pix = (
                round((-p[0] + 0.5) * self.win.size[0], 0),
                round((-p[1] + 0.5) * self.win.size[1], 0),
            )
            if units == "pix":
                return p_pix
            elif units == "cm":
                return tuple(pix2cm(pos, self.win.monitor) for pos in p_pix)
            elif units == "deg":
                return tuple(pix2deg(pos, self.win.monitor) for pos in p_pix)
            else:
                return tuple(
                    pix2deg(np.array(p_pix),
                            self.win.monitor,
                            correctFlat=True))
        else:
            raise ValueError("unit ({}) is not supported.".format(units))

    def _flush_to_file(self):
        """Write data to disk.

        Args:
            None

        Returns:
            None
        """
        self.datafile.flush()  # internal buffer to RAM
        os.fsync(self.datafile.fileno())  # RAM file cache to disk

    def _collect_calibration_data(self, p):
        """Callback function used by Tobii calibration in run_calibration.

        Args:
            p: the calibration point

        Returns:
            None
        """
        self.calibration.collect_data(*self._get_tobii_pos(p))

    def _collect_validation_data(self, p):
        """Callback function used by Tobii Pro SDK addons."""
        self.validation.start_collecting_data(Point2(*self._get_tobii_pos(p)))
        # wait a bit for data collection
        while self.validation.is_collecting_data:
            core.wait(0.5, 0.0)

    def _open_datafile(self):
        """Open a file for gaze data.

        Args:
            None

        Returns:
            None
        """
        if os.path.exists(self.filename):
            self.datafile = open(self.filename, "w")
        else:
            self.datafile = open(self.filename, "x")
        _write_buffer = "Recording date:\t{}\n".format(
            datetime.now().strftime("%Y/%m/%d"))
        _write_buffer += "Recording time:\t{}\n".format(
            datetime.now().strftime("%H:%M:%S"))
        _write_buffer += "Recording resolution:\t{} x {}\n".format(
            *self.win.size)
        _write_buffer += "PsychoPy units:\t{}\n".format(self.win.units)
        if self.validation_result_buffers is not None:
            _write_buffer += "\n".join(
                    f"{k}:{v}\t"
                    for d in self.validation_result_buffers
                    for k, v in d.items()
                )
            self.validation_result_buffers = None

        self.datafile.write(_write_buffer)
        self._flush_to_file()

    def start_recording(self, filename=None, newfile=True):
        """Start recording

        Args:
            filename: the name of the data file. If None, use default name.
                Default is None.
            newfile: open a new file to save data. Default is True.

        Returns:
            None
        """
        if filename is not None:
            self.filename = filename

        if newfile:
            self._open_datafile()

        self.gaze_data = []
        self.event_data = []
        self.eyetracker.subscribe_to(self.tr.EYETRACKER_GAZE_DATA,
                                     self._on_gaze_data,
                                     as_dictionary=True)
        core.wait(1)  # wait a bit for the eye tracker to get ready
        self.recording = True
        self.t0 = self.tr.get_system_time_stamp()

    def stop_recording(self, pause=False):
        """Stop recording.

        Args:
            None

        Returns:
            None
        """
        if not self.recording:
            raise RuntimeWarning("Not recording now.")

        self.eyetracker.unsubscribe_from(self.tr.EYETRACKER_GAZE_DATA,
                                         self._on_gaze_data)
        self.recording = False
        # time correction for event data
        self.event_data = [(round((x[0] - self.t0) / 1000.0, 2), x[1])
                           for x in self.event_data]
        self._flush_data()

    def get_current_gaze_position(self):
        """Get the newest gaze position.

        Args:
            None

        Returns:
            A tuple of the newest gaze position in PsychoPy coordinate system.
            For example: (0, 0).
        """
        if not self.gaze_data:
            return (np.nan, np.nan)
        else:
            gaze_data = self.gaze_data[-1]
            lp = self._get_psychopy_pos(
                gaze_data["left_gaze_point_on_display_area"])
            rp = self._get_psychopy_pos(
                gaze_data["right_gaze_point_on_display_area"])
            if not (gaze_data["left_gaze_point_validity"]
                    or gaze_data["right_gaze_point_validity"]):  # not detected
                return (np.nan, np.nan)
            elif not gaze_data["left_gaze_point_validity"]:
                ave = rp  # use right eye
            elif not gaze_data["right_gaze_point_validity"]:
                ave = lp  # use left eye
            else:
                ave = ((lp[0] + rp[0]) / 2.0, (lp[1] + rp[1]) / 2.0)

            return tuple(round(pos, 4) for pos in ave)

    def get_current_pupil_size(self):
        """Get the newest pupil size.

        Args:
            None

        Returns:
            The newest pupil diameter (mm) reported by the eye-tracker.
            If both eyes are detected, return the average pupil size. If
            either of the eyes is detected, it will be returned.
            For example: 3.1542.
        """
        if not self.gaze_data:
            return np.nan
        else:
            gaze_data = self.gaze_data[-1]
            if not (gaze_data["left_pupil_validity"]
                    or gaze_data["right_pupil_validity"]):  # not detected
                pup = np.nan
            elif not gaze_data["left_pupil_validity"]:
                pup = gaze_data["right_pupil_diameter"]  # use right pupil
            elif not gaze_data["right_pupil_validity"]:
                pup = gaze_data["left_pupil_diameter"]  # use left pupil
            else:
                pup = ((gaze_data["left_pupil_diameter"] +
                        gaze_data["right_pupil_diameter"]) / 2.0)

            return round(pup, 4)

    def record_event(self, event):
        """Record events with timestamp.

            This method works only during recording.

        Args:
            event: the event

        Returns:
            None
        """
        if not self.recording:
            raise RuntimeWarning("Not recording now.")

        self.event_data.append([self.tr.get_system_time_stamp(), event])

    def close(self):
        """Close the data file.

        Args:
            None

        Returns:
            None
        """
        # stop recording if not already
        if self.recording:
            self.stop_recording()
        elif self.datafile is not None:
            self.datafile.close()

    def run_calibration(self,
                        calibration_points,
                        focus_time=0.5,
                        decision_key="space",
                        result_msg_color="white"):
        """Run calibration

        Args:
            calibration_points: list of position of the calibration points.
            focus_time: the duration allowing the subject to focus in seconds.
                        Default is 0.5.
            decision_key: key to leave the procedure. Default is space.
            result_msg_color: Color to be used for calibration result text.
                Accepts any PsychoPy color specification. Default is white.

        Returns:
            bool: The status of calibration. True for success, False otherwise.
        """
        if self.eyetracker is None:
            raise ValueError("Eyetracker is not found.")

        if not (2 <= len(calibration_points) <= 9):
            raise ValueError(
                "The number of calibration points must be between 2 and 9.")

        else:
            self.numkey_dict = {
                k: v
                for k, v in self.numkey_dict.items()
                if v < len(calibration_points)
            }
        # prepare calibration stimuli
        self.calibration_target_dot = visual.Circle(
            self.win,
            radius=self.calibration_dot_size,
            fillColor=self.calibration_dot_color,
            lineColor=self.calibration_dot_color,
        )
        self.calibration_target_disc = visual.Circle(
            self.win,
            radius=self.calibration_disc_size,
            fillColor=self.calibration_disc_color,
            lineColor=self.calibration_disc_color,
        )
        self.retry_marker = visual.Circle(
            self.win,
            radius=self.calibration_dot_size,
            fillColor=self.calibration_dot_color,
            lineColor=self.calibration_disc_color,
            autoLog=False,
        )
        if self.win.units == "norm":  # fix oval
            self.calibration_target_dot.setSize(
                [float(self.win.size[1]) / self.win.size[0], 1.0])
            self.calibration_target_disc.setSize(
                [float(self.win.size[1]) / self.win.size[0], 1.0])
            self.retry_marker.setSize(
                [float(self.win.size[1]) / self.win.size[0], 1.0])
        result_msg = visual.TextStim(
            self.win,
            pos=(0, -self.win.size[1] / 4),
            color=result_msg_color,
            units="pix",
            alignText="left",
            autoLog=False,
        )

        self.original_calibration_points = calibration_points[:]
        # set all points
        cp_num = len(self.original_calibration_points)
        self.retry_points = list(range(cp_num))

        in_calibration_loop = True
        event.clearEvents()

        self.calibration.enter_calibration_mode()
        while in_calibration_loop:
            self.calibration_points = [
                self.original_calibration_points[x] for x in self.retry_points
            ]

            # clear the display
            self.win.flip()
            self.update_calibration(_focus_time=focus_time)
            self.calibration_result = self.calibration.compute_and_apply()
            self.win.flip()

            result_img = self._show_calibration_result()
            result_msg.setText(
                "Accept/Retry: {k}\n"
                "Select/Deselect all points: 0\n"
                "Select/Deselect recalibration points: 1-{p} key\n"
                "Abort: esc".format(k=decision_key, p=cp_num))

            waitkey = True
            self.retry_points = []
            while waitkey:
                for key in event.getKeys():
                    if key in [decision_key, "escape"]:
                        waitkey = False
                    elif key in self.numkey_dict:
                        if self.numkey_dict[key] == -1:
                            if len(self.retry_points) == cp_num:
                                self.retry_points = []
                            else:
                                self.retry_points = list(range(cp_num))
                        else:
                            key_index = self.numkey_dict[key]
                            if key_index < cp_num:
                                if key_index in self.retry_points:
                                    self.retry_points.remove(key_index)
                                else:
                                    self.retry_points.append(key_index)

                result_img.draw()
                if len(self.retry_points) > 0:
                    for retry_p in self.retry_points:
                        self.retry_marker.setPos(
                            self.original_calibration_points[retry_p])
                        self.retry_marker.draw()

                result_msg.draw()
                self.win.flip()

            if key == decision_key:
                if len(self.retry_points) == 0:
                    retval = True
                    in_calibration_loop = False
                else:  # retry
                    for point_index in self.retry_points:
                        x, y = self._get_tobii_pos(
                            self.original_calibration_points[point_index])
                        self.calibration.discard_data(x, y)
            elif key == "escape":
                retval = False
                in_calibration_loop = False

        self.calibration.leave_calibration_mode()

        return retval

    def _validation_metrics(self,prefix, left, right, monitor):
        """Generate degrees + pixels metrics for either mean or point values."""
        return {
            f"{prefix}_degrees_left":  round(left, 4),
            f"{prefix}_degrees_right": round(right, 4),
            f"{prefix}_pixels_left":   round(deg2pix(left, monitor), 4),
            f"{prefix}_pixels_right":  round(deg2pix(right, monitor), 4),
        }

    def _process_validation_result(self, validation_result, validation_event):
        """Process validation result"""
        mon = self.win.monitor
        mean_metrics = self._validation_metrics(
            prefix="Mean_accuracy",
            left=validation_result.average_accuracy_left,
            right=validation_result.average_accuracy_right,
            monitor=mon,
        )
        result_buffer = {
            "Validation_time": datetime.now().strftime("%H:%M:%S"),
            **mean_metrics,
        }

        self.validation_summary_buffers.append({
            "validation_step": validation_event,
            "point": "mean",
            "stimulus_x": None,
            "stimulus_y": None,
            **mean_metrics,
        })

        # -------------------------
        # Per-point metrics
        # -------------------------
        for idx, (_, values) in enumerate(validation_result.points.items(), start=1):
            last = values[-1]

            point_metrics = self._validation_metrics(
                prefix=f"Point_{idx}_accuracy",
                left=last.accuracy_left_eye,
                right=last.accuracy_right_eye,
                monitor=mon,
            )
            result_buffer.update(point_metrics)

            point_mean_metrics = self._validation_metrics(
                prefix="Mean_accuracy",
                left=last.accuracy_left_eye,
                right=last.accuracy_right_eye,
                monitor=mon,
            )
            p = last.screen_point
            stimulus_positions = self._tobii_to_pixels(
                        p.x, p.y, p.x, p.y)
            self.validation_summary_buffers.append({
                "validation_step": validation_event,
                "point": f"Point {idx}",
                "stimulus_x": stimulus_positions["ave_x"],
                "stimulus_y": stimulus_positions["ave_y"],
                **point_mean_metrics,  
            })

        return result_buffer
  

    def _show_validation_result_full(self, result_buffer, show_results,
                                save_to_file, decision_key, result_msg_color, validation_event):
        if save_to_file:
            if self.validation_result_buffers is None:
                self.validation_result_buffers = list()

        if show_results:
            # Create calibration visualization image
            img = Image.new("RGBA", tuple(self.win.size))
            img_draw = ImageDraw.Draw(img)
            result_img = visual.SimpleImageStim(self.win, img, autoLog=False)
            img_draw.rectangle(((0, 0), tuple(self.win.size)), fill=(0, 0, 0, 0))
            # Draw calibration points if available
            for (key, value) in self.validation_result.points.items():
                this_point = value[-1]
                p = this_point.screen_point
                
                for this_sample in this_point.gaze_data:
                    lp = this_sample.left_eye.gaze_point.position_on_display_area
                    rp = this_sample.right_eye.gaze_point.position_on_display_area
                    pixel_positions = self._tobii_to_pixels(
                        lp[0], lp[1], rp[0], rp[1])
                    stimulus_positions = self._tobii_to_pixels(
                        p.x, p.y, p.x, p.y
                    )
                    this_sample_record = {
                        "system_time_stamp": this_sample.system_time_stamp,
                        "left_x": pixel_positions["left_x"],
                        "right_x": pixel_positions["right_x"],
                        "left_y": pixel_positions["left_y"],
                        "left_valid": pixel_positions["left_x"] != np.nan,
                        "right_valid": pixel_positions["right_x"] != np.nan,
                        "right_y": pixel_positions["right_y"],
                        "gaze_x": pixel_positions["ave_x"],
                        "gaze_y": pixel_positions["ave_y"],
                        "stimulus_x": stimulus_positions["ave_x"],
                        "stimulus_y": stimulus_positions["ave_y"],
                    }
                    self.validation_result_buffers.append({
                        "validation_step": validation_event,
                        **this_sample_record
                    })
                    img_draw.line(
                        (
                            (p.x * self.win.size[0],
                            p.y * self.win.size[1]),
                            (
                                lp[0] * self.win.size[0],
                                lp[1] * self.win.size[1],
                            ),
                        ),
                        fill=(0, 255, 0, 255),
                    )
                    
                    img_draw.line(
                        (
                            (p.x * self.win.size[0],
                            p.y * self.win.size[1]),
                            (
                                rp[0] * self.win.size[0],
                                rp[1] * self.win.size[1],
                            ),
                        ),
                        fill=(255, 0, 0, 255),
                    )
                
                img_draw.ellipse(
                    (
                        (p.x * self.win.size[0] - 3,
                        p.y * self.win.size[1] - 3),
                        (p.x * self.win.size[0] + 3,
                        p.y * self.win.size[1] + 3),
                    ),
                    outline=(0, 0, 0, 255),
                )
            
            # Update image and draw it
            result_img.setImage(img)
            result_img.draw()
            
            # Create and draw text message
            result_msg = visual.TextStim(self.win,
                                        pos=(0, -self.win.size[1] / 4),
                                        color=result_msg_color,
                                        units="pix",
                                        alignText="left",
                                        wrapWidth=self.win.size[0] * 0.6,
                                        autoLog=False)
            result_msg.setText(str(result_buffer))
            result_msg.draw()
            self.win.flip()

            waitkey = True
            while waitkey:
                for key in event.getKeys():
                    if key == decision_key:
                        waitkey = False
                        break
    
    def _show_validation_result(self, result_buffer, show_results,
                                save_to_file, decision_key, result_msg_color):
        if save_to_file:
            if self.validation_result_buffers is None:
                self.validation_result_buffers = list()
            self.validation_result_buffers.append(result_buffer)

        if show_results:
            result_msg = visual.TextStim(self.win,
                                         pos=(0, -self.win.size[1] / 4),
                                         color=result_msg_color,
                                         units="pix",
                                         alignText="left",
                                         wrapWidth=self.win.size[0] * 0.6,
                                         autoLog=False)
            result_msg.setText(str(result_buffer))
            result_msg.draw()
            self.win.flip()

            waitkey = True
            while waitkey:
                for key in event.getKeys():
                    if key == decision_key:
                        waitkey = False
                        break 
    
    def display_text(self, text, color="black", duration=5):
        result_msg = visual.TextStim(self.win,
                                        pos=(0, -self.win.size[1] / 4),
                                        color=color,
                                        units="pix",
                                        alignText="left",
                                        wrapWidth=self.win.size[0] * 0.6,
                                        autoLog=False)
        result_msg.setText(str(text))
        result_msg.draw()
        self.win.flip()
        core.wait(duration)

    def _update_validation_auto(self, validation_points, _focus_time=0.5):
        """Automatic validation procedure."""
        # start
        clock = core.Clock()
        for current_validation_point in validation_points:
            self.calibration_target_disc.setPos(current_validation_point)
            self.calibration_target_dot.setPos(current_validation_point)
            clock.reset()
            while True:
                t = clock.getTime() * self.shrink_speed
                self.calibration_target_disc.setRadius([
                    (np.sin(t)**2 + self.calibration_target_min) *
                    self.calibration_disc_size
                ])
                self.calibration_target_dot.setRadius([
                    (np.sin(t)**2 + self.calibration_target_min) *
                    self.calibration_dot_size
                ])
                self.calibration_target_disc.draw()
                self.calibration_target_dot.draw()
                if clock.getTime() >= self._shrink_sec:
                    core.wait(_focus_time, 0.0)
                    self._collect_validation_data(current_validation_point)
                    break

                self.win.flip()

    def _show_calibration_result(self):
        img = Image.new("RGBA", tuple(self.win.size))
        img_draw = ImageDraw.Draw(img)
        result_img = visual.SimpleImageStim(self.win, img, autoLog=False)
        img_draw.rectangle(((0, 0), tuple(self.win.size)), fill=(0, 0, 0, 0))
        if self.calibration_result.status == self.tr.CALIBRATION_STATUS_FAILURE:
            # computeCalibration failed.
            pass
        else:
            if len(self.calibration_result.calibration_points) == 0:
                pass
            else:

                for this_point in self.calibration_result.calibration_points:
                    p = this_point.position_on_display_area
                    for this_sample in this_point.calibration_samples:
                        lp = this_sample.left_eye.position_on_display_area
                        rp = this_sample.right_eye.position_on_display_area
                        if (this_sample.left_eye.validity ==
                                self.tr.VALIDITY_VALID_AND_USED):
                            img_draw.line(
                                (
                                    (p[0] * self.win.size[0],
                                     p[1] * self.win.size[1]),
                                    (
                                        lp[0] * self.win.size[0],
                                        lp[1] * self.win.size[1],
                                    ),
                                ),
                                fill=(0, 255, 0, 255),
                            )
                        if (this_sample.right_eye.validity ==
                                self.tr.VALIDITY_VALID_AND_USED):
                            img_draw.line(
                                (
                                    (p[0] * self.win.size[0],
                                     p[1] * self.win.size[1]),
                                    (
                                        rp[0] * self.win.size[0],
                                        rp[1] * self.win.size[1],
                                    ),
                                ),
                                fill=(255, 0, 0, 255),
                            )
                    img_draw.ellipse(
                        (
                            (p[0] * self.win.size[0] - 3,
                             p[1] * self.win.size[1] - 3),
                            (p[0] * self.win.size[0] + 3,
                             p[1] * self.win.size[1] + 3),
                        ),
                        outline=(0, 0, 0, 255),
                    )

        result_img.setImage(img)
        return result_img

    def _update_calibration_auto(self, _focus_time=0.5):
        """Automatic calibration procedure."""
        # start calibration
        event.clearEvents()
        clock = core.Clock()
        for point_idx in self.retry_points:
            this_pos = self.original_calibration_points[point_idx]
            self.calibration_target_disc.setPos(this_pos)
            self.calibration_target_dot.setPos(this_pos)
            clock.reset()
            while True:
                t = clock.getTime() * self.shrink_speed
                self.calibration_target_disc.setRadius([
                    (np.sin(t)**2 + self.calibration_target_min) *
                    self.calibration_disc_size
                ])
                self.calibration_target_dot.setRadius([
                    (np.sin(t)**2 + self.calibration_target_min) *
                    self.calibration_dot_size
                ])
                self.calibration_target_disc.draw()
                self.calibration_target_dot.draw()
                if clock.getTime() >= self._shrink_sec:
                    core.wait(_focus_time, 0.0)
                    self._collect_calibration_data(this_pos)
                    break

                self.win.flip()

    def show_status(self, decision_key="space"):
        """Showing the participant's gaze position in track box.

        Args:
            decision_key: key to leave the procedure. Default is space.

        Returns:
            None
        """
        bgrect = visual.Rect(self.win,
                             pos=(0, 0.4),
                             width=0.25,
                             height=0.2,
                             lineColor="white",
                             fillColor="black",
                             units="height",
                             autoLog=False)

        leye = visual.Circle(self.win,
                             size=0.02,
                             units="height",
                             lineColor=None,
                             fillColor="green",
                             autoLog=False)

        reye = visual.Circle(self.win,
                             size=0.02,
                             units="height",
                             lineColor=None,
                             fillColor="red",
                             autoLog=False)

        zbar = visual.Rect(self.win,
                           pos=(0, 0.28),
                           width=0.25,
                           height=0.03,
                           lineColor="green",
                           fillColor="green",
                           units="height",
                           autoLog=False)

        zc = visual.Rect(self.win,
                         pos=(0, 0.28),
                         width=0.01,
                         height=0.03,
                         lineColor="white",
                         fillColor="white",
                         units="height",
                         autoLog=False)

        zpos = visual.Rect(self.win,
                           pos=(0, 0.28),
                           width=0.005,
                           height=0.03,
                           lineColor="black",
                           fillColor="black",
                           units="height",
                           autoLog=False)

        if self.eyetracker is None:
            raise ValueError("Eyetracker is not found.")

        self.eyetracker.subscribe_to(self.tr.EYETRACKER_USER_POSITION_GUIDE,
                                     self._on_gaze_data,
                                     as_dictionary=True)
        core.wait(1)  # wait a bit for the eye tracker to get ready

        b_show_status = True

        while b_show_status:
            bgrect.draw()
            zbar.draw()
            zc.draw()
            gaze_data = self.gaze_data[-1]
            lv = gaze_data["left_user_position_validity"]
            rv = gaze_data["right_user_position_validity"]
            lx, ly, lz = gaze_data["left_user_position"]
            rx, ry, rz = gaze_data["right_user_position"]
            if lv:
                lx, ly = self._get_psychopy_pos_from_trackbox([lx, ly],
                                                              units="height")
                leye.setPos((round(lx * 0.25, 4), round(ly * 0.2 + 0.4, 4)))
                leye.draw()
            if rv:
                rx, ry = self._get_psychopy_pos_from_trackbox([rx, ry],
                                                              units="height")
                reye.setPos((round(rx * 0.25, 4), round(ry * 0.2 + 0.4, 4)))
                reye.draw()
            if lv or rv:
                zpos.setPos((
                    round((((lz * int(lv) + rz * int(rv)) /
                            (int(lv) + int(rv))) - 0.5) * 0.125, 4),
                    0.28,
                ))
                zpos.draw()

            for key in event.getKeys():
                if key == decision_key:
                    b_show_status = False
                    break

            self.win.flip()

        self.eyetracker.unsubscribe_from(self.tr.EYETRACKER_USER_POSITION_GUIDE,
                                         self._on_gaze_data)

    # property getters and setters for parameter changes
    @property
    def shrink_speed(self):
        return self._shrink_speed

    @shrink_speed.setter
    def shrink_speed(self, value):
        self._shrink_speed = value
        # adjust the duration of shrinking
        self._shrink_sec = 3 / self._shrink_speed

    @property
    def shrink_sec(self):
        return self._shrink_sec

    @shrink_sec.setter
    def shrink_sec(self, value):
        self._shrink_sec = value

class TobiiInfantController(TobiiController):
    """Tobii controller with children-friendly calibration procedure.

        This is a subclass of TobiiController, with some modification for
        developmental research.

    Args:
        win: psychopy.visual.Window object.
        id: the id of eyetracker.
        filename: the name of the data file.

    Attributes:
        shrink_speed: the shrinking speed of target in calibration.
            Default is 1.
        numkey_dict: keys used for calibration. Default is the number pad.
    """
    def __init__(self, win, id=0, filename="gaze_TOBII_output.tsv", calibration_disc_size=15, mock=False):
        super().__init__(win, id, filename, mock)
        self.update_calibration = self._update_calibration_infant_auto
        self.mock = mock
        self.shrink_speed = 1
        self.Events = []
        self.calibration_disc_size = calibration_disc_size
        if _has_addons:
            self.update_validation = self._update_validation_infant
    
    def _convert_tobii_record_csv(self, record):
        """Convert tobii record to CSV-friendly format with separate columns"""
        left_x, left_y = record['left_gaze_point_on_display_area']
        right_x, right_y = record['right_gaze_point_on_display_area']
        pixel_positions = self._tobii_to_pixels(
            left_x, left_y, right_x, right_y)
        # Calculate average pupil
        if not (record["left_pupil_validity"] or record["right_pupil_validity"]):
            pup = np.nan
        elif not record["left_pupil_validity"]:
            pup = record["right_pupil_diameter"]
        elif not record["right_pupil_validity"]:
            pup = record["left_pupil_diameter"]
        else:
            pup = (record["left_pupil_diameter"] + record["right_pupil_diameter"]) / 2.0
        return {
            'system_time_stamp': record["system_time_stamp"],
            'study_time': round((record["system_time_stamp"] - self.t0) / 1000.0, 2),
            'trial_time': np.nan,
            'left_x': round(pixel_positions["left_x"], 4),
            'left_y': round(pixel_positions["left_y"], 4),
            'left_valid': int(record["left_gaze_point_validity"]),
            'left_pupil': round(record["left_pupil_diameter"], 4),
            'left_pupil_valid': int(record["left_pupil_validity"]),
            'right_x': round(pixel_positions["right_x"], 4),
            'right_y': round(pixel_positions["right_y"], 4),
            'right_valid': int(record["right_gaze_point_validity"]),
            'right_pupil': round(record["right_pupil_diameter"], 4),
            'right_pupil_valid': int(record["right_pupil_validity"]),
            'gaze_x': round(pixel_positions["ave_x"], 4),
            'gaze_y': round(pixel_positions["ave_y"], 4),
            'pupil_size': round(pup, 4)
        }
    
    def _flush_data_csv(self):
        """Write data to CSV file, only appending new rows."""

        if not self.gaze_data:
            raise RuntimeWarning("No data were collected.")
        
        # ---------------------------
        # Gaze Data
        # ---------------------------
        data_records = [self._convert_tobii_record_csv(gd) for gd in self.gaze_data]
        data_df = pd.DataFrame(data_records)
        data_df['events'] = ''

        # Process events
        start_trial_events, end_trial_events = [], []
        if len(self.event_data) > 0:
            event_rows = []
            for evt_timestamp, evt_label in self.event_data:
                event_row = {
                    'system_time_stamp': evt_timestamp,
                    'study_time': round((evt_timestamp - self.t0) / 1000.0, 2),
                    'trial_time': np.nan,
                    'left_x': np.nan, 'left_y': np.nan, 'left_valid': 0,
                    'left_pupil': np.nan, 'left_pupil_valid': 0,
                    'right_x': np.nan, 'right_y': np.nan, 'right_valid': 0,
                    'right_pupil': np.nan, 'right_pupil_valid': 0,
                    'gaze_x': np.nan, 'gaze_y': np.nan, 'pupil_size': np.nan,
                    'events': evt_label
                }
                if evt_label.startswith("Trial_Start_"):
                    start_trial_events.append(evt_timestamp)
                if evt_label.startswith("Trial_End_"):
                    end_trial_events.append(evt_timestamp)
                event_rows.append(event_row)
            if event_rows:
                event_df = pd.DataFrame(event_rows)
                data_df = pd.concat([data_df, event_df], ignore_index=True)
                data_df = data_df.sort_values('system_time_stamp').reset_index(drop=True)

        # Compute trial_time
        for start, end in zip(start_trial_events, end_trial_events):
            mask = (data_df['system_time_stamp'] >= start) & (data_df['system_time_stamp'] <= end)
            data_df.loc[mask, 'trial_time'] = round((data_df.loc[mask, 'system_time_stamp'] - start) / 1000.0, 2)

        # Select columns
        columns_order = ['system_time_stamp', 'study_time', 'trial_time', 
                        'left_x', 'left_y', 'left_valid', 'left_pupil', 'left_pupil_valid',
                        'right_x', 'right_y', 'right_valid', 'right_pupil', 'right_pupil_valid',
                        'gaze_x', 'gaze_y', 'pupil_size', 'events']
        data_df = data_df[columns_order]

        # Write new gaze data to CSV
        # The data_df already contains only new data from self.gaze_data
        if not data_df.empty:
            data_df.to_csv(self.filename, mode='a', index=False, 
                        header=not os.path.isfile(self.filename))

        # Clear processed gaze data to free memory
        self.gaze_data = []
        self.event_data = []
        core.wait(0.1)
        # ---------------------------
        # Validation Results
        # ---------------------------
        if self.validation_result_buffers is not None and len(self.validation_result_buffers) != 0:
            validation_columns_order = ['stimulus_x', 'stimulus_y', 'validation_step', 'system_time_stamp', 'study_time',
                                        'left_x', 'left_y', 'left_valid',
                                        'right_x', 'right_y', 'right_valid',
                                        'gaze_x', 'gaze_y']
            
            val_records = []
            prev_validation_step = None
            validation_start_time = 0
            
            for val_buffer in self.validation_result_buffers:
                if val_buffer['validation_step'] != prev_validation_step:
                    validation_start_time = val_buffer['system_time_stamp']
                val_buffer['study_time'] = round((val_buffer['system_time_stamp'] - self.t0) / 1000.0, 2)
                val_buffer['validation_time'] = round((val_buffer['system_time_stamp'] - validation_start_time) / 1000.0, 2)
                val_records.append(val_buffer)
                prev_validation_step = val_buffer['validation_step']
            
            val_df = pd.DataFrame(val_records)
            val_df = val_df[validation_columns_order]

            # Write new validation data to CSV
            if not val_df.empty:
                val_df.to_csv(self.validation_filename, mode='a', index=False, 
                            header=not os.path.isfile(self.validation_filename))
            
            # Clear processed validation data to free memory
            self.validation_result_buffers = list()

        # ---------------------------
        # Validation Summary
        # ---------------------------
        if self.validation_summary_buffers is not None and len(self.validation_summary_buffers) != 0:
            validation_summary_columns_order = ['point', 'stimulus_x', 'stimulus_y', 'validation_step', 
                                            'Mean_accuracy_degrees_left', 'Mean_accuracy_degrees_right',
                                            'Mean_accuracy_pixels_left', 'Mean_accuracy_pixels_right']
            
            val_summary_df = pd.DataFrame(self.validation_summary_buffers)
            val_summary_df = val_summary_df[validation_summary_columns_order]

            # Write new validation summary to CSV
            if not val_summary_df.empty:
                val_summary_df.to_csv(self.validation_summary_filename, mode='a', index=False, 
                                    header=not os.path.isfile(self.validation_summary_filename))
            
            # Clear processed validation summary data to free memory
            self.validation_summary_buffers = list()

    def start_recording(self, filename=None, newfile=True):
        """Start recording with CSV support"""
        if filename is not None:
            self.filename = filename

        # For CSV files, we don't write headers in _open_datafile
        if newfile and not self.filename.endswith('.csv'):
            self._open_datafile()
        
        self.gaze_data = []
        self.event_data = []
        self.eyetracker.subscribe_to(self.tr.EYETRACKER_GAZE_DATA,
                                     self._on_gaze_data,
                                     as_dictionary=True)
        core.wait(1)
        self.recording = True
        self.t0 = self.tr.get_system_time_stamp()
    
    def stop_recording(self):
        """Stop recording with CSV support"""
        if not self.recording:
            return  # Silently return if not recording

        self.eyetracker.unsubscribe_from(self.tr.EYETRACKER_GAZE_DATA,
                                         self._on_gaze_data)
        self.recording = False
        
        # Time correction for event data
        self.event_data = [(x[0], x[1]) for x in self.event_data]
        
        # Choose appropriate flush method based on file extension
        if self.filename.endswith('.csv'):
            self._flush_data_csv()
        else:
            self._flush_data()
    
    def record_event(self, event):
        """Record events with timestamp"""
        if not self.recording:
            return  # Silently return if not recording
        
        self.event_data.append([self.tr.get_system_time_stamp(), event])

    def _update_calibration_infant(self,
                                   _focus_time=0.5,
                                   collect_key="space",
                                   exit_key="return"):
        """The calibration procedure designed for infants.

            An implementation of run_calibration().

        Args:
            focus_time: the duration allowing the subject to focus in seconds.
                            Default is 0.5.
            collect_key: key to start collecting samples. Default is space.
            exit_key: key to finish and leave the current calibration
                procedure. It should not be confused with `decision_key`, which
                is used to leave the whole calibration process. `exit_key` is
                used to leave the current calibration, the user may recalibrate
                or accept the result afterwards. Default is return (Enter)

        Returns:
            None
        """
        # start calibration
        event.clearEvents()
        point_idx = -1
        in_calibration = True
        clock = core.Clock()
        while in_calibration:
            # get keys
            core.wait(0.001)
            keys = event.getKeys()
            for key in keys:
                if key in self.numkey_dict:
                    point_idx = self.numkey_dict[key]

                    # play the sound if it exists
                    # only start playing sound once
                    if self._audio is not None:
                        if point_idx in self.retry_points:
                            self._audio.play()
                elif key == collect_key:
                    # allow the participant to focus
                    core.wait(_focus_time, 0.0)
                    # collect samples when space is pressed
                    if point_idx in self.retry_points:
                        self._collect_calibration_data(
                            self.original_calibration_points[point_idx])
                        point_idx = -1
                elif key == exit_key:
                    # exit calibration when return is pressed
                    in_calibration = False
                    if self._audio is not None:
                        self._audio.pause()
                    break

            # draw calibration target
            if point_idx in self.retry_points:
                this_target = self.targets.get_stim(point_idx)
                this_pos = self.original_calibration_points[point_idx]
                this_target.setPos(this_pos)
                t = clock.getTime() * self.shrink_speed
                newsize = [
                    (np.sin(t)**2 + self.calibration_target_min) * e
                    for e in self.targets.get_stim_original_size(point_idx)
                ]
                this_target.setSize(newsize)
                this_target.draw()
            self.win.flip()

    def _update_validation_infant(self,
                                validation_points,
                                _focus_time=0.5,
                                collect_key="space"):
        """Semi-automatic validation procedure for infants."""        
        for idx, current_validation_point in enumerate(validation_points):
            event.clearEvents()
            if self._audio is not None:
                self._audio.play()            
            deg = 0
            this_target = self.targets.get_stim(idx)
            orig_size = self.targets.get_stim_original_size(idx)
            this_target.setSize(
                (self.calibration_disc_size,
                self.calibration_disc_size * (orig_size[0] / orig_size[1])))
            this_target.setPos(current_validation_point)
            in_validation = True
            
            while in_validation:
                deg += 0.5
                this_target.setOri(ceil(deg))
                this_target.draw()
                self.win.flip()

                keys = event.getKeys()
                for key in keys:
                    if key == collect_key:
                        # stop audio after each point
                        if self._audio is not None:
                            self._audio.stop()
                        core.wait(_focus_time)
                        self._collect_validation_data(current_validation_point)
                        in_validation = False
                        break
        
        # Stop audio after ALL points are done
        if self._audio is not None:
            try:
                self._audio.stop()
            except Exception:
                pass

    def _update_calibration_infant_auto(self, _focus_time=0.5, collect_key="space"):
        """Semi-automatic calibration procedure.
        
        Shows each calibration point with animation, waits for space bar press
        to collect data and move to next point.
        
        Args:
            _focus_time: duration allowing the subject to focus in seconds.
                        Default is 0.5.
            collect_key: key to start collecting samples and move to next point.
                        Default is space.
        
        Returns:
            None
        """
        # start calibration and play sound once for entire calibration
        event.clearEvents()
        clock = core.Clock()
        
        # Play sound once at the beginning
        if self._audio is not None:
            self._audio.play()
        
        for point_idx in self.retry_points:
            # Set position for this calibration point
            this_pos = self.original_calibration_points[point_idx]
            this_target = self.targets.get_stim(point_idx)
            this_target.setPos(this_pos)
            
            # Reset clock for animation
            clock.reset()
            in_calibration = True
            
            # Show animated target until space is pressed
            while in_calibration:
                # Animate target
                t = clock.getTime() * self.shrink_speed
                newsize = [
                    (np.sin(t)**2 + self.calibration_target_min) * e
                    for e in self.targets.get_stim_original_size(point_idx)
                ]
                this_target.setSize(newsize)
                this_target.draw()
                self.win.flip()
                
                core.wait(0.01)
                # Check for key press
                keys = event.getKeys()
                for key in keys:
                    if key == collect_key:
                        # Collect data immediately
                        try:
                            core.wait(_focus_time)
                            self._collect_calibration_data(this_pos)
                        except Exception:
                            pass
                        # Move to next point immediately
                        in_calibration = False
                        break
        
        # Stop audio after all points are done
        if self._audio is not None:
            self._audio.stop()

    def run_calibration(self,
                        calibration_points,
                        infant_stims,
                        shuffle=True,
                        audio=None,
                        focus_time=0.5,
                        decision_key="space",
                        result_msg_color="white",
                        *kwargs):
        """Run calibration.

            How to use:
                - Press 1-9 to present calibration stimulus (press 0 to hide
                  it).
                - Press space to start collect calibration samples.
                - Press Enter to finish the calibration and show the
                  calibration result.
                - Choose the points to recalibrate with 1-9. If no points are
                  selected, the calibration result will be accepted and
                  applied.
                - Press decision_key (default is space) to accept the
                  calibration result or recalibrate.

            The experimenter should manually show the stimulus and collect data
            when the subject is paying attention to the stimulus.

        Args:
            calibration_points: list of position of the calibration points.
            infant_stims: list of images to attract the infant. If the number
                of images is equal to or larger than the number of calibration
                points, the images will be used in order. If not, the images
                will be repeated.
            shuffle: whether to shuffle the presentation order of the stimuli.
                Default is True.
            audio: the psychopy.sound.Sound object to play during calibration.
                If None, no sound will be played. Default is None.
            focus_time: the duration allowing the subject to focus in seconds.
                        Default is 0.5.
            decision_key: key to leave the procedure. Default is space.
            result_msg_color: Color to be used for calibration result text.
                Accepts any PsychoPy color specification. Default is white.
            *kwargs: other arguments to pass into psychopy.visual.ImageStim.
        Returns:
            bool: The status of calibration. True for success, False otherwise.
        """
        if self.eyetracker is None:
            raise ValueError("Eyetracker is not found.")

        if not (2 <= len(calibration_points) <= 9):
            raise ValueError("Calibration points must be between 2 and 9")

        else:
            self.numkey_dict = {
                k: v
                for k, v in self.numkey_dict.items()
                if v < len(calibration_points)
            }

        # prepare calibration stimuli
        self.targets = InfantStimuli(self.win,
                                     infant_stims,
                                     shuffle=shuffle,
                                     *kwargs)
        self._audio = audio

        self.retry_marker = visual.Circle(
            self.win,
            radius=self.calibration_dot_size,
            fillColor=self.calibration_dot_color,
            lineColor=self.calibration_disc_color,
            autoLog=False,
        )
        if self.win.units == "norm":  # fix oval
            self.retry_marker.setSize(
                [float(self.win.size[1]) / self.win.size[0], 1.0])
        result_msg = visual.TextStim(
            self.win,
            pos=(0, -self.win.size[1] / 4),
            color=result_msg_color,
            units="pix",
            autoLog=False,
        )

        self.calibration.enter_calibration_mode()

        self.original_calibration_points = calibration_points[:]
        # set all points
        cp_num = len(self.original_calibration_points)
        self.retry_points = list(range(cp_num))

        in_calibration_loop = True
        event.clearEvents()
        while in_calibration_loop:
            self.calibration_points = [
                self.original_calibration_points[x] for x in self.retry_points
            ]

            # clear the display
            self.win.flip()
            self.update_calibration(_focus_time=focus_time)
            self.calibration_result = self.calibration.compute_and_apply()
            self.win.flip()

            result_img = self._show_calibration_result()
            result_msg.setText(
                "Accept/Retry: {k}\n"
                "Select/Deselect all points: 0\n"
                "Select/Deselect recalibration points: 1-{p} key\n"
                "Abort: esc".format(k=decision_key, p=cp_num))

            waitkey = True
            self.retry_points = []
            while waitkey:
                for key in event.getKeys():
                    if key in [decision_key, "escape"]:
                        waitkey = False
                    elif key in self.numkey_dict:
                        if self.numkey_dict[key] == -1:
                            if len(self.retry_points) == cp_num:
                                self.retry_points = []
                            else:
                                self.retry_points = list(range(cp_num))
                        else:
                            key_index = self.numkey_dict[key]
                            if key_index < cp_num:
                                if key_index in self.retry_points:
                                    self.retry_points.remove(key_index)
                                else:
                                    self.retry_points.append(key_index)

                result_img.draw()
                if len(self.retry_points) > 0:
                    for retry_p in self.retry_points:
                        self.retry_marker.setPos(
                            self.original_calibration_points[retry_p])
                        self.retry_marker.draw()

                result_msg.draw()
                self.win.flip()

            if key == decision_key:
                if len(self.retry_points) == 0:
                    retval = True
                    in_calibration_loop = False
                else:  # retry
                    for point_index in self.retry_points:
                        x, y = self._get_tobii_pos(
                            self.original_calibration_points[point_index])
                        self.calibration.discard_data(x, y)
            elif key == "escape":
                retval = False
                in_calibration_loop = False

        self.calibration.leave_calibration_mode()

        return retval

    def run_validation(self,
                       validation_points=None,
                       infant_stims=None,
                       shuffle=True,
                       sample_count=30,
                       timeout=1,
                       focus_time=0.5,
                       decision_key="space",
                       show_results=False,
                       save_to_file=True,
                       result_msg_color="white",
                       event="default",
                       audio=None,
                       *kwargs):
        """Run validation.
        Press space to start collect valdiation samples.

        Args:
            validation_points: list of position of the validation points. If
                None, the calibration points are used. Default is None.
            infant_stims: list of images to attract the infant. If None,
                stimuli used in the latest calibration procedure are used.
                Default is None.
            shuffle: whether to shuffle the presentation order of the stimuli.
                Default is True. Has no effects if infant_stims is set to None.
            sample_count: The number of samples to collect. Default is 30,
                minimum 10, maximum 3000.
            timeout: Timeout in seconds. Default is 1, minimum 0.1, maximum 3.
            focus_time: the duration allowing the subject to focus in seconds.
                        Default is 0.5.
            decision_key: key to leave the procedure. Default is space.
            show_results: Whether to show the validation result. Default is
                False.
            save_to_file: Whether to save the validation result to the data
                file. Default is True.
            result_msg_color: Color to be used for calibration result text.
                Accepts any PsychoPy color specification. Default is white.
            event: the event label to mark the validation in the data file.
                Default is "default".
            *kwargs: other arguments to pass into psychopy.visual.ImageStim.
                Has no effects if infant_stims is set to None.
        Returns:
            tobii_research_addons.ScreenBasedCalibrationValidation.CalibrationValidationResult
        """
        if self.update_validation is None:
            raise ModuleNotFoundError("tobii_research_addons is not found.")
        
        if audio is not None:
            self._audio = audio

        # setup the procedure
        self.validation = ScreenBasedCalibrationValidation(
            self.eyetracker, sample_count, int(1000 * timeout), self.mock)

        if validation_points is None:
            validation_points = self.original_calibration_points

        if infant_stims is not None:
            self.targets = InfantStimuli(self.win,
                                         infant_stims,
                                         shuffle=shuffle,
                                         *kwargs)

        # clear the display
        self.win.flip()

        self.validation.enter_validation_mode()
        self.update_validation(validation_points=validation_points,
                               _focus_time=focus_time)
        self.validation_result = self.validation.compute()
        self.validation.leave_validation_mode()
        self.win.flip()

        if not (save_to_file or show_results):
            return self.validation_result
        result_buffer = self._process_validation_result(self.validation_result, event)
        self._show_validation_result_full(result_buffer, show_results, save_to_file,
                                     decision_key, result_msg_color, event)

        return result_buffer

    def write_buffer_to_file(self, gaze_data_buffer, filename):
        global Events

        # Swap buffers - get current data and start fresh
        saving_data, gaze_data_buffer = gaze_data_buffer, []
        saving_events, self.Events = self.Events, []
    
        # Convert lists to dataframes
        data_df = pd.DataFrame(saving_data)
        events_df = pd.DataFrame(saving_events)
    
        # Match events with eye tracking data
        idx = np.searchsorted(data_df['system_time_stamp'].values,
                            events_df['system_time_stamp'].values,
                            side='left')
        data_df['events'] = ''
        data_df.loc[idx, 'events'] = events_df['label'].values
        
        # Split coordinate tuples into separate columns
        data_df[['left_x', 'left_y']] = data_df['left_gaze_point_on_display_area'].tolist()
        data_df[['right_x', 'right_y']] = data_df['right_gaze_point_on_display_area'].tolist()
        
        # Convert and adjust coordinates
        data_df['time'] = data_df['system_time_stamp'] / 1000.0
        data_df['left_x_new'] = data_df['left_x'] * self.win.size[0]
        data_df['left_y_new'] = self.win.size[1] - (data_df['left_y'] * self.win.size[1])  # Flip y-axis
        data_df['right_x_new'] = data_df['right_x'] * self.win.size[0]
        data_df['right_y_new'] = self.win.size[1] - (data_df['right_y'] * self.win.size[1])  # Flip y-axis

        # Rename columns for clarity
        data_df = data_df.rename(columns={
            'left_gaze_point_validity': 'left_valid',
            'right_gaze_point_validity': 'right_valid',
            'left_pupil_diameter': 'left_pupil',
            'right_pupil_diameter': 'right_pupil',
            'left_pupil_validity': 'left_pupil_valid',
            'right_pupil_validity': 'right_pupil_valid'
        })
        
        # Keep only essential columns
        data_df = data_df[['time', 'left_x_new', 'left_y_new', 'left_valid', 'left_pupil', 'left_pupil_valid', 'right_x_new', 'right_y_new', 'right_valid', 'right_pupil', 'right_pupil_valid', 'events', 'left_x', 'left_y', 'right_x', 'right_y']]
        
        # Save to CSV
        if (os.path.exists(filename)):
            data_df.to_csv(filename, mode='a', index=False, header=not os.path.isfile(filename))
        else:
            data_df.to_csv(filename, index=False)

    # Collect looking time
    def collect_lt(self, max_time, min_away, blink_dur=1):
        """Collect looking time data in runtime.

            Collect and calculate looking time in runtime. Also end the trial
            automatically when the participant look away.

        Args:
            max_time: maximum looking time in seconds.
            min_away: minimum duration to stop in seconds.
            blink_dur: the tolerable duration of missing data in seconds.

        Returns:
            lt (float): The looking time in the trial.
        """
        trial_timer = core.Clock()
        absence_timer = core.Clock()
        away_time = []

        looking = True
        trial_timer.reset()
        absence_timer.reset()

        while trial_timer.getTime() <= max_time:
            gaze_data = self.gaze_data[-1]
            lv = gaze_data["left_gaze_point_validity"]
            rv = gaze_data["right_gaze_point_validity"]

            if any((lv, rv)):
                # if the last sample is missing
                if not looking:
                    away_dur = absence_timer.getTime()
                    if away_dur >= min_away:
                        away_time.append(away_dur)
                        lt = trial_timer.getTime() - np.sum(away_time)
                        # stop the trial
                        return round(lt, 3)
                    elif away_dur >= blink_dur:
                        away_time.append(away_dur)
                    # if missing samples are tolerable
                    else:
                        pass
                looking = True
                absence_timer.reset()
            else:
                if absence_timer.getTime() >= min_away:
                    away_dur = absence_timer.getTime()
                    away_time.append(away_dur)
                    lt = trial_timer.getTime() - np.sum(away_time)
                    # terminate the trial
                    return round(lt, 3)
                else:
                    pass
                looking = False

            self.win.flip()
        # if the loop is completed, return the looking time
        else:
            lt = max_time - np.sum(away_time)
            return round(lt, 3)
    
    def collect_lt_with_calibration(self, max_time, min_away, blink_dur=1, calibration_key='c', escape_key='escape', next_key='n', pause_key='space'):
        """
        Collect looking time but ALSO allow operator-triggered calibration
        by pressing the calibration_key at ANY time for example.

        Returns:
            (lt, status)
            lt: float
            status: "normal", "calibration", "looking_away", "pause", "escape", "next_trial"
        """
        trial_timer = core.Clock()
        absence_timer = core.Clock()
        away_time = []

        looking = True
        trial_timer.reset()
        absence_timer.reset()

        while trial_timer.getTime() <= max_time:

            # check calibration key ANY TIME during the trial
            keys = event.getKeys()
            if calibration_key in keys:
                lt = trial_timer.getTime() 
                return round(lt, 3), "calibration"
            elif escape_key in keys:
                lt = trial_timer.getTime()
                return round(lt, 3), "escape"
            elif pause_key in keys:
                lt = trial_timer.getTime() 
                return round(lt, 3), "pause"
            elif next_key in keys:
                lt = trial_timer.getTime() 
                return round(lt, 3), "next_trial"

            if not self.gaze_data:
                # No gaze data yet, wait a bit
                core.wait(0.01)
                continue
            gaze_data = self.gaze_data[-1]
            lv = gaze_data["left_gaze_point_validity"]
            rv = gaze_data["right_gaze_point_validity"]

            if any((lv, rv)):
                # valid gaze sample
                if not looking:
                    away_dur = absence_timer.getTime()
                    if away_dur >= min_away:
                        away_time.append(away_dur)
                        lt = trial_timer.getTime() 
                        return round(lt, 3), "looking_away"
                    elif away_dur >= blink_dur:
                        away_time.append(away_dur)
                looking = True
                absence_timer.reset()

            else:
                # missing gaze sample
                if absence_timer.getTime() >= min_away:
                    away_dur = absence_timer.getTime()
                    away_time.append(away_dur)
                    lt = trial_timer.getTime() 
                    return round(lt, 3), "looking_away"
                looking = False

            # REMOVED: self.win.flip()
            # Let the caller handle flipping so movie can be drawn
            core.wait(0.001)  # Small wait to prevent CPU spinning

        # Trial ended normally by time limit
        lt = trial_timer.getTime()  
        return round(lt, 3), "normal"

class MockTobiiInfantController(TobiiInfantController):
    """Mock infant controller that mimics the real TobiiInfantController API"""
    def __init__(self, win, id=0, filename="mock_gaze_output.csv", calibration_disc_size=200):
         # Now call parent __init__ which will use our MockTobiiResearch
        super().__init__(win, id, filename, calibration_disc_size, mock=True)

# backward compatible
tobii_controller = TobiiController
tobii_infant_controller = TobiiInfantController
tobii_mock_infant_controller = MockTobiiInfantController

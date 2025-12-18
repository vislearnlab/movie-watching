import threading
import time
import numpy as np

# Mock Tobii Research module
class MockTobiiResearch:
    """Mock tobii_research module for testing without hardware"""
    EYETRACKER_GAZE_DATA = "gaze_data"
    EYETRACKER_USER_POSITION_GUIDE = "user_position"
    CALIBRATION_STATUS_FAILURE = 0
    CALIBRATION_STATUS_SUCCESS = 1
    VALIDITY_VALID_AND_USED = 1
    
    class MockEyeTracker:
        def __init__(self):
            self.callbacks = {}
            self.frequency = 250
            self._generate_gaze = False
            self._gaze_thread = None
            self._stop_thread = False
            
        def get_display_area(self):
            """Return mock display area for 14-inch MacBook Pro (M3)"""
            class DisplayArea:
                # Physical display size (approximate, in millimeters)
                width = 302   # mm
                height = 196  # mm

                # Display plane positioned 600 mm in front of origin
                top_left = (0, 0, 600)
                top_right = (302, 0, 600)
                bottom_left = (0, 196, 600)
                bottom_right = (302, 196, 600)

            return DisplayArea()
        
        def _generate_mock_gaze_data(self):
            """Generate realistic mock gaze data in a background thread"""
            # Center of screen with some drift
            center_x, center_y = 0.5, 0.5
            drift_speed = 0.001
            
            while not self._stop_thread and self._generate_gaze:
                # Add random drift to simulate natural eye movement
                center_x += np.random.normal(0, drift_speed)
                center_y += np.random.normal(0, drift_speed)
                
                # Keep within bounds
                center_x = np.clip(center_x, 0.1, 0.9)
                center_y = np.clip(center_y, 0.1, 0.9)
                
                # Add noise for each eye
                left_x = center_x + np.random.normal(0, 0.01)
                left_y = center_y + np.random.normal(0, 0.01)
                right_x = center_x + np.random.normal(0, 0.01)
                right_y = center_y + np.random.normal(0, 0.01)
                
                # Clip to valid range
                left_x = np.clip(left_x, 0, 1)
                left_y = np.clip(left_y, 0, 1)
                right_x = np.clip(right_x, 0, 1)
                right_y = np.clip(right_y, 0, 1)
                
                # Simulate occasional data loss (10% of the time)
                left_valid = np.random.random() > 0.1
                right_valid = np.random.random() > 0.1
                
                # Convert normalized coordinates to user coordinates (mm from display center)
                display = self.get_display_area()
                
                # Calculate 3D position on screen for gaze points
                left_gaze_3d = (
                    left_x * display.width - display.width/2,
                    left_y * display.height - display.height/2,
                    600  # Display is 600mm away
                )
                right_gaze_3d = (
                    right_x * display.width - display.width/2,
                    right_y * display.height - display.height/2,
                    600
                )
                
                # Eye origins (approximate positions, in mm from center of trackbox)
                left_origin = (-30, 0, -50)  # Left eye, 30mm left, 50mm closer than screen
                right_origin = (30, 0, -50)  # Right eye, 30mm right
                
                # Create mock gaze data with proper object structure that also supports dict access
                class GazeData:
                    def __init__(self):
                        self.system_time_stamp = int(time.time() * 1000000)
                        self.device_time_stamp = int(time.time() * 1000000)
                        
                        # Left eye data
                        self.left_eye = type('obj', (object,), {
                            'gaze_point': type('obj', (object,), {
                                'position_on_display_area': (left_x, left_y),
                                'position_in_user_coordinates': left_gaze_3d,
                                'validity': left_valid
                            })(),
                            'gaze_origin': type('obj', (object,), {
                                'position_in_user_coordinates': left_origin,
                                'position_in_track_box_coordinates': (0.3, 0.5, 0.5),
                                'validity': left_valid
                            })(),
                            'pupil': type('obj', (object,), {
                                'diameter': np.random.normal(3.5, 0.3) if left_valid else np.nan,
                                'validity': left_valid
                            })()
                        })()
                        
                        # Right eye data
                        self.right_eye = type('obj', (object,), {
                            'gaze_point': type('obj', (object,), {
                                'position_on_display_area': (right_x, right_y),
                                'position_in_user_coordinates': right_gaze_3d,
                                'validity': right_valid
                            })(),
                            'gaze_origin': type('obj', (object,), {
                                'position_in_user_coordinates': right_origin,
                                'position_in_track_box_coordinates': (0.7, 0.5, 0.5),
                                'validity': right_valid
                            })(),
                            'pupil': type('obj', (object,), {
                                'diameter': np.random.normal(3.5, 0.3) if right_valid else np.nan,
                                'validity': right_valid
                            })()
                        })()
                    
                    def __getitem__(self, key):
                        """Support dictionary-style access for backward compatibility"""
                        # Map dictionary keys to object attributes
                        key_map = {
                            'system_time_stamp': self.system_time_stamp,
                            'device_time_stamp': self.device_time_stamp,
                            'left_gaze_point_on_display_area': self.left_eye.gaze_point.position_on_display_area,
                            'left_gaze_point_validity': self.left_eye.gaze_point.validity,
                            'left_gaze_point_in_user_coordinate_system': self.left_eye.gaze_point.position_in_user_coordinates,
                            'left_gaze_origin_in_user_coordinate_system': self.left_eye.gaze_origin.position_in_user_coordinates,
                            'left_gaze_origin_in_trackbox_coordinate_system': self.left_eye.gaze_origin.position_in_track_box_coordinates,
                            'left_gaze_origin_validity': self.left_eye.gaze_origin.validity,
                            'left_pupil_diameter': self.left_eye.pupil.diameter,
                            'left_pupil_validity': self.left_eye.pupil.validity,
                            'right_gaze_point_on_display_area': self.right_eye.gaze_point.position_on_display_area,
                            'right_gaze_point_validity': self.right_eye.gaze_point.validity,
                            'right_gaze_point_in_user_coordinate_system': self.right_eye.gaze_point.position_in_user_coordinates,
                            'right_gaze_origin_in_user_coordinate_system': self.right_eye.gaze_origin.position_in_user_coordinates,
                            'right_gaze_origin_in_trackbox_coordinate_system': self.right_eye.gaze_origin.position_in_track_box_coordinates,
                            'right_gaze_origin_validity': self.right_eye.gaze_origin.validity,
                            'right_pupil_diameter': self.right_eye.pupil.diameter,
                            'right_pupil_validity': self.right_eye.pupil.validity,
                        }
                        if key in key_map:
                            return key_map[key]
                        raise KeyError(f"Key '{key}' not found in GazeData")
                
                gaze_data = GazeData()
                
                # Call the callback if registered
                if MockTobiiResearch.EYETRACKER_GAZE_DATA in self.callbacks:
                    callback = self.callbacks[MockTobiiResearch.EYETRACKER_GAZE_DATA]
                    callback(gaze_data)
                
                # Sleep to match frequency (Hz)
                time.sleep(1.0 / self.frequency)
        
        def _generate_mock_position_data(self):
            """Generate mock user position data for show_status"""
            while not self._stop_thread:
                if MockTobiiResearch.EYETRACKER_USER_POSITION_GUIDE in self.callbacks:
                    # Simulate user position in trackbox
                    position_data = {
                        'left_user_position': (0.45, 0.45, 0.5),  # (x, y, z) in trackbox
                        'left_user_position_validity': True,
                        'right_user_position': (0.55, 0.45, 0.5),
                        'right_user_position_validity': True,
                    }
                    callback = self.callbacks[MockTobiiResearch.EYETRACKER_USER_POSITION_GUIDE]
                    callback(position_data)
                
                time.sleep(0.033)  # ~30 Hz
        
        def subscribe_to(self, data_type, callback, as_dictionary=True):
            """Mock subscription with background thread for data generation"""
            self.callbacks[data_type] = callback
            if data_type == MockTobiiResearch.EYETRACKER_GAZE_DATA:
                self._generate_gaze = True
                self._stop_thread = False
                if self._gaze_thread is None or not self._gaze_thread.is_alive():
                    self._gaze_thread = threading.Thread(
                        target=self._generate_mock_gaze_data,
                        daemon=True
                    )
                    self._gaze_thread.start()
            elif data_type == MockTobiiResearch.EYETRACKER_USER_POSITION_GUIDE:
                self._stop_thread = False
                position_thread = threading.Thread(
                    target=self._generate_mock_position_data,
                    daemon=True
                )
                position_thread.start()
                
        def unsubscribe_from(self, data_type, callback):
            """Mock unsubscription"""
            if data_type in self.callbacks:
                del self.callbacks[data_type]
            if data_type == MockTobiiResearch.EYETRACKER_GAZE_DATA:
                self._generate_gaze = False
                self._stop_thread = True
                
        def set_gaze_output_frequency(self, freq):
            """Mock frequency setting"""
            self.frequency = freq
    
    @staticmethod
    def find_all_eyetrackers():
        """Return a mock eye tracker"""
        return [MockTobiiResearch.MockEyeTracker()]
    
    @staticmethod
    def get_system_time_stamp():
        """Return current timestamp in microseconds"""
        return int(time.time() * 1000000)
    
    class ScreenBasedCalibration:
        def __init__(self, eyetracker):
            self.eyetracker = eyetracker
            self.calibration_points = []

        class CalibPoint:
            """Mock calibration point with samples"""
            def __init__(self, x, y):
                self.position_on_display_area = (x, y)
                self.calibration_samples = []
                
                # Add mock samples for visualization
                for _ in range(3):
                    sample = type('obj', (object,), {
                        'left_eye': type('obj', (object,), {
                            'position_on_display_area': (
                                x + np.random.normal(0, 0.02),
                                y + np.random.normal(0, 0.02)
                            ),
                            'validity': 1  # VALIDITY_VALID_AND_USED
                        })(),
                        'right_eye': type('obj', (object,), {
                            'position_on_display_area': (
                                x + np.random.normal(0, 0.02),
                                y + np.random.normal(0, 0.02)
                            ),
                            'validity': 1  # VALIDITY_VALID_AND_USED
                        })()
                    })()
                    self.calibration_samples.append(sample)
        
        class CalibrationResult:
            """Mock calibration result - always succeeds"""
            def __init__(self, points):
                self.status = 1  # CALIBRATION_STATUS_SUCCESS
                self.calibration_points = [
                    MockTobiiResearch.ScreenBasedCalibration.CalibPoint(x, y) 
                    for x, y in points
                ]
            
        def enter_calibration_mode(self):
            pass
            
        def leave_calibration_mode(self):
            pass
            
        def collect_data(self, x, y):
            """Mock collect calibration data"""
            self.calibration_points.append((x, y))
            
        def discard_data(self, x, y):
            """Mock discard data"""
            self.calibration_points = [(px, py) for px, py in self.calibration_points 
                                      if not (abs(px - x) < 0.01 and abs(py - y) < 0.01)]
            
        def compute_and_apply(self):
            """Mock compute and apply calibration"""
            return MockTobiiResearch.ScreenBasedCalibration.CalibrationResult(self.calibration_points)
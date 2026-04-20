import cv2
import numpy as np
import json
import os
import time
from collections import deque

class BallTracker:
    def __init__(self, settings_file="hsv_settings.json"):
        # --- CONFIGURATION ---
        self.BUFFER_SIZE = 32
        self.MIN_RADIUS = 6
        self.MAX_DISTANCE = 50
        self.MAX_DISAPPEARED = 20
        self.WARMUP_FRAMES = 60
        self.SETTINGS_FILE = settings_file

        # --- STATE VARIABLES ---
        self.trails = []
        self.next_object_id = 0
        self.frame_count = 0
        
        # FPS Calculation
        self.prev_frame_time = 0
        self.current_fps = 0

        # Processing Tools
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Load Settings
        self.settings = self._load_settings()
        
        # State for display
        self.current_frame = None
        self.current_mask = None
        self.sliders_window_created = False

    def _load_settings(self):
        default = {
            "h_min": 0, "h_max": 30,
            "s_min": 100, "s_max": 255,
            "v_min": 100, "v_max": 255,
            "use_motion": 1
        }
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return default

    def _save_settings(self):
        with open(self.SETTINGS_FILE, 'w') as f:
            json.dump(self.settings, f)
        print(f"Settings saved to {self.SETTINGS_FILE}")

    def _nothing(self, x):
        pass

    def _ensure_sliders_window(self):
        if not self.sliders_window_created:
            cv2.namedWindow("Settings")
            cv2.resizeWindow("Settings", 300, 350)
            cv2.createTrackbar("Hue Min", "Settings", self.settings["h_min"], 179, self._nothing)
            cv2.createTrackbar("Hue Max", "Settings", self.settings["h_max"], 179, self._nothing)
            cv2.createTrackbar("Sat Min", "Settings", self.settings["s_min"], 255, self._nothing)
            cv2.createTrackbar("Sat Max", "Settings", self.settings["s_max"], 255, self._nothing)
            cv2.createTrackbar("Val Min", "Settings", self.settings["v_min"], 255, self._nothing)
            cv2.createTrackbar("Val Max", "Settings", self.settings["v_max"], 255, self._nothing)
            cv2.createTrackbar("Use Motion", "Settings", self.settings["use_motion"], 1, self._nothing)
            self.sliders_window_created = True

    def _update_settings_from_sliders(self):
        self.settings["h_min"] = cv2.getTrackbarPos("Hue Min", "Settings")
        self.settings["h_max"] = cv2.getTrackbarPos("Hue Max", "Settings")
        self.settings["s_min"] = cv2.getTrackbarPos("Sat Min", "Settings")
        self.settings["s_max"] = cv2.getTrackbarPos("Sat Max", "Settings")
        self.settings["v_min"] = cv2.getTrackbarPos("Val Min", "Settings")
        self.settings["v_max"] = cv2.getTrackbarPos("Val Max", "Settings")
        self.settings["use_motion"] = cv2.getTrackbarPos("Use Motion", "Settings")

    def update(self, frame, img_format='bgr'):
        """
        Process a single frame.
        Args:
            frame: The image array (numpy array).
            img_format (str): 'bgr' (default) or 'rgb'.
        """
        if frame is None:
            return

        # Handle Format Conversion
        if img_format.lower() == 'rgb':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.frame_count += 1
        
        # FPS Calc
        new_frame_time = time.time()
        if self.prev_frame_time != 0:
            diff = new_frame_time - self.prev_frame_time
            if diff > 0:
                self.current_fps = 1 / diff
        self.prev_frame_time = new_frame_time

        # Resize and Store
        frame = cv2.resize(frame, (640, 480))
        self.current_frame = frame 
        
        # Warmup Check
        if self.frame_count < self.WARMUP_FRAMES:
            return

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Get settings
        h_min, h_max = self.settings["h_min"], self.settings["h_max"]
        s_min, s_max = self.settings["s_min"], self.settings["s_max"]
        v_min, v_max = self.settings["v_min"], self.settings["v_max"]
        
        # Masking
        color_mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        
        if self.settings["use_motion"]:
            motion_mask = self.fgbg.apply(blurred)
            final_mask = cv2.bitwise_and(color_mask, color_mask, mask=motion_mask)
        else:
            final_mask = color_mask

        final_mask = cv2.erode(final_mask, None, iterations=2)
        final_mask = cv2.dilate(final_mask, None, iterations=2)
        self.current_mask = final_mask

        # Detection
        cnts, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centers = []
        
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > self.MIN_RADIUS:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    current_centers.append(center)

        # Tracking (Distance Matching)
        used_centers = set()
        
        for t_idx, trail in enumerate(self.trails):
            if len(trail['pts']) == 0:
                continue
            
            last_pos = trail['pts'][0]
            best_dist = self.MAX_DISTANCE
            best_c_idx = -1

            for c_idx, center in enumerate(current_centers):
                if c_idx in used_centers:
                    continue
                dist = np.linalg.norm(np.array(last_pos) - np.array(center))
                if dist < best_dist:
                    best_dist = dist
                    best_c_idx = c_idx

            if best_c_idx != -1:
                trail['pts'].appendleft(current_centers[best_c_idx])
                trail['disappeared'] = 0
                used_centers.add(best_c_idx)
            else:
                trail['disappeared'] += 1

        # Create new trails
        for c_idx, center in enumerate(current_centers):
            if c_idx not in used_centers:
                new_trail = {
                    'id': self.next_object_id, 
                    'pts': deque(maxlen=self.BUFFER_SIZE),
                    'disappeared': 0
                }
                new_trail['pts'].appendleft(center)
                self.trails.append(new_trail)
                self.next_object_id += 1

        # Cleanup
        self.trails = [t for t in self.trails if t['disappeared'] < self.MAX_DISAPPEARED]

    def get_trails(self):
        """Returns the list of currently active trails."""
        return self.trails

    def show_feed(self, debug=False):
        """
        Displays the tracked feed.
        Args:
            debug (bool): If True, shows Mask view and Settings window.
        Returns:
            bool: False if user pressed 'q', True otherwise.
        """
        if self.current_frame is None:
            return True

        display_frame = self.current_frame.copy()

        # Draw Warmup
        if self.frame_count < self.WARMUP_FRAMES:
            cv2.putText(display_frame, f"Warming Up: {int(self.frame_count/self.WARMUP_FRAMES*100)}%", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Draw Trails
            for trail in self.trails:
                pts = trail['pts']
                if len(pts) < 2: continue
                
                np.random.seed(trail['id'])
                color = np.random.randint(0, 255, 3).tolist()
                
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None: continue
                    thickness = int(np.sqrt(self.BUFFER_SIZE / float(i + 1)) * 2.0)
                    cv2.line(display_frame, pts[i - 1], pts[i], color, thickness)
            
            # Draw FPS
            cv2.putText(display_frame, f"FPS: {int(self.current_fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Debug Mode Logic
        if debug:
            self._ensure_sliders_window()
            self._update_settings_from_sliders()
            
            if self.current_mask is not None:
                mask_bgr = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR)
                display_frame = np.hstack([display_frame, mask_bgr])
            
            cv2.imshow("Tracker (Debug Mode)", display_frame)
        else:
            # Cleanup debug windows if switching from True -> False
            if self.sliders_window_created:
                cv2.destroyWindow("Settings")
                cv2.destroyWindow("Tracker (Debug Mode)")
                self.sliders_window_created = False
            
            cv2.imshow("Tracker", display_frame)

        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            self._save_settings()

    def release(self):
        cv2.destroyAllWindows()
import cv2
import numpy as np
import json
import os
import time
from collections import deque

class BallTracker:
    def __init__(self, settings_file="hsv_settings.json", calibration_file="calibration_params.json"):
        # --- CONFIGURATION ---
        self.BUFFER_SIZE = 32
        self.MIN_RADIUS = 6
        self.MAX_DISTANCE = 50
        self.MAX_DISAPPEARED = 20
        self.WARMUP_FRAMES = 60
        
        # --- PATH RESOLUTION ---
        # Ensures files are read relative to this script, critical for ROS 2
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.SETTINGS_FILE = os.path.join(package_dir, settings_file)
        self.CALIBRATION_FILE = os.path.join(package_dir, calibration_file)

        print(f"[Tracker] Loading settings from: {self.SETTINGS_FILE}")
        print(f"[Tracker] Loading calibration from: {self.CALIBRATION_FILE}")

        # --- STATE VARIABLES ---
        self.trails = []       # List of dictionaries for ball trails
        self.objects = {}      # Dictionary for ArUco markers {id: (x,y)}
        self.next_object_id = 0
        self.frame_count = 0
        self.prev_frame_time = 0
        self.current_fps = 0

        # Processing Tools
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.settings = self._load_settings()
        
        # ArUco Setup (DICT_4x4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Perspective Warping State
        self.perspective_marker_id = -1
        self.marker_real_size = 0.0
        self.pixels_per_meter = 0
        self.perspective_matrix = None
        self.warped_size = (640, 480) 
        
        # Calibration State
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_calibration() 

        # Display State
        self.raw_frame = None      
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
        print(f"[Tracker] Settings saved to {self.SETTINGS_FILE}")
    
    def _load_calibration(self):
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                with open(self.CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                    self.camera_matrix = np.array(data["camera_matrix"])
                    self.dist_coeffs = np.array(data["dist_coeffs"])
                    print("[Tracker] Camera calibration loaded successfully.")
            except Exception as e:
                print(f"[Tracker] Failed to load calibration: {e}")
        else:
            print("[Tracker] No calibration file found. Running with raw (distorted) images.")

    def _save_calibration(self):
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist()
        }
        with open(self.CALIBRATION_FILE, 'w') as f:
            json.dump(data, f)
        print(f"[Tracker] Calibration parameters saved to {self.CALIBRATION_FILE}")

    def calibrate_camera(self, cap_source, rows, cols, marker_id=None, marker_size_m=None):
        """
        Runs interactive calibration.
        1. Accumulates checkerboard corners from raw frames.
        2. Calculates Distortion Matrix.
        3. Undistorts a reference frame to calculate accurate pixels_per_meter.
        """
        print("--- STARTING CALIBRATION ---")
        print(f"Please hold a {rows}x{cols} checkerboard.")
        if marker_id is not None:
            print(f"For scale, ensure ArUco ID {marker_id} is visible in at least one capture.")
        print("Press 'c' to capture. Press 'q' to calculate.")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

        objpoints, imgpoints = [], []
        
        # We need to save a frame that DEFINITELY has the marker to measure scale LATER
        frame_with_marker = None 
        
        last_valid_frame = None 
        captured_count = 0

        while True:
            ret, frame = cap_source.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()
            
            # Detect Checkerboard
            ret_corners, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

            # Visual feedback for ArUco (Just so user knows it's visible)
            marker_visible = False
            if marker_id is not None:
                acorners, aids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                if aids is not None and marker_id in aids:
                    marker_visible = True
                    cv2.aruco.drawDetectedMarkers(display, acorners, aids)

            # Draw Instructions
            if ret_corners:
                cv2.drawChessboardCorners(display, (rows, cols), corners, ret_corners)
                msg = "Press 'c' to Capture"
                if marker_id and not marker_visible:
                    msg += " (Marker NOT Visible!)"
                cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.putText(display, f"Captured: {captured_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            if frame_with_marker is not None:
                cv2.putText(display, "Scale Ref Frame: OK", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Camera Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and ret_corners:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # If this frame has the marker, save it for the post-calibration step
                if marker_visible:
                    frame_with_marker = frame.copy()
                    print(" * Frame captured (Contains ArUco for scale)")
                else:
                    print(" * Frame captured (No ArUco)")

                last_valid_frame = frame.copy()
                captured_count += 1
                cv2.imshow("Camera Calibration", 255 - display)
                cv2.waitKey(50)
            elif key == ord('q'): break

        cv2.destroyWindow("Camera Calibration")

        if captured_count > 0:
            print("Calculating distortion matrix...")
            h, w = last_valid_frame.shape[:2]
            ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
            
            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                print("Distortion calculated successfully.")

                # --- NEW: POST-CALIBRATION SCALE CALCULATION ---
                if frame_with_marker is not None and marker_size_m is not None:
                    print("Calculating scale on UNDISTORTED frame...")
                    
                    # 1. Undistort the reference frame
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                    undistorted_frame = cv2.undistort(frame_with_marker, mtx, dist, None, newcameramtx)
                    
                    # 2. Detect Marker on the FLAT image
                    ugray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
                    u_corners, u_ids, _ = cv2.aruco.detectMarkers(ugray, self.aruco_dict, parameters=self.aruco_params)
                    
                    if u_ids is not None and marker_id in u_ids:
                        idx = np.where(u_ids == marker_id)[0][0]
                        c = u_corners[idx][0] # Corners: TL, TR, BR, BL
                        
                        # Calculate perimeter or side length in pixels
                        perimeter = cv2.arcLength(c, True)
                        side_px = perimeter / 4.0
                        
                        self.pixels_per_meter = side_px / marker_size_m
                        print(f"FINAL SCALE: {self.pixels_per_meter:.2f} px/m (calculated on undistorted image)")
                        
                        # Show the user the frame we used
                        cv2.polylines(undistorted_frame, [c.astype(int)], True, (0, 255, 0), 2)
                        cv2.imshow("Undistorted Reference", cv2.resize(undistorted_frame, (0,0), fx=0.5, fy=0.5))
                        cv2.waitKey(2000)
                        cv2.destroyWindow("Undistorted Reference")
                    else:
                        print("Error: Could not redetect marker after undistortion.")
                else:
                    if marker_id:
                        print("Warning: No captured frames contained the ArUco marker. Scale not updated.")

                self._save_calibration()
            else:
                print("Calibration failed.")
        else:
            print("Calibration cancelled.")

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

    def set_perspective_marker(self, marker_id, marker_real_width_meters, pixels_per_meter=500):
        """
        Configures the anchor marker for automatic table flattening.
        """
        self.perspective_marker_id = marker_id
        self.marker_real_size = marker_real_width_meters
        self.pixels_per_meter = pixels_per_meter
        self.perspective_matrix = None # Reset matrix
        print(f"[Tracker] Perspective Enabled: Anchor ID={marker_id}, {pixels_per_meter} px/m")

    def update(self, frame, img_format='bgr'):
        """
        Main processing pipeline:
        Raw -> Undistort -> Perspective Warp -> ArUco Detect -> Ball Track
        """
        if frame is None: return

        if img_format.lower() == 'rgb':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 1. Capture Raw (for Debug View)
        self.raw_frame = cv2.resize(frame.copy(), (640, 480))

        # 2. Undistort (Lens Correction)
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        # 3. Perspective Warping (Planar Transformation)
        if self.perspective_marker_id != -1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            # Recalculate matrix ONLY if we find the marker (otherwise use cached)
            if ids is not None and self.perspective_marker_id in ids:
                index = np.where(ids == self.perspective_marker_id)[0][0]
                marker_corners = corners[index][0] 

                # Auto-Sizing Logic
                px_size = self.marker_real_size * self.pixels_per_meter
                dst_marker = np.array([
                    [0, 0], [px_size, 0], [px_size, px_size], [0, px_size]
                ], dtype="float32")

                H = cv2.getPerspectiveTransform(marker_corners, dst_marker)

                # Find bounding box of transformed image
                h_img, w_img = frame.shape[:2]
                img_corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype="float32").reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(img_corners, H)
                
                [x_min, y_min] = transformed_corners.min(axis=0).min(axis=0)
                [x_max, y_max] = transformed_corners.max(axis=0).max(axis=0)

                # Shift to origin
                translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
                self.perspective_matrix = translation_matrix.dot(H)
                
                # Set output size
                new_w = int(x_max - x_min)
                new_h = int(y_max - y_min)
                
                # Safety Cap (prevent infinite size on horizon views)
                MAX_DIM = 2000 
                if new_w > MAX_DIM or new_h > MAX_DIM:
                    scale_factor = MAX_DIM / max(new_w, new_h)
                    new_w = int(new_w * scale_factor)
                    new_h = int(new_h * scale_factor)
                    scale_m = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
                    self.perspective_matrix = scale_m.dot(self.perspective_matrix)

                self.warped_size = (new_w, new_h)

            # Apply Warping
            if self.perspective_matrix is not None:
                frame = cv2.warpPerspective(frame, self.perspective_matrix, self.warped_size)
        
        # --- Frame is now "Perfect" (Undistorted + Top-Down) ---
        
        self.frame_count += 1
        new_frame_time = time.time()
        if self.prev_frame_time != 0:
            diff = new_frame_time - self.prev_frame_time
            if diff > 0: self.current_fps = 1 / diff
        self.prev_frame_time = new_frame_time

        # Resize if not warping (to keep speed up)
        if self.perspective_marker_id == -1:
            frame = cv2.resize(frame, (640, 480))
        
        self.current_frame = frame 
        
        if self.frame_count < self.WARMUP_FRAMES:
            return

        # 4. ArUco Detection (On Warped Image - for Object Tracking)
        self.objects = {} 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            ids = ids.flatten()
            for (marker_corners, marker_id) in zip(corners, ids):
                c = marker_corners[0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                self.objects[int(marker_id)] = (center_x, center_y)

        # 5. Ball Tracking
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        h_min, h_max = self.settings["h_min"], self.settings["h_max"]
        s_min, s_max = self.settings["s_min"], self.settings["s_max"]
        v_min, v_max = self.settings["v_min"], self.settings["v_max"]
        
        color_mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        
        if self.settings["use_motion"]:
            motion_mask = self.fgbg.apply(blurred)
            final_mask = cv2.bitwise_and(color_mask, color_mask, mask=motion_mask)
        else:
            final_mask = color_mask

        final_mask = cv2.erode(final_mask, None, iterations=2)
        final_mask = cv2.dilate(final_mask, None, iterations=2)
        self.current_mask = final_mask

        cnts, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centers = []
        
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > self.MIN_RADIUS:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    current_centers.append(center)

        # Trail Matching Logic
        used_centers = set()
        for t_idx, trail in enumerate(self.trails):
            if len(trail['pts']) == 0: continue
            last_pos = trail['pts'][0]
            best_dist = self.MAX_DISTANCE
            best_c_idx = -1
            for c_idx, center in enumerate(current_centers):
                if c_idx in used_centers: continue
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

        for c_idx, center in enumerate(current_centers):
            if c_idx not in used_centers:
                new_trail = {'id': self.next_object_id, 'pts': deque(maxlen=self.BUFFER_SIZE), 'disappeared': 0}
                new_trail['pts'].appendleft(center)
                self.trails.append(new_trail)
                self.next_object_id += 1
        self.trails = [t for t in self.trails if t['disappeared'] < self.MAX_DISAPPEARED]

    def get_trails(self):
        return self.trails
    
    def get_objects(self):
        return self.objects

    def show_feed(self, debug=False):
        if self.current_frame is None: return True

        display_frame = self.current_frame.copy()

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
            
            # Draw Objects
            for mid, pos in self.objects.items():
                cv2.circle(display_frame, pos, 5, (0, 255, 0), -1) 
                cv2.putText(display_frame, f"ID: {mid}", (pos[0] + 10, pos[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Stats
            cv2.putText(display_frame, f"FPS: {int(self.current_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.perspective_matrix is not None:
                cv2.putText(display_frame, "Planar View", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            elif self.camera_matrix is not None:
                cv2.putText(display_frame, "Undistorted", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        if debug:
            self._ensure_sliders_window()
            self._update_settings_from_sliders()
            
            # Triple Split View: Raw | Processed | Mask
            # Resize Raw
            raw_h, raw_w = self.raw_frame.shape[:2]
            scale_raw = 300 / raw_w
            raw_view = cv2.resize(self.raw_frame, (0,0), fx=scale_raw, fy=scale_raw)
            cv2.putText(raw_view, "Raw", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # Resize Processed
            proc_h, proc_w = display_frame.shape[:2]
            scale_proc = 300 / proc_w
            proc_view = cv2.resize(display_frame, (0,0), fx=scale_proc, fy=scale_proc)
            cv2.putText(proc_view, "Output", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # Resize Mask
            if self.current_mask is not None:
                mask_h, mask_w = self.current_mask.shape[:2]
                scale_mask = 300 / mask_w
                mask_view = cv2.resize(self.current_mask, (0,0), fx=scale_mask, fy=scale_mask)
                mask_bgr = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_bgr, "Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                mask_bgr = np.zeros_like(proc_view)

            # Ensure heights match for hstack (pad if necessary)
            max_h = max(raw_view.shape[0], proc_view.shape[0], mask_bgr.shape[0])
            
            def pad_img(img, target_h):
                pad = target_h - img.shape[0]
                if pad > 0: return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
                return img
            
            debug_stack = np.hstack([pad_img(raw_view, max_h), pad_img(proc_view, max_h), pad_img(mask_bgr, max_h)])
            cv2.imshow("Tracker (Debug Mode)", debug_stack)

        else:
            if self.sliders_window_created:
                cv2.destroyWindow("Settings")
                cv2.destroyWindow("Tracker (Debug Mode)")
                self.sliders_window_created = False
            cv2.imshow("Tracker", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            self._save_settings()
        
        return True

    def release(self):
        cv2.destroyAllWindows()
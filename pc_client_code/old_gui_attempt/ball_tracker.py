import cv2
import numpy as np
import json
import os
import time
from collections import deque
from scipy.optimize import curve_fit

class BallTracker:
    def __init__(self, settings_file="hsv_settings.json", calibration_file="calibration_params.json"):
        # --- CONFIGURATION ---
        self.BUFFER_SIZE = 32
        self.MIN_RADIUS = 6
        self.MAX_DISTANCE = 50
        self.MAX_DISAPPEARED = 20
        self.WARMUP_FRAMES = 60
        
        # --- PATHS ---
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.SETTINGS_FILE = os.path.join(package_dir, settings_file)
        self.CALIBRATION_FILE = os.path.join(package_dir, calibration_file)

        # --- STATE ---
        self.trails = []       
        self.objects = {}      
        self.next_object_id = 0
        self.frame_count = 0
        self.current_fps = 30.0 
        self.prev_frame_time = 0

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.settings = self._load_settings()
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # --- TRACKING STATE ---
        self.roi_marker_ids = None 
        self.roi_offsets = (0, 0)
        self.roi_goal_offset = 0   
        
        self.origin_marker_ids = None 
        self.origin_offset_mm = (0, 0)
        
        # Caches
        self.cached_roi_rects = None      
        self.cached_goal_line_y = None 
        self.cached_origin_px = None
        
        # Constraints
        self.robot_travel_mm = 300.0 
        self.robot_width_mm = 50.0   
        self.safe_margin_mm = 10.0   
        
        # Calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.pixels_per_meter = 0 
        self._load_calibration() 

        self.current_frame = None  
        self.current_mask = None
        self.sliders_window_created = False

    def _load_settings(self):
        default = {"h_min": 0, "h_max": 30, "s_min": 100, "s_max": 255, "v_min": 100, "v_max": 255, "use_motion": 1}
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r') as f: return json.load(f)
            except: pass
        return default

    def _save_settings(self):
        with open(self.SETTINGS_FILE, 'w') as f: json.dump(self.settings, f)
    
    def _load_calibration(self):
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                with open(self.CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get("camera_matrix"):
                        self.camera_matrix = np.array(data["camera_matrix"])
                    if data.get("dist_coeffs"):
                        self.dist_coeffs = np.array(data["dist_coeffs"])
                    self.pixels_per_meter = data.get("pixels_per_meter", 0)
            except: pass

    def _save_calibration(self):
        data = {
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else [],
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else [],
            "pixels_per_meter": self.pixels_per_meter
        }
        with open(self.CALIBRATION_FILE, 'w') as f: json.dump(data, f)

    def delete_calibration_key(self, key):
        if key == 'distortion':
            self.camera_matrix = None
            self.dist_coeffs = None
        elif key == 'scale':
            self.pixels_per_meter = 0
        self._save_calibration()

    def calibrate_camera(self, cap_source, rows, cols, marker_id=None, marker_size_m=None, mode='both'):
        # Calibration runs in its own window loop, blocking the main GUI loop is okay for this task usually,
        # or it can be run in a thread. For simplicity, we use cv2.imshow here.
        print(f"--- CALIBRATION START ({mode}) ---")
        
        # 1. Distortion
        if mode in ['both', 'distortion']:
            print("STEP: Distortion.")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
            objpoints, imgpoints = [], []
            captured = 0

            while True:
                ret, frame = cap_source.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = frame.copy()
                ret_corners, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

                if ret_corners:
                    cv2.drawChessboardCorners(display, (rows, cols), corners, ret_corners)
                    cv2.putText(display, f"Captured: {captured}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Calib: Distortion", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and ret_corners:
                    objpoints.append(objp)
                    imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
                    captured += 1
                    cv2.waitKey(100)
                elif key == ord('q'): break
            
            cv2.destroyWindow("Calib: Distortion")
            if captured > 0:
                h, w = frame.shape[:2]
                ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
                if ret:
                    self.camera_matrix = mtx
                    self.dist_coeffs = dist
                    print("Distortion Saved.")

        # 2. Scale
        if mode in ['both', 'scale']:
            if self.camera_matrix is None:
                print("Error: Calibrate Distortion first.")
                return
            print("STEP: Scale.")
            while True:
                ret, frame = cap_source.read()
                if not ret: break
                h, w = frame.shape[:2]
                newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                
                if ids is not None: cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, "Press 'c' to Calc Scale", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Calib: Scale", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and ids is not None and marker_id in ids:
                    idx = np.where(ids == marker_id)[0][0]
                    perimeter = cv2.arcLength(corners[idx][0], True)
                    self.pixels_per_meter = (perimeter / 4.0) / marker_size_m
                    print(f"Scale Saved: {self.pixels_per_meter:.2f}")
                    break
                elif key == ord('q'): break
            cv2.destroyWindow("Calib: Scale")

        self._save_calibration()

    def _nothing(self, x): pass

    def _ensure_sliders_window(self):
        if not self.sliders_window_created:
            cv2.namedWindow("HSV Settings")
            cv2.resizeWindow("HSV Settings", 300, 350)
            cv2.createTrackbar("Hue Min", "HSV Settings", self.settings["h_min"], 179, self._nothing)
            cv2.createTrackbar("Hue Max", "HSV Settings", self.settings["h_max"], 179, self._nothing)
            cv2.createTrackbar("Sat Min", "HSV Settings", self.settings["s_min"], 255, self._nothing)
            cv2.createTrackbar("Sat Max", "HSV Settings", self.settings["s_max"], 255, self._nothing)
            cv2.createTrackbar("Val Min", "HSV Settings", self.settings["v_min"], 255, self._nothing)
            cv2.createTrackbar("Val Max", "HSV Settings", self.settings["v_max"], 255, self._nothing)
            cv2.createTrackbar("Use Motion", "HSV Settings", self.settings["use_motion"], 1, self._nothing)
            self.sliders_window_created = True

    def _update_settings_from_sliders(self):
        self.settings["h_min"] = cv2.getTrackbarPos("Hue Min", "HSV Settings")
        self.settings["h_max"] = cv2.getTrackbarPos("Hue Max", "HSV Settings")
        self.settings["s_min"] = cv2.getTrackbarPos("Sat Min", "HSV Settings")
        self.settings["s_max"] = cv2.getTrackbarPos("Sat Max", "HSV Settings")
        self.settings["v_min"] = cv2.getTrackbarPos("Val Min", "HSV Settings")
        self.settings["v_max"] = cv2.getTrackbarPos("Val Max", "HSV Settings")
        self.settings["use_motion"] = cv2.getTrackbarPos("Use Motion", "HSV Settings")

    def set_roi_markers(self, id1, id2, offset_x=50, offset_y=50, goal_offset_y=50):
        self.roi_marker_ids = (id1, id2)
        self.roi_offsets = (offset_x, offset_y)
        self.roi_goal_offset = goal_offset_y

    def set_origin_markers(self, id1, id2, offset_x_mm=0, offset_y_mm=0):
        self.origin_marker_ids = (id1, id2)
        self.origin_offset_mm = (offset_x_mm, offset_y_mm)

    def set_robot_constraints(self, travel_mm, width_mm, margin_mm=10):
        self.robot_travel_mm = travel_mm
        self.robot_width_mm = width_mm
        self.safe_margin_mm = margin_mm

    def get_ball_position_mm(self):
        if not self.trails or len(self.trails[0]['pts']) == 0: return None
        if self.cached_origin_px is None or self.pixels_per_meter == 0: return None
        ball_px_x, ball_px_y = self.trails[0]['pts'][0]
        origin_px_x, origin_px_y = self.cached_origin_px
        dx_px = ball_px_x - origin_px_x
        dy_px = ball_px_y - origin_px_y 
        scale = 1000.0 / self.pixels_per_meter
        return (int(dx_px * scale), int(dy_px * scale * -1))

    def update(self, frame, velocity_method='poly', rotation=0):
        if frame is None: return
        
        # 1. Undistort (Use Raw Frame Dimensions)
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        # 2. Rotate (After Undistort)
        if rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.current_frame = frame 
        self.frame_count += 1
        if self.frame_count < self.WARMUP_FRAMES: return

        # 3. ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        self.objects = {}
        if ids is not None:
            for (c, mid) in zip(corners, ids.flatten()):
                self.objects[int(mid)] = (int(c[0][:, 0].mean()), int(c[0][:, 1].mean()))
            
            # Caches
            if self.roi_marker_ids:
                id1, id2 = self.roi_marker_ids
                if id1 in self.objects and id2 in self.objects:
                    p1, p2 = self.objects[id1], self.objects[id2]
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    off_x, off_y = self.roi_offsets
                    outer = (x1, y1, x2-x1, y2-y1)
                    if (x2-x1) > 2*off_x and (y2-y1) > 2*off_y:
                        inner = (x1+off_x, y1+off_y, (x2-x1)-2*off_x, (y2-y1)-2*off_y)
                    else: inner = outer
                    self.cached_roi_rects = {'outer': outer, 'inner': inner}
                    self.cached_goal_line_y = (inner[1] + inner[3]) - self.roi_goal_offset

            if self.origin_marker_ids:
                oid1, oid2 = self.origin_marker_ids
                if oid1 in self.objects and oid2 in self.objects:
                    op1, op2 = np.array(self.objects[oid1]), np.array(self.objects[oid2])
                    mid = (op1 + op2) / 2.0
                    if self.pixels_per_meter > 0:
                        ox_px = (self.origin_offset_mm[0] / 1000.0) * self.pixels_per_meter
                        oy_px = (self.origin_offset_mm[1] / 1000.0) * self.pixels_per_meter
                        self.cached_origin_px = (int(mid[0] + ox_px), int(mid[1] + oy_px))

        # 4. Ball Tracking
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.settings["h_min"], self.settings["s_min"], self.settings["v_min"]), 
                                (self.settings["h_max"], self.settings["s_max"], self.settings["v_max"]))
        if self.settings["use_motion"]:
            mask = cv2.bitwise_and(mask, mask, mask=self.fgbg.apply(blurred))
        mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
        
        if self.cached_roi_rects:
            rx, ry, rw, rh = self.cached_roi_rects['inner']
            roi_mask = np.zeros_like(mask)
            roi_mask[ry:ry+rh, rx:rx+rw] = 255
            mask = cv2.bitwise_and(mask, roi_mask)
        self.current_mask = mask

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curr = []
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            if r > self.MIN_RADIUS:
                M = cv2.moments(c)
                if M["m00"] > 0: curr.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))

        # Update Trails
        used = set()
        for t in self.trails:
            if not t['pts']: continue
            last = t['pts'][0]
            best_d, best_i = self.MAX_DISTANCE, -1
            for i, c in enumerate(curr):
                if i in used: continue
                d = np.linalg.norm(np.array(last)-np.array(c))
                if d < best_d: best_d, best_i = d, i
            if best_i != -1:
                t['pts'].appendleft(curr[best_i])
                t['disappeared'] = 0
                used.add(best_i)
            else: t['disappeared'] += 1
        
        for i, c in enumerate(curr):
            if i not in used:
                self.trails.append({'id': self.next_object_id, 'pts': deque([c], maxlen=self.BUFFER_SIZE), 'disappeared':0})
                self.next_object_id += 1
        self.trails = [t for t in self.trails if t['disappeared'] < self.MAX_DISAPPEARED]

    def show_feed(self, debug=False, scale=1.0, return_rgb=False):
        if self.current_frame is None: 
            return None if return_rgb else True
            
        display = self.current_frame.copy()

        # Draw Elements
        if self.cached_roi_rects:
            ix, iy, iw, ih = self.cached_roi_rects['inner']
            cv2.rectangle(display, (ix,iy), (ix+iw, iy+ih), (0,255,255), 2)
        
        if self.cached_goal_line_y:
            cv2.line(display, (0, self.cached_goal_line_y), (display.shape[1], self.cached_goal_line_y), (0,0,255), 2)

        if self.cached_origin_px and self.pixels_per_meter > 0 and self.cached_goal_line_y:
            ox, oy = self.cached_origin_px
            cv2.drawMarker(display, (ox, oy), (255,0,255), cv2.MARKER_CROSS, 20, 2)
            
            lim_px = int((self.robot_travel_mm/2000.0) * self.pixels_per_meter)
            wid_px = int((self.robot_width_mm/2000.0) * self.pixels_per_meter)
            
            lx, rx = ox - lim_px, ox + lim_px
            cv2.line(display, (lx, self.cached_goal_line_y), (rx, self.cached_goal_line_y), (255,255,0), 3)
            # Bumpers
            cv2.line(display, (lx-wid_px, self.cached_goal_line_y-10), (lx-wid_px, self.cached_goal_line_y+10), (0,0,255), 2)
            cv2.line(display, (rx+wid_px, self.cached_goal_line_y-10), (rx+wid_px, self.cached_goal_line_y+10), (0,0,255), 2)

        pos = self.get_ball_position_mm()
        if pos and self.trails:
            bx, by = self.trails[0]['pts'][0]
            cv2.putText(display, f"X:{pos[0]} Y:{pos[1]}", (bx+10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        for mid, p in self.objects.items():
            cv2.circle(display, p, 4, (0,255,0), -1)
            cv2.putText(display, str(mid), (p[0]+5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.putText(display, f"FPS: {int(self.current_fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # --- OUTPUT SELECTION ---
        if return_rgb:
            # For GUI: Return RGB Array
            return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        else:
            # For Standalone: Open Window
            if debug:
                self._ensure_sliders_window()
                self._update_settings_from_sliders()
                cv2.imshow("Debug Feed", display)
                if self.current_mask is not None: cv2.imshow("Mask", self.current_mask)
            else:
                final = cv2.resize(display, (0,0), fx=scale, fy=scale)
                cv2.imshow("Tracker Feed", final)
            
            if cv2.waitKey(1) & 0xFF == ord('s'): self._save_settings()
            return True

    def release(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = BallTracker()
    tracker.set_roi_markers(0, 2, offset_x=50, offset_y=30, goal_offset_y=100)
    tracker.set_origin_markers(0, 2, offset_x_mm=100, offset_y_mm=0)
    tracker.calibrate_camera(cap, 6, 8, 1, 0.03, "both")
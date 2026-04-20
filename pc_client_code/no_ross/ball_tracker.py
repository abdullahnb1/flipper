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
        self.TABLE_SLOPE_DEG = 3.5
        
        # --- VELOCITY FILTER CONFIG ---
        self.VEL_FILTER_SIZE = 2    
        self.VEL_DECAY_WEIGHT = 0.1  
        self.MAX_VELOCITY_MM_S = 500.0 
        
        # --- PATH RESOLUTION ---
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.SETTINGS_FILE = os.path.join(package_dir, settings_file)
        self.CALIBRATION_FILE = os.path.join(package_dir, calibration_file)

        print(f"[Tracker] Loading settings from: {self.SETTINGS_FILE}")
        print(f"[Tracker] Loading calibration from: {self.CALIBRATION_FILE}")

        # --- STATE VARIABLES ---
        self.trails = []       
        self.objects = {}      
        self.next_object_id = 0
        self.frame_count = 0
        self.prev_frame_time = 0
        self.current_fps = 30.0 
        
        self.vel_history = deque(maxlen=self.VEL_FILTER_SIZE)

        # Processing Tools
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.settings = self._load_settings()
        
        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Perspective / Physics State
        self.perspective_marker_id = -1
        self.marker_real_size = 0.0
        self.pixels_per_meter = 0 
        self.perspective_matrix = None
        self.warped_size = (640, 480) 
        
        # ROI & Goal Line State
        self.roi_marker_ids = None 
        self.roi_offsets = (0, 0)
        self.roi_goal_offset = 0   
        
        # CACHED STATE
        self.cached_roi_rects = None      
        self.cached_goal_line_y = None 
        self.cached_origin_px = None
        
        # Origin & Constraint State
        self.origin_marker_ids = None 
        self.origin_offset_mm = (0, 0)
        
        # Robot Physical Constraints (Asymmetric)
        self.robot_min_x_mm = -150.0 
        self.robot_max_x_mm = 150.0  
        self.robot_width_mm = 50.0   
        self.safe_margin_mm = 10.0   
        
        # VISUALIZATION EXTRAS
        self.flipper_boundary_x_mm = None 

        # Calibration State
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_calibration() 

        # Display State
        self.current_frame = None  
        self.current_mask = None
        self.raw_frame = None
        self.sliders_window_created = False

    # ------------------------------------------------------------------
    #   SETTINGS & CALIBRATION
    # ------------------------------------------------------------------
    def _load_settings(self):
        default = {"h_min": 0, "h_max": 30, "s_min": 100, "s_max": 255, "v_min": 100, "v_max": 255, "use_motion": 1}
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r') as f: return json.load(f)
            except: pass
        return default

    def save_settings(self):
        try:
            with open(self.SETTINGS_FILE, 'w') as f: 
                json.dump(self.settings, f)
            print("[Tracker] Settings Saved Successfully.")
            return True
        except Exception as e:
            print(f"[Tracker] Save Error: {e}")
            return False
    
    def _load_calibration(self):
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                with open(self.CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                    
                    cm = data.get("camera_matrix")
                    if cm: self.camera_matrix = np.array(cm)
                    else: self.camera_matrix = None
                        
                    dc = data.get("dist_coeffs")
                    if dc: self.dist_coeffs = np.array(dc)
                    else: self.dist_coeffs = None

                    self.pixels_per_meter = data.get("pixels_per_meter", 0)
                    
                    pm = data.get("perspective_matrix")
                    if pm: self.perspective_matrix = np.array(pm)
                    
                    self.warped_size = tuple(data.get("warped_size", (1280, 720)))
                    self.perspective_marker_id = data.get("perspective_marker_id", -1)
                    self.marker_real_size = data.get("marker_real_size", 0.0)
            except Exception as e:
                print(f"[Tracker] Calibration Load Error: {e}")

    def _save_calibration(self):
        data = {
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else [],
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else [],
            "pixels_per_meter": self.pixels_per_meter,
            "perspective_matrix": self.perspective_matrix.tolist() if self.perspective_matrix is not None else [],
            "warped_size": self.warped_size,
            "perspective_marker_id": self.perspective_marker_id,
            "marker_real_size": self.marker_real_size
        }
        with open(self.CALIBRATION_FILE, 'w') as f: json.dump(data, f)

    def calibrate_camera(self, cap_source, rows, cols, marker_id=None, marker_size_m=None, mode='both', rotation=0):
        print(f"--- CALIBRATION START ({mode}) ---")
        frame_for_scale = None
        
        # --- 1. DISTORTION ---
        if mode in ['both', 'distortion']:
            print("STEP 1: Lens Distortion. Using RAW FRAME (Ignore Rotation).")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
            objpoints, imgpoints = [], []
            captured_count = 0

            while True:
                ret, frame = cap_source.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = frame.copy()
                ret_corners, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

                if ret_corners:
                    cv2.drawChessboardCorners(display, (rows, cols), corners, ret_corners)
                    cv2.putText(display, f"Captured: {captured_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                cv2.imshow("Calibration: Distortion (RAW)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and ret_corners:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    captured_count += 1
                    frame_for_scale = frame.copy()
                    print(f"Captured {captured_count}")
                    cv2.waitKey(100)
                elif key == ord('q'): break
            
            cv2.destroyWindow("Calibration: Distortion (RAW)")

            if captured_count > 0:
                h, w = frame.shape[:2]
                ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
                if ret:
                    self.camera_matrix = mtx
                    self.dist_coeffs = dist
                    print("Distortion Saved.")
                    self._save_calibration()

        def apply_rotation(img, angle):
            if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img

        # --- 2. SCALE ---
        if mode in ['both', 'scale']:
            print("STEP 2: Scale. Show ArUco Marker.")
            
            def get_processed_frame(raw_frame):
                if self.camera_matrix is not None and self.camera_matrix.size > 0:
                    h, w = raw_frame.shape[:2]
                    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
                    undistorted = cv2.undistort(raw_frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
                    return apply_rotation(undistorted, rotation)
                else:
                    return apply_rotation(raw_frame, rotation)

            if frame_for_scale is None:
                while True:
                    ret, frame = cap_source.read()
                    if not ret: break
                    
                    display = get_processed_frame(frame)
                    
                    status = "Calibrated" if (self.camera_matrix is not None) else "RAW (No Dist. Calib)"
                    cv2.putText(display, f"Mode: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(display, "Press 'c' with Marker Visible", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    cv2.imshow("Calibration: Scale", display)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        frame_for_scale = frame
                        break
                cv2.destroyWindow("Calibration: Scale")

            if frame_for_scale is not None:
                processed_frame = get_processed_frame(frame_for_scale)
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                
                if ids is not None and marker_id in ids:
                    idx = np.where(ids == marker_id)[0][0]
                    perimeter = cv2.arcLength(corners[idx][0], True)
                    self.pixels_per_meter = (perimeter / 4.0) / marker_size_m
                    print(f"Scale Saved: {self.pixels_per_meter:.2f} px/m")
                else:
                    print("Error: Marker not found in frame.")
        
        self._save_calibration()

    def _nothing(self, x): pass

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
        self.perspective_marker_id = marker_id
        self.marker_real_size = marker_real_width_meters
        if self.pixels_per_meter == 0: self.pixels_per_meter = pixels_per_meter
        self.perspective_matrix = None 

    def set_roi_markers(self, id1, id2, offset_x=50, offset_y=50, goal_offset_y=50):
        self.roi_marker_ids = (id1, id2)
        self.roi_offsets = (offset_x, offset_y)
        self.roi_goal_offset = goal_offset_y

    def set_origin_markers(self, id1, id2, offset_x_mm=0, offset_y_mm=0):
        self.origin_marker_ids = (id1, id2)
        self.origin_offset_mm = (offset_x_mm, offset_y_mm)

    def set_robot_constraints(self, min_x_mm, max_x_mm, width_mm, margin_mm=10):
        self.robot_min_x_mm = float(min_x_mm)
        self.robot_max_x_mm = float(max_x_mm)
        self.robot_width_mm = float(width_mm)
        self.safe_margin_mm = float(margin_mm)

    def set_flipper_boundary(self, x_mm):
        self.flipper_boundary_x_mm = x_mm

    def get_ball_position_mm(self):
        if not self.trails or len(self.trails[0]['pts']) == 0: return None
        if self.cached_origin_px is None or self.pixels_per_meter == 0: return None

        ball_px_x, ball_px_y = self.trails[0]['pts'][0]
        origin_px_x, origin_px_y = self.cached_origin_px
        
        dx_px = ball_px_x - origin_px_x
        dy_px = ball_px_y - origin_px_y 
        
        scale = 1000.0 / self.pixels_per_meter
        x_mm = dx_px * scale
        y_mm = dy_px * scale * -1 
        
        return (int(x_mm), int(y_mm))

    # ------------------------------------------------------------------
    #   VELOCITY & PREDICTION METHODS
    # ------------------------------------------------------------------
    def _calculate_filtered_velocity(self, trail):
        if len(trail) < 2: return None
            
        dt = 1.0 / self.current_fps if self.current_fps > 0 else 0.033
        p0 = np.array(trail[0]) # Current
        p1 = np.array(trail[1]) # Previous
        instant_vel = (p0 - p1) / dt
        
        if self.pixels_per_meter > 0:
            max_px_s = (self.MAX_VELOCITY_MM_S / 1000.0) * self.pixels_per_meter
            speed = np.linalg.norm(instant_vel)
            if speed > max_px_s and speed > 0:
                instant_vel = (instant_vel / speed) * max_px_s
        
        self.vel_history.append(instant_vel)
        
        count = len(self.vel_history)
        weights = np.array([self.VEL_DECAY_WEIGHT ** i for i in range(count)])[::-1]
        weights /= weights.sum()
        
        vx_avg = 0.0
        vy_avg = 0.0
        for i, v in enumerate(self.vel_history):
            vx_avg += v[0] * weights[i]
            vy_avg += v[1] * weights[i]
            
        return (vx_avg, vy_avg)

    def _predict_linear_intersection(self, pos, vel):
        if self.cached_goal_line_y is None or vel is None: return None
        
        bx, by = pos
        vx, vy = vel
        
        if abs(vy) < 1.0: return None
        
        dist_y = self.cached_goal_line_y - by
        t_impact = dist_y / vy
        
        if t_impact <= 0: return None
        
        pred_x_px = bx + vx * t_impact
        
        # Check Bounds
        if self.cached_origin_px is not None and self.pixels_per_meter > 0:
            origin_x = self.cached_origin_px[0]
            min_x_offset_px = (self.robot_min_x_mm / 1000.0) * self.pixels_per_meter
            max_x_offset_px = (self.robot_max_x_mm / 1000.0) * self.pixels_per_meter
            margin_px = (self.safe_margin_mm / 1000.0) * self.pixels_per_meter
            
            abs_min_x = origin_x + min_x_offset_px - margin_px
            abs_max_x = origin_x + max_x_offset_px + margin_px
            
            if not (abs_min_x <= pred_x_px <= abs_max_x): return None

        return int(pred_x_px)

    # --- UPDATED: HYBRID PREDICTION (3-Point Fallback + Poly) ---
    def _predict_polynomial_intersection(self, trail, num_points=10):
        if self.cached_goal_line_y is None: return None, None
        
        dt = 1.0 / self.current_fps if self.current_fps > 0 else 0.033
        
        # --- FALLBACK: If < 4 points, use 3-Point Backwards Diff ---
        if len(trail) < 4:
            if len(trail) < 3: return None, None
            
            # Use 3-Point Formula for better local derivative than 2-Point
            # f'(x) = (3f(x) - 4f(x-h) + f(x-2h)) / 2h
            p0 = np.array(trail[0])
            p1 = np.array(trail[1])
            p2 = np.array(trail[2])
            
            vel = (3*p0 - 4*p1 + p2) / (2*dt)
            vx, vy = vel[0], vel[1]
            
            # Simple Linear Projection with this robust velocity
            dist_y = self.cached_goal_line_y - p0[1]
            
            if abs(vy) < 1.0: return None, (vx, vy)
            t_impact = dist_y / vy
            
            if t_impact <= 0: return None, (vx, vy)
            
            pred_x_raw = int(p0[0] + vx * t_impact)
            
            # Bounds Check
            if self.cached_origin_px is not None and self.pixels_per_meter > 0:
                origin_x = self.cached_origin_px[0]
                min_x_offset_px = (self.robot_min_x_mm / 1000.0) * self.pixels_per_meter
                max_x_offset_px = (self.robot_max_x_mm / 1000.0) * self.pixels_per_meter
                margin_px = (self.safe_margin_mm / 1000.0) * self.pixels_per_meter
                abs_min_x = origin_x + min_x_offset_px - margin_px
                abs_max_x = origin_x + max_x_offset_px + margin_px
                
                if not (abs_min_x <= pred_x_raw <= abs_max_x): return None, (vx, vy)
                
            return pred_x_raw, (vx, vy)

        # --- STANDARD: If >= 4 points, use Curve Fit ---
        try:
            recent = list(trail)[:num_points][::-1] 
            x_data = np.array([p[0] for p in recent])
            y_data = np.array([p[1] for p in recent])
            t_data = np.linspace(-(len(recent)-1)*dt, 0, len(recent))

            # X(t) = v_x * t + x_0 (Linear)
            coeff_x = np.polyfit(t_data, x_data, 1) 
            
            # Y(t) = 0.5*a*t^2 + v_y*t + y_0 (Quadratic for gravity)
            coeff_y = np.polyfit(t_data, y_data, 2) 
            
            vx_fit = coeff_x[0]
            vy_fit = coeff_y[1]
            current_vel = (vx_fit, vy_fit)

            # Solve: 0.5*a*t^2 + v_y*t + (y_0 - goal_y) = 0
            A = coeff_y[0]
            B = coeff_y[1]
            C = coeff_y[2] - self.cached_goal_line_y
            
            delta = B**2 - 4*A*C
            if delta < 0: return None, current_vel 
            
            sqrt_delta = np.sqrt(delta)
            if abs(A) < 0.001:
                if abs(B) < 0.001: return None, current_vel
                t_impact = -C / B
            else:
                t1 = (-B - sqrt_delta) / (2*A)
                t2 = (-B + sqrt_delta) / (2*A)
                opts = [t for t in [t1, t2] if t > 0]
                if not opts: return None, current_vel
                t_impact = min(opts)

            pred_x_raw = coeff_x[0] * t_impact + coeff_x[1]
            
            # Bounds Check (Same logic)
            if self.cached_origin_px is not None and self.pixels_per_meter > 0:
                origin_x = self.cached_origin_px[0]
                min_x_offset_px = (self.robot_min_x_mm / 1000.0) * self.pixels_per_meter
                max_x_offset_px = (self.robot_max_x_mm / 1000.0) * self.pixels_per_meter
                margin_px = (self.safe_margin_mm / 1000.0) * self.pixels_per_meter
                abs_min_x = origin_x + min_x_offset_px - margin_px
                abs_max_x = origin_x + max_x_offset_px + margin_px
                
                if not (abs_min_x <= pred_x_raw <= abs_max_x):
                    return None, current_vel

            return int(pred_x_raw), current_vel

        except:
            return None, None

    # ------------------------------------------------------------------
    #   UPDATE LOOP
    # ------------------------------------------------------------------
    def update(self, frame, velocity_method='velocity', img_format='bgr', rotation=0):
        if frame is None: return
        if img_format.lower() == 'rgb': frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.raw_frame = frame.copy()

        # Undistort & Rotate
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        if rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.perspective_matrix is not None:
            frame = cv2.warpPerspective(frame, self.perspective_matrix, self.warped_size)
        
        self.frame_count += 1
        new_frame_time = time.time()
        if self.prev_frame_time != 0:
            diff = new_frame_time - self.prev_frame_time
            if diff > 0: self.current_fps = 1 / diff
        self.prev_frame_time = new_frame_time

        self.current_frame = frame 
        if self.frame_count < self.WARMUP_FRAMES: return

        # ArUco Detect
        self.objects = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            for (c, mid) in zip(corners, ids.flatten()):
                cx, cy = int(c[0][:, 0].mean()), int(c[0][:, 1].mean())
                self.objects[int(mid)] = (cx, cy)
            
            if self.roi_marker_ids:
                id1, id2 = self.roi_marker_ids
                if id1 in self.objects and id2 in self.objects:
                    p1 = self.objects[id1]
                    p2 = self.objects[id2]
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    off_x, off_y = self.roi_offsets
                    w, h = x2 - x1, y2 - y1
                    self.cached_roi_rects = {'outer': (x1, y1, w, h), 
                                             'inner': (x1+off_x, y1+off_y, w-2*off_x, h-2*off_y)}
                    self.cached_goal_line_y = (y1 + off_y + (h-2*off_y)) - self.roi_goal_offset

            if self.origin_marker_ids:
                oid1, oid2 = self.origin_marker_ids
                if oid1 in self.objects and oid2 in self.objects:
                    op1 = np.array(self.objects[oid1])
                    op2 = np.array(self.objects[oid2])
                    mid = (op1 + op2) / 2.0
                    if self.pixels_per_meter > 0:
                        ox = int(mid[0] + (self.origin_offset_mm[0]/1000.0)*self.pixels_per_meter)
                        oy = int(mid[1] + (self.origin_offset_mm[1]/1000.0)*self.pixels_per_meter)
                        self.cached_origin_px = (ox, oy)

        # Ball Detect
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.settings["h_min"], self.settings["s_min"], self.settings["v_min"]), 
                                (self.settings["h_max"], self.settings["s_max"], self.settings["v_max"]))
        if self.settings["use_motion"]:
            motion = self.fgbg.apply(blurred)
            mask = cv2.bitwise_and(mask, mask, mask=motion)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        if self.cached_roi_rects:
            rx, ry, rw, rh = self.cached_roi_rects['inner']
            roi_mask = np.zeros_like(mask)
            roi_mask[ry:ry+rh, rx:rx+rw] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        self.current_mask = mask

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centers = []
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            
            # Ignore below goal
            if self.cached_goal_line_y is not None and y > self.cached_goal_line_y:
                continue
            
            if r > self.MIN_RADIUS:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    current_centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))

        # Tracking Association
        used_centers = set()
        for t in self.trails:
            if not t['pts']: continue
            last = t['pts'][0]
            best_dist, best_idx = self.MAX_DISTANCE, -1
            for i, c in enumerate(current_centers):
                if i in used_centers: continue
                d = np.linalg.norm(np.array(last) - np.array(c))
                if d < best_dist: best_dist, best_idx = d, i
            if best_idx != -1:
                t['pts'].appendleft(current_centers[best_idx])
                t['disappeared'] = 0
                used_centers.add(best_idx)
            else: t['disappeared'] += 1

        for i, c in enumerate(current_centers):
            if i not in used_centers:
                self.trails.append({'id': self.next_object_id, 'pts': deque(maxlen=self.BUFFER_SIZE), 'disappeared': 0})
                self.trails[-1]['pts'].appendleft(c)
                self.next_object_id += 1
        
        self.trails = [t for t in self.trails if t['disappeared'] < self.MAX_DISAPPEARED]

        # PREDICTION LOGIC
        for t in self.trails:
            pred_x = None
            vel = None
            
            if len(t['pts']) >= 2:
                # SELECT METHOD HERE
                if velocity_method == 'poly_curve':
                    pred_x, vel = self._predict_polynomial_intersection(t['pts'])
                else:
                    # Default: Velocity Linear
                    vel = self._calculate_filtered_velocity(t['pts'])
                    if vel is not None:
                        pred_x = self._predict_linear_intersection(t['pts'][0], vel)
            else:
                self.vel_history.clear() 

            t['physics'] = {'pred_x': pred_x, 'vel': vel}

    # ------------------------------------------------------------------
    #   VISUALIZATION
    # ------------------------------------------------------------------
    def get_trails(self): return self.trails
    def get_objects(self): return self.objects

    def show_feed(self, debug=False, scale=1.0, return_rgb=False):
        if self.current_frame is None: return None if return_rgb else True
        display = self.current_frame.copy()

        if self.frame_count < self.WARMUP_FRAMES:
            cv2.putText(display, f"Warming Up... {int(self.frame_count/self.WARMUP_FRAMES*100)}%", (20,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            # ROI
            if self.cached_roi_rects:
                ix, iy, iw, ih = self.cached_roi_rects['inner']
                cv2.rectangle(display, (ix, iy), (ix+iw, iy+ih), (0, 255, 255), 2)
            if self.cached_goal_line_y is not None:
                cv2.line(display, (0, self.cached_goal_line_y), (display.shape[1], self.cached_goal_line_y), (0, 0, 255), 2)
            
            # Robot & Flipper Bounds
            if self.cached_origin_px is not None:
                ox, oy = self.cached_origin_px
                cv2.drawMarker(display, (ox, oy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                if self.pixels_per_meter > 0 and self.cached_goal_line_y is not None:
                    gy = self.cached_goal_line_y
                    min_x_px = int(ox + (self.robot_min_x_mm / 1000.0) * self.pixels_per_meter)
                    max_x_px = int(ox + (self.robot_max_x_mm / 1000.0) * self.pixels_per_meter)
                    half_width_px = int((self.robot_width_mm / 2.0 / 1000.0) * self.pixels_per_meter)
                    
                    # Travel Line (Cyan)
                    cv2.line(display, (min_x_px, gy), (max_x_px, gy), (255, 255, 0), 4)
                    
                    # Left Limit + Paddle Box
                    cv2.line(display, (min_x_px, gy-15), (min_x_px, gy+15), (0,0,255), 3) # Red Tick
                    cv2.rectangle(display, (min_x_px-half_width_px, gy-10), (min_x_px+half_width_px, gy+10), (0,255,0), 2) # Green Box
                    
                    # Right Limit + Paddle Box
                    cv2.line(display, (max_x_px, gy-15), (max_x_px, gy+15), (0,0,255), 3) # Red Tick
                    cv2.rectangle(display, (max_x_px-half_width_px, gy-10), (max_x_px+half_width_px, gy+10), (0,255,0), 2) # Green Box
                    
                    if self.flipper_boundary_x_mm is not None:
                        bound_px = int(ox + (self.flipper_boundary_x_mm / 1000.0) * self.pixels_per_meter)
                        for y in range(0, display.shape[0], 20):
                            cv2.line(display, (bound_px, y), (bound_px, y+10), (255, 255, 0), 2)

            pos_mm = self.get_ball_position_mm()
            if pos_mm and self.trails:
                bx, by = self.trails[0]['pts'][0]
                cv2.putText(display, f"X:{pos_mm[0]} Y:{pos_mm[1]}", (bx + 15, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for t in self.trails:
                if 'physics' in t:
                    if t['physics']['vel'] is not None:
                        bx, by = t['pts'][0]
                        vx, vy = t['physics']['vel']
                        end_x = int(bx + vx * 0.2)
                        end_y = int(by + vy * 0.2)
                        cv2.arrowedLine(display, (bx, by), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

                    if t['physics']['pred_x'] is not None and self.cached_goal_line_y is not None:
                        pred_x = int(t['physics']['pred_x'])
                        gy = self.cached_goal_line_y
                        cv2.line(display, (pred_x, gy - 30), (pred_x, gy + 30), (150, 150, 255), 2)
                        cv2.circle(display, (pred_x, gy), 5, (150, 150, 255), -1)
                        cv2.putText(display, "HIT", (pred_x + 5, gy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)

            cv2.putText(display, f"FPS: {int(self.current_fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if return_rgb:
            return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        if debug:
            self._ensure_sliders_window()
            self._update_settings_from_sliders()
            
            # --- MULTI-VIEW DEBUG ---
            h, w = display.shape[:2]
            if self.raw_frame is not None:
                raw_resized = cv2.resize(self.raw_frame, (w, h))
            else:
                raw_resized = np.zeros_like(display)
                
            if self.current_mask is not None:
                mask_bgr = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR)
                mask_resized = cv2.resize(mask_bgr, (w, h))
            else:
                mask_resized = np.zeros_like(display)
                
            combined = np.hstack([raw_resized, mask_resized, display])
            final_debug = cv2.resize(combined, (0,0), fx=0.6, fy=0.6)
            cv2.imshow("Tracker (Debug)", final_debug)
        else:
            final_display = cv2.resize(display, (0,0), fx=scale, fy=scale)
            cv2.imshow("Tracker", final_display)
            
        return True

    def tune_settings(self, cap_source):
        print("--- TUNING MODE ---")
        print("Press 's' to SAVE settings.")
        print("Press 'q' to QUIT.")
        
        self._ensure_sliders_window()
        
        while True:
            ret, frame = cap_source.read()
            if not ret: 
                print("Error reading frame.")
                break
            
            self.update(frame, velocity_method='poly')
            self.show_feed(debug=True, scale=0.8)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_settings()
            elif key == ord('q'):
                print("Exiting tuner...")
                break
                
        cv2.destroyAllWindows()

    def clear_perspective_calibration(self):
        self.perspective_matrix = None
        self.warped_size = (1280, 720) 
        self.perspective_marker_id = -1
        self.marker_real_size = 0.0
        self._save_calibration()

    def release(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    tracker = BallTracker()
    tracker.set_roi_markers(0, 2, offset_x=50, offset_y=30, goal_offset_y=100)
    tracker.set_origin_markers(0, 2, offset_x_mm=100, offset_y_mm=0)
    
    # --- MODE SELECTION ---
    # mode = "calibrate"
    mode = "tune"

    if mode == "calibrate":
        tracker.calibrate_camera(cap, 6, 8, 1, 0.03, "scale") 
    elif mode == "tune":
        tracker.tune_settings(cap)
        
    cap.release()
import cv2
import numpy as np
import json
import os
import time
from collections import deque

# --- CONFIGURATION ---
BUFFER_SIZE = 32        # Length of the "tail"
MIN_RADIUS = 6          # Minimum size of ball to detect
MAX_DISTANCE = 50       # Max distance (pixels) to link a ball to an existing trail
MAX_DISAPPEARED = 20    # Frames to keep a trail alive if ball is lost
WARMUP_FRAMES = 60      # Frames to skip at startup (let camera adjust light)
SETTINGS_FILE = "hsv_settings.json"

# --- GLOBAL VARIABLES ---
trails = []
next_object_id = 0
frame_count = 0

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Initialize Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

def load_settings():
    default_settings = {
        "h_min": 0, "h_max": 30,
        "s_min": 100, "s_max": 255,
        "v_min": 100, "v_max": 255,
        "use_motion": 1
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return default_settings

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)
    print(f"Settings saved to {SETTINGS_FILE}")

def nothing(x):
    pass

# Setup Window
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 300, 350)

current_settings = load_settings()

cv2.createTrackbar("Hue Min", "Settings", current_settings["h_min"], 179, nothing)
cv2.createTrackbar("Hue Max", "Settings", current_settings["h_max"], 179, nothing)
cv2.createTrackbar("Sat Min", "Settings", current_settings["s_min"], 255, nothing)
cv2.createTrackbar("Sat Max", "Settings", current_settings["s_max"], 255, nothing)
cv2.createTrackbar("Val Min", "Settings", current_settings["v_min"], 255, nothing)
cv2.createTrackbar("Val Max", "Settings", current_settings["v_max"], 255, nothing)
cv2.createTrackbar("Use Motion", "Settings", current_settings["use_motion"], 1, nothing)

cap = cv2.VideoCapture(0)

print("Starting Camera...")
print(f"Warming up for {WARMUP_FRAMES} frames (ignoring input)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 1. PRE-PROCESSING
    frame = cv2.resize(frame, (640, 480))
    
    # FPS Calculation
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Skip processing during warmup to avoid noise
    if frame_count < WARMUP_FRAMES:
        cv2.putText(frame, f"Warming Up: {int(frame_count/WARMUP_FRAMES*100)}%", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Multi-Ball Tracker", frame)
        cv2.waitKey(1)
        continue

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. READ SETTINGS
    h_min = cv2.getTrackbarPos("Hue Min", "Settings")
    h_max = cv2.getTrackbarPos("Hue Max", "Settings")
    s_min = cv2.getTrackbarPos("Sat Min", "Settings")
    s_max = cv2.getTrackbarPos("Sat Max", "Settings")
    v_min = cv2.getTrackbarPos("Val Min", "Settings")
    v_max = cv2.getTrackbarPos("Val Max", "Settings")
    use_motion = cv2.getTrackbarPos("Use Motion", "Settings")

    # 3. MASKING
    color_mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
    
    if use_motion:
        motion_mask = fgbg.apply(blurred)
        final_mask = cv2.bitwise_and(color_mask, color_mask, mask=motion_mask)
    else:
        final_mask = color_mask

    final_mask = cv2.erode(final_mask, None, iterations=2)
    final_mask = cv2.dilate(final_mask, None, iterations=2)

    # 4. DETECTION
    cnts, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_centers = []
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > MIN_RADIUS:
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                current_centers.append(center)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # 5. TRACKING & CLEANUP
    used_centers = set()
    used_trails = set()

    # Match existing trails
    for t_idx, trail in enumerate(trails):
        if len(trail['pts']) == 0:
            continue
        
        last_pos = trail['pts'][0]
        best_dist = MAX_DISTANCE
        best_c_idx = -1

        for c_idx, center in enumerate(current_centers):
            if c_idx in used_centers:
                continue
            dist = np.linalg.norm(np.array(last_pos) - np.array(center))
            if dist < best_dist:
                best_dist = dist
                best_c_idx = c_idx

        if best_c_idx != -1:
            # Found a match: Update trail and reset 'disappeared' counter
            trail['pts'].appendleft(current_centers[best_c_idx])
            trail['disappeared'] = 0
            used_centers.add(best_c_idx)
            used_trails.add(t_idx)
        else:
            # No match: Increment 'disappeared' counter
            trail['disappeared'] += 1

    # Create new trails for unmatched centers
    for c_idx, center in enumerate(current_centers):
        if c_idx not in used_centers:
            new_trail = {
                'id': next_object_id, 
                'pts': deque(maxlen=BUFFER_SIZE),
                'disappeared': 0
            }
            new_trail['pts'].appendleft(center)
            trails.append(new_trail)
            next_object_id += 1

    # Remove dead trails (Filter list)
    trails = [t for t in trails if t['disappeared'] < MAX_DISAPPEARED]

    # 6. DRAWING
    for trail in trails:
        pts = trail['pts']
        if len(pts) < 2:
            continue
            
        np.random.seed(trail['id'])
        color = np.random.randint(0, 255, 3).tolist()
        
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.0)
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    # 7. DISPLAY
    # Draw FPS on frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack([frame, mask_bgr])
    cv2.imshow("Multi-Ball Tracker", stacked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        settings_to_save = {
            "h_min": h_min, "h_max": h_max,
            "s_min": s_min, "s_max": s_max,
            "v_min": v_min, "v_max": v_max,
            "use_motion": use_motion
        }
        save_settings(settings_to_save)

cap.release()
cv2.destroyAllWindows()
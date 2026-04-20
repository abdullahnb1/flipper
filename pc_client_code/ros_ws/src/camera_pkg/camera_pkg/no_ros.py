#!/usr/bin/env python3
import cv2
import time
import copy
import logging

# Assumes these custom packages exist in your python path
from hsv_class import BallTracker
from pico_controller import PicoController

# Setup basic logging to replace ROS logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("RobotController")

class RobotController:
    def __init__(self):
        # --- CONFIGURATION ---
        # Note: You might need to change the camera index (0, 1, 2) depending on your setup
        self.CAMERA_INDEX = 0
        self.SERIAL_PORT = "/dev/ttyACM0"
        
        # --- ROBOT TRAVEL LIMITS ---
        self.LIMIT_MIN_X_MM = -135.0  
        self.LIMIT_MAX_X_MM = 140.0   
        self.ROBOT_WIDTH_MM = 160.0
        
        self.MECH_OFFSET_MM = 0
        self.HIT_DISTANCE_MM = 140
        self.HIT_RATIO = 2
        self.HIT_DIR = -1

        self.JITTER_THRESHOLD_MM = 1.5
        
        # Flipper Logic
        self.FLIPPER_BOUNDARY_X_MM = 140.0 
        self.FLIPPER_OFFSET_MM = 35.0 
        self.SWING_RESET_DELAY = 1
        
        # Prediction
        self.WEIGHT_CURRENT = 1.0
        self.WEIGHT_PRED = 1.0 - self.WEIGHT_CURRENT
        
        # Timing settings
        self.move_rate = 15
        self.hit_rate = 30
        self.move_period = 1.0 / self.move_rate
        self.hit_period = 1.0 / self.hit_rate

        # Track last run times for non-blocking loop
        self.last_move_time = 0.0
        self.last_hit_time = 0.0

        # --- HARDWARE INIT ---
        self.pico = None
        try:
            self.pico = PicoController(port=self.SERIAL_PORT)
            self.pico.start()
            logger.info("Pico connected. Homing...")
            time.sleep(0.5)
            self.pico.home()
            time.sleep(10) # Wait for homing to complete
            self.pico.set_rpm(400)
        except Exception as e:
            logger.error(f"Failed to connect to Pico: {e}")

        # --- STATE VARIABLES ---
        self.swing_triggered = False
        self.last_swing_time = 0.0
        self.last_sent_target = -9999.0
        self.next_hit_direction = 1
        self.hit_degree = 45 # Default initialization
        
        self.latest_ball_pos = None 

        # --- VISION INIT ---
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.tracker.set_roi_markers(1, 3, offset_x=10, offset_y=10, goal_offset_y=50)
        self.tracker.set_origin_markers(1, 7, offset_x_mm=0, offset_y_mm=0)
        
        self.tracker.set_robot_constraints(
            min_x_mm=self.LIMIT_MIN_X_MM, 
            max_x_mm=self.LIMIT_MAX_X_MM, 
            margin_mm=10.0, 
            width_mm=self.ROBOT_WIDTH_MM
        )
        self.tracker.set_flipper_boundary(self.FLIPPER_BOUNDARY_X_MM)

        # Initialize Camera
        self.cap = cv2.VideoCapture(self.CAMERA_INDEX)
        # Optional: Set resolution if needed
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        logger.info(f"System Ready. Limits: [{self.LIMIT_MIN_X_MM}, {self.LIMIT_MAX_X_MM}]")

    def process_frame(self):
        """Captures frame, rotates it, and updates tracker."""
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read from camera")
            return

        try:
            # Rotate 90 CCW as per original logic
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            self.tracker.update(frame, velocity_method='poly_curve')
            self.latest_ball_pos = self.tracker.get_ball_position_mm()
            
            self.tracker.show_feed(debug=False, scale=1)
            
        except Exception as e:
            logger.error(f"Vision Processing Error: {e}")

    def update_movement(self):
        """Logic to calculate target position and move stepper."""
        if self.latest_ball_pos is None: return

        try:
            ball_x_mm, _ = self.latest_ball_pos
            
            # 1. Prediction
            target_x_raw = ball_x_mm 
            trails = self.tracker.get_trails()
            if trails and len(trails) > 0:
                physics = trails[0].get('physics', {})
                pred_x = physics.get('pred_x')
                if pred_x is not None:
                    target_x_raw = (ball_x_mm * self.WEIGHT_CURRENT) + (pred_x * self.WEIGHT_PRED)

            # 2. Flipper Logic (Simple Check - No Hysteresis)
            if target_x_raw < self.FLIPPER_BOUNDARY_X_MM:
                # Ball is on Left -> Move Robot Right (+ Offset) so Left flipper hits
                target_x_raw += self.FLIPPER_OFFSET_MM
                self.hit_degree = 45/self.HIT_RATIO
                self.hit_dwell = 100
                self.next_hit_direction = 1*self.HIT_DIR
            else:
                # Ball is on Right -> Move Robot Left (- Offset) so Right flipper hits
                target_x_raw -= self.FLIPPER_OFFSET_MM
                self.hit_degree = 45/self.HIT_RATIO
                self.hit_dwell = 100
                self.next_hit_direction = -1*self.HIT_DIR

            # 3. SAFETY CLAMP
            safe_x = max(self.LIMIT_MIN_X_MM, min(self.LIMIT_MAX_X_MM, target_x_raw))
            stepper_target = self.MECH_OFFSET_MM + safe_x
            
            # 4. Send Command
            if abs(stepper_target - self.last_sent_target) > self.JITTER_THRESHOLD_MM:
                if self.pico:
                    self.pico.set_target(stepper_target)
                self.last_sent_target = stepper_target

        except Exception as e:
            logger.error(f"Move Logic Error: {e}")

    def update_flipper(self):
        """Logic to check distance and trigger swing."""
        if self.swing_triggered:
            if (time.time() - self.last_swing_time) > self.SWING_RESET_DELAY:
                self.swing_triggered = False
            return 

        if self.latest_ball_pos is None: return

        try:
            _, ball_y_mm = self.latest_ball_pos
            dist_to_goal = abs(ball_y_mm)
            
            if dist_to_goal < self.HIT_DISTANCE_MM:
                if self.pico:
                    self.pico.hit(direction=self.next_hit_direction, swing_deg=self.hit_degree)
                
                self.swing_triggered = True
                self.last_swing_time = time.time()
                logger.info(f"SWING ({self.next_hit_direction})! Dist={dist_to_goal:.1f}mm")

        except Exception as e:
            logger.error(f"Hit Logic Error: {e}")

    def run(self):
        """Main Loop replacing ROS Spin"""
        try:
            while True:
                current_time = time.time()

                # 1. Vision runs as fast as possible (or constrained by camera FPS)
                self.process_frame()

                # 2. Movement Timer Check (30Hz)
                if (current_time - self.last_move_time) >= self.move_period:
                    self.update_movement()
                    self.last_move_time = current_time

                # 3. Hit Timer Check (30Hz)
                if (current_time - self.last_hit_time) >= self.hit_period:
                    self.update_flipper()
                    self.last_hit_time = current_time

                # GUI Event Loop
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("Releasing resources...")
        if self.pico: 
            self.pico.close()
        if self.cap:
            self.cap.release()
        self.tracker.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    bot = RobotController()
    bot.run()
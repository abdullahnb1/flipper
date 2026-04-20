#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import time
import copy
from camera_pkg.hsv_class import BallTracker
from camera_pkg.pico_controller import PicoController

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        
        # --- CONFIGURATION ---
        SERIAL_PORT = "/dev/ttyACM0"
        
        # --- ROBOT TRAVEL LIMITS ---
        # The motor will NEVER go beyond these values
        self.LIMIT_MIN_X_MM = -135.0  
        self.LIMIT_MAX_X_MM = 150.0   
        self.ROBOT_WIDTH_MM = 160.0
        
        self.MECH_OFFSET_MM = 0
        self.HIT_DISTANCE_MM = 140
        
        # Increased threshold slightly to reduce "buzzing" at the center
        self.JITTER_THRESHOLD_MM = 1.5
        
        # Flipper Logic
        self.FLIPPER_BOUNDARY_X_MM = 140.0 
        self.FLIPPER_OFFSET_MM = 30.0 
        self.SWING_RESET_DELAY = 1
        
        # Prediction
        self.WEIGHT_CURRENT = 1.0
        self.WEIGHT_PRED = 1.0 - self.WEIGHT_CURRENT
        
        # Timers
        self.move_rate = 30
        self.hit_rate = 30
        self.move_period = 1/self.move_rate
        self.hit_period = 1/self.hit_rate

        # --- HARDWARE INIT ---
        self.pico = None
        try:
            self.pico = PicoController(port=SERIAL_PORT)
            self.pico.start()
            self.get_logger().info("Pico connected. Homing...")
            time.sleep(0.5)
            self.pico.home()
            time.sleep(10)
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Pico: {e}")

        # --- STATE VARIABLES ---
        self.swing_triggered = False
        self.last_swing_time = 0.0
        self.last_sent_target = -9999.0
        self.next_hit_direction = 1
        
        self.latest_ball_pos = None 

        # --- VISION INIT ---
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.tracker.set_roi_markers(1, 3, offset_x=10, offset_y=10, goal_offset_y=50)
        self.tracker.set_origin_markers(1, 7, offset_x_mm=0, offset_y_mm=0)
        
        # Visualization Limits matches Motor Limits
        self.tracker.set_robot_constraints(
            min_x_mm=self.LIMIT_MIN_X_MM, 
            max_x_mm=self.LIMIT_MAX_X_MM, 
            margin_mm=10.0, 
            width_mm=self.ROBOT_WIDTH_MM
        )
        self.tracker.set_flipper_boundary(self.FLIPPER_BOUNDARY_X_MM)

        # --- ROS SETUP ---
        self.br = CvBridge()
        self.subscriber_ = self.create_subscription(CompressedImage, "camera_capture", self.img_callback, 10) 
        self.create_timer(self.move_period, self.move_callback)
        self.create_timer(self.hit_period, self.hit_callback)
        
        self.get_logger().info(f"System Ready. Limits: [{self.LIMIT_MIN_X_MM}, {self.LIMIT_MAX_X_MM}]")

    def img_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            self.tracker.update(frame, velocity_method='velocity')
            self.latest_ball_pos = self.tracker.get_ball_position_mm()
            
            self.tracker.show_feed(debug=False, scale=1)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Img Callback Error: {e}")

    def move_callback(self):
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
                self.hit_degree = 45
                self.hit_dwell = 100
                self.next_hit_direction = 1
            else:
                # Ball is on Right -> Move Robot Left (- Offset) so Right flipper hits
                target_x_raw -= self.FLIPPER_OFFSET_MM
                self.hit_degree = 60
                self.hit_dwell = 500
                self.next_hit_direction = -1

            # 3. SAFETY CLAMP
            # Ensures target is strictly within [MIN_X, MAX_X]
            safe_x = max(self.LIMIT_MIN_X_MM, min(self.LIMIT_MAX_X_MM, target_x_raw))
            stepper_target = self.MECH_OFFSET_MM + safe_x
            
            # 4. Send Command
            if abs(stepper_target - self.last_sent_target) > self.JITTER_THRESHOLD_MM:
                if self.pico:
                    self.pico.set_target(stepper_target)
                self.last_sent_target = stepper_target

        except Exception as e:
            self.get_logger().error(f"Move Callback Error: {e}")

    def hit_callback(self):
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
                self.get_logger().info(f"SWING ({self.next_hit_direction})! Dist={dist_to_goal:.1f}mm")

        except Exception as e:
            self.get_logger().error(f"Hit Callback Error: {e}")

    def stop_tracker(self):
        if self.pico: self.pico.close()
        self.tracker.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop_tracker()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
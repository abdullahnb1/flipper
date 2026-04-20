#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import time
from camera_pkg.hsv_class import BallTracker
from camera_pkg.pico_controller import PicoController

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        
        # --- CONFIGURATION ---
        SERIAL_PORT = "/dev/ttyACM0"
        
        self.MECH_OFFSET_MM = 0  # Center of travel in stepper coordinates
        
        # Hit Threshold (Distance from Goal Line)
        self.HIT_DISTANCE_MM = 10.0 
        
        # Safety Limits (MM relative to center origin)
        self.MAX_TRAVEL_MM = 420.0 # +/- 150mm
        
        # Jitter Control: Minimum change required to move motor
        self.JITTER_THRESHOLD_MM = 5.0 
        
        # --- HARDWARE INIT ---
        self.pico = PicoController(port=SERIAL_PORT)
        if self.pico.connected:
            self.get_logger().info("Homing...")
            self.pico.home()
        
        # --- STATE VARIABLES ---
        self.swing_triggered = False
        self.last_sent_target = -9999.0 # Initialize far away

        # --- VISION INIT ---
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.tracker.set_roi_markers(1, 3, offset_x=15, offset_y=10, goal_offset_y=30)
        self.tracker.set_origin_markers(1, 7, offset_x_mm=5, offset_y_mm=0)
        
        # Set visual constraints
        self.tracker.set_robot_constraints(travel_mm=self.MAX_TRAVEL_MM, width_mm=50.0)

        self.subscriber_ = self.create_subscription(CompressedImage, "camera_capture", self.img_callback, 10)
        self.br = CvBridge()
        self.get_logger().info("Tracking Mode: ALWAYS FOLLOW (Anti-Jitter Enabled)")

    def img_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            self.tracker.update(frame, velocity_method='poly')
            
            ball_pos = self.tracker.get_ball_position_mm()
            
            # --- LOGIC START ---
            if ball_pos:
                ball_x_mm, ball_y_mm = ball_pos
                
                # 1. ALWAYS TRACK X (Constrained)
                # Clamp target to safe range [-150, +150]
                limit = self.MAX_TRAVEL_MM / 2.0
                safe_x = max(-limit, min(limit, ball_x_mm))
                
                # Convert to Stepper Coordinates
                stepper_target = self.MECH_OFFSET_MM + safe_x
                
                # --- ANTI-JITTER FILTER ---
                # Only move if the difference is greater than the threshold
                if abs(stepper_target - self.last_sent_target) > self.JITTER_THRESHOLD_MM:
                    if self.pico.connected:
                        self.pico.goto(stepper_target)
                    # Update the "last sent" position
                    self.last_sent_target = stepper_target
                
                # 2. TRIGGER HIT (Based on Distance Y)
                dist_to_goal = abs(ball_y_mm)
                
                if dist_to_goal < self.HIT_DISTANCE_MM and not self.swing_triggered:
                    if self.pico.connected:
                        self.pico.hit(direction=1)
                    self.swing_triggered = True
                    self.get_logger().info(f"SWING! Dist={dist_to_goal:.1f}mm")
                
                # Reset Trigger if ball moves away
                if dist_to_goal > (self.HIT_DISTANCE_MM + 50):
                    self.swing_triggered = False
            
            # --- LOGIC END ---

            self.tracker.show_feed(debug=False, scale=2)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def stop_tracker(self):
        if self.pico.connected: self.pico.close()
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
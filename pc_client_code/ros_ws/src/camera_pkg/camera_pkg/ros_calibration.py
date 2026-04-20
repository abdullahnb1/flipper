#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import threading
import time
from camera_pkg.hsv_class import BallTracker

# --- HELPER CLASS: MOCK VIDEO CAPTURE ---
class RosImageAdapter:
    """
    Acts like cv2.VideoCapture but feeds images received from ROS.
    This allows us to use the existing blocking calibrate_camera function
    without rewriting the library.
    """
    def __init__(self):
        self.current_frame = None
        self.new_frame_event = threading.Event()
        self.running = True

    def update_frame(self, frame):
        """Called by ROS callback to update the image."""
        self.current_frame = frame
        self.new_frame_event.set() # Notify reader that data is ready

    def read(self):
        """Called by BallTracker.calibrate_camera."""
        if not self.running:
            return False, None
        
        # Block until a new frame arrives (timeout 2 sec)
        if self.new_frame_event.wait(timeout=2.0):
            self.new_frame_event.clear()
            return True, self.current_frame
        else:
            print("[CalibNode] Waiting for video stream...")
            return False, None

    def release(self):
        self.running = False

class CalibrationNode(Node):
    def __init__(self):
        super().__init__("camera_calibration_node")
        
        # --- CONFIGURATION ---
        self.CHECKERBOARD_ROWS = 6
        self.CHECKERBOARD_COLS = 8
        self.ARUCO_ID = 1
        self.ARUCO_SIZE_M = 0.03
        self.CALIB_MODE = "both" # 'distortion', 'scale', or 'both'
        
        # --- SETUP ---
        self.br = CvBridge()
        self.adapter = RosImageAdapter()
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        
        # Subscribe to same topic as main node
        self.subscriber_ = self.create_subscription(
            CompressedImage, 
            "camera/image_raw/compressed", # camera_capture
            self.img_callback, 
            10
        )
        
        self.get_logger().info("Waiting for camera stream...")
        self.get_logger().info("Controls: Press 'c' to capture, 'q' to quit.")

        # Run calibration in a separate thread so it doesn't block ROS callbacks
        self.calib_thread = threading.Thread(target=self.run_calibration_process)
        self.calib_thread.start()

    def img_callback(self, msg):
        try:
            # 1. Decode
            frame = self.br.compressed_imgmsg_to_cv2(msg)
            
            # 2. Rotate (MUST match main node rotation!)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # 3. Feed to adapter
            self.adapter.update_frame(frame)
            
        except Exception as e:
            self.get_logger().error(f"Img Callback Error: {e}")

    def run_calibration_process(self):
        """
        Runs the blocking calibration routine in a background thread.
        """
        # Give ROS time to connect
        time.sleep(2.0)
        
        self.get_logger().info("Starting Calibration Routine...")
        
        # Note: We pass rotation=0 here because we already rotated the image 
        # in the img_callback. The tracker expects raw-ish frames for this function.
        self.tracker.calibrate_camera(
            cap_source=self.adapter, 
            rows=self.CHECKERBOARD_ROWS, 
            cols=self.CHECKERBOARD_COLS, 
            marker_id=self.ARUCO_ID, 
            marker_size_m=self.ARUCO_SIZE_M, 
            mode=self.CALIB_MODE,
            rotation=0 # Already rotated in callback
        )
        
        self.get_logger().info("Calibration finished/saved.")
        self.adapter.release()
        
        # Kill the node when done
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        cv2.destroyAllWindows()
        # rclpy.shutdown() is handled in the thread or exception

if __name__ == "__main__":
    main()
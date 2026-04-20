#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
from camera_pkg.hsv_class import BallTracker  # Import your class
from camera_pkg.pico_controller import PicoController

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        SERIAL_PORT = "/dev/ttyACM0"
        MECH_OFFSET_MM = 0
        HIT_TIME_THRESHOLD = 0.15
        pico = PicoController(port=SERIAL_PORT)

        if pico.connected:
            pico.home()
            time.sleep(1)

        self.swing_triggered = False

        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.tracker.set_roi_markers(1, 3, offset_x=30, offset_y=20, goal_offset_y=30)
        self.tracker.set_origin_markers(1, 7)

        self.subscriber_ = self.create_subscription(CompressedImage, "camera_capture", self.img_callback, 10)
        self.br = CvBridge()
        self.get_logger().info("Camera subscriber node is initialized.")

    def img_callback(self, msg):
        frame = self.br.compressed_imgmsg_to_cv2(msg)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        self.tracker.update(frame, img_format="bgr", velocity_method="poly")
        balls = self.tracker.get_trails()
        objects = self.tracker.get_objects()
        self.get_logger().info(f"Received new frame:\n Balls: {balls}\n Objects: {objects}\n")
        self.tracker.show_feed(debug=False, scale=1)
        
        # cv2.imshow("camera", frame)
        # cv2.waitKey(1)

    def stop_tracker(self):
        self.get_logger().info("Stopping tracker and releasing windows...")
        self.tracker.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_tracker()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__("camera_publisher")
        self.declare_parameter("fps", 30)
        self.fps = self.get_parameter("fps").value
        self.period = 1/self.fps
        self.timer = self.create_timer(self.period, self.timer_callback)
        self.publisher_ = self.create_publisher(CompressedImage, "camera_capture", 10)

        self.cap = cv2.VideoCapture(0) #This index may change between devices
        self.br = CvBridge()

        self.get_logger().info("Camera publisher node is initialized.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret == True:
            msg = self.br.cv2_to_compressed_imgmsg(frame, dst_format="png")
            self.get_logger().info(f"Sending new frame: {self.fps} fps")
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
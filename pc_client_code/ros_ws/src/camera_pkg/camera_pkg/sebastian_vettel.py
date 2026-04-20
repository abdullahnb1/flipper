#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import time
import io
import copy

# Audio Imports
import pygame
from pydub import AudioSegment

from camera_pkg.hsv_class import BallTracker
from camera_pkg.pico_controller import PicoController

# --- AUDIO CONFIGURATION CLASS ---
class AudioHandler:
    def __init__(self, file_path):
        self.enabled = False
        try:
            pygame.mixer.init()
            full_audio = AudioSegment.from_file(file_path)
            
            # --- DEFINE SLICES (Start Sec, End Sec) ---
            self.sound_approach = self._create_sound(full_audio, 2.5, 5.5)
            self.sound_hit      = self._create_sound(full_audio, 5.5, 7.5)
            self.sound_miss     = self._create_sound(full_audio, 5, 12)
            # NEW: Sound for when ball leaves RoI (e.g. side out)
            self.sound_lost     = self._create_sound(full_audio, 7.0, 9) 
            
            self.enabled = True
            print("[Audio] Slices loaded successfully.")
        except Exception as e:
            print(f"[Audio] Error loading sound: {e}")

    def _create_sound(self, audio_seg, start_sec, end_sec):
        segment = audio_seg[int(start_sec*1000):int(end_sec*1000)]
        wav_io = io.BytesIO()
        segment.export(wav_io, format="wav")
        wav_io.seek(0)
        return pygame.mixer.Sound(wav_io)

    def play_approach(self):
        if self.enabled: self.sound_approach.play()
    def play_hit(self):
        if self.enabled: self.sound_hit.play()
    def play_miss(self):
        if self.enabled: self.sound_miss.play()
    def play_lost(self):
        if self.enabled: self.sound_lost.play()


class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        
        # --- CONFIGURATION ---
        SERIAL_PORT = "/dev/ttyACM0"
        AUDIO_FILE_PATH = "/home/ogan/Music/Sebastian_Vettel_Turkiye_GP_Okay_Karacan_Aman_Aman.mp3"
        
        self.audio = AudioHandler(AUDIO_FILE_PATH)
        
        # Limits
        self.LIMIT_MIN_X_MM = -135.0  
        self.LIMIT_MAX_X_MM = 150.0   
        self.ROBOT_WIDTH_MM = 160.0
        self.MECH_OFFSET_MM = 0
        
        self.HIT_DISTANCE_MM = 140
        self.JITTER_THRESHOLD_MM = 1.5
        self.FLIPPER_BOUNDARY_X_MM = 140.0 
        self.FLIPPER_OFFSET_MM = 30.0 
        self.SWING_RESET_DELAY = 0.5
        
        self.WEIGHT_CURRENT = 1.0
        self.WEIGHT_PRED = 1.0 - self.WEIGHT_CURRENT
        
        # --- HARDWARE INIT ---
        self.pico = None
        try:
            self.pico = PicoController(port=SERIAL_PORT)
            self.pico.start()
            self.get_logger().info("Pico connected. Homing...")
            time.sleep(0.5)
            self.pico.home()
            time.sleep(2) 
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Pico: {e}")

        # --- STATE VARIABLES ---
        self.swing_triggered = False
        self.last_swing_time = 0.0
        self.last_sent_target = -9999.0
        self.next_hit_direction = 1
        
        self.latest_ball_pos = None 
        
        # --- AUDIO STATE FLAGS ---
        self.last_valid_y = None # Tracks Y pos even when ball disappears
        self.audio_approach_played = False
        self.audio_miss_played = False
        self.audio_lost_played = False

        # --- VISION INIT ---
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.tracker.set_roi_markers(1, 3, offset_x=10, offset_y=10, goal_offset_y=50)
        self.tracker.set_origin_markers(1, 7, offset_x_mm=0, offset_y_mm=0)
        
        self.tracker.set_robot_constraints(
            min_x_mm=self.LIMIT_MIN_X_MM, 
            max_x_mm=self.LIMIT_MAX_X_MM, 
            width_mm=self.ROBOT_WIDTH_MM
        )
        self.tracker.set_flipper_boundary(self.FLIPPER_BOUNDARY_X_MM)

        # --- ROS SETUP ---
        self.br = CvBridge()
        self.subscriber_ = self.create_subscription(CompressedImage, "camera_capture", self.img_callback, 10) 
        self.create_timer(1.0/30.0, self.move_callback)
        self.create_timer(1.0/30.0, self.hit_callback)
        
        self.get_logger().info(f"System Ready.")

    def img_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.tracker.update(frame, velocity_method='velocity')
            self.latest_ball_pos = self.tracker.get_ball_position_mm()
            self.tracker.show_feed(debug=False, scale=1)
            cv2.waitKey(1)
        except Exception as e:
            pass

    def check_audio_triggers(self, current_pos):
        """
        Handles Approach, Miss, and Lost ball sounds.
        current_pos: Tuple (x, y) or None
        """
        # --- CASE 1: Ball is Visible ---
        if current_pos is not None:
            ball_y_mm = current_pos[1]
            self.last_valid_y = ball_y_mm # Remember where it was
            
            # Reset "Lost" flag since we see it
            self.audio_lost_played = False

            # Reset Approach/Miss if ball is far away (reset game)
            if ball_y_mm > 800:
                self.audio_approach_played = False
                self.audio_miss_played = False

            # APPROACH TRIGGER (Incoming fast)
            trails = self.tracker.get_trails()
            vel_y = 0
            if trails and len(trails) > 0:
                physics = trails[0].get('physics', {})
                vel = physics.get('vel')
                if vel: vel_y = vel[1]

            if (150 < ball_y_mm < 800) and (vel_y < -10) and not self.audio_approach_played:
                self.audio.play_approach()
                self.audio_approach_played = True

            # MISS TRIGGER (Passed flipper line)
            # -50 ensures it's actually behind the robot
            if ball_y_mm < -50 and not self.swing_triggered and not self.audio_miss_played:
                self.audio.play_miss()
                self.audio_miss_played = True
                self.get_logger().info("Audio: Miss!")

        # --- CASE 2: Ball is LOST (None) ---
        else:
            # We assume "Lost" if we haven't played it yet, and we knew where the ball was recently
            if self.last_valid_y is not None and not self.audio_lost_played:
                
                # Only play "Lost" if the ball was still in the playfield (y > 0)
                # If y < 0, it was a "Miss", not a "Lost" (avoid double sound)
                if self.last_valid_y > 0:
                    self.audio.play_lost()
                    self.get_logger().info("Audio: Lost (Left RoI)")
                
                # Lock flag so it doesn't loop
                self.audio_lost_played = True
                self.audio_approach_played = False # Reset for next ball

    def move_callback(self):
        # 1. Always check audio (handles "Lost" state when pos is None)
        self.check_audio_triggers(self.latest_ball_pos)

        if self.latest_ball_pos is None: return

        try:
            ball_x_mm, _ = self.latest_ball_pos
            
            target_x_raw = ball_x_mm 
            trails = self.tracker.get_trails()
            if trails and len(trails) > 0:
                physics = trails[0].get('physics', {})
                pred_x = physics.get('pred_x')
                if pred_x is not None:
                    target_x_raw = (ball_x_mm * self.WEIGHT_CURRENT) + (pred_x * self.WEIGHT_PRED)

            if target_x_raw < self.FLIPPER_BOUNDARY_X_MM:
                target_x_raw += self.FLIPPER_OFFSET_MM
                self.next_hit_direction = 1
            else:
                target_x_raw -= self.FLIPPER_OFFSET_MM
                self.next_hit_direction = -1

            safe_x = max(self.LIMIT_MIN_X_MM, min(self.LIMIT_MAX_X_MM, target_x_raw))
            stepper_target = self.MECH_OFFSET_MM + safe_x
            
            if abs(stepper_target - self.last_sent_target) > self.JITTER_THRESHOLD_MM:
                if self.pico:
                    self.pico.set_target(stepper_target)
                self.last_sent_target = stepper_target

        except Exception as e:
            self.get_logger().error(f"Move Error: {e}")

    def hit_callback(self):
        if self.swing_triggered:
            if (time.time() - self.last_swing_time) > self.SWING_RESET_DELAY:
                self.swing_triggered = False
            return 

        if self.latest_ball_pos is None: return

        try:
            _, ball_y_mm = self.latest_ball_pos
            if abs(ball_y_mm) < self.HIT_DISTANCE_MM:
                if self.pico:
                    self.pico.hit(direction=self.next_hit_direction)
                
                self.audio.play_hit()
                
                self.swing_triggered = True
                self.last_swing_time = time.time()
                self.get_logger().info(f"SWING ({self.next_hit_direction})")

        except Exception as e:
            pass

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
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import threading
import time
import json
import os

# Import your modules
from hsv_class import BallTracker
from pico_controller import PicoController

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Mission Control")
        self.root.geometry("1400x900")
        
        # --- STATE ---
        self.pico = None
        self.cap = None
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.running = True
        self.tracking_enabled = False
        self.swing_triggered = False
        self.warp_enabled = True # New toggle
        
        # Default Params
        self.params = {
            "video_source": 0,
            "rotation": 0, 
            "serial_port": "/dev/ttyACM0",
            "mech_offset": 200.0,
            "hit_dist": 150.0,
            "travel_limit": 300.0,
            "roi_id1": 1, "roi_id2": 3,
            "origin_id1": 1, "origin_id2": 7
        }
        self.load_gui_settings()

        # --- LAYOUT ---
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel (Video)
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Live Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with black background for letterboxing
        self.canvas = tk.Canvas(self.video_frame, bg="#202020")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right Panel (Controls)
        self.control_panel = ttk.Frame(self.main_frame, width=400)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        self._build_connection_panel()
        self._build_motor_panel()
        self._build_calibration_panel()
        self._build_log_panel()
        
        self.btn_estop = tk.Button(self.control_panel, text="EMERGENCY STOP", bg="red", fg="white", 
                                   font=("Arial", 14, "bold"), command=self.emergency_stop)
        self.btn_estop.pack(fill=tk.X, pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_video_loop()
        self.update_log_loop()

    # --- GUI BUILDERS ---
    def _build_connection_panel(self):
        frame = ttk.LabelFrame(self.control_panel, text="Hardware Connection")
        frame.pack(fill=tk.X, pady=5)
        
        # Camera Source
        ttk.Label(frame, text="Cam Idx:").grid(row=0, column=0, padx=2)
        self.ent_cam = ttk.Entry(frame, width=3)
        self.ent_cam.insert(0, str(self.params["video_source"]))
        self.ent_cam.grid(row=0, column=1)
        
        # Rotation Dropdown
        ttk.Label(frame, text="Rot:").grid(row=0, column=2, padx=2)
        self.combo_rot = ttk.Combobox(frame, values=["0", "90", "180", "270"], width=5, state="readonly")
        self.combo_rot.set(str(self.params.get("rotation", 0)))
        self.combo_rot.grid(row=0, column=3)
        self.combo_rot.bind("<<ComboboxSelected>>", self._save_rotation)

        self.btn_cam = ttk.Button(frame, text="Start Cam", command=self.toggle_camera)
        self.btn_cam.grid(row=0, column=4, padx=5)
        
        # Pico Port
        ttk.Label(frame, text="Port:").grid(row=1, column=0, columnspan=2, sticky="e")
        self.ent_port = ttk.Entry(frame, width=15)
        self.ent_port.insert(0, self.params["serial_port"])
        self.ent_port.grid(row=1, column=2, columnspan=2)
        self.btn_pico = ttk.Button(frame, text="Connect", command=self.toggle_pico)
        self.btn_pico.grid(row=1, column=4, padx=5)
        
        # Status LEDs
        self.lbl_cam_status = ttk.Label(frame, text="CAM: OFF", foreground="red", font=("Arial", 10, "bold"))
        self.lbl_cam_status.grid(row=2, column=0, columnspan=5, pady=2)
        self.lbl_pico_status = ttk.Label(frame, text="PICO: OFF", foreground="red", font=("Arial", 10, "bold"))
        self.lbl_pico_status.grid(row=3, column=0, columnspan=5, pady=2)

    def _build_motor_panel(self):
        frame = ttk.LabelFrame(self.control_panel, text="Motor Control")
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame, text="HOME AXIS", command=self.cmd_home).pack(fill=tk.X, padx=5, pady=5)
        
        # Jogging
        jog_frame = ttk.Frame(frame)
        jog_frame.pack(fill=tk.X, pady=2)
        ttk.Button(jog_frame, text="<< -50", command=lambda: self.cmd_move(-50)).pack(side=tk.LEFT, expand=True)
        ttk.Button(jog_frame, text="-10", command=lambda: self.cmd_move(-10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(jog_frame, text="+10", command=lambda: self.cmd_move(10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(jog_frame, text="+50 >>", command=lambda: self.cmd_move(50)).pack(side=tk.LEFT, expand=True)
        
        # Manual Hit
        hit_frame = ttk.Frame(frame)
        hit_frame.pack(fill=tk.X, pady=5)
        ttk.Label(hit_frame, text="Swing Deg:").pack(side=tk.LEFT)
        self.ent_angle = ttk.Entry(hit_frame, width=5)
        self.ent_angle.insert(0, "45")
        self.ent_angle.pack(side=tk.LEFT)
        ttk.Button(hit_frame, text="FIRE!", command=self.cmd_hit).pack(side=tk.LEFT, padx=10)
        
        # Auto-Tracking Checkbox
        self.chk_track = ttk.Checkbutton(frame, text="Enable Auto-Tracking", command=self.toggle_tracking)
        self.chk_track.pack(pady=5)

    def _build_calibration_panel(self):
        frame = ttk.LabelFrame(self.control_panel, text="Calibration & Display")
        frame.pack(fill=tk.X, pady=5)
        
        # Warp Toggle (CRITICAL FIX)
        self.chk_warp = ttk.Checkbutton(frame, text="Show Warped (Planar) View", command=self.toggle_warp)
        self.chk_warp.state(['!alternate'])
        self.chk_warp.state(['selected']) # Default ON
        self.chk_warp.pack(pady=2)
        
        # IDs Input
        id_frame = ttk.Frame(frame)
        id_frame.pack(fill=tk.X)
        ttk.Label(id_frame, text="ROI IDs:").grid(row=0, column=0)
        self.ent_roi1 = ttk.Entry(id_frame, width=3)
        self.ent_roi1.insert(0, self.params["roi_id1"])
        self.ent_roi1.grid(row=0, column=1)
        self.ent_roi2 = ttk.Entry(id_frame, width=3)
        self.ent_roi2.insert(0, self.params["roi_id2"])
        self.ent_roi2.grid(row=0, column=2)
        
        ttk.Label(id_frame, text="Origin IDs:").grid(row=1, column=0)
        self.ent_org1 = ttk.Entry(id_frame, width=3)
        self.ent_org1.insert(0, self.params["origin_id1"])
        self.ent_org1.grid(row=1, column=1)
        self.ent_org2 = ttk.Entry(id_frame, width=3)
        self.ent_org2.insert(0, self.params["origin_id2"])
        self.ent_org2.grid(row=1, column=2)
        
        ttk.Button(frame, text="Calibrate Arena (ROI/Origin)", command=self.apply_calibration).pack(fill=tk.X, pady=5)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Advanced Calibrations
        ttk.Button(frame, text="Run Lens Calibration (Scale)", command=self.run_lens_cal).pack(fill=tk.X)
        ttk.Button(frame, text="Run Perspective Calibration", command=self.run_persp_cal).pack(fill=tk.X)
        ttk.Button(frame, text="RESET Perspective", command=self.reset_perspective).pack(fill=tk.X)

    def _build_log_panel(self):
        frame = ttk.LabelFrame(self.control_panel, text="System Log")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.txt_log = tk.Text(frame, height=8, state="disabled", bg="#f0f0f0", font=("Consolas", 8))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    # --- LOGIC ---
    def log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

    def _save_rotation(self, event=None):
        self.params["rotation"] = int(self.combo_rot.get())
        self.save_gui_settings()
        # Warn user
        self.log(f"Rotation set to {self.params['rotation']}. Recalibrate Perspective if changed!")

    def toggle_warp(self):
        self.warp_enabled = "selected" in self.chk_warp.state()
        if not self.warp_enabled:
            # Temporarily disable perspective in tracker so we see raw rotated view
            self.tracker.perspective_matrix = None
            self.log("Warp Disabled - Showing Raw Rotated Feed")
        else:
            # Reload from file to restore
            self.tracker._load_calibration()
            self.log("Warp Enabled - Reloaded Matrix")

    def toggle_camera(self):
        if self.cap is None:
            src = self.ent_cam.get()
            try: src = int(src)
            except: pass
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if self.cap.isOpened():
                self.lbl_cam_status.config(text="CAM: ON", foreground="green")
                self.btn_cam.config(text="Stop Cam")
                self.log("Camera Started")
            else:
                self.cap = None
                self.log("Error: Camera not found")
        else:
            self.cap.release()
            self.cap = None
            self.lbl_cam_status.config(text="CAM: OFF", foreground="red")
            self.btn_cam.config(text="Start Cam")
            self.canvas.delete("all")

    def toggle_pico(self):
        if self.pico is None:
            port = self.ent_port.get()
            self.pico = PicoController(port=port)
            if self.pico.connected:
                self.lbl_pico_status.config(text="PICO: CONNECTED", foreground="green")
                self.btn_pico.config(text="Disconnect")
                self.log("Pico Connected")
            else:
                self.pico = None
                self.log("Error: Pico connection failed")
        else:
            self.pico.close()
            self.pico = None
            self.lbl_pico_status.config(text="PICO: OFF", foreground="red")
            self.btn_pico.config(text="Connect Pico")

    def cmd_home(self):
        if self.pico:
            self.log("Homing...")
            self.pico.home()

    def cmd_move(self, mm):
        if self.pico:
            self.pico.send_command("MOVE", mm)

    def cmd_hit(self):
        if self.pico:
            angle = float(self.ent_angle.get())
            self.pico.hit(direction=1, swing_deg=angle)

    def toggle_tracking(self):
        if "selected" in self.chk_track.state():
            self.tracking_enabled = True
            self.log("AUTO-TRACKING ENABLED")
        else:
            self.tracking_enabled = False
            self.log("Auto-Tracking Disabled")

    def emergency_stop(self):
        self.tracking_enabled = False
        if self.pico:
            # Send stop command if firmware supports it, otherwise close port
            self.pico.close() 
            self.pico = None
        self.lbl_pico_status.config(text="ESTOP TRIPPED", foreground="red")
        self.log("!!! EMERGENCY STOP !!!")

    def apply_calibration(self):
        r1 = int(self.ent_roi1.get())
        r2 = int(self.ent_roi2.get())
        o1 = int(self.ent_org1.get())
        o2 = int(self.ent_org2.get())
        
        self.tracker.set_roi_markers(r1, r2, offset_x=30, offset_y=20, goal_offset_y=30)
        self.tracker.set_origin_markers(o1, o2, offset_x_mm=0, offset_y_mm=0)
        
        self.params.update({"roi_id1": r1, "roi_id2": r2, "origin_id1": o1, "origin_id2": o2})
        self.save_gui_settings()
        self.log("Arena Calibration Applied")

    def run_lens_cal(self):
        if self.cap:
            self.log("Starting Lens Calibration...")
            threading.Thread(target=self.tracker.calibrate_camera, 
                             args=(self.cap, 6, 8, 1, 0.03, "scale", 0)).start()

    def run_persp_cal(self):
        if self.cap:
            rot = int(self.combo_rot.get())
            self.log(f"Starting Perspective Calibration (Rot={rot})...")
            # Pass rotation so calibration window is upright
            threading.Thread(target=self.tracker.calibrate_camera, 
                             args=(self.cap, 6, 8, 1, 0.03, "perspective", rot)).start()

    def reset_perspective(self):
        self.tracker.clear_perspective_calibration()
        self.log("Perspective Cleared")

    def update_video_loop(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                rot = int(self.combo_rot.get())
                
                # 1. Update Tracking (Includes Undistort -> Rotate -> Warp)
                self.tracker.update(frame, velocity_method='poly', rotation=rot)
                
                # 2. Logic
                if self.tracking_enabled and self.pico and self.pico.connected:
                    self._run_tracking_logic()

                # 3. Get Processed Image
                img_rgb = self.tracker.show_feed(debug=False, return_rgb=True)
                
                # 4. Display with Correct Aspect Ratio
                if img_rgb is not None:
                    c_width = self.canvas.winfo_width()
                    c_height = self.canvas.winfo_height()
                    
                    if c_width > 10 and c_height > 10:
                        img_pil = PIL.Image.fromarray(img_rgb)
                        
                        # Calculate Aspect Ratio to prevent "Messed Up" stretching
                        img_w, img_h = img_pil.size
                        ratio = min(c_width/img_w, c_height/img_h)
                        new_w = int(img_w * ratio)
                        new_h = int(img_h * ratio)
                        
                        img_pil = img_pil.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                        img_tk = PIL.ImageTk.PhotoImage(image=img_pil)
                        
                        # Center image
                        pos_x = (c_width - new_w) // 2
                        pos_y = (c_height - new_h) // 2
                        
                        self.canvas.delete("all") # Clear previous frame
                        self.canvas.create_image(pos_x, pos_y, image=img_tk, anchor=tk.NW)
                        self.canvas.image = img_tk

        self.root.after(30, self.update_video_loop)

    def _run_tracking_logic(self):
        # ... logic identical to before ...
        ball_pos = self.tracker.get_ball_position_mm()
        if ball_pos:
            ball_x, ball_y = ball_pos
            mech_offset = float(self.params["mech_offset"])
            limit = float(self.params["travel_limit"]) / 2.0
            
            safe_x = max(-limit, min(limit, ball_x))
            target = mech_offset + safe_x
            
            # Anti-Jitter logic from previous step could be added here
            self.pico.goto(target)
            
            hit_dist = float(self.params["hit_dist"])
            dist = abs(ball_y)
            if dist < hit_dist and not self.swing_triggered:
                self.pico.hit(1)
                self.swing_triggered = True
                self.log(f"Hit! {dist:.0f}mm")
            
            if dist > (hit_dist + 50):
                self.swing_triggered = False

    def update_log_loop(self):
        if self.pico:
            while not self.pico.rx_queue.empty():
                msg = self.pico.rx_queue.get()
        self.root.after(100, self.update_log_loop)

    def save_gui_settings(self):
        with open("gui_settings.json", "w") as f:
            json.dump(self.params, f)

    def load_gui_settings(self):
        if os.path.exists("gui_settings.json"):
            try:
                with open("gui_settings.json", "r") as f:
                    self.params.update(json.load(f))
            except: pass

    def on_close(self):
        self.running = False
        if self.pico: self.pico.close()
        if self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()
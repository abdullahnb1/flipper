import tkinter as tk
from tkinter import ttk, scrolledtext
import cv2
import PIL.Image, PIL.ImageTk
import sys
import json
import time
from ball_tracker_for_gui import BallTracker
from pico_controller import PicoController
import os

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
    def write(self, str):
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", str, (self.tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except: pass
    def flush(self): pass

class RobotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Little Daisies - Ultimate Control")
        self.geometry("1600x900")
        
        self.is_tracking = False
        self.is_emergency = False
        self.cap = None
        self.pico = None
        self.tracker = BallTracker()
        
        self.last_sent_target = -9999.0
        self.swing_triggered = False
        self.last_swing_time = 0
        self.next_hit_dir = 1
        
        # --- PARAMETERS ---
        self.params = {
            # Connections
            "cam_index": tk.IntVar(value=2),
            "rotation": tk.IntVar(value=90), 
            "pico_port": tk.StringVar(value="/dev/ttyACM0"),
            
            # Robot Config
            "limit_min_x": tk.DoubleVar(value=-135.0),
            "limit_max_x": tk.DoubleVar(value=150.0),
            "robot_width": tk.DoubleVar(value=160.0),
            "safe_margin": tk.DoubleVar(value=10.0),
            "offset_x": tk.DoubleVar(value=0.0),
            "offset_y": tk.DoubleVar(value=0.0),
            "pixels_per_meter": tk.DoubleVar(value=self.tracker.pixels_per_meter), # New!
            
            # Game Logic
            "hit_dist": tk.DoubleVar(value=140.0),
            "flipper_bound": tk.DoubleVar(value=140.0),
            "flipper_offset": tk.DoubleVar(value=25.0),
            "jitter": tk.DoubleVar(value=1.5),
            "weight_curr": tk.DoubleVar(value=1.0),
            "weight_pred": tk.DoubleVar(value=0.0),
            
            # Manual / Specific Angles
            "motor_target": tk.DoubleVar(value=0.0),
            "servo_target": tk.DoubleVar(value=0.0),
            "left_hit_angle": tk.DoubleVar(value=45.0),
            "right_hit_angle": tk.DoubleVar(value=45.0),
            "manual_hit_dir": tk.IntVar(value=1),
            
            # Vision
            "h_min": tk.IntVar(value=0), "h_max": tk.IntVar(value=30),
            "s_min": tk.IntVar(value=100), "s_max": tk.IntVar(value=255),
            "v_min": tk.IntVar(value=100), "v_max": tk.IntVar(value=255),
            "debug_view": tk.BooleanVar(value=False)
        }
        self._load_gui_defaults()

        self._build_ui()
        sys.stdout = TextRedirector(self.txt_log, "stdout")
        sys.stderr = TextRedirector(self.txt_log, "stderr")
        print("[System] Dashboard Initialized.")
        self.update_loop()

    def _build_ui(self):
        split = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5, bg="#333")
        split.pack(fill=tk.BOTH, expand=True)
        
        self.frm_video = ttk.Frame(split, width=900)
        split.add(self.frm_video, stretch="always")
        self.lbl_video = ttk.Label(self.frm_video, text="NO SIGNAL", anchor="center", background="black", foreground="white")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        frm_ctrl = ttk.Frame(split, width=600)
        split.add(frm_ctrl, stretch="never")
        
        # Connection
        cxn_frame = ttk.LabelFrame(frm_ctrl, text="Hardware")
        cxn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(cxn_frame, text="Cam:").grid(row=0, column=0)
        ttk.Entry(cxn_frame, textvariable=self.params["cam_index"], width=3).grid(row=0, column=1)
        ttk.Label(cxn_frame, text="Rot:").grid(row=0, column=2)
        ttk.OptionMenu(cxn_frame, self.params["rotation"], 0, 0, 90, 180, 270).grid(row=0, column=3)
        ttk.Button(cxn_frame, text="Init Cam", command=self.start_camera).grid(row=0, column=4, padx=5)
        ttk.Label(cxn_frame, text="Port:").grid(row=1, column=0)
        ttk.Entry(cxn_frame, textvariable=self.params["pico_port"], width=10).grid(row=1, column=1, columnspan=2)
        self.btn_pico = ttk.Button(cxn_frame, text="Connect Pico", command=self.toggle_pico)
        self.btn_pico.grid(row=1, column=3, columnspan=2, padx=5, pady=2)

        # Game State
        game_frame = ttk.LabelFrame(frm_ctrl, text="Game Status")
        game_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_track = ttk.Button(game_frame, text="START TRACKING", command=self.toggle_tracking, state="disabled")
        self.btn_track.pack(fill=tk.X, padx=5, pady=2)
        self.btn_estop = tk.Button(game_frame, text="EMERGENCY STOP", bg="red", fg="white", font=("Arial", 12, "bold"), command=self.toggle_estop)
        self.btn_estop.pack(fill=tk.X, padx=5, pady=5)

        # Tabs
        nb = ttk.Notebook(frm_ctrl)
        nb.pack(fill=tk.BOTH, expand=True, padx=5)
        
        t_robot = ttk.Frame(nb); nb.add(t_robot, text="Robot")
        self._add_slider(t_robot, "Min X (mm)", "limit_min_x", -300, 0)
        self._add_slider(t_robot, "Max X (mm)", "limit_max_x", 0, 300)
        self._add_slider(t_robot, "Origin X Off (mm)", "offset_x", -100, 100)
        self._add_slider(t_robot, "Origin Y Off (mm)", "offset_y", -100, 100)
        self._add_slider(t_robot, "Scale (PPM)", "pixels_per_meter", 0, 1000) # IMPORTANT: MANUAL SCALE
        self._add_slider(t_robot, "Robot Width (mm)", "robot_width", 50, 300)
        self._add_slider(t_robot, "Safe Margin (mm)", "safe_margin", 0, 50)
        
        t_game = ttk.Frame(nb); nb.add(t_game, text="Game")
        self._add_slider(t_game, "Pred Weight (0=All Pred)", "weight_pred", 0.0, 1.0, 0.1)
        self._add_slider(t_game, "Curr Weight (1=All Real)", "weight_curr", 0.0, 1.0, 0.1)
        self._add_slider(t_game, "Hit Dist Y (mm)", "hit_dist", 50, 300)
        self._add_slider(t_game, "Flip Bound X (mm)", "flipper_bound", 0, 200)
        self._add_slider(t_game, "Flip Offset X (mm)", "flipper_offset", 0, 100)
        self._add_slider(t_game, "Jitter Thresh (mm)", "jitter", 0, 10, 0.1)
        
        t_man = ttk.Frame(nb); nb.add(t_man, text="Manual")
        mf1 = ttk.LabelFrame(t_man, text="Motor"); mf1.pack(fill=tk.X, padx=5)
        ttk.Entry(mf1, textvariable=self.params["motor_target"], width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(mf1, text="Go X", command=lambda: self._send_pico("GOTO", self.params["motor_target"].get())).pack(side=tk.LEFT)
        ttk.Button(mf1, text="Home", command=lambda: self._send_pico("HOME")).pack(side=tk.RIGHT, padx=5)
        
        mf2 = ttk.LabelFrame(t_man, text="Servo / Hit Angles"); mf2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mf2, text="L Angle:").grid(row=0, column=0)
        ttk.Entry(mf2, textvariable=self.params["left_hit_angle"], width=5).grid(row=0, column=1)
        ttk.Label(mf2, text="R Angle:").grid(row=0, column=2)
        ttk.Entry(mf2, textvariable=self.params["right_hit_angle"], width=5).grid(row=0, column=3)
        ttk.Button(mf2, text="< HIT L", command=lambda: self.manual_hit(1)).grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(mf2, text="HIT R >", command=lambda: self.manual_hit(-1)).grid(row=1, column=2, columnspan=2, pady=5)
        ttk.Label(mf2, text="Servo Abs:").grid(row=2, column=0)
        ttk.Entry(mf2, textvariable=self.params["servo_target"], width=5).grid(row=2, column=1)
        ttk.Button(mf2, text="Set", command=lambda: self._send_pico("SERVO", self.params["servo_target"].get())).grid(row=2, column=2)

        t_vis = ttk.Frame(nb); nb.add(t_vis, text="Vision")
        ttk.Checkbutton(t_vis, text="Show Debug Mask", variable=self.params["debug_view"]).pack(anchor="w")
        self._add_slider(t_vis, "Hue Min", "h_min", 0, 179)
        self._add_slider(t_vis, "Hue Max", "h_max", 0, 179)
        self._add_slider(t_vis, "Sat Min", "s_min", 0, 255)
        self._add_slider(t_vis, "Sat Max", "s_max", 0, 255)
        self._add_slider(t_vis, "Val Min", "v_min", 0, 255)
        self._add_slider(t_vis, "Val Max", "v_max", 0, 255)

        self.txt_log = scrolledtext.ScrolledText(frm_ctrl, height=8, bg="black", fg="#00ff00", font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Button(frm_ctrl, text="Save Settings", command=self.save_settings).pack(fill=tk.X, padx=5, pady=5)

    def _add_slider(self, parent, label, key, min_v, max_v, res=1):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f, text=label, width=15).pack(side=tk.LEFT)
        tk.Scale(f, from_=min_v, to=max_v, variable=self.params[key], resolution=res, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(f, textvariable=self.params[key], width=5).pack(side=tk.RIGHT)

    def start_camera(self):
        if self.cap: self.cap.release()
        try:
            idx = self.params["cam_index"].get()
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                print(f"[Sys] Camera {idx} Started.")
                self.btn_track.config(state="normal")
            else:
                print(f"[Error] Failed to open Camera {idx}")
        except Exception as e: print(f"[Error] Cam Init: {e}")

    def toggle_pico(self):
        if self.pico and self.pico.connected:
            self.pico.close()
            self.pico = None
            self.btn_pico.config(text="Connect Pico")
            print("[Sys] Pico Disconnected.")
        else:
            try:
                print(f"[Sys] Connecting to {self.params['pico_port'].get()}...")
                self.pico = PicoController(port=self.params['pico_port'].get())
                self.pico.start()
                time.sleep(0.5)
                self.pico.home()
                self.btn_pico.config(text="Disconnect Pico")
                print("[Sys] Pico Ready.")
            except Exception as e:
                print(f"[Error] Pico Connection: {e}")

    def manual_hit(self, direction):
        angle = self.params["left_hit_angle"].get() if direction == 1 else self.params["right_hit_angle"].get()
        if self.pico and self.pico.connected:
            self.pico.hit(direction) # Update this line if PicoController accepts angle
            print(f"[Manual] Hit {direction} (Angle {angle})")
        else:
            print("[Error] Pico not connected")

    def _send_pico(self, type, val=None):
        if not self.pico or not self.pico.connected: 
            print("[Error] No Pico"); return
        if type == "GOTO": self.pico.set_target(float(val))
        elif type == "HOME": self.pico.home()
        elif type == "SERVO": 
            print(f"[Sys] Servo -> {val}") 
            # self.pico.set_servo(val) # Implement in PicoController

    def toggle_estop(self):
        self.is_emergency = not self.is_emergency
        if self.is_emergency:
            self.btn_estop.config(text="RESUME", bg="green")
            self.is_tracking = False
            self.btn_track.config(state="disabled")
            if self.pico: self.pico.set_target(0)
            print("[URGENT] EMERGENCY STOP")
        else:
            self.btn_estop.config(text="EMERGENCY STOP", bg="red")
            self.btn_track.config(state="normal")
            print("[Sys] Resumed.")

    def toggle_tracking(self):
        self.is_tracking = not self.is_tracking
        self.btn_track.config(text="STOP TRACKING" if self.is_tracking else "START TRACKING")
        print(f"[Sys] Tracking: {self.is_tracking}")

    def save_settings(self):
        data = {k: v.get() for k, v in self.params.items()}
        try:
            with open("gui_config.json", "w") as f: json.dump(data, f)
            self.tracker.settings["h_min"] = data["h_min"]
            self.tracker.settings["h_max"] = data["h_max"]
            self.tracker.settings["s_min"] = data["s_min"]
            self.tracker.settings["s_max"] = data["s_max"]
            self.tracker.settings["v_min"] = data["v_min"]
            self.tracker.settings["v_max"] = data["v_max"]
            self.tracker.save_settings()
            print("[Sys] All Settings Saved.")
        except Exception as e: print(f"[Error] Save: {e}")

    def _load_gui_defaults(self):
        if os.path.exists("gui_config.json"):
            try:
                with open("gui_config.json", "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k in self.params: self.params[k].set(v)
            except: pass

    def update_loop(self):
        # Update Tracker from GUI
        self.tracker.settings.update({
            "h_min": self.params["h_min"].get(), "h_max": self.params["h_max"].get(),
            "s_min": self.params["s_min"].get(), "s_max": self.params["s_max"].get(),
            "v_min": self.params["v_min"].get(), "v_max": self.params["v_max"].get()
        })
        self.tracker.set_origin_markers(0, 0, offset_x_mm=self.params["offset_x"].get(), offset_y_mm=self.params["offset_y"].get())
        self.tracker.set_robot_constraints(
            self.params["limit_min_x"].get(), self.params["limit_max_x"].get(),
            self.params["robot_width"].get(), self.params["safe_margin"].get()
        )
        self.tracker.set_flipper_boundary(self.params["flipper_bound"].get())
        # MANUAL SCALE OVERRIDE (For "lines not displayed" fix)
        self.tracker.pixels_per_meter = self.params["pixels_per_meter"].get()

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                r = self.params["rotation"].get()
                if r == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif r == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif r == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                self.tracker.update(frame, velocity_method='poly')
                
                if self.is_tracking and not self.is_emergency:
                    self._run_game_logic()
                
                rgb_frame = self.tracker.get_annotated_frame(debug_view=self.params["debug_view"].get())
                
                if rgb_frame is not None:
                    win_w = self.lbl_video.winfo_width()
                    win_h = self.lbl_video.winfo_height()
                    if win_w > 10 and win_h > 10:
                        ih, iw, _ = rgb_frame.shape
                        scale = min(win_w/iw, win_h/ih)
                        nw, nh = int(iw*scale), int(ih*scale)
                        rgb_frame = cv2.resize(rgb_frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    
                    img = PIL.Image.fromarray(rgb_frame)
                    imgtk = PIL.ImageTk.PhotoImage(image=img)
                    self.lbl_video.imgtk = imgtk 
                    self.lbl_video.configure(image=imgtk, text="")

        self.after(20, self.update_loop)

    def _run_game_logic(self):
        pos = self.tracker.get_ball_position_mm()
        if not pos: return
        bx, by = pos
        
        w_curr = self.params["weight_curr"].get()
        w_pred = self.params["weight_pred"].get()
        tx = bx
        trails = self.tracker.trails
        if trails and trails[0].get('physics', {}).get('pred_x'):
            tx = (bx * w_curr) + (trails[0]['physics']['pred_x'] * w_pred)
            
        bound = self.params["flipper_bound"].get()
        offset = self.params["flipper_offset"].get()
        
        # NO HYSTERESIS
        if tx < bound:
            tx += offset; self.next_hit_dir = 1
        else:
            tx -= offset; self.next_hit_dir = -1
            
        safe_x = max(self.params["limit_min_x"].get(), min(self.params["limit_max_x"].get(), tx))
        
        if abs(safe_x - self.last_sent_target) > self.params["jitter"].get():
            if self.pico: self.pico.set_target(safe_x)
            self.last_sent_target = safe_x
            
        if abs(by) < self.params["hit_dist"].get():
            if not self.swing_triggered:
                if self.pico: self.pico.hit(self.next_hit_dir)
                self.swing_triggered = True
                self.last_swing_time = time.time()
                print(f"[Game] Hit {self.next_hit_dir}")
        elif (time.time() - self.last_swing_time) > 0.5:
            self.swing_triggered = False

if __name__ == "__main__":
    app = RobotGUI()
    app.mainloop()
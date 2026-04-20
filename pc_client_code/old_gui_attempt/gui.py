import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
import json
import os

# --- IMPORTS ---
from ball_tracker import BallTracker
from pico_controller import PicoController

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Mission Control")
        self.root.geometry("600x950") 
        
        self.pico = None
        self.cap = None
        self.tracker = BallTracker(settings_file="hsv_settings.json")
        self.running = True
        
        # --- STATE ---
        self.tracking_enabled = False
        self.estop_active = False
        self.swing_triggered = False
        self.last_sent_target = -9999.0
        
        # --- PARAMETERS (Default) ---
        self.params = {
            # Hardware
            "video_source": 0,
            "req_width": 640,
            "req_height": 480,
            "rotation": 0,
            "serial_port": "/dev/ttyACM0",
            
            # Physics / Game
            "mech_offset": 0.0,    
            "hit_dist": 150.0,     
            "travel_limit": 300.0, 
            "jitter_th": 5.0,      
            "rpm_fast": 120.0,     # Default RPM
            
            # ROI / Arena Setup
            "roi_id1": 1, "roi_id2": 3, 
            "roi_off_x": 30, "roi_off_y": 20, "goal_off": 30,
            
            # Origin Setup
            "org_id1": 1, "org_id2": 7, 
            "org_off_x": 0, "org_off_y": 0,
            
            "rob_width": 50.0, "safe_margin": 10.0
        }
        
        # Load settings
        self.load_settings()
        self._push_params_to_tracker()

        # --- LAYOUT ---
        self._build_header(root)
        
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_run = ttk.Frame(self.tabs)
        self.tab_arena = ttk.Frame(self.tabs)
        self.tab_cam = ttk.Frame(self.tabs)
        
        self.tabs.add(self.tab_run, text="  RUN GAME  ")
        self.tabs.add(self.tab_arena, text="  ARENA SETUP  ")
        self.tabs.add(self.tab_cam, text="  CAMERA CALIB  ")
        
        self._build_tab_run(self.tab_run)
        self._build_tab_arena(self.tab_arena)
        self._build_tab_cam(self.tab_cam)
        
        self._build_footer(root)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start Loops
        self.update_video_loop()
        self.update_log_loop()

    # =================================================================
    #   GUI BUILDERS
    # =================================================================
    
    def _build_header(self, parent):
        f = ttk.LabelFrame(parent, text="Hardware Connection", padding=5)
        f.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1: Camera
        f1 = ttk.Frame(f); f1.pack(fill=tk.X)
        ttk.Label(f1, text="Cam ID:").pack(side=tk.LEFT)
        self.ent_cam = ttk.Entry(f1, width=3); self.ent_cam.insert(0, self.params["video_source"]); self.ent_cam.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f1, text="Rot:").pack(side=tk.LEFT)
        self.cb_rot = ttk.Combobox(f1, values=["0","90","180","270"], width=4, state="readonly")
        self.cb_rot.set(self.params["rotation"]); self.cb_rot.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f1, text="Scale:").pack(side=tk.LEFT)
        self.ent_scale = ttk.Entry(f1, width=4); self.ent_scale.insert(0, "1.0"); self.ent_scale.pack(side=tk.LEFT, padx=2)
        
        self.btn_cam = ttk.Button(f1, text="Start Cam", command=self.toggle_cam)
        self.btn_cam.pack(side=tk.LEFT, padx=10)
        
        # Row 2: Pico
        f2 = ttk.Frame(f); f2.pack(fill=tk.X, pady=5)
        ttk.Label(f2, text="Port:").pack(side=tk.LEFT)
        self.ent_port = ttk.Entry(f2, width=15); self.ent_port.insert(0, self.params["serial_port"]); self.ent_port.pack(side=tk.LEFT, padx=2)
        
        self.btn_pico = ttk.Button(f2, text="Connect Pico", command=self.toggle_pico)
        self.btn_pico.pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = ttk.Label(f, text="System Ready", foreground="blue")
        self.lbl_status.pack(anchor="w")

    def _build_tab_run(self, parent):
        # Motor Control
        lf = ttk.LabelFrame(parent, text="Manual Control")
        lf.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(lf, text="HOME MOTOR", command=self.cmd_home).pack(fill=tk.X, padx=5, pady=5)
        
        fj = ttk.Frame(lf); fj.pack(fill=tk.X, pady=2)
        ttk.Button(fj, text="<< -50", command=lambda: self.cmd_move(-50)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(fj, text="-10", command=lambda: self.cmd_move(-10)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(fj, text="+10", command=lambda: self.cmd_move(10)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(fj, text="+50 >>", command=lambda: self.cmd_move(50)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        fh = ttk.Frame(lf); fh.pack(fill=tk.X, pady=5)
        ttk.Label(fh, text="Swing Deg:").pack(side=tk.LEFT)
        self.e_ang = ttk.Entry(fh, width=5); self.e_ang.insert(0, "45"); self.e_ang.pack(side=tk.LEFT, padx=5)
        ttk.Button(fh, text="FIRE!", command=self.cmd_hit).pack(side=tk.LEFT, padx=5)

        # Game Control
        self.btn_track = tk.Button(parent, text="START THE GAME", bg="green", fg="white", 
                                   font=("Arial", 14, "bold"), height=2, command=self.toggle_tracking)
        self.btn_track.pack(fill=tk.X, padx=5, pady=10)
        
        self.btn_estop = tk.Button(parent, text="EMERGENCY STOP", bg="red", fg="white", 
                                   font=("Arial", 16, "bold"), height=2, command=self.emergency_stop)
        self.btn_estop.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_unlock = tk.Button(parent, text="UNLOCK SYSTEM", bg="orange", fg="black", 
                                    font=("Arial", 12, "bold"), command=self.unlock_estop)

    def _build_tab_arena(self, parent):
        def add_row(p, r, lbl, key):
            ttk.Label(p, text=lbl).grid(row=r, column=0, sticky="e", pady=2)
            e = ttk.Entry(p, width=6); e.insert(0, self.params[key]); e.grid(row=r, column=1, sticky="w", padx=5)
            e.bind("<FocusOut>", lambda ev: self.apply_settings(silent=True)) # Auto-preview
            return e

        lf1 = ttk.LabelFrame(parent, text="1. Play Area (ROI) Setup")
        lf1.pack(fill=tk.X, padx=5, pady=5)
        
        f = ttk.Frame(lf1); f.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(f, text="Marker IDs:").pack(side=tk.LEFT)
        self.e_r1 = ttk.Entry(f, width=3); self.e_r1.insert(0, self.params["roi_id1"]); self.e_r1.pack(side=tk.LEFT, padx=2)
        self.e_r2 = ttk.Entry(f, width=3); self.e_r2.insert(0, self.params["roi_id2"]); self.e_r2.pack(side=tk.LEFT, padx=2)
        
        self.e_rofx = add_row(lf1, 1, "Margin X (mm):", "roi_off_x")
        self.e_rofy = add_row(lf1, 2, "Margin Y (mm):", "roi_off_y")
        self.e_goff = add_row(lf1, 3, "Goal Line Offset:", "goal_off")

        lf2 = ttk.LabelFrame(parent, text="2. Origin (0,0) Setup")
        lf2.pack(fill=tk.X, padx=5, pady=5)
        
        f = ttk.Frame(lf2); f.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(f, text="Marker IDs:").pack(side=tk.LEFT)
        self.e_o1 = ttk.Entry(f, width=3); self.e_o1.insert(0, self.params["origin_id1"]); self.e_o1.pack(side=tk.LEFT, padx=2)
        self.e_o2 = ttk.Entry(f, width=3); self.e_o2.insert(0, self.params["origin_id2"]); self.e_o2.pack(side=tk.LEFT, padx=2)
        
        self.e_oox = add_row(lf2, 1, "Shift X (mm):", "org_off_x")
        self.e_ooy = add_row(lf2, 2, "Shift Y (mm):", "org_off_y")

        lf3 = ttk.LabelFrame(parent, text="3. Motion Physics")
        lf3.pack(fill=tk.X, padx=5, pady=5)
        
        self.e_trav = add_row(lf3, 0, "Total Travel (mm):", "travel_limit")
        self.e_mech = add_row(lf3, 1, "Center Offset (mm):", "mech_offset")
        self.e_jit  = add_row(lf3, 2, "Jitter Thresh (mm):", "jitter_th")
        self.e_hitd = add_row(lf3, 3, "Hit Distance (mm):", "hit_dist")
        
        # --- NEW RPM FIELD ---
        self.e_rpm = add_row(lf3, 4, "Motor RPM:", "rpm_fast")
        
        ttk.Button(parent, text="APPLY & UPDATE VIEW", command=self.apply_settings).pack(fill=tk.X, padx=10, pady=10)

    def _build_tab_cam(self, parent):
        lf = ttk.LabelFrame(parent, text="Lens Calibration")
        lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(lf, text="Run Distortion Calib", command=lambda: self.run_calib('distortion')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(lf, text="Run Scale Calib", command=lambda: self.run_calib('scale')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Separator(lf).pack(fill=tk.X, pady=5)
        ttk.Button(lf, text="DELETE Distortion Data", command=lambda: self.del_calib('distortion')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(lf, text="DELETE Scale Data", command=lambda: self.del_calib('scale')).pack(fill=tk.X, padx=5, pady=2)

    def _build_footer(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        lf = ttk.LabelFrame(f, text="System Log")
        lf.pack(fill=tk.BOTH, expand=True)
        self.txt_log = tk.Text(lf, height=6, state="disabled", bg="#f0f0f0", font=("Consolas", 8))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(parent, text="SAVE CONFIG TO DISK", bg="#444", fg="white", 
                  command=self.save_settings_disk).pack(fill=tk.X, padx=5, pady=5)

    # =================================================================
    #   LOGIC
    # =================================================================
    
    def log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

    def _push_params_to_tracker(self):
        p = self.params
        self.tracker.set_roi_markers(p["roi_id1"], p["roi_id2"], p["roi_off_x"], p["roi_off_y"], p["goal_off"])
        self.tracker.set_origin_markers(p["origin_id1"], p["origin_id2"], p["org_off_x"], p["org_off_y"])
        self.tracker.set_robot_constraints(p["travel_limit"], p["rob_width"], p["safe_margin"])

    def apply_settings(self, silent=False):
        try:
            # ROI
            self.params["roi_id1"] = int(self.e_r1.get())
            self.params["roi_id2"] = int(self.e_r2.get())
            self.params["roi_off_x"] = int(self.e_rofx.get())
            self.params["roi_off_y"] = int(self.e_rofy.get())
            self.params["goal_off"] = int(self.e_goff.get())
            
            # Origin
            self.params["origin_id1"] = int(self.e_o1.get())
            self.params["origin_id2"] = int(self.e_o2.get())
            self.params["org_off_x"] = int(self.e_oox.get())
            self.params["org_off_y"] = int(self.e_ooy.get())
            
            # Limits & Physics
            self.params["travel_limit"] = float(self.e_trav.get())
            self.params["mech_offset"] = float(self.e_mech.get())
            self.params["jitter_th"] = float(self.e_jit.get())
            self.params["hit_dist"] = float(self.e_hitd.get())
            self.params["rotation"] = int(self.cb_rot.get())
            
            # RPM (New)
            new_rpm = float(self.e_rpm.get())
            self.params["rpm_fast"] = new_rpm

            # Push to Objects
            self._push_params_to_tracker()
            if self.pico and self.pico.connected:
                self.pico.set_rpm(new_rpm)
                if not silent: self.log(f"RPM set to {new_rpm}")

            if not silent: self.log("Settings Applied")
            
        except ValueError:
            if not silent: messagebox.showerror("Error", "Invalid number in fields")

    def save_settings_disk(self):
        self.apply_settings(silent=True) 
        with open("gui_settings.json", "w") as f:
            json.dump(self.params, f)
        self.log("CONFIGURATION SAVED.")

    def run_calib(self, mode):
        if not self.cap: self.log("Start Cam First"); return
        wt = self.tracking_enabled; self.tracking_enabled = False
        def task():
            self.tracker.calibrate_camera(self.cap, 6, 8, 1, 0.03, mode)
            if wt: self.tracking_enabled = True
        threading.Thread(target=task).start()

    def del_calib(self, key):
        if messagebox.askyesno("Confirm", f"Delete {key}?"): self.tracker.delete_calibration_key(key)

    # --- CONTROL ---
    def cmd_home(self): 
        if self.pico and self.pico.connected:
            if not self.estop_active:
                self.log("Sending HOME...")
                self.pico.home()
            else:
                self.log("Cannot Home: E-STOP ACTIVE")
        else:
            self.log("Cannot Home: PICO NOT CONNECTED")

    def cmd_move(self, m): 
        if self.pico and self.pico.connected:
            if not self.estop_active:
                self.pico.send_command("MOVE", m)
            else:
                self.log("Stopped")
        else:
            self.log("Not Connected")

    def cmd_hit(self): 
        if self.pico and self.pico.connected:
            if not self.estop_active:
                try: ang = float(self.e_ang.get())
                except: ang = 45
                self.log(f"Firing Servo {ang} deg")
                self.pico.hit(1, ang)
            else:
                self.log("E-STOP Active")
        else:
            self.log("Not Connected")

    def toggle_tracking(self):
        if self.estop_active: return
        self.tracking_enabled = not self.tracking_enabled
        self.btn_track.config(text="STOP THE GAME" if self.tracking_enabled else "START THE GAME", 
                              bg="orange" if self.tracking_enabled else "green")
    def emergency_stop(self):
        self.estop_active = True; self.tracking_enabled = False; self.btn_track.config(bg="gray")
        if self.pico: self.pico.close(); self.pico=None
        self.btn_unlock.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_status.config(text="E-STOP ACTIVE", foreground="red")
        self.log("!!! EMERGENCY STOP !!!")

    def unlock_estop(self):
        if messagebox.askyesno("Safety", "Area Clear?"):
            self.estop_active = False; self.btn_track.config(bg="green")
            self.btn_unlock.pack_forget(); self.toggle_pico()
            self.lbl_status.config(text="System Unlocked", foreground="blue")

    def toggle_cam(self):
        if self.cap:
            self.cap.release(); self.cap=None; self.btn_cam.config(text="Start Cam")
            cv2.destroyAllWindows()
        else:
            try:
                self.cap = cv2.VideoCapture(int(self.ent_cam.get()))
                # Requested Res
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.params["req_width"])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params["req_height"])
                self.btn_cam.config(text="Stop Cam")
                self.log("Camera Started")
            except: self.log("Cam Error")

    def toggle_pico(self):
        if self.pico:
            self.pico.close(); self.pico=None; self.btn_pico.config(text="Connect")
            self.lbl_status.config(text="Disconnected", foreground="red")
        else:
            self.pico = PicoController(self.ent_port.get())
            if self.pico.connected: 
                self.btn_pico.config(text="Disconnect")
                self.log("Pico Connected")
                self.lbl_status.config(text="Pico Ready", foreground="green")
            else: 
                self.log("Pico Err")

    # --- LOOPS ---
    def update_video_loop(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rot = int(self.cb_rot.get())
                self.tracker.update(frame, velocity_method='poly', rotation=rot)
                
                if self.tracking_enabled and self.pico and self.pico.connected and not self.estop_active:
                    self._run_tracking_logic()
                
                try: s = float(self.ent_scale.get())
                except: s = 1.0
                self.tracker.show_feed(debug=False, scale=s)
                cv2.waitKey(1)
        self.root.after(20, self.update_video_loop)

    def _run_tracking_logic(self):
        pos = self.tracker.get_ball_position_mm()
        if pos:
            bx, by = pos
            limit_mm = self.params["travel_limit"] / 2.0
            safe_x = max(-limit_mm, min(limit_mm, bx))
            target_pos = self.params["mech_offset"] + safe_x
            
            jitter = self.params["jitter_th"]
            if abs(target_pos - self.last_sent_target) > jitter:
                self.pico.goto(target_pos)
                self.last_sent_target = target_pos
            
            hit_d = self.params["hit_dist"]
            dist = abs(by)
            if dist < hit_d and not self.swing_triggered:
                self.pico.hit(1)
                self.swing_triggered = True
                self.log(f"HIT! Dist={dist:.0f}mm")
            if dist > (hit_d + 50): self.swing_triggered = False

    def update_log_loop(self):
        if self.pico:
            while not self.pico.rx_queue.empty():
                msg = self.pico.rx_queue.get()
                # self.log(f"Pico: {msg}")
        self.root.after(100, self.update_log_loop)

    def load_settings(self):
        if os.path.exists("gui_settings.json"):
            try:
                with open("gui_settings.json", "r") as f:
                    self.params.update(json.load(f))
            except: pass

    def save_settings(self):
        with open("gui_settings.json", "w") as f:
            json.dump(self.params, f)

    def on_close(self):
        self.running = False
        if self.pico: self.pico.close()
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()
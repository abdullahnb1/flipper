# pico_controller.py — PC-side controller for Pico USB-serial JSON protocol
# Optimized for high-frequency ROS updates.

import json
import threading
import time
from typing import Optional, Callable
import serial
import queue

class PicoController:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        
        # Reader state
        self._rx_thread = None
        self._stop = threading.Event()
        self.on_message: Optional[Callable[[dict], None]] = None
        self.last_status: Optional[dict] = None
        self.connected = True # Assume connected if serial opened without error

        # Writer state (Throttling)
        self._tx_thread = None
        self._latest_target_mm = None
        self._last_sent_target_mm = None
        self._write_lock = threading.Lock()

    def start(self):
        """Starts both Reader and Writer threads."""
        self._stop.clear()
        
        # 1. Reader Thread
        self._rx_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._rx_thread.start()
        
        # 2. Writer Thread (for throttling GOTO)
        self._tx_thread = threading.Thread(target=self._target_sender_loop, daemon=True)
        self._tx_thread.start()

    def close(self):
        self._stop.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        if self._tx_thread:
            self._tx_thread.join(timeout=1.0)
        self.ser.close()
        self.connected = False

    # ---------- Background Loops ----------

    def _reader_loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)

                while b"\n" in buf:
                    raw, _, rest = buf.partition(b"\n")
                    buf = bytearray(rest)
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        msg = json.loads(raw.decode("utf-8", errors="ignore"))
                    except Exception:
                        continue

                    if msg.get("type") == "status":
                        self.last_status = msg

                    if self.on_message:
                        self.on_message(msg)
            except Exception:
                break

    def _target_sender_loop(self):
        """
        Background loop to send GOTO commands at a max rate.
        Prevents spamming the serial port if ROS runs faster than Serial can handle.
        """
        while not self._stop.is_set():
            target = self._latest_target_mm
            
            # Only send if target exists and has changed
            if target is not None and target != self._last_sent_target_mm:
                self.send_line(f"GOTO {target}")
                self._last_sent_target_mm = target
            
            # Limit update rate to ~50Hz (20ms) to let Pico breathe
            time.sleep(0.02) 

    # ---------- Low-level Send ----------
    def send_line(self, line: str):
        if not line.endswith("\n"):
            line += "\n"
        try:
            with self._write_lock:
                self.ser.write(line.encode("utf-8"))
        except Exception:
            pass

    # ---------- Command Helpers ----------
    def help(self):
        self.send_line("HELP")

    def ping(self):
        self.send_line("PING")

    def status(self):
        self.send_line("STATUS")

    def home(self, rpm_fast=120, rpm_slow=100, timeout_ms=20000):
        self.send_line(f"HOME {rpm_fast} {rpm_slow} {timeout_ms}")

    def set_rpm(self, rpm: float):
        self.send_line(f"SET_RPM {rpm}")

    def servo_goto(self, signed_deg: float):
        self.send_line(f"SERVO_GOTO {signed_deg}")

    def hit(self, direction: int, swing_deg: Optional[float] = None, dwell_ms: int = 150):
        if swing_deg is None:
            self.send_line(f"HIT {direction} {dwell_ms}")
        else:
            self.send_line(f"HIT {direction} {swing_deg} {dwell_ms}")

    # --- Throttled Target Setter ---
    def set_target(self, mm: float):
        """
        Updates the target variable. 
        The background thread will pick this up and send it efficiently.
        """
        self._latest_target_mm = round(mm, 2)

    # --- Direct Move (Unthrottled) ---
    def move(self, delta_mm: float):
        self.send_line(f"MOVE {delta_mm}")

if __name__ == "__main__":
    PORT = "/dev/ttyACM0"
    pico = PicoController(PORT)
    pico.start()
    print("Connecting...")
    time.sleep(2)
    pico.ping()
    pico.set_rpm(500)
    
    print("Testing Stream...")
    for i in range(50):
        pico.set_target(i) # This won't flood serial now
        time.sleep(0.005)  # 200Hz loop
        
    pico.close()
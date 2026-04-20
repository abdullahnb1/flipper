import serial
import json
import time
import threading
import queue

class PicoController:
    def __init__(self, port='/dev/ttyACM0', baud=115200):
        self.rx_queue = queue.Queue()
        self.running = False
        self.connected = False
        self.ser = None

        print(f"[Pico] Connecting to {port}...")
        
        try:
            # 1. Open Port
            self.ser = serial.Serial(port, baud, timeout=0.01)
            
            # 2. Force Reset (Toggle DTR)
            self.ser.dtr = False
            time.sleep(0.1)
            self.ser.dtr = True
            
            self.running = True
            
            # 3. Start Reading Thread
            self.rx_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.rx_thread.start()
            
            # 4. AUTO-WAIT: Block until Pico finishes booting (up to 5s)
            print("[Pico] Waiting for boot handshake...")
            if self._wait_for_ready(timeout=5.0):
                print("[Pico] Handshake received! System Online.")
                self.connected = True
            else:
                print("[Pico] WARNING: Handshake timed out (Pico didn't say 'ready').")
                # We assume connected anyway, in case it was already running
                self.connected = True 
                
        except serial.SerialException as e:
            print(f"[Pico] Hardware not found: {e}")

    def _wait_for_ready(self, timeout):
        """Sends PING and waits for 'ready' or 'pong' response."""
        start = time.time()
        next_ping = time.time()
        
        while time.time() - start < timeout:
            # Ping every 0.5s to wake it up
            if time.time() > next_ping:
                try:
                    self.ser.write(b"PING\n")
                    self.ser.flush()
                except: pass
                next_ping = time.time() + 0.5

            # Check queue
            try:
                msg = self.rx_queue.get(timeout=0.1)
                # If we get ANYTHING valid back, the Pico is alive
                if msg.get("type") in ["ready", "ack", "pong", "status"]:
                    return True
            except queue.Empty:
                continue
        return False

    def _read_loop(self):
        buffer = ""
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    chunk = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            try:
                                self.rx_queue.put(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"[Pico Log] {line}")
            except OSError:
                break
            time.sleep(0.001)

    def send_command(self, cmd, *args):
        if not self.connected: return
        try:
            full_cmd = f"{cmd} {' '.join(map(str, args))}\n"
            self.ser.write(full_cmd.encode('utf-8'))
            self.ser.flush()
        except:
            print("[Pico] Write Error")

    # --- API ---
    def home(self):
        print("[Pico] Homing...")
        self.send_command("HOME")

    def goto(self, mm):
        self.send_command("GOTO", round(mm, 2))

    def hit(self, direction=1, swing_deg=45):
        # Trigger the servo flip
        self.send_command("HIT", direction, swing_deg, 100) # dir, angle, dwell_ms
        
    def set_rpm(self, rpm):
        self.send_command("SET_RPM", rpm)

    def close(self):
        self.running = False
        if self.ser: self.ser.close()

# --- DIAGNOSTIC MAIN ---
if __name__ == "__main__":
    # Use the port that works for you (/dev/ttyACM0 or COMx)
    pico = PicoController(port="/dev/ttyACM0")

    if pico.connected:
        print("\n--- SENDING HOME COMMAND ---")
        pico.home()
        
        # Listen for the specific Ack/Done
        start_time = time.time()
        while time.time() - start_time < 30: # 30s timeout
            try:
                msg = pico.rx_queue.get(timeout=0.5)
                print(f"[RX] {msg}")
                
                if msg.get("type") == "error":
                    print("!!! PICO ERROR !!!")
                    break
                    
                if msg.get("cmd") == "HOME" and msg.get("done"):
                    print("--- HOMING SUCCESSFUL ---")
                    # Try a small move to verify
                    time.sleep(1)
                    print("Moving to 10mm...")
                    pico.goto(10)
                    
            except queue.Empty:
                pass
    else:
        print("Failed to initialize Pico.")
    
    pico.close()
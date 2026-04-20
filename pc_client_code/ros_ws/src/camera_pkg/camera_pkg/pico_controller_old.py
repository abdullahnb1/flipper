    # pico_controller.py — PC-side controller for Pico USB-serial JSON protocol
    # Sends commands like: "GOTO 12.3\n"
    # Reads JSON replies (one per line) in a background thread.

    import json
    import threading
    import time
    from typing import Optional, Callable

    import serial


    class PicoController:
        def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.05):
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            self._rx_thread = None
            self._stop = threading.Event()

            self.on_message: Optional[Callable[[dict], None]] = None
            self.last_status: Optional[dict] = None

            # for convenience
            self._lock = threading.Lock()

        def start(self):
            self._stop.clear()
            self._rx_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._rx_thread.start()

        def close(self):
            self._stop.set()
            if self._rx_thread:
                self._rx_thread.join(timeout=1.0)
            self.ser.close()

        # ---------- low-level ----------
        def send_line(self, line: str):
            if not line.endswith("\n"):
                line += "\n"
            with self._lock:
                self.ser.write(line.encode("utf-8"))

        def _reader_loop(self):
            buf = bytearray()
            while not self._stop.is_set():
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
                        # ignore malformed
                        continue

                    if msg.get("type") == "status":
                        self.last_status = msg

                    if self.on_message:
                        self.on_message(msg)

        # ---------- command helpers (names MUST match Pico) ----------
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

        # IMPORTANT: streaming target override
        def set_target_mm(self, mm: float):
            # Uses GOTO as "streaming target" (non-blocking on Pico)
            self.send_line(f"GOTO {mm}")

        def move_relative_mm(self, delta_mm: float):
            self.send_line(f"MOVE {delta_mm}")


    if __name__ == "__main__":
        # Example usage
        PORT = "COM6"  # change: e.g. "COM6" on Windows or "/dev/ttyACM0" on Linux
        pico = PicoController(PORT)

        def printer(msg):
            t = msg.get("type")
            if t == "ready":
                print("READY:", msg)
            elif t == "ack":
                print("ACK:", msg)
            elif t == "error":
                print("ERR:", msg)
            elif t == "status":
                # keep it light
                pass

        pico.on_message = printer
        pico.start()

        pico.ping()
        time.sleep(0.2)
        pico.status()

        # Example: stream target (simulate ball tracking)
        # Send a new target every 20 ms
        t0 = time.time()
        while time.time() - t0 < 5.0:
            # dummy moving target
            target = 10.0 * (time.time() - t0)  # mm
            pico.set_target_mm(target)
            time.sleep(0.02)

        pico.close()
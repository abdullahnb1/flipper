# main.py — Pico / MicroPython
# Serial-command controller (USB CDC) for:
# - NEMA17 + TB6600 (STEP/DIR)
# - 2x limit switches (NC fail-safe recommended)
# - Two-way servo flipper (signed angles)
#
# PC sends commands like:
#   HOME
#   HOME 120 40 20000
#   GOTO 25.0
#   MOVE -5.0
#   SET_RPM 150
#   HIT +1 45 150
#   SERVO_GOTO -10
#   STATUS
#
# Pico replies as JSON lines, e.g.:
#   {"type":"ack","cmd":"HOME","ok":true}
#   {"type":"status","pos_mm":0.0,"pos_steps":0,"homed":true,"servo_signed_deg":0.0}

import machine
import time
import sys
import uselect

try:
    import ujson as json
except ImportError:
    import json

# =========================================================
# 1) LIMIT SWITCH
# =========================================================
class MicroSwitch:
    """
    For NC fail-safe wiring with PULL_UP:
      - Normal (not hit): pin reads 0 (connected to GND through NC contact)
      - Hit OR wire break: pin reads 1  => TRIGGERED
    """
    def __init__(self, pin_num, pull=machine.Pin.PULL_UP, triggered_level=1):
        self.pin = machine.Pin(pin_num, machine.Pin.IN, pull)
        self.triggered_level = 1 if int(triggered_level) else 0

    def raw_level(self) -> int:
        return int(self.pin.value())

    def is_triggered_raw(self) -> bool:
        return self.raw_level() == self.triggered_level


# =========================================================
# 2) SERVO (two-way flipper)
# =========================================================
class TwoWayServo:
    def __init__(self, pin_num, freq=50, min_us=500, max_us=2500,
                 abs_range_deg=180, center_angle=90, max_swing=60):
        self.pwm = machine.PWM(machine.Pin(pin_num))
        self.pwm.freq(int(freq))

        self.min_us = int(min_us)
        self.max_us = int(max_us)
        self.abs_range_deg = float(abs_range_deg)

        self.center_angle = float(center_angle)
        self.max_swing = float(max_swing)

        self.current_signed = 0.0
        self.goto_signed(0)

    def _abs_angle_to_duty_u16(self, abs_angle_deg: float) -> int:
        a = max(0.0, min(self.abs_range_deg, float(abs_angle_deg)))
        pulse_us = self.min_us + (a / self.abs_range_deg) * (self.max_us - self.min_us)
        period_us = 1_000_000 / self.pwm.freq()
        duty = int((pulse_us / period_us) * 65535)
        return max(0, min(65535, duty))

    def goto_abs(self, abs_angle_deg: float):
        self.pwm.duty_u16(self._abs_angle_to_duty_u16(abs_angle_deg))

    def goto_signed(self, signed_angle_deg: float):
        s = max(-self.max_swing, min(self.max_swing, float(signed_angle_deg)))
        self.current_signed = s
        self.goto_abs(self.center_angle + s)

    def hit(self, direction=+1, swing_deg=None, dwell_ms=120, return_ms=0):
        d = +1 if direction >= 0 else -1
        swing = self.max_swing if swing_deg is None else float(swing_deg)
        swing = max(0.0, min(self.max_swing, swing))

        self.goto_signed(d * swing)
        time.sleep_ms(int(dwell_ms))
        self.goto_signed(0)

        if return_ms > 0:
            time.sleep_ms(int(return_ms))


# =========================================================
# 3) STEPPER (TB6600) — TICKS-BASED TIMING (NO sleep_us in loops)
# =========================================================
class StepperBeltAxis:
    """
    Uses ticks_us scheduling to reduce timing jitter.
    No time.sleep_us() inside step loops (busy-wait scheduling instead).
    """
    def __init__(self,
                 step_pin, dir_pin,
                 rpm=60,
                 motor_steps_per_rev=200,
                 microstep=16,                 # match TB6600 DIP
                 belt_pitch_mm=2.0,
                 pulley_teeth=20,
                 left_switch=None, right_switch=None,
                 left_dir=0, right_dir=1,
                 backoff_mm=5.0,
                 step_high_us=5                # TB6600: 2–10us OK, 5us safe
                 ):
        self.step = machine.Pin(step_pin, machine.Pin.OUT)
        self.dir  = machine.Pin(dir_pin,  machine.Pin.OUT)

        self.motor_steps_per_rev = int(motor_steps_per_rev)
        self.microstep = int(microstep)

        self.belt_pitch_mm = float(belt_pitch_mm)
        self.pulley_teeth = int(pulley_teeth)

        self.mm_per_rev = self.belt_pitch_mm * self.pulley_teeth
        self.steps_per_rev = self.motor_steps_per_rev * self.microstep
        self.steps_per_mm = self.steps_per_rev / self.mm_per_rev

        self.step_high_us = int(step_high_us)

        self.sw_left  = left_switch
        self.sw_right = right_switch

        self.left_dir = int(left_dir) & 1
        self.right_dir = int(right_dir) & 1

        self.backoff_steps = max(1, int(round(float(backoff_mm) * self.steps_per_mm)))

        self.pos_steps = 0
        self.homed = False
        self.travel_steps = None
        self.left_limit_steps = None
        self.right_limit_steps = None

        self.set_rpm(rpm)
        self.step.value(0)

    # ---------- speed ----------
    def set_rpm(self, rpm, min_low_us=5):
        """
        Set axis speed in RPM with a safe minimum LOW time between pulses.

        min_low_us:
          Minimum time STEP stays LOW between pulses (in addition to step_high_us).
          5–10 us is a reasonable safety margin for TB6600 in MicroPython.
        """
        rpm = float(rpm)
        rpm = max(0.1, rpm)

        # Minimum allowed period (rising edge to rising edge)
        # Must be at least HIGH time + LOW time
        min_period_us = int(self.step_high_us + int(min_low_us))
        if min_period_us < self.step_high_us + 1:
            min_period_us = self.step_high_us + 1

        # Max steps/sec allowed by min period
        max_sps = 1_000_000.0 / float(min_period_us)

        # Convert that to max RPM for current steps_per_rev
        max_rpm_by_period = (max_sps * 60.0) / float(self.steps_per_rev)

        # Clamp RPM to achievable range
        # (If you want, you can also clamp to some global max like 800 here)
        rpm = min(rpm, max_rpm_by_period)
        self.rpm = rpm

        # Compute period from RPM
        sps = (self.rpm * self.steps_per_rev) / 60.0
        sps = max(1.0, sps)
        period_us = int(1_000_000 / sps)

        # Final safety clamp (should rarely trigger now)
        if period_us < min_period_us:
            period_us = min_period_us

        self.step_period_us = period_us


    # ---------- direction ----------
    def _set_dir(self, d):
        self.dir.value(1 if int(d) else 0)

    # ---------- position helpers ----------
    def pos_mm(self):
        return self.pos_steps / self.steps_per_mm

    def _clamp_target_steps(self, target_steps):
        if not self.homed or self.left_limit_steps is None or self.right_limit_steps is None:
            return target_steps
        return max(self.left_limit_steps, min(self.right_limit_steps, target_steps))

    # ---------- switch helpers ----------
    def _left_hit(self):
        return (self.sw_left is not None) and self.sw_left.is_triggered_raw()

    def _right_hit(self):
        return (self.sw_right is not None) and self.sw_right.is_triggered_raw()

    def _hit_in_direction(self, step_inc):
        if step_inc < 0 and self._left_hit():
            return True
        if step_inc > 0 and self._right_hit():
            return True
        return False

    # ---------- ticks helpers ----------
    def _busy_wait_until(self, t_us):
        while time.ticks_diff(t_us, time.ticks_us()) > 0:
            pass

    def _pulse_at(self, t_edge_us):
        self._busy_wait_until(t_edge_us)
        self.step.value(1)

        t_low = time.ticks_add(t_edge_us, self.step_high_us)
        self._busy_wait_until(t_low)
        self.step.value(0)

    # ---------- safety backoff ----------
    def _backoff_from_limit(self, step_inc):
        backoff_inc = +1 if step_inc < 0 else -1
        self._set_dir(self.right_dir if backoff_inc > 0 else self.left_dir)

        period = self.step_period_us
        next_edge = time.ticks_us()

        for _ in range(self.backoff_steps):
            if self._hit_in_direction(backoff_inc):
                break
            self._pulse_at(next_edge)
            self.pos_steps += backoff_inc
            next_edge = time.ticks_add(next_edge, period)

    # ---------- blocking motion (steps) ----------
    def move_steps(self, steps):
        steps = int(steps)
        if steps == 0:
            return True

        step_inc = +1 if steps > 0 else -1
        n = abs(steps)

        self._set_dir(self.right_dir if step_inc > 0 else self.left_dir)

        # clamp to homed limits
        if self.homed and self.left_limit_steps is not None and self.right_limit_steps is not None:
            final = self.pos_steps + step_inc * n
            final = self._clamp_target_steps(final)
            n = abs(final - self.pos_steps)
            if n == 0:
                return True
            step_inc = +1 if final > self.pos_steps else -1
            self._set_dir(self.right_dir if step_inc > 0 else self.left_dir)

        period = self.step_period_us
        next_edge = time.ticks_us()

        for _ in range(n):
            if self._hit_in_direction(step_inc):
                self._backoff_from_limit(step_inc)
                return False

            self._pulse_at(next_edge)
            self.pos_steps += step_inc
            next_edge = time.ticks_add(next_edge, period)

        return True

    # ---------- blocking motion (mm) ----------
    def move_mm(self, mm):
        target_steps = int(round(float(mm) * self.steps_per_mm))
        return self.move_steps(target_steps)

    def goto_mm(self, target_mm):
        target_steps = int(round(float(target_mm) * self.steps_per_mm))
        target_steps = self._clamp_target_steps(target_steps)
        return self.move_steps(target_steps - self.pos_steps)

    # =========================================================
    # HOMING
    # =========================================================
    def home(self, rpm_fast=120, rpm_slow=40, timeout_ms=20000):
        if self.sw_left is None or self.sw_right is None:
            raise RuntimeError("Need BOTH left and right switches for homing.")

        t0 = time.ticks_ms()
        def timed_out():
            return time.ticks_diff(time.ticks_ms(), t0) > int(timeout_ms)

        # release if starting on switch
        self.set_rpm(rpm_slow)
        period = self.step_period_us
        next_edge = time.ticks_us()

        if self._left_hit():
            self._set_dir(self.right_dir)
            while self._left_hit():
                if timed_out():
                    raise RuntimeError("Homing timeout while releasing LEFT.")
                self._pulse_at(next_edge)
                self.pos_steps += 1
                next_edge = time.ticks_add(next_edge, period)

        if self._right_hit():
            self._set_dir(self.left_dir)
            while self._right_hit():
                if timed_out():
                    raise RuntimeError("Homing timeout while releasing RIGHT.")
                self._pulse_at(next_edge)
                self.pos_steps -= 1
                next_edge = time.ticks_add(next_edge, period)

        # 1) seek left
        self.set_rpm(rpm_fast)
        period = self.step_period_us
        next_edge = time.ticks_us()
        self._set_dir(self.left_dir)

        while not self._left_hit():
            if timed_out():
                raise RuntimeError("Homing timeout seeking LEFT.")
            self._pulse_at(next_edge)
            self.pos_steps -= 1
            next_edge = time.ticks_add(next_edge, period)

        self._backoff_from_limit(step_inc=-1)
        self.pos_steps = 0

        # 2) seek right
        self.set_rpm(rpm_fast)
        period = self.step_period_us
        next_edge = time.ticks_us()
        self._set_dir(self.right_dir)

        while not self._right_hit():
            if timed_out():
                raise RuntimeError("Homing timeout seeking RIGHT.")
            self._pulse_at(next_edge)
            self.pos_steps += 1
            next_edge = time.ticks_add(next_edge, period)

        self.travel_steps = int(self.pos_steps)
        if self.travel_steps <= 0:
            raise RuntimeError("Homing failed: travel_steps <= 0 (swap left_dir/right_dir).")

        self._backoff_from_limit(step_inc=+1)

        # 3) go to center
        center_from_left = self.travel_steps // 2
        self.set_rpm(rpm_slow)
        self.move_steps(-(center_from_left))

        # define center as 0
        self.pos_steps = 0
        half = self.travel_steps // 2
        self.left_limit_steps = -half
        self.right_limit_steps = +half
        self.homed = True
        return True


# =========================================================
# 4) CONFIG
# =========================================================
PIN_STEP = 16
PIN_DIR  = 13

PIN_SERVO = 10

PIN_SW_LEFT  = 28
PIN_SW_RIGHT = 27

MICROSTEP = 2
INITIAL_RPM = 120

BELT_PITCH_MM = 2.0
PULLEY_TEETH = 20

LEFT_DIR  = 0
RIGHT_DIR = 1

BACKOFF_MM = 5.0


# =========================================================
# 5) SERIAL COMMS + COMMANDS
# =========================================================
BUSY = False

def send(obj):
    # Always send one JSON object per line
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
    except Exception:
        # fallback minimal
        sys.stdout.write('{"type":"error","msg":"json_fail"}\n')

def status_payload(axis, servo):
    return {
        "type": "status",
        "pos_mm": float(axis.pos_mm()),
        "pos_steps": int(axis.pos_steps),
        "homed": bool(axis.homed),
        "servo_signed_deg": float(servo.current_signed),
        "busy": bool(BUSY),
    }

def ack(cmd, ok=True, **extra):
    out = {"type": "ack", "cmd": cmd, "ok": bool(ok)}
    for k, v in extra.items():
        out[k] = v
    send(out)

def err(cmd, msg, **extra):
    out = {"type": "error", "cmd": cmd, "msg": str(msg)}
    for k, v in extra.items():
        out[k] = v
    send(out)

def parse_line(line):
    # strip CRLF and split by whitespace
    line = line.strip()
    if not line:
        return None, []
    parts = line.split()
    cmd = parts[0].upper()
    args = parts[1:]
    return cmd, args

def to_float(s, default=None):
    try:
        return float(s)
    except:
        return default

def to_int(s, default=None):
    try:
        return int(s)
    except:
        return default

def handle_command(cmd, args, axis, servo):
    global BUSY

    # Always allow STATUS / PING even if busy
    if cmd in ("STATUS",):
        send(status_payload(axis, servo))
        return
    if cmd in ("PING",):
        ack("PING", ok=True)
        return
    if cmd in ("HELP", "?"):
        send({
            "type": "help",
            "commands": [
                "HOME [rpm_fast rpm_slow timeout_ms]",
                "GOTO <mm>",
                "MOVE <mm>",
                "SET_RPM <rpm>",
                "SERVO_GOTO <signed_deg>",
                "HIT <dir(+1/-1)> [swing_deg] [dwell_ms]",
                "STATUS",
                "PING",
            ]
        })
        return

    if BUSY:
        err(cmd, "busy")
        return

    # ---- Motion commands (blocking) ----
    if cmd == "HOME":
        rpm_fast = 120.0
        rpm_slow = 40.0
        timeout  = 20000

        if len(args) >= 1:
            v = to_float(args[0], None)
            if v is None: return err(cmd, "bad rpm_fast")
            rpm_fast = v
        if len(args) >= 2:
            v = to_float(args[1], None)
            if v is None: return err(cmd, "bad rpm_slow")
            rpm_slow = v
        if len(args) >= 3:
            v = to_int(args[2], None)
            if v is None: return err(cmd, "bad timeout_ms")
            timeout = v

        BUSY = True
        ack(cmd, ok=True, started=True)
        try:
            axis.home(rpm_fast=rpm_fast, rpm_slow=rpm_slow, timeout_ms=timeout)
            ack(cmd, ok=True, done=True)
            send(status_payload(axis, servo))
        except Exception as e:
            err(cmd, str(e))
        finally:
            BUSY = False
        return

    if cmd == "GOTO":
        if len(args) < 1:
            return err(cmd, "missing mm")
        mm = to_float(args[0], None)
        if mm is None:
            return err(cmd, "bad mm")

        BUSY = True
        ack(cmd, ok=True, started=True, target_mm=float(mm))
        try:
            ok = axis.goto_mm(mm)
            ack(cmd, ok=bool(ok), done=True)
            send(status_payload(axis, servo))
        except Exception as e:
            err(cmd, str(e))
        finally:
            BUSY = False
        return

    if cmd == "MOVE":
        if len(args) < 1:
            return err(cmd, "missing mm")
        mm = to_float(args[0], None)
        if mm is None:
            return err(cmd, "bad mm")

        BUSY = True
        ack(cmd, ok=True, started=True, delta_mm=float(mm))
        try:
            ok = axis.move_mm(mm)
            ack(cmd, ok=bool(ok), done=True)
            send(status_payload(axis, servo))
        except Exception as e:
            err(cmd, str(e))
        finally:
            BUSY = False
        return

    if cmd == "SET_RPM":
        if len(args) < 1:
            return err(cmd, "missing rpm")
        rpm = to_float(args[0], None)
        if rpm is None:
            return err(cmd, "bad rpm")
        axis.set_rpm(rpm)
        ack(cmd, ok=True, rpm=float(axis.rpm), step_period_us=int(axis.step_period_us))
        return

    # ---- Servo commands ----
    if cmd == "SERVO_GOTO":
        if len(args) < 1:
            return err(cmd, "missing signed_deg")
        deg = to_float(args[0], None)
        if deg is None:
            return err(cmd, "bad signed_deg")
        servo.goto_signed(deg)
        ack(cmd, ok=True)
        send(status_payload(axis, servo))
        return

    if cmd == "HIT":
        if len(args) < 1:
            return err(cmd, "missing dir")
        direction = to_int(args[0], None)
        if direction is None or direction == 0:
            return err(cmd, "bad dir (use +1 or -1)")

        swing = None
        dwell = 150

        if len(args) >= 2:
            swing = to_float(args[1], None)
            if swing is None:
                return err(cmd, "bad swing_deg")
        if len(args) >= 3:
            dwell = to_int(args[2], None)
            if dwell is None:
                return err(cmd, "bad dwell_ms")

        BUSY = True
        ack(cmd, ok=True, started=True)
        try:
            servo.hit(direction=direction, swing_deg=swing, dwell_ms=dwell)
            ack(cmd, ok=True, done=True)
            send(status_payload(axis, servo))
        except Exception as e:
            err(cmd, str(e))
        finally:
            BUSY = False
        return

    return err(cmd, "unknown_command")


def serial_loop(axis, servo):
    # Non-blocking line reader over USB serial (stdin)
    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)

    # announce ready
    send({"type": "ready", "msg": "pico_controller_ready"})
    send(status_payload(axis, servo))

    buf = ""

    while True:
        # poll with small timeout (ms)
        events = poll.poll(20)
        if not events:
            # optional: you can periodically stream status here if you want
            continue

        for _fd, ev in events:
            if ev & uselect.POLLIN:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch == "\r":
                    continue
                if ch == "\n":
                    cmd, args = parse_line(buf)
                    buf = ""
                    if cmd is None:
                        continue
                    handle_command(cmd, args, axis, servo)
                else:
                    # prevent runaway buffer
                    if len(buf) < 200:
                        buf += ch
                    else:
                        buf = ""
                        err("RX", "line_too_long")


# =========================================================
# 6) BOOT
# =========================================================
if __name__ == "__main__":
    left_sw  = MicroSwitch(PIN_SW_LEFT,  pull=machine.Pin.PULL_UP, triggered_level=1)
    right_sw = MicroSwitch(PIN_SW_RIGHT, pull=machine.Pin.PULL_UP, triggered_level=1)

    servo = TwoWayServo(
        PIN_SERVO,
        center_angle=90,
        max_swing=60
    )
    servo.goto_signed(0)
    time.sleep_ms(200)

    axis = StepperBeltAxis(
        step_pin=PIN_STEP,
        dir_pin=PIN_DIR,
        rpm=INITIAL_RPM,
        microstep=MICROSTEP,
        belt_pitch_mm=BELT_PITCH_MM,
        pulley_teeth=PULLEY_TEETH,
        left_switch=left_sw,
        right_switch=right_sw,
        left_dir=LEFT_DIR,
        right_dir=RIGHT_DIR,
        backoff_mm=BACKOFF_MM,
        step_high_us=5
    )

    serial_loop(axis, servo)

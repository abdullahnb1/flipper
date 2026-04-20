import serial
import time

PORT = "/dev/ttyACM0"

print(f"Waiting for {PORT}...")

while True:
    try:
        # Try to connect until successful
        ser = serial.Serial(PORT, 115200, timeout=1)
        print("Connected! Waiting for Pico 'Ready' signal...")
        break
    except serial.SerialException:
        time.sleep(0.5)

# Clear any old garbage
ser.reset_input_buffer()

# Listen loop
start = time.time()
while True:
    # 1. Read
    if ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"RX < {line}")
    
    # 2. Periodic Ping
    if time.time() - start > 1.0:
        print("TX > PING")
        ser.write(b"PING\n")
        ser.flush()
        
        print("TX > MOVE")
        ser.write(b"MOVE\n")
        ser.flush()
        
        start = time.time()
    
    time.sleep(0.01)
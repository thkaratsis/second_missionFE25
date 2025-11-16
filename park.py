import cv2
import numpy as np
from picamera2 import Picamera2
import time
import threading
import smbus2

from pca9685_control import set_servo_angle, set_motor_speed

# ToF deps
import board
import busio
import digitalio
import adafruit_vl53l0x

# ==== CONSTANTS ====
MOTOR_FWD = 1
MOTOR_REV = 2
SERVO_CHANNEL = 0
CENTER_ANGLE = 90

# motor speeds (you can tweak)
ALIGN_SPEED = 10     # slow speed during side-distance alignment

# IMU/servo correction constants (you can tweak)
YAW_DEADBAND_DEG_ALIGN = 2.0
YAW_KP_ALIGN = 1.0           # proportional gain for yaw → servo during alignment
SERVO_CORR_LIMIT_ALIGN = 25  # max correction in degrees

# Side alignment target
SIDE_TARGET_CM = 26.0        # we align until side < 26 cm
SIDE_ALIGN_YAW_PER_CM = 1.0  # kept (not used in new logic but not removed)

# ToF thresholds for sanity
MIN_VALID_CM = 5.0
MAX_VALID_CM = 150.0

# How parallel we want (IMU)
YAW_PARALLEL_TOL = 1.0       # deg, considered "parallel" when |yaw| < this
PARALLEL_STABLE_COUNT = 5    # number of consecutive checks inside tol
PARALLEL_ALIGN_MAX_TIME = 3.0  # seconds max for IMU straighten phase

# Steering angles for hard turns
RIGHT_TURN_ANGLE = 60        # servo angle for turning right
LEFT_TURN_ANGLE = 120        # servo angle for turning left

# ==== PINK MASK ====
# (H low corrected to 160 as you said)
PINK_LO = np.array([160, 200, 120], dtype=np.uint8)
PINK_HI = np.array([170, 240, 180], dtype=np.uint8)

# ==== IMU SETUP (MPU6050) ====
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
GYRO_ZOUT_H = 0x47
bus = smbus2.SMBus(1)

def mpu6050_init():
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_raw_data(addr):
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr+1)
    value = (high << 8) | low
    if value > 32767:
        value -= 65536
    return value

def get_gyro_z_bias(samples=200):
    total = 0.0
    for _ in range(samples):
        raw = read_raw_data(GYRO_ZOUT_H)
        total += raw / 131.0
        time.sleep(0.005)
    return total / samples

mpu6050_init()
print("Measuring gyro bias, keep sensor still...")
gyro_z_bias = get_gyro_z_bias()
print(f"Gyro Z bias: {gyro_z_bias:.3f} deg/s")

# ==== YAW TRACKING ====
yaw = 0.0
yaw_lock = threading.Lock()
last_time = time.time()

def imu_update():
    """Call this often to integrate yaw."""
    global yaw, last_time
    now = time.time()
    dt = now - last_time
    last_time = now

    raw_gz_dps = read_raw_data(GYRO_ZOUT_H) / 131.0
    Gz = raw_gz_dps - gyro_z_bias

    with yaw_lock:
        yaw += Gz * dt
        current_yaw = yaw
    return current_yaw

def imu_center_servo(current_yaw_deg: float, deadband: float, kp: float, limit: float) -> int:
    """
    IMU-based centering:
      - current_yaw_deg > 0  → robot rotated one way
      - we steer in the opposite direction to bring yaw back to 0.
    """
    if abs(current_yaw_deg) <= deadband:
        return CENTER_ANGLE
    # IMPORTANT: minus sign so we cancel yaw, not amplify it
    corr = -kp * current_yaw_deg
    corr = max(-limit, min(limit, corr))
    return int(CENTER_ANGLE + corr)

def reset_yaw():
    global yaw, last_time
    with yaw_lock:
        yaw = 0.0
    last_time = time.time()

# ==== SERVO SETUP ====
set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)

# ==== CAMERA SETUP ====
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (1920, 1080)})
)
picam2.start()

# ==== ToF SETUP ====
i2c = busio.I2C(board.SCL, board.SDA)
xshut_pins = {
    "left":    board.D16,
    "right":   board.D25,
    "front":   board.D26,
    "back":    board.D8,
}
addresses = {
    "left":    0x30,
    "right":   0x31,
    "front":   0x32,
    "back":    0x33,
}

xshuts = {}
for name, pin in xshut_pins.items():
    x = digitalio.DigitalInOut(pin)
    x.direction = digitalio.Direction.OUTPUT
    x.value = False
    xshuts[name] = x
time.sleep(0.1)

sensors = {}
for name in ["left", "right", "front", "back"]:
    xshuts[name].value = True
    time.sleep(0.05)
    s = adafruit_vl53l0x.VL53L0X(i2c)
    s.set_address(addresses[name])
    s.start_continuous()
    sensors[name] = s
    print(f"[TOF] {name.upper()} active at {hex(addresses[name])}")

def tof_cm(sensor):
    try:
        val = sensor.range / 10.0
        if val <= 0 or val > MAX_VALID_CM:
            return 999
        return val
    except:
        return 999

# ==== SIDE ALIGNMENT ROUTINE ====
def side_distance_align(direction, duration=3.0):
    """
    NEW behavior (no logic removed elsewhere):
      - If direction == "right":
          turn right until right < 26 cm,
          then use yaw to steer back to yaw ≈ 0,
          then go straight.
      - If direction == "left":
          same but mirrored with left sensor.
    """
    print(f"[ALIGN] Starting side-distance alignment for direction = {direction}")
    reset_yaw()  # yaw=0 is our reference heading

    # -------- Phase 1: turn TOWARD chosen side until side < SIDE_TARGET_CM --------
    end_time = time.time() + duration
    while time.time() < end_time:
        current_yaw = imu_update()

        if direction == "left":
            side_cm = tof_cm(sensors["left"])
        else:
            side_cm = tof_cm(sensors["right"])

        if side_cm != 999:
            print(f"[ALIGN-P1] dir={direction}, side={side_cm:.1f}cm, yaw={current_yaw:.2f}")
            # stop this phase once we are closer than target
            if side_cm <= SIDE_TARGET_CM:
                print("[ALIGN-P1] Reached target side distance.")
                break

        # steer towards the wall
        if direction == "left":
            steer_angle = LEFT_TURN_ANGLE 
        else:
            steer_angle = RIGHT_TURN_ANGLE

        set_servo_angle(SERVO_CHANNEL, steer_angle)
        set_motor_speed(MOTOR_FWD, MOTOR_REV, ALIGN_SPEED)
        time.sleep(0.05)

    # -------- Phase 2: keep moving, straighten yaw back to 0 using IMU --------
    print("[ALIGN-P2] Straightening yaw to 0 while moving forward.")
    stable_count = 0
    phase2_start = time.time()
    while time.time() - phase2_start < PARALLEL_ALIGN_MAX_TIME:
        current_yaw = imu_update()

        servo_angle = imu_center_servo(
            current_yaw,
            YAW_DEADBAND_DEG_ALIGN,
            YAW_KP_ALIGN,
            SERVO_CORR_LIMIT_ALIGN
        )
        servo_angle = max(60, min(120, servo_angle))

        set_servo_angle(SERVO_CHANNEL, servo_angle)
        set_motor_speed(MOTOR_FWD, MOTOR_REV, ALIGN_SPEED)

        print(f"[ALIGN-P2] yaw={current_yaw:.2f}, servo={servo_angle}")

        if abs(current_yaw) < YAW_PARALLEL_TOL:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= PARALLEL_STABLE_COUNT:
            print("[ALIGN-P2] Yaw stable near 0; parallel achieved.")
            break

        time.sleep(0.05)

    # -------- Phase 3: servo straight, keep going forward (no stop) --------
    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
    set_motor_speed(MOTOR_FWD, MOTOR_REV, ALIGN_SPEED)
    reset_yaw()
    print("[ALIGN] Done side-distance alignment. Servo centered, still moving straight.")

# ==== STARTUP: CHOOSE DIRECTION & ALIGN SIDE DISTANCE ====
print("\n=== STARTUP: Choosing direction & aligning to side wall ===")

left_dist  = tof_cm(sensors["left"])
right_dist = tof_cm(sensors["right"])
print(f"[START] Left={left_dist:.1f} cm | Right={right_dist:.1f} cm")

# Choose direction: more open side
if left_dist > right_dist:
    direction = "left"
else:
    direction = "right"

print(f"[START] Direction chosen: {direction.upper()} (more open side)")

# Align side distance for chosen direction
side_distance_align(direction, duration=3.0)

print("=== ENTERING PINK-SQUARE DETECTION LOOP ===")
print("Press Q to quit.")

# ==== MAIN LOOP: PINK SQUARE DETECTION ====
try:
    while True:
        # Keep IMU time base up to date
        imu_update()

        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Pink mask
        mask = cv2.inRange(hsv, PINK_LO, PINK_HI)

        # Morphological cleanup
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pink_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:   # filter small noise
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if 0.6 < aspect < 1.4:   # roughly square
                pink_boxes.append((x, y, w, h))
                cv2.rectangle(bgr, (x, y), (x+w, y+h), (255, 0, 255), 3)
                cv2.putText(bgr, "PINK", (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

        cv2.putText(bgr, f"PINK COUNT: {len(pink_boxes)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)

        cv2.imshow("Pink Squares", bgr)
        cv2.imshow("Pink Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
    set_motor_speed(MOTOR_FWD, MOTOR_REV, 0)
    print("Clean exit. Motors stopped, servo centered.")

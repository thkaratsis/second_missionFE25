# 2nd mission
# v1.1: add state print, add turn/lap count and stop after 3 laps, variables cleanup
#       add ultrasonic for side checks during cruise



# =========================
# IMPORTS
# =========================
import cv2
import numpy as np
from picamera2 import Picamera2
import time
from pca9685_control import set_servo_angle, set_motor_speed
import smbus2
import threading
import board
import busio
import digitalio
import adafruit_vl53l0x
from gpiozero import Device, Button, DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory   # Uses /dev/gpiochip*
Device.pin_factory = LGPIOFactory()


# =========================
# CONFIGURABLE PARAMETERS
# =========================

# ---- Speed settings ----
NORMAL_SPEED          = 11.5   # Base forward cruising speed during normal driving
AVOID_SPEED           = 13   # Reverse speed when doing obstacle-avoidance backup
TURN_MOTOR_SPEED      = 13   # Motor speed while performing a 90° line-based turn
EMERGENCY_BACK_SPEED  = 14   # Reverse speed in emergency “stuck” escape
BLUE_BACK_SPEED       = 14   # Reverse speed when box triggers blue-backward escape
UNPARK_STRAIGHT_SPEED = 18   # Speed used during smart unpark phases
STOP_SPEED            = 0    # Zero-speed (motors off)

# ---- Motor / servo basics ----
MOTOR_FWD       = 1         # PCA9685 motor channel for forward direction
MOTOR_REV       = 2         # PCA9685 motor channel for reverse direction
SERVO_CHANNEL   = 0         # PCA9685 channel used by steering servo
CENTER_ANGLE    = 90        # Servo angle for going straight
LEFT_FAR        = 105       # Steering angle for a far red box (mild left)
LEFT_NEAR       = 120       # Steering angle for a near red box (strong left)
RIGHT_FAR       = 75        # Steering angle for a far green box (mild right)
RIGHT_NEAR      = 60        # Steering angle for a near green box (strong right)
new_servo_angle = 90        # Helper variable used in blue-backward servo updates
LEFT_COLLIDE_ANGLE = 120    # Steering angle on left side collision correctio
RIGHT_COLLIDE_ANGLE = 60    # Steering angle on right side collision correctio

# ---- Obstacle detection (vision) ----
MIN_AREA          = 2000    # Minimum contour area to accept as a box
MAX_AREA          = 20000   # Area at which box is considered "very close"
COLOR_HOLD_FRAMES = 3       # Frames the same color must persist to be “locked”

# ---- ToF thresholds (general) ----
SIDE_COLLIDE_CM       = 30.0  # If side < this, we steer away to avoid collision
TURN_SIDE_REQ_CM      = 40.0  # (legacy) side distance needed to allow turn
TURN_FRONT_REQ_CM     = 60.0  # (legacy) front distance needed to allow turn
TOF_TURN_FRONT_MAX_CM = 80.0  # Max front distance at which line-turn is valid
TOF_TURN_LEFT_MIN_CM  = 80.0  # (legacy) min left distance for some turn logic

# ---- Stuck / emergency escape thresholds ----
FRONT_STUCK_CM    = 10.0    # If front ToF < this and no boxes/line -> emergency reverse
BACK_CLEAR_CM     = 10.0   # Reverse until back ToF >= this
EMERGENCY_TURN_DEG = 60.0  # Degrees to turn during emergency escape after reversing

# ---- Blue-backward escape (box very close to front) ----
BLUE_BACK_DURATION = 0.7   # How long to reverse in blue-backward mode (seconds)

# ---- Obstacle avoidance (non-blue-backward) ----
AVOID_BACK_DURATION = 0.7  # Reverse duration in normal avoidance (seconds)

# ---- Line-turn (shape / Hough based) ----
TURN_LEFT_SERVO       = 60    # Servo angle for a hard left line-turn
TURN_IMU_TARGET_DEG   = 75.0  # (legacy) target yaw for IMU-based turn completion
TURN_COOLDOWN_S       = 0.7   # Minimal cooldown after a line detection
TURN_MIN_INTERVAL_S   = 5.0   # Minimum time between two turns
TURN_COMPLETE_DEG     = 90.0  # (legacy) yaw angle for considering turn complete
CANNY_LO, CANNY_HI    = 60, 160  # Canny edge thresholds for line band pre-processing
BLUR_KSIZE            = 5     # Gaussian blur kernel size for edge preprocessing
HOUGH_THRESHOLD       = 60    # HoughLinesP threshold
HOUGH_MIN_LENGTH      = 120   # Min line length to accept from HoughLinesP
HOUGH_MAX_GAP         = 20    # Max allowed gap in HoughLinesP segments
LINE_DETECT_CONSEC_FRAMES = 2 # Frames of consistent line detection to confirm
LINE_ORIENT_MIN_DEG   = 25    # Minimum angle (deg) to accept a line as “diagonal”
LINE_ORIENT_MAX_DEG   = 65    # Maximum angle (deg) to accept a line as “diagonal”
LINE_MASK_THICKNESS   = 9     # Thickness of mask drawn over detected line band

# ---- Turn-related constants ----
LINE_CENTER_Y_MIN          = 500  # (legacy) minimal Y for generic line center
LINE_CENTER_BLUE_Y_MIN     = 550  # Minimal Y for blue line to be valid for turn
LINE_CENTER_ORANGE_Y_MIN   = 600  # Minimal Y for orange line to be valid for turn
TURN_RIGHT_SERVO           = 120  # Servo angle for a hard right line-turn
TURN_MIN_YAW_DEG           = 75.0 # Min yaw before we allow stopping a turn (if used)
TURN_FAILSAFE_MAX_DEG      = 80.0 # Failsafe yaw to stop turn even without box

YAW_RESET_AFTER_LEFT  = -10.0     # Yaw offset after finishing a left turn
YAW_RESET_AFTER_RIGHT = 10.0      # Yaw offset after finishing a right turn

TURN_COOLDOWN_SEC = 6.0           # Cooldown after each completed turn (no new turns)

# HSV thresholds (tweak for venue lighting)
RED1_LO    = np.array([0,   120, 110], dtype=np.uint8)  # Red lower hue range 1
RED1_HI    = np.array([5,  255, 255], dtype=np.uint8)  # Red upper hue range 1
RED2_LO    = np.array([0, 120, 110], dtype=np.uint8)  # Red lower hue range 2
RED2_HI    = np.array([5, 255, 255], dtype=np.uint8)  # Red upper hue range 2
GREEN_LO   = np.array([70, 145, 70],  dtype=np.uint8)   # Green lower HSV bound
GREEN_HI   = np.array([80, 200, 160], dtype=np.uint8)   # Green upper HSV bound
ORANGE_LO  = np.array([6,  170, 170], dtype=np.uint8)   # Orange lower HSV bound
ORANGE_HI  = np.array([14, 210, 205], dtype=np.uint8)   # Orange upper HSV bound
BLUE_LO    = np.array([110, 100, 110], dtype=np.uint8)  # Blue lower HSV bound
BLUE_HI    = np.array([120, 211, 150], dtype=np.uint8)  # Blue upper HSV bound

# ---- Dynamic yaw / line trigger tuning ----
BLUE_Y_TRIGGER_FRAC = 0.78  # Fraction of image height for blue line trigger Y
BLUE_MIN_LEN_PX     = 70    # Minimal Hough line length (blue/orange) to trigger turn
TURN_YAW_MIN_DEG    = 80.0  # Minimum yaw target for dynamic line-turn yaw
TURN_YAW_MAX_DEG    = 90.0  # Maximum yaw target for dynamic line-turn yaw

# ---- Post-reverse / settling behavior ----
SETTLE_DURATION      = 0.0  # Seconds to force CENTER_ANGLE after certain events
settle_until_ts      = 0.0  # Timestamp until which settle is active
POST_BACK_FOLLOW_S   = 0.7  # Time to bias steering toward box after blue-backward

# ---- Extra IMU gain near boxes ----
BOX_YAW_GAIN_MIN = 1.5  # Minimum yaw gain when box just appears
BOX_YAW_GAIN_MAX = 3.0  # Maximum yaw gain when box very close

# ---- Drift control ----
DRIFT_GZ_THRESH       = 0.8   # Max |Gz| to consider for drift/bias update
BIAS_ALPHA            = 0.002 # Smoothing factor for gyro bias update
STRAIGHT_SERVO_WINDOW = 8     # Servo must be within this of CENTER to update bias
DRIFT_DECAY_RATE      = 0.20  # Yaw decay factor when re-centering

# ---- Stability gates ----
YAW_CLAMP_DEG     = 120.0 # Limit |yaw| to avoid runaway integration
SIDE_CLEAR_CM     = 40.0  # Both side distances > this -> path considered clear
SIDE_BALANCE_CM   = 8.0   # Max difference between left/right to consider balanced
CALM_GZ_THRESH    = 0.35  # |Gz| threshold for applying strong yaw decay
HARD_DECAY_RATE   = 2.0   # Strong yaw decay factor when sides are clear & calm
SOFT_DECAY_RATE   = 0.6   # Softer yaw decay factor used during bias update
CENTER_LOCK_WINDOW = 4    # Servo must be this close to CENTER for hard decay

# ---- Post-turn grace periods (avoid instant re-trigger) ----
POST_TURN_GRACE_GREEN_S = 0.75  # Ignore green box triggers for this long after turn
POST_TURN_GRACE_RED_S   = 0.75 # Ignore red box triggers for this long after turn

# ---- IMU keep-straight gains ----
YAW_KP_BASE             = 1.2  # Base proportional gain for yaw correction
SERVO_CORR_LIMIT_BASE   = 25   # Max correction (deg) from base yaw controller
YAW_DEADBAND_DEG_BASE   = 4.0  # Deadband for base yaw correction (small error ignored)
YAW_DEADBAND_DEG_STRONG = 6.0  # Deadband when using stronger yaw correction
YAW_KP_STRONG           = 1.3  # Stronger proportional gain when no boxes
SERVO_CORR_LIMIT_STRONG = 24   # Max correction (deg) for strong yaw controller

# ---- Smart Unpark configuration ----
UNPARK_CENTER_ANGLE     = CENTER_ANGLE # Servo angle for going straight during unpark
UNPARK_LEFT_TURN_ANGLE  = 55           # Initial unpark steering angle if turning left
UNPARK_RIGHT_TURN_ANGLE = 120          # Initial unpark steering angle if turning right

# ---- Turn / lap counting ----
TURNS_PER_LAP = 4   # Number of line-based turns per lap
TOTAL_LAPS    = 3   # Number of laps to complete
TOTAL_TURNS   = TURNS_PER_LAP * TOTAL_LAPS  # Convenience (total planned turns)

# ---- Line / ToF gates for turning ----
TURN_FRONT_MIN_CM = 18.0   # Min front distance to allow starting a line-based turn
TURN_FRONT_MAX_CM = 140.0  # Max front distance to allow starting a line-based turn
FIRST_TURN_FRONT_THRESH_CM = 70.0  # Extra gate for the very first turn


# =========================
# HARDWARE SETUP
# =========================

# ---- IMU SETUP ----
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


# ---- YAW TRACKING -----
yaw = 0.0
yaw_lock = threading.Lock()
last_time = time.time()

def reset_yaw_listener():
    global yaw
    while True:
        input("Press ENTER to reset yaw to 0°")
        with yaw_lock:
            yaw = 0.0
            print("Yaw reset to 0°")

threading.Thread(target=reset_yaw_listener, daemon=True).start()


def imu_center_servo(current_yaw_deg: float, deadband: float, kp: float, limit: float) -> int:
    if abs(current_yaw_deg) <= deadband:
        return CENTER_ANGLE
    corr = kp * current_yaw_deg
    corr = max(-limit, min(limit, corr))
    return int(CENTER_ANGLE + corr)

# ----- SERVO SETUP -----
set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
current_servo_angle = CENTER_ANGLE

# --- servo stabilizer ---
SERVO_SMOOTH_ALPHA = 0.35
SERVO_MIN_DELTA_DEG = 2.0

# ---- CAMERA SETUP ----
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (800, 480)})) #(main={"size": (640, 480)}))
picam2.set_controls({"FrameRate": 50})
picam2.start()

# ---- ToF SETUP ----
i2c = busio.I2C(board.SCL, board.SDA)
xshut_pins = {
    "left":    board.D16,
    "right":   board.D25,
    "front":   board.D26,
    "back":    board.D8,
    "front_l": board.D7,
    "front_r": board.D24
}
addresses = {
    "left":    0x30,
    "right":   0x31,
    "front":   0x32,
    "back":    0x33,
    "front_l": 0x34,
    "front_r": 0x35
}

xshuts = {}
for name, pin in xshut_pins.items():
    x = digitalio.DigitalInOut(pin)
    x.direction = digitalio.Direction.OUTPUT
    x.value = False
    xshuts[name] = x
time.sleep(0.1)

sensors = {}
for name in ["left", "right", "front", "back", "front_l", "front_r"]:
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
        if val <= 0 or val > 150:
            return 999
        return val
    except:
        return 999


# ---- Ultrasonic setup ----
# ==== ULTRASONIC PINS (FRONT, LEFT, RIGHT) ====
TRIG_FRONT, ECHO_FRONT = 22, 23  # GPIO pins for front sensor
TRIG_LEFT,  ECHO_LEFT  = 27, 17  # GPIO pins for left sensor
TRIG_RIGHT, ECHO_RIGHT = 5,  6   # GPIO pins for right sensor

# ==== ULTRASONIC SENSORS (gpiozero DistanceSensor) ====
front_ultra = DistanceSensor(echo=ECHO_FRONT, trigger=TRIG_FRONT, max_distance=2.4, queue_len=3)
left_ultra  = DistanceSensor(echo=ECHO_LEFT,  trigger=TRIG_LEFT,  max_distance=1.2, queue_len=3)
right_ultra = DistanceSensor(echo=ECHO_RIGHT, trigger=TRIG_RIGHT, max_distance=1.2, queue_len=3)

def ultra_cm(sensor, max_cm=200.0):
    """
    Read a gpiozero DistanceSensor and return distance in cm.
    Returns 999 on invalid / out-of-range.
    """
    d_m = sensor.distance  # in meters, between 0 and max_distance
    if d_m is None:
        return 999
    d_cm = d_m * 100.0
    if d_cm <= 0 or d_cm > max_cm:
        return 999
    return d_cm

def get_front_ultra_cm():
    return ultra_cm(front_ultra)

def get_left_ultra_cm():
    return ultra_cm(left_ultra)

def get_right_ultra_cm():
    return ultra_cm(right_ultra)



# ==== HELPERS ====

# ==== STATE PRINTING ====
run_state = None

def set_run_state(new_state: str):
    """
    Print state changes only when the label actually changes.
    """
    global run_state
    if new_state != run_state:
        run_state = new_state
        print(f"[STATE] {new_state}", flush=True)



def get_largest_contour(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(largest_contour)
    return largest_contour, (x, y), (x + w, y + h), w*h

def thin_shape_reject(candidate, min_extent=0.30, ar_lo=0.35, ar_hi=3.0):
    if not candidate:
        return None
    cnt, tl, br, area = candidate
    x1, y1 = tl; x2, y2 = br
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    ar = w / float(h)
    if not (ar_lo <= ar <= ar_hi):
        return None
    extent = cv2.contourArea(cnt) / float(w * h)
    if extent < min_extent:
        return None
    return candidate

def compute_servo_angle(color, area):
    norm_area = max(MIN_AREA, min(MAX_AREA, area))
    closeness = (norm_area - MIN_AREA) / (MAX_AREA - MIN_AREA)
    if color == "Red":
        return int(LEFT_FAR + closeness * (LEFT_NEAR - LEFT_FAR))
    else:
        return int(RIGHT_FAR - closeness * (RIGHT_FAR - RIGHT_NEAR))

def boxes_intersect(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def valid_orientation(x1,y1,x2,y2):
    dx = x2-x1; dy = y2-y1
    ang = abs(np.degrees(np.arctan2(dy, dx)))
    return LINE_ORIENT_MIN_DEG <= ang <= LINE_ORIENT_MAX_DEG

def max_line_len(lines):
    if lines is None:
        return 0
    m = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        L = int(np.hypot(x2 - x1, y2 - y1))
        if L > m:
            m = L
    return m

def preprocess_edges(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    return edges

def detect_line_and_mask(edges, h, w):
    band_y1, band_y2 = int(h*0.45), int(h*0.90)
    roi = edges.copy()
    roi[:band_y1,:] = 0
    roi[band_y2:,:] = 0
    lines = cv2.HoughLinesP(
        roi, rho=1, theta=np.pi/180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )
    line_mask = np.zeros((h, w), dtype=np.uint8)
    seg = None
    if lines is not None:
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            if valid_orientation(x1,y1,x2,y2):
                if seg is None:
                    seg = (x1,y1,x2,y2)
                cv2.line(line_mask, (x1,y1), (x2,y2), 255, LINE_MASK_THICKNESS)
    return (seg is not None), seg, line_mask

def y_at_center(lines, center_x):
    if lines is None:
        return -1
    ys = []
    for x1,y1,x2,y2 in lines[:,0]:
        if x1 != x2:
            m = (y2 - y1) / float(x2 - x1)
            y = m * (center_x - x1) + y1
            ys.append(y)
        else:
            if x1 == center_x:
                ys.append(max(y1, y2))
    return max(ys) if ys else -1

# ===============================
# SMART UNPARK (two-phase; faster; left-case phase 2 = 65° + extra ~60° left if direction == "left")
# ===============================
start_button = Button(20)


print("\n=== SMART UNPARK START ===")
start_button.wait_for_press()
print("Button pressed! Starting sequence...")
time.sleep(1)

left_dist = tof_cm(sensors["left"])
right_dist = tof_cm(sensors["right"])
print(f"Left={left_dist:.1f}cm | Right={right_dist:.1f}cm")

if left_dist > right_dist:
    direction = "left"
    print("➡ Choosing LEFT (more open).")
else:
    direction = "right"
    print("➡ Choosing RIGHT (more open).")

# Phase 1: ~45°
with yaw_lock:
    yaw = 0.0
last_time = time.time()
first_angle = UNPARK_LEFT_TURN_ANGLE if direction == "left" else UNPARK_RIGHT_TURN_ANGLE
set_servo_angle(SERVO_CHANNEL, first_angle)
set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_STRAIGHT_SPEED)
while True:
    now = time.time(); dt = now - last_time; last_time = now
    Gz = (read_raw_data(GYRO_ZOUT_H)/131.0) - gyro_z_bias
    with yaw_lock:
        yaw += Gz * dt; current_yaw = yaw
    if abs(current_yaw) >= 45.0:
        break

# Phase 2: ~40°
with yaw_lock:
    yaw = 0.0
last_time = time.time()
if direction == "left":
    second_angle = UNPARK_RIGHT_TURN_ANGLE; target_abs_yaw = 40.0
else:
    second_angle = UNPARK_LEFT_TURN_ANGLE;  target_abs_yaw = 40.0
set_servo_angle(SERVO_CHANNEL, second_angle)
set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_STRAIGHT_SPEED)
while True:
    now = time.time(); dt = now - last_time; last_time = now
    Gz = (read_raw_data(GYRO_ZOUT_H)/131.0) - gyro_z_bias
    with yaw_lock:
        yaw += Gz * dt; current_yaw = yaw
    if abs(current_yaw) >= target_abs_yaw:
        break

# Extra only for initial "left": ~60° left
if direction == "left":
    with yaw_lock:
        yaw = 0.0
    last_time = time.time()
    set_servo_angle(SERVO_CHANNEL, UNPARK_LEFT_TURN_ANGLE)
    set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_STRAIGHT_SPEED)
    while True:
        now = time.time(); dt = now - last_time; last_time = now
        Gz = (read_raw_data(GYRO_ZOUT_H)/131.0) - gyro_z_bias
        with yaw_lock:
            yaw += Gz * dt; current_yaw = yaw
        if abs(current_yaw) >= 60.0:
            break

# Straighten & reset yaw
set_servo_angle(SERVO_CHANNEL, UNPARK_CENTER_ANGLE)
set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_STRAIGHT_SPEED)
with yaw_lock:
    yaw = 0.0
print("[DONE] Unpark sequence complete. Entering vision/avoid loop...")
set_run_state("cruise")

# ==== STATE VARIABLES ====
last_color = None
frame_count = 0
last_update_time = time.time()
state = "normal"
state_start = time.time()

line_seen_streak = 0
last_turn_end_time = -1.0
turn_active = False
turn_dir = None
box_seen_while_turning = False
next_turn_allowed_time = 0.0
straight_lock_time = 0.0

# track if a box was seen during the current turn
turn_box_seen = False
turn_target_deg_active = None

# post-reverse follow state
back_follow_angle = None
post_back_follow_until = 0.0

# turn confirmation state
blue_gate_streak = 0

# first-turn gate remains
first_turn_gate_active = True

# NEW: emergency stuck escape state
emergency_mode = False
emergency_phase = None  # "reverse" -> "turn"
emergency_direction = None  # "left" or "right"

blue_backward_start = None
in_blue_backward = False

lines_blue = 0
lines_orange = 0

# ==== LAP & TURN COUNTERS ====
turn_count = 1   # total turns completed
lap_count = 1    # current lap (1-based)


try:
    motors_started = True
    avoidance_mode = False
    avoid_direction = None
    avoid_start_time = None

    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
    current_servo_angle = CENTER_ANGLE
    last_update_time = time.time()

    set_motor_speed(MOTOR_FWD, MOTOR_REV, STOP_SPEED)
    time.sleep(0.2)

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        raw_gz_dps = read_raw_data(GYRO_ZOUT_H)/131.0
        Gz = raw_gz_dps - gyro_z_bias
        with yaw_lock:
            yaw += Gz * dt
            if yaw > YAW_CLAMP_DEG: yaw = YAW_CLAMP_DEG
            elif yaw < -YAW_CLAMP_DEG: yaw = -YAW_CLAMP_DEG
            current_yaw = yaw

        if (not in_blue_backward) and (not avoidance_mode) and (not turn_active) and (not emergency_mode):
            near_center = abs(current_servo_angle - CENTER_ANGLE) <= STRAIGHT_SERVO_WINDOW
            if near_center and abs(raw_gz_dps) < DRIFT_GZ_THRESH:
                gyro_z_bias = (1.0 - BIAS_ALPHA)*gyro_z_bias + BIAS_ALPHA*raw_gz_dps
                yaw *= (1.0 - min(1.0, SOFT_DECAY_RATE * dt))

        img = picam2.capture_array()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_img, w_img = img_bgr.shape[:2]
        blue_y_trig = int(h_img * BLUE_Y_TRIGGER_FRAC)

        edges = preprocess_edges(img_bgr)
        line_seen_raw, line_seg, line_mask = detect_line_and_mask(edges, h_img, w_img)
        time_since_last_turn = (time.time() - last_turn_end_time)
        out_of_cooldown = (time_since_last_turn >= TURN_COOLDOWN_S)
        out_of_min_interval = (time_since_last_turn >= TURN_MIN_INTERVAL_S)

        if not turn_active and out_of_cooldown and out_of_min_interval and not emergency_mode:
            line_seen_streak = line_seen_streak + 1 if line_seen_raw else 0
            line_seen = line_seen_streak >= LINE_DETECT_CONSEC_FRAMES
        else:
            line_seen_streak = 0
            line_seen = False

        # ====== COLOR & LINE MASKS (using global HSV thresholds) ======

        # ORANGE & BLUE for line detection (Hough)
        mask_orange = cv2.inRange(imgHSV, ORANGE_LO, ORANGE_HI)
        edges_orange = cv2.Canny(mask_orange, 50, 150)

        mask_blue = cv2.inRange(imgHSV, BLUE_LO, BLUE_HI)
        edges_blue = cv2.Canny(mask_blue, 50, 150)

        lines_orange = cv2.HoughLinesP(
            edges_orange, 1, np.pi / 180,
            threshold=50, minLineLength=50, maxLineGap=10
        )
        lines_blue = cv2.HoughLinesP(
            edges_blue, 1, np.pi / 180,
            threshold=50, minLineLength=50, maxLineGap=10
        )

        blue_len_max   = max_line_len(lines_blue)
        orange_len_max = max_line_len(lines_orange)

        img_lines = img.copy()

        center_x = img.shape[1] // 2
        orange_y = y_at_center(lines_orange, center_x)
        blue_y   = y_at_center(lines_blue, center_x)

        # PINK (exclude from red)
        mask_pink = cv2.inRange(
            imgHSV,
            np.array([140, 60, 120], dtype=np.uint8),
            np.array([170, 255, 255], dtype=np.uint8),
        )

        # RED using global thresholds
        mask_red1 = cv2.inRange(imgHSV, RED1_LO, RED1_HI)
        mask_red2 = cv2.inRange(imgHSV, RED2_LO, RED2_HI)
        mask_red  = cv2.bitwise_or(mask_red1, mask_red2)

        # Remove orange and pink from red
        mask_red  = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_orange))
        mask_red  = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_pink))
        if line_mask is not None:
            mask_red = cv2.bitwise_and(mask_red, cv2.bitwise_not(line_mask))

        # GREEN using global thresholds
        mask_green = cv2.inRange(imgHSV, GREEN_LO, GREEN_HI)
        if line_mask is not None:
            mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(line_mask))

        k3 = np.ones((3,3), np.uint8)
        k5 = np.ones((5,5), np.uint8)
        def morph(m):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
            return m
        mask_red   = morph(mask_red)
        mask_green = morph(mask_green)

        img_contours = img_bgr.copy()
        boxes = []

        red_data = thin_shape_reject(get_largest_contour(mask_red, min_area=MIN_AREA))
        if red_data:
            _, top_left, bottom_right, area = red_data
            boxes.append(("Red", area, (*top_left, *bottom_right)))
            cv2.rectangle(img_contours, top_left, bottom_right, (0,0,255), 2)
            cv2.putText(img_contours, f"Red {int(area)}", (top_left[0], max(20, top_left[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        green_data = thin_shape_reject(get_largest_contour(mask_green, min_area=MIN_AREA))
        if green_data:
            _, top_left, bottom_right, area = green_data
            boxes.append(("Green", area, (*top_left, *bottom_right)))
            cv2.rectangle(img_contours, top_left, bottom_right, (0,255,0), 2)
            cv2.putText(img_contours, f"Green {int(area)}", (top_left[0], max(20, top_left[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        red_seen = red_data is not None
        green_seen = green_data is not None

        # mark that we saw a box during the active turn
        if turn_active and len(boxes) > 0:
            turn_box_seen = True

        center_x = img_contours.shape[1] // 2
        car_width, car_height = 500, 150
        bottom_y = img_contours.shape[0] - 10
        car_box = (center_x - car_width//2, bottom_y - car_height,
                   center_x + car_width//2, bottom_y)

        if line_seg:
            x1,y1,x2,y2 = line_seg
            cv2.line(img_contours, (x1,y1), (x2,y2), (0,255,255), 3)

        cv2.rectangle(img_contours, (car_box[0],car_box[1]), (car_box[2],car_box[3]), (255,255,255), 1)

        target_angle = CENTER_ANGLE
        motor_speed = 0

        # --------- NEW: EMERGENCY STUCK ESCAPE ----------
        f_cm = tof_cm(sensors["front"])
        l_cm = tof_cm(sensors["left"])
        r_cm = tof_cm(sensors["right"])
        b_cm = tof_cm(sensors["back"])

        # Ultrasonic distances (front / left / right) in cm
        f_ultra = get_front_ultra_cm()
        l_ultra = get_left_ultra_cm()
        r_ultra = get_right_ultra_cm()


        #if (not emergency_mode) and (not turn_active) and (not in_blue_backward) and (not avoidance_mode):
        #    no_boxes = (len(boxes) == 0)
        #    no_line  = (not line_seen)
        #    if no_boxes and no_line and f_cm < FRONT_STUCK_CM:
        #        emergency_mode = True
        #        emergency_phase = "reverse"
        #        set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
        #        set_motor_speed(MOTOR_FWD, MOTOR_REV, -EMERGENCY_BACK_SPEED)

        if emergency_mode:
            if emergency_phase == "reverse":
                set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -EMERGENCY_BACK_SPEED)
                b_cm = tof_cm(sensors["back"])
                if b_cm >= BACK_CLEAR_CM:
                    emergency_phase = "turn"
                    #emergency_direction = "left" if l_cm >= r_cm else "right"
                    emergency_direction = "left" if l_ultra >= r_ultra else "right"
                    with yaw_lock:
                        yaw = 0.0
                    last_time = time.time()
                    turn_servo = LEFT_NEAR if emergency_direction == "left" else RIGHT_NEAR
                    set_servo_angle(SERVO_CHANNEL, turn_servo)
                    set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)

            elif emergency_phase == "turn":
                target_angle = LEFT_NEAR if emergency_direction == "left" else RIGHT_NEAR
                set_servo_angle(SERVO_CHANNEL, target_angle)
                set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)
                if abs(current_yaw) >= EMERGENCY_TURN_DEG:
                    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
                    with yaw_lock:
                        yaw = 0.0
                    emergency_mode = False
                    emergency_phase = None
                    emergency_direction = None
                    settle_until_ts = time.time() + SETTLE_DURATION
                if cv2.waitKey(1) in [27, ord('q')]:
                    break
                continue
        # --------- END: EMERGENCY STUCK ESCAPE ----------

        # ----- TURN decision (BLUE OR ORANGE, turn side from parking 'direction') -----
        Turn = "No"

        blue_trigger = (blue_y  >= LINE_CENTER_BLUE_Y_MIN   and blue_len_max   >= BLUE_MIN_LEN_PX)
        orange_trigger = (orange_y >= LINE_CENTER_ORANGE_Y_MIN and orange_len_max >= BLUE_MIN_LEN_PX)

        line_trigger = blue_trigger or orange_trigger

        if line_trigger:
            # use initial parking direction ONLY for deciding left/right
            if direction == "left":
                Turn = "Left"
            else:
                Turn = "Right"

        if line_trigger and TURN_FRONT_MIN_CM <= f_ultra <= TURN_FRONT_MAX_CM:
            blue_gate_streak += 1
        else:
            blue_gate_streak = 0

        ok_to_turn = (
            blue_gate_streak >= LINE_DETECT_CONSEC_FRAMES and
            out_of_cooldown and out_of_min_interval and
            (not in_blue_backward) and (not turn_active)
        )

        # ToF gate so we DON'T start the turn when already at the wall
        #tof_line_turn_gate = (TURN_FRONT_MIN_CM <= f_cm <= TURN_FRONT_MAX_CM)
        tof_line_turn_gate = (TURN_FRONT_MIN_CM <= f_ultra <= TURN_FRONT_MAX_CM)
       
        # choose which line y-position to use (whichever of blue/orange is lower in the image)
        valid_ys = []
        if blue_trigger and blue_y >= 0:
            valid_ys.append(blue_y)
        if orange_trigger and orange_y >= 0:
            valid_ys.append(orange_y)
        line_y_for_turn = max(valid_ys) if valid_ys else (blue_y if blue_y >= 0 else orange_y)

        now_ts = time.time()

        # ----- TURN state machine (yaw-based, no time.sleep) -----
        if (Turn in ("Left", "Right")
            and not turn_active
            and not in_blue_backward
            and not avoidance_mode
            and not emergency_mode
            and now_ts >= next_turn_allowed_time
            and ok_to_turn
            and tof_line_turn_gate):

            # start a turn
            turn_active = True
            turn_dir = Turn
            box_seen_while_turning = False

            # which line color triggered?
            if blue_trigger and not orange_trigger:
                line_label = "blue line found – turning " + Turn.lower()
            elif orange_trigger and not blue_trigger:
                line_label = "orange line found – turning " + Turn.lower()
            else:
                line_label = "line found – turning " + Turn.lower()

            set_run_state(line_label)


            # dynamic yaw target (kept for reference)
            y0 = int(0.65 * h_img)
            yr = max(1, int(0.35 * h_img))
            t_dyn = (line_y_for_turn - y0) / float(yr)
            if t_dyn < 0: t_dyn = 0.0
            if t_dyn > 1: t_dyn = 1.0
            turn_target_deg_active = TURN_YAW_MIN_DEG + t_dyn * (TURN_YAW_MAX_DEG - TURN_YAW_MIN_DEG)

            with yaw_lock:
                yaw = 0.0      # reset yaw at turn start
            last_time = now_ts

        if turn_active:
            # steer hard according to direction and drive forward
            if turn_dir == "Left":
                target_angle = TURN_LEFT_SERVO
            else:
                target_angle = TURN_RIGHT_SERVO

            set_servo_angle(SERVO_CHANNEL, target_angle)
            set_motor_speed(MOTOR_FWD, MOTOR_REV, TURN_MOTOR_SPEED)

            # if any red/green box seen while turning, mark it (kept but not used for stop)
            if red_seen or green_seen:
                box_seen_while_turning = True

            # stop rule: full ~90° yaw rotation (failsafe only)
            failsafe_yaw = abs(current_yaw) >= TURN_FAILSAFE_MAX_DEG

            if failsafe_yaw:
                # straighten wheels
                set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)

                # set yaw to a defined offset after turn
                if turn_dir == "Left":
                    with yaw_lock:
                        yaw = YAW_RESET_AFTER_LEFT
                else:
                    with yaw_lock:
                        yaw = YAW_RESET_AFTER_RIGHT

                current_servo_angle = CENTER_ANGLE
                turn_active = False
                turn_dir = None
                box_seen_while_turning = False

                # ====== TURN / LAP COUNTING ======
                turn_count += 1

                # turn number within current lap (1–4)
                turn_in_lap = ((turn_count - 1) % TURNS_PER_LAP) + 1
                # lap number (1–3)
                lap_count = (turn_count - 1) // TURNS_PER_LAP + 1

                print(f"[LAP] turn {turn_in_lap} / lap {lap_count}", flush=True)

                # If we have finished ALL laps (3 laps * 4 turns = 12 turns)
                if turn_count >= TOTAL_TURNS:
                    print("[LAP] All 3 laps completed. Driving straight for 1s then stopping.", flush=True)
                    # go straight for 1 second
                    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
                    set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)
                    time.sleep(3.5)
                    # stop completely
                    set_motor_speed(MOTOR_FWD, MOTOR_REV, STOP_SPEED)
                    break  # exit the while True loop (will go to finally:)

                # ====== NORMAL COOL-DOWN AFTER TURN (if more laps remain) ======
                next_turn_allowed_time = time.time() + TURN_COOLDOWN_SEC
                last_turn_end_time = time.time()
                settle_until_ts = time.time() + SETTLE_DURATION

                if first_turn_gate_active:
                    first_turn_gate_active = False

                set_run_state("cruise")

                # skip the rest of your normal logic this frame
                continue


        # ==== BLUE-BACKWARD LOGIC (immediate if intersect) ====
        if not in_blue_backward and not turn_active and not emergency_mode:
            now_ts = time.time()
            if red_data and boxes_intersect(car_box, (*red_data[1], *red_data[2])):
                if now_ts - last_turn_end_time >= POST_TURN_GRACE_RED_S:
                    in_blue_backward = True
                    blue_backward_start = now_ts
                    target_angle = LEFT_NEAR
                    set_motor_speed(MOTOR_FWD, MOTOR_REV, -BLUE_BACK_SPEED)
                    set_run_state("red obstacle found – blue-backward escape")
                    # Center while reversing and remember follow direction
                    back_follow_angle = LEFT_NEAR
                    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
                    current_servo_angle = CENTER_ANGLE
            elif green_data and boxes_intersect(car_box, (*green_data[1], *green_data[2])):
                if now_ts - last_turn_end_time >= POST_TURN_GRACE_GREEN_S:
                    in_blue_backward = True
                    blue_backward_start = now_ts
                    target_angle = RIGHT_NEAR
                    set_motor_speed(MOTOR_FWD, MOTOR_REV, -BLUE_BACK_SPEED)
                    set_run_state("green obstacle found – blue-backward escape")
                    # Center while reversing and remember follow direction
                    back_follow_angle = RIGHT_NEAR
                    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
                    current_servo_angle = CENTER_ANGLE

        if in_blue_backward:
            # Hold CENTER while reversing
            current_servo_angle = CENTER_ANGLE
            set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)

            new_servo_angle += CENTER_ANGLE - current_servo_angle
            set_servo_angle(SERVO_CHANNEL, new_servo_angle)
            set_servo_angle(SERVO_CHANNEL, current_servo_angle)

            if time.time() - blue_backward_start >= BLUE_BACK_DURATION:
                with yaw_lock:
                    yaw = 0.0 
                in_blue_backward = False
                new_servo_angle = CENTER_ANGLE
                new_servo_angle += CENTER_ANGLE - current_servo_angle
                set_servo_angle(SERVO_CHANNEL, new_servo_angle)
                settle_until_ts = time.time() + SETTLE_DURATION
                motor_speed = NORMAL_SPEED
                set_motor_speed(MOTOR_FWD, MOTOR_REV, motor_speed)
                set_run_state("cruise")
                # Follow the remembered box direction for a short window
                if back_follow_angle is not None:
                    set_servo_angle(SERVO_CHANNEL, back_follow_angle)
                    current_servo_angle = back_follow_angle
                    post_back_follow_until = time.time() + POST_BACK_FOLLOW_S
            else:
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -BLUE_BACK_SPEED)
                if cv2.waitKey(1) in [27, ord('q')]:
                    break
                continue

        # ==== AVOIDANCE LOGIC (boxes) ====
        if not avoidance_mode and not emergency_mode:
            now_ts = time.time()
            for color, _, box_coords in boxes:
                if boxes_intersect(car_box, box_coords):
                    grace = POST_TURN_GRACE_GREEN_S if color == "Green" else POST_TURN_GRACE_RED_S
                    if now_ts - last_turn_end_time < grace:
                        continue
                    avoidance_mode = True
                    avoid_direction = "right" if color == "Green" else "left"
                    avoid_start_time = now_ts
                    set_motor_speed(MOTOR_FWD, MOTOR_REV, -AVOID_SPEED)
                    target_angle = RIGHT_NEAR if avoid_direction == "right" else LEFT_NEAR
                    state = "avoid"
                    state_start = now_ts
                    set_run_state(f"avoidance – {color.lower()} obstacle")
                    break


        if avoidance_mode:
            elapsed = time.time() - (avoid_start_time or time.time())
            if elapsed < AVOID_BACK_DURATION:
                set_servo_angle(SERVO_CHANNEL, RIGHT_NEAR if avoid_direction == "right" else LEFT_NEAR)
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -AVOID_SPEED)
            else:
                set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)
                target_angle = imu_center_servo(current_yaw, YAW_DEADBAND_DEG_BASE, YAW_KP_BASE, SERVO_CORR_LIMIT_BASE)
                if not any(boxes_intersect(car_box, b[2]) for b in boxes):
                    avoidance_mode = False
                    avoid_direction = None
                    state = "normal"
                    set_run_state("cruise")
                    with yaw_lock:
                        yaw = 0.0
                    settle_until_ts = time.time() + SETTLE_DURATION

        # ==== NORMAL FORWARD + IMU STRAIGHTENING =====
        if not avoidance_mode and not in_blue_backward and not turn_active and state == "normal" and not emergency_mode:
            motor_speed = NORMAL_SPEED
            set_motor_speed(MOTOR_FWD, MOTOR_REV, motor_speed)

            if len(boxes) == 0:
                set_run_state("cruise")

                target_angle = imu_center_servo(
                    current_yaw,
                    YAW_DEADBAND_DEG_STRONG ,
                    YAW_KP_STRONG,
                    SERVO_CORR_LIMIT_STRONG
                )

                l_cm = tof_cm(sensors["left"])
                r_cm = tof_cm(sensors["right"])
                #l = l_cm
                #r = r_cm
                l = l_ultra
                r = r_ultra
                if l < SIDE_COLLIDE_CM and r >= SIDE_COLLIDE_CM:
                    target_angle = LEFT_COLLIDE_ANGLE
                elif r < SIDE_COLLIDE_CM and l >= SIDE_COLLIDE_CM:
                    target_angle = RIGHT_COLLIDE_ANGLE

                if (l > SIDE_CLEAR_CM and r > SIDE_CLEAR_CM and abs(l - r) < SIDE_BALANCE_CM
                    and abs(raw_gz_dps) < CALM_GZ_THRESH
                    and abs(current_servo_angle - CENTER_ANGLE) <= CENTER_LOCK_WINDOW):
                    with yaw_lock:
                        yaw *= (1.0 - min(1.0, HARD_DECAY_RATE * dt))
                        if abs(yaw) < 1.2:
                            yaw = 0.0

            else:
                target_angle = imu_center_servo(
                    current_yaw,
                    YAW_DEADBAND_DEG_BASE,
                    YAW_KP_BASE,
                    SERVO_CORR_LIMIT_BASE
                )
                boxes.sort(key=lambda b: b[1], reverse=True)
                chosen_color, chosen_area, box_coords = boxes[0]

                if last_color == chosen_color:
                    frame_count += 1
                else:
                    frame_count = 0
                    last_color = chosen_color

                if frame_count >= COLOR_HOLD_FRAMES:
                    # lock onto this obstacle color
                    if chosen_color == "Red":
                        set_run_state("red obstacle found")
                    elif chosen_color == "Green":
                        set_run_state("green obstacle found")
                    else:
                        set_run_state(f"{chosen_color.lower()} obstacle found")
                    
                    # --- AREA-ONLY scaling (far -> less turn, near -> more turn) + IMU yaw term ---
                    area_angle = compute_servo_angle(chosen_color, chosen_area)

                    # compute IMU yaw gain based ONLY on box area (Option 1 but IMU-only)
                    norm_area = max(MIN_AREA, min(MAX_AREA, chosen_area))
                    closeness = (norm_area - MIN_AREA) / (MAX_AREA - MIN_AREA + 1e-6)
                    yaw_gain = BOX_YAW_GAIN_MIN + closeness * (BOX_YAW_GAIN_MAX - BOX_YAW_GAIN_MIN)
                    

                    if chosen_color == "Red":
                        target_angle = max(60, min(130, int(area_angle + int(current_yaw * yaw_gain))))
                    elif chosen_color == "Green":
                        target_angle = max(60, min(130, int(area_angle - int(current_yaw * yaw_gain))))
                    else:
                        target_angle = max(75, min(105, area_angle))

        # After reverse, briefly bias steering toward the box direction
        t_now = time.time()
        if (not avoidance_mode) and (not in_blue_backward) and (not turn_active) and (not emergency_mode):
            if t_now < post_back_follow_until and back_follow_angle is not None:
                target_angle = back_follow_angle

        # ==== SERVO OUTPUT (smoothing & deadband) ====
        #if time.time() - last_update_time > SERVO_UPDATE_DELAY:
        #    fast = (avoidance_mode or in_blue_backward or turn_active or emergency_mode)
        #    step = FAST_SERVO_STEP if fast else STEP

        #    if time.time() < settle_until_ts and not turn_active and not in_blue_backward and not emergency_mode:
        #        desired = CENTER_ANGLE
        #    else:
        #        desired = target_angle

        #    smoothed = int(round((1.0 - SERVO_SMOOTH_ALPHA) * current_servo_angle + SERVO_SMOOTH_ALPHA * desired))
        #    if abs(smoothed - current_servo_angle) <= SERVO_MIN_DELTA_DEG:
        #        smoothed = current_servo_angle

        #    if abs(current_servo_angle - smoothed) > step:
        #        current_servo_angle += step if current_servo_angle < smoothed else -step
        #    else:
        #        current_servo_angle = smoothed

        #    current_servo_angle = max(50, min(130, current_servo_angle))
        #    set_servo_angle(SERVO_CHANNEL, current_servo_angle)
        #    last_update_time = time.time()
        
        # ==== SERVO OUTPUT (direct angle) ====
        # Directly set the servo to the desired angle (no smoothing)
        if time.time() < settle_until_ts and not turn_active and not in_blue_backward and not emergency_mode:
            desired = CENTER_ANGLE
        else:
            desired = target_angle

        current_servo_angle = max(65, min(115, desired))
        set_servo_angle(SERVO_CHANNEL, current_servo_angle)


        # cv2.imshow('Contours', img_contours)

        if lines_orange is not None:
            for x1, y1, x2, y2 in lines_orange[:, 0]:
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if lines_blue is not None:
            for x1, y1, x2, y2 in lines_blue[:, 0]:
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 165, 255), 2)
        if cv2.waitKey(1) in [27, ord('q')]:
            break

    # cv2.imshow('Lines', img_lines)
    time.sleep(0.1)

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
    set_motor_speed(MOTOR_FWD, MOTOR_REV, STOP_SPEED)
    # Close ultrasonic sensors
    front_ultra.close()
    left_ultra.close()
    right_ultra.close()
#red = 13
#green = 19  
#blue = 11


# ===============================================
# VIVA LaVida – FE 2025 Combined Navigation Script
# - BOX AVOIDANCE (camera red/green)
# - TURN DETECTION (blue/orange line) with "keep turning" rule
# - SMART UNPARK (ToF + button)
# - DEBUG LEDs: red (GPIO13), green (GPIO19), blue (GPIO11)
# - Area-based steering: farther box -> less turn, closest -> full turn
# Raspberry Pi 5 + Picamera2 + Adafruit PCA9685 + MPU6050 + VL53L0X
# ===============================================

import time
import threading
import cv2
import numpy as np
import smbus2
import RPi.GPIO as GPIO
from gpiozero import Button
from picamera2 import Picamera2
import board
import busio
import digitalio
import adafruit_vl53l0x
from adafruit_pca9685 import PCA9685

# ---------- CONSTANTS ----------
LOOP_DT_TARGET = 0.02
SHOW_WINDOWS = True

MOTOR_FWD = 1
MOTOR_REV = 2
SERVO_CHANNEL = 0

CENTER_ANGLE = 90
LEFT_NEAR  = 120
RIGHT_NEAR = 60
LEFT_FAR   = 110
RIGHT_FAR  = 70
SERVO_MIN  = 60
SERVO_MAX  = 120
STEP = 3
FAST_SERVO_STEP = 8
SERVO_UPDATE_DELAY = 0.015

NORMAL_SPEED = 20
AVOID_SPEED  = 20
BLUE_BACK_SPEED = 20

BLUE_BACK_DURATION = 1.0
AVOID_BACK_DURATION = 1.0

MIN_AREA = 1500
MAX_AREA = 9000
COLOR_HOLD_FRAMES = 4

AREA_RESPONSE_GAMMA = 0.9
AREA_GAIN = 1.35

CAM_SIZE = (1920, 1080)
CAR_BOX_WIDTH  = 350
CAR_BOX_HEIGHT = 100
CAR_BOX_BOTTOM_MARGIN = 10

LINE_CENTER_Y_MIN = 500
TURN_LEFT_SERVO  = 60
TURN_RIGHT_SERVO = 120
TURN_MIN_YAW_DEG = 75.0
TURN_FAILSAFE_MAX_DEG = 90.0
TURN_MOTOR_SPEED = 22

YAW_RESET_AFTER_LEFT  = 0.0
YAW_RESET_AFTER_RIGHT = 0.0

MPU6050_ADDR = 0x68
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C
INT_ENABLE   = 0x38
GYRO_ZOUT_H  = 0x47
YAW_CLAMP_DEG = 120.0
YAW_KP            = 1.3
YAW_DEADBAND_DEG  = 2.0
SERVO_CORR_LIMIT  = 28
YAW_BOX_GAIN      = 3.8

RED1_LO = np.array([0,   140, 120], dtype=np.uint8)
RED1_HI = np.array([5,  255, 190], dtype=np.uint8)
RED2_LO = np.array([0, 140, 120], dtype=np.uint8)
RED2_HI = np.array([5, 255, 190], dtype=np.uint8)
GREEN_LO = np.array([65, 90, 90], dtype=np.uint8)
GREEN_HI = np.array([80, 255, 150], dtype=np.uint8)
ORANGE_LO = np.array([6,  140, 180], dtype=np.uint8)
ORANGE_HI = np.array([14, 200, 210], dtype=np.uint8)
BLUE_LO   = np.array([112,  140, 120], dtype=np.uint8)
BLUE_HI   = np.array([118, 230, 160], dtype=np.uint8)

LED_BLUE_PIN  = 11
LED_RED_PIN   = 13
LED_GREEN_PIN = 19

START_BUTTON_PIN = 20
UNPARK_SPEED = 25
UNPARK_LEFT_TURN_ANGLE  = 58
UNPARK_RIGHT_TURN_ANGLE = 120
UNPARK_PHASE1_TARGET_DEG = 45.0
UNPARK_PHASE2_TARGET_DEG = 40.0
UNPARK_EXTRA_LEFT_DEG    = 55.0

XSHUT_PINS = {
    "left":         board.D16,
    "right":        board.D25,
    "front":        board.D26,
    "front_right":  board.D24,
    "front_left":   board.D7,
    "back":         board.D8,
}
TOF_ADDR = {
    "left":        0x30,
    "right":       0x31,
    "front":       0x32,
    "back":        0x33,
    "front_right": 0x34,
    "front_left":  0x35,
}
TOF_RANGE_BAD = 999.0
TOF_VALID_MAX = 150.0

# 5s no-repeat turn cooldown (no time.sleep in loop)
TURN_COOLDOWN_SEC = 5.0

# Pre-emptive hard steer before reverse
PREEMPT_AREA = int(MAX_AREA * 0.55)
DANGER_MARGIN = 60

# Adafruit PCA9685 pulse mapping
SERVO_PULSE_MIN = 1000
SERVO_PULSE_MAX = 2000
SERVO_PERIOD = 20000

# ---------- UTILS ----------
# Hough params for lines
HOUGH_THRESH = 50
MIN_LINE_LEN = 50
MAX_LINE_GAP = 10

# Morph kernel to clean masks a bit (optional)
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def draw_boxes(frame_bgr, mask, bgr_color, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), bgr_color, 2)
        cv2.putText(frame_bgr, f"{label} {int(area)}", (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1, cv2.LINE_AA)

def draw_lines(frame_bgr, mask, bgr_color, label):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESH,
                            minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(frame_bgr, (x1, y1), (x2, y2), bgr_color, 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.putText(frame_bgr, label, (cx+4, cy-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1, cv2.LINE_AA)
            

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

def _area_closeness(area):
    a = clamp(area, MIN_AREA, MAX_AREA)
    frac = (a - MIN_AREA) / float(MAX_AREA - MIN_AREA + 1e-6)
    return pow(clamp(frac, 0.0, 1.0), AREA_RESPONSE_GAMMA)

def compute_servo_angle(color, area):
    w = _area_closeness(area)
    near = LEFT_NEAR if color == "Red" else RIGHT_NEAR
    raw = CENTER_ANGLE + AREA_GAIN * w * (near - CENTER_ANGLE)
    return int(round(clamp(raw, SERVO_MIN, SERVO_MAX)))

def get_largest_contour(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    x,y,w,h = cv2.boundingRect(c)
    return c, (x,y), (x+w, y+h), w*h

def boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def imu_center_servo(yaw_deg):
    if abs(yaw_deg) <= YAW_DEADBAND_DEG:
        return CENTER_ANGLE
    corr = clamp(YAW_KP * yaw_deg, -SERVO_CORR_LIMIT, SERVO_CORR_LIMIT)
    return int(CENTER_ANGLE + corr)

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

def set_leds(red=False, green=False, blue=False):
    GPIO.output(LED_RED_PIN,   GPIO.HIGH if red else GPIO.LOW)
    GPIO.output(LED_GREEN_PIN, GPIO.HIGH if green else GPIO.LOW)
    GPIO.output(LED_BLUE_PIN,  GPIO.HIGH if blue else GPIO.LOW)

# ---------- HARDWARE INIT ----------
GPIO.setmode(GPIO.BCM)
for p in [LED_BLUE_PIN, LED_RED_PIN, LED_GREEN_PIN]:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, GPIO.LOW)

# Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": CAM_SIZE}))
picam2.start()
time.sleep(1.5)

# I2C + PCA9685 (same style as your script)
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50
I2C_LOCK = threading.Lock()

def set_servo_angle(channel, angle):
    target = max(SERVO_MIN, min(SERVO_MAX, int(round(angle))))
    pulse = int(
        SERVO_PULSE_MIN + (SERVO_PULSE_MAX - SERVO_PULSE_MIN) *
        ((target - SERVO_MIN) / float(SERVO_MAX - SERVO_MIN))
    )
    with I2C_LOCK:
        pca.channels[channel].duty_cycle = int(pulse * 65535 / SERVO_PERIOD)

def set_motor_speed(channel_forward, channel_reverse, speed):
    s = int(max(-100, min(100, speed)))
    duty = int(abs(s) * 65535 / 100)
    with I2C_LOCK:
        if s > 0:
            pca.channels[channel_forward].duty_cycle = duty
            pca.channels[channel_reverse].duty_cycle = 0
        elif s < 0:
            pca.channels[channel_forward].duty_cycle = 0
            pca.channels[channel_reverse].duty_cycle = duty
        else:
            pca.channels[channel_forward].duty_cycle = 0
            pca.channels[channel_reverse].duty_cycle = 0

# Center servo
set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
current_servo_angle = CENTER_ANGLE
last_servo_update = time.time()

# IMU (MPU6050 via smbus2)
bus = smbus2.SMBus(1)
def mpu6050_init():
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
    bus.write_byte_data(MPU6050_ADDR, SMPLRT_DIV, 7)
    bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, 0)
    bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0)
    bus.write_byte_data(MPU6050_ADDR, INT_ENABLE, 1)

def read_raw(addr):
    hi = bus.read_byte_data(MPU6050_ADDR, addr)
    lo = bus.read_byte_data(MPU6050_ADDR, addr+1)
    val = (hi << 8) | lo
    if val > 32767: val -= 65536
    return val

mpu6050_init()
def get_gyro_z_bias(samples=200):
    tot = 0.0
    for _ in range(samples):
        tot += read_raw(GYRO_ZOUT_H) / 131.0
        time.sleep(0.005)
    return tot / samples
gyro_z_bias = get_gyro_z_bias()

yaw = 0.0
yaw_lock = threading.Lock()
last_time = time.time()

# ToF sensors
xshuts = {}
for name, pin in XSHUT_PINS.items():
    x = digitalio.DigitalInOut(pin)
    x.direction = digitalio.Direction.OUTPUT
    x.value = False
    xshuts[name] = x
time.sleep(0.1)

sensors = {}
for name in ["left", "right", "front", "back", "front_right", "front_left"]:
    xshuts[name].value = True
    time.sleep(0.05)
    s = adafruit_vl53l0x.VL53L0X(i2c)
    s.set_address(TOF_ADDR[name])
    s.start_continuous()
    sensors[name] = s
    print(f"[ToF] {name.upper()} @ {hex(TOF_ADDR[name])}")

def tof_cm(sensor):
    try:
        v = sensor.range / 10.0
        if v <= 0 or v > TOF_VALID_MAX: return TOF_RANGE_BAD
        return v
    except:
        return TOF_RANGE_BAD
    
# --- Drawing helpers ---
def draw_hough_lines(dst, lines, bgr):
    """Draw Hough line segments (from cv2.HoughLinesP)."""
    if lines is None:
        return
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(dst, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2, cv2.LINE_AA)

def label_text(dst, text, org, bgr=(255,255,255)):
    cv2.putText(dst, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)


# ---------- SMART UNPARK ----------
start_button = Button(START_BUTTON_PIN)
print("\n=== SMART UNPARK READY ===")
print("Press the start button to begin...")
start_button.wait_for_press()
print("Button pressed! Starting unpark...")
time.sleep(1)

left_d  = tof_cm(sensors["left"])
right_d = tof_cm(sensors["right"])
direction = "left" if left_d > right_d else "right"
print(f"[UNPARK] Choosing {direction.upper()}")

# Phase 1
with yaw_lock: yaw = 0.0
last_time = time.time()
first_angle = UNPARK_LEFT_TURN_ANGLE if direction=="left" else UNPARK_RIGHT_TURN_ANGLE
set_servo_angle(SERVO_CHANNEL, first_angle)
set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_SPEED)
while True:
    now = time.time(); dt = now - last_time; last_time = now
    Gz = (read_raw(GYRO_ZOUT_H) / 131.0) - gyro_z_bias
    with yaw_lock:
        yaw += Gz * dt
        if abs(yaw) >= UNPARK_PHASE1_TARGET_DEG: break

# Phase 2
with yaw_lock: yaw = 0.0
last_time = time.time()
second_angle = UNPARK_RIGHT_TURN_ANGLE if direction=="left" else UNPARK_LEFT_TURN_ANGLE
set_servo_angle(SERVO_CHANNEL, second_angle)
set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_SPEED)
while True:
    now = time.time(); dt = now - last_time; last_time = now
    Gz = (read_raw(GYRO_ZOUT_H) / 131.0) - gyro_z_bias
    with yaw_lock:
        yaw += Gz * dt
        if abs(yaw) >= UNPARK_PHASE2_TARGET_DEG: break

# Extra for left
if direction == "left":
    with yaw_lock: yaw = 0.0
    last_time = time.time()
    set_servo_angle(SERVO_CHANNEL, UNPARK_LEFT_TURN_ANGLE)
    set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_SPEED)
    while True:
        now = time.time(); dt = now - last_time; last_time = now
        Gz = (read_raw(GYRO_ZOUT_H) / 131.0) - gyro_z_bias
        with yaw_lock:
            yaw += Gz * dt
            if abs(yaw) >= UNPARK_EXTRA_LEFT_DEG: break

# Straighten and go
set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
set_motor_speed(MOTOR_FWD, MOTOR_REV, UNPARK_SPEED)
with yaw_lock: yaw = 0.0
print("[UNPARK] Complete. Entering main loop...")

# ---------- MAIN LOOP STATE ----------
last_color = None
frame_count = 0
in_blue_backward = False
blue_backward_start = None
bb_color = None
bb_area_at_trigger = None

avoidance_mode = False
avoid_start_time = 0.0
avoid_direction = None
avoid_color = None
avoid_area_at_trigger = None
avoid_peak_area = None

turn_active = False
turn_dir = None
box_seen_while_turning = False
next_turn_allowed_time = 0.0

try:
    set_motor_speed(MOTOR_FWD, MOTOR_REV, 0)

    while True:
        # ----- Timing & IMU -----
        now = time.time()
        dt = now - last_time
        last_time = now

        Gz = (read_raw(GYRO_ZOUT_H) / 131.0) - gyro_z_bias
        with yaw_lock:
            yaw += Gz * dt
            yaw = clamp(yaw, -YAW_CLAMP_DEG, YAW_CLAMP_DEG)
            current_yaw = yaw

        # ----- Camera -----
        img = picam2.capture_array()
        h, w = img.shape[:2]
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        center_x = w // 2

        # ----- Box masks -----
        mask_red   = cv2.bitwise_or(cv2.inRange(imgHSV, RED1_LO, RED1_HI),
                                    cv2.inRange(imgHSV, RED2_LO, RED2_HI))
        mask_green = cv2.inRange(imgHSV, GREEN_LO, GREEN_HI)

        red_data   = get_largest_contour(mask_red,   min_area=MIN_AREA)
        green_data = get_largest_contour(mask_green, min_area=MIN_AREA)

        boxes = []
        if red_data:
            _, tl, br, area = red_data
            boxes.append(("Red", area, (*tl, *br)))
        if green_data:
            _, tl, br, area = green_data
            boxes.append(("Green", area, (*tl, *br)))

        red_seen = red_data is not None
        green_seen = green_data is not None

        # Car footprint
        car_box = (center_x - CAR_BOX_WIDTH//2,
                   h - CAR_BOX_BOTTOM_MARGIN - CAR_BOX_HEIGHT,
                   center_x + CAR_BOX_WIDTH//2,
                   h - CAR_BOX_BOTTOM_MARGIN)

        # ----- TURN detection via orange/blue lines -----
        mask_orange = cv2.inRange(imgHSV, ORANGE_LO, ORANGE_HI)
        mask_blue   = cv2.inRange(imgHSV, BLUE_LO, BLUE_HI)
        edges_orange = cv2.Canny(mask_orange, 50, 150)
        edges_blue   = cv2.Canny(mask_blue,   50, 150)
        lines_orange = cv2.HoughLinesP(edges_orange, 1, np.pi/180,
                                       threshold=50, minLineLength=50, maxLineGap=10)
        lines_blue   = cv2.HoughLinesP(edges_blue,   1, np.pi/180,
                                       threshold=50, minLineLength=50, maxLineGap=10)
        orange_y = y_at_center(lines_orange, center_x)
        blue_y   = y_at_center(lines_blue,   center_x)

        Turn = "No"
        if orange_y > blue_y and orange_y > LINE_CENTER_Y_MIN:
            Turn = "Right"
        elif blue_y > orange_y and blue_y > LINE_CENTER_Y_MIN:
            Turn = "Left"

        # ----- TURN state machine (no time.sleep) -----
        if (Turn in ("Left","Right")) and not turn_active and not in_blue_backward and not avoidance_mode and (now >= next_turn_allowed_time):
            turn_active = True
            turn_dir = Turn
            box_seen_while_turning = False
            with yaw_lock: yaw = 0.0
            last_time = time.time()

        set_leds(blue=turn_active, red=red_seen, green=green_seen)

        if turn_active:
            target_angle = TURN_LEFT_SERVO if turn_dir=="Left" else TURN_RIGHT_SERVO
            set_servo_angle(SERVO_CHANNEL, target_angle)
            set_motor_speed(MOTOR_FWD, MOTOR_REV, TURN_MOTOR_SPEED)

            if red_seen or green_seen:
                box_seen_while_turning = True

            stop_for_box = box_seen_while_turning and (abs(current_yaw) >= TURN_MIN_YAW_DEG)
            failsafe_yaw = abs(current_yaw) >= TURN_FAILSAFE_MAX_DEG

            if stop_for_box or failsafe_yaw:
                set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
                if turn_dir == "Left":
                    with yaw_lock: yaw = YAW_RESET_AFTER_LEFT
                else:
                    with yaw_lock: yaw = YAW_RESET_AFTER_RIGHT
                current_servo_angle = CENTER_ANGLE
                turn_active = False
                next_turn_allowed_time = time.time() + TURN_COOLDOWN_SEC
                turn_dir = None
            # continue to next iteration (no sleep)
            goto_next = True
        else:
            goto_next = False

        if goto_next:
            continue

        # ----- Immediate reverse if intersecting a box -----
        if not in_blue_backward:
            if red_data and boxes_intersect(car_box, (*red_data[1], *red_data[2])):
                in_blue_backward = True
                blue_backward_start = time.time()
                bb_color = "Red"
                bb_area_at_trigger = red_data[3]
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -BLUE_BACK_SPEED)
            elif green_data and boxes_intersect(car_box, (*green_data[1], *green_data[2])):
                in_blue_backward = True
                blue_backward_start = time.time()
                bb_color = "Green"
                bb_area_at_trigger = green_data[3]
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -BLUE_BACK_SPEED)

        if in_blue_backward:
            cur_area = (red_data[3] if (bb_color=="Red" and red_data) else
                        green_data[3] if (bb_color=="Green" and green_data) else
                        bb_area_at_trigger)
            steer = compute_servo_angle(bb_color, cur_area)
            set_servo_angle(SERVO_CHANNEL, steer)
            set_motor_speed(MOTOR_FWD, MOTOR_REV, -BLUE_BACK_SPEED)
            if (time.time() - blue_backward_start) >= BLUE_BACK_DURATION:
                in_blue_backward = False
                bb_color = None
                bb_area_at_trigger = None
                with yaw_lock: yaw = 0.0
                set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)
            continue

        # ----- Reverse-avoidance trigger -----
        if not avoidance_mode:
            for color, area, box_coords in boxes:
                if boxes_intersect(car_box, box_coords):
                    avoidance_mode = True
                    avoid_direction = "right" if color == "Green" else "left"
                    avoid_color = color
                    avoid_area_at_trigger = area
                    avoid_peak_area = area
                    avoid_start_time = time.time()
                    break

        if avoidance_mode and avoid_color == "Red":
            GPIO.output(LED_RED_PIN, GPIO.HIGH)
        if avoidance_mode and avoid_color == "Green":
            GPIO.output(LED_GREEN_PIN, GPIO.HIGH)

        if avoidance_mode:
            elapsed = time.time() - avoid_start_time
            if elapsed < AVOID_BACK_DURATION:
                cur_area = (red_data[3] if (avoid_color=="Red" and red_data) else
                            green_data[3] if (avoid_color=="Green" and green_data) else
                            avoid_area_at_trigger)
                if avoid_peak_area is None:
                    avoid_peak_area = cur_area
                else:
                    avoid_peak_area = max(avoid_peak_area, cur_area)
                steer = compute_servo_angle(avoid_color, avoid_peak_area)
                set_servo_angle(SERVO_CHANNEL, steer)
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -AVOID_SPEED)
            else:
                set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)
                target_angle = imu_center_servo(current_yaw)
                set_servo_angle(SERVO_CHANNEL, clamp(target_angle, SERVO_MIN, SERVO_MAX))
                if not any(boxes_intersect(car_box, b[2]) for b in boxes):
                    avoidance_mode = False
                    avoid_direction = None
                    avoid_color = None
                    avoid_area_at_trigger = None
                    avoid_peak_area = None
                    with yaw_lock: yaw = 0.0
            continue

        # ----- Normal forward + IMU straighten + area guidance -----
        set_motor_speed(MOTOR_FWD, MOTOR_REV, NORMAL_SPEED)
        target_angle = imu_center_servo(current_yaw)

        danger_mode = False
        if boxes:
            boxes.sort(key=lambda b: b[1], reverse=True)
            chosen_color, chosen_area, chosen_box = boxes[0]
            if last_color == chosen_color:
                frame_count += 1
            else:
                frame_count = 0
                last_color = chosen_color

            if frame_count >= COLOR_HOLD_FRAMES:
                area_angle = compute_servo_angle(chosen_color, chosen_area)
                if chosen_color == "Red":
                    target_angle = clamp(int(round(area_angle + YAW_BOX_GAIN * current_yaw)),
                                         SERVO_MIN, SERVO_MAX)
                elif chosen_color == "Green":
                    target_angle = clamp(int(round(area_angle - YAW_BOX_GAIN * current_yaw)),
                                         SERVO_MIN, SERVO_MAX)

                box_bottom_y = chosen_box[3]
                near_car = box_bottom_y >= (h - (CAR_BOX_BOTTOM_MARGIN + CAR_BOX_HEIGHT + DANGER_MARGIN))
                if (chosen_area >= PREEMPT_AREA) or near_car:
                    if chosen_color == "Red":
                        target_angle = clamp(max(target_angle, LEFT_NEAR - 2), SERVO_MIN, SERVO_MAX)
                    else:
                        target_angle = clamp(min(target_angle, RIGHT_NEAR + 2), SERVO_MIN, SERVO_MAX)
                    danger_mode = True

        # ----- Servo smoothing/output -----
        if (time.time() - last_servo_update) >= SERVO_UPDATE_DELAY:
            delta = target_angle - current_servo_angle
            step = FAST_SERVO_STEP if abs(delta) >= 12 else STEP
            if danger_mode:
                step = max(step, FAST_SERVO_STEP * 2)
            if abs(delta) > step:
                current_servo_angle += step if delta > 0 else -step
            else:
                current_servo_angle = target_angle
            current_servo_angle = clamp(current_servo_angle, SERVO_MIN, SERVO_MAX)
            set_servo_angle(SERVO_CHANNEL, current_servo_angle)
            last_servo_update = time.time()

        # NOTE: no time.sleep() anywhere in the main loop

                    # Clean up a bit
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, KERNEL)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, KERNEL)
            mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, KERNEL)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, KERNEL)

            # Draw boxes (red/green)
            draw_boxes(frame_bgr, mask_red,   (0, 0, 255),   "Red")
            draw_boxes(frame_bgr, mask_green, (0, 255, 0),   "Green")

            # Draw lines (orange/blue)
            draw_lines(frame_bgr, mask_orange, (0, 165, 255), "Orange")
            draw_lines(frame_bgr, mask_blue,   (255, 0, 0),   "Blue")

            # Show
            cv2.imshow("Camera annotated", frame_bgr)
            cv2.imshow("Mask - Red", mask_red)
            cv2.imshow("Mask - Green", mask_green)
            cv2.imshow("Mask - Orange", mask_orange)
            cv2.imshow("Mask - Blue", mask_blue)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
finally:
    try:
        set_motor_speed(MOTOR_FWD, MOTOR_REV, 0)
        set_servo_angle(SERVO_CHANNEL, CENTER_ANGLE)
    except Exception:
        pass
    set_leds(False, False, False)
    GPIO.cleanup()
    try:
        picam2.stop()
        cv2.destroyAllWindows()
    except Exception:
        pass
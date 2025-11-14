#-----------------------------------------------------------------------------------------------------------------------
# 1st Mission WRO 2025 FE - VivaLaVida
# Final Version
#-----------------------------------------------------------------------------------------------------------------------

# ===============================
# IMPORTS                       
# ===============================


import os, sys

VENV_PY = "/home/stem/env/bin/python"  # exact path to Python in your venv

if sys.executable != VENV_PY and os.path.exists(VENV_PY):
    print(f"[INFO] Relaunching under virtual environment: {VENV_PY}", flush=True)
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

# --- your normal robot code below ---
print(f"[INFO] Now running inside: {sys.executable}")


import threading
import time
from collections import deque
import busio
from adafruit_pca9685 import PCA9685
import adafruit_mpu6050
import csv
from datetime import datetime
import json

import board
import digitalio
import adafruit_vl53l0x
from enum import Enum, auto
import numpy as np
from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory   # Uses /dev/gpiochip*
Device.pin_factory = LGPIOFactory()
from gpiozero import Button, LED
from gpiozero import DistanceSensor
from threading import Event
import cv2
import numpy as np
from picamera2 import Picamera2

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get directory of the running script
CONFIG_FILE = os.path.join(BASE_DIR, "1st_mission_variables.json")

# ===============================
# CONFIGURATION VARIABLES
# ===============================

# ---------- Initialization ----------
USE_GUI = 0                   # 1 = Debugging mode (run with GUI), 0 = COMPETITION MODE (run headless, no GUI)
DEBUG = 0                     # 1 = enable prints, 0 = disable all prints
USE_TOF_SIDES = 0             # Side sensors:  0 = Ultrasonic, 1 = ToF
USE_TOF_FRONT = 0             # Front sensor:  0 = Ultrasonic, 1 = ToF
USE_TOF_CORNERS = 0           # Front side sensor: 0 = disable, 1 = enable

NARROW_SUM_THRESHOLD = 60     # cm; left+right distance threshold to decide if in between narrow walls
NARROW_HYSTERESIS = 10        # cm; prevents rapid toggling
NARROW_FACTOR_SPEED = 0.6     # multiply state speeds when in narrow corridors
NARROW_FACTOR_DIST  = 0.5     # multiply distance-based thresholds/corrections when in narrow corridors

# ---------- Speeds ----------
SPEED_IDLE = 0
SPEED_STOPPED = 0
SPEED_CRUISE = 22             # Motor speed for normal straight driving (0-100%)
SPEED_TURN_INIT = 22         # Motor speed while waiting for open side to turn
SPEED_TURN = 18               # Motor speed while turning
SPEED_POST_TURN = 18          # Motor speed following a turn

# ---------- Driving ----------
SOFT_MARGIN = 26              # Distance from wall where small steering corrections start (cm)
MAX_CORRECTION = 7            # Maximum servo correction applied for wall-following (degrees)
CORR_EPS = 1.5                # cm: treat the side as "steady" if within ±1.5 cm of trigger value
CORRECTION_MULTIPLIER = 1.4   # Proportional gain (servo degrees per cm of error 0 default: 2). Higher = snappier; lower = smoother&slower. 

STOP_THRESHOLD = 20           # Front distance (cm) at which robot stops immediately
OBSTACLE_WAIT_TIME = 5.0      # seconds to wait before retrying after a front-stop

# ---------- Turn management ----------
FRONT_TURN_TRIGGER = 90       # Front distance (cm) at which a turn is triggered
TURN_DECISION_THRESHOLD = 80  # Minimum side distance (cm) to allow turn in that direction
TURN_ANGLE_LEFT = 61          # Servo angle for left turn
TURN_ANGLE_RIGHT = 119        # Servo angle for right turn
TURN_TIMEOUT = 2.5            # Maximum time allowed for a turn (seconds)
TURN_LOCKOUT = 1.5            # Minimum interval between consecutive turns (seconds)
POST_TURN_DURATION = 0.5      # Time to drive straight after a turn (seconds)
LOCK_TURN_DIRECTION = 1       # 1 = enable turn lock direction after 1st turn, 0 = disable
TARGET_TURN_ANGLE = 85        # Degrees to turn per corner
TURN_ANGLE_TOLERANCE = 6      # Acceptable overshoot (degrees)

# ---------- Turn angle constraints ----------
MIN_TURN_ANGLE = 75           # Minimum yaw change (degrees) before turn can stop
MAX_TURN_ANGLE = 105          # Maximum yaw change (degrees) to force stop turn

# ---------- Laps ----------
MAX_LAPS = 3                  # Maximum number of laps before stopping (0 = unlimited)
POST_LAP_DURATION = 0.85         # Time to drive forward after final lap before stopping (seconds)

# ---------- Sensor data filtering ----------
N_READINGS = 5                # Number of readings stored for median filtering
US_QUEUE_LEN = 1              # Queue_len for Ultrasonic gpiozero.DistanceSensor
FILTER_ALPHA = 1.0            # Ultrasonic 0.1 = smoother, 0.5 = faster reaction, 1 to ignore filter
FILTER_JUMP = 9999            # Ultrasonic maximum jump (cm) allowed between readings
FILTER_ALPHA_TOF = 1.0        # ToF 0.1 = smoother, 0.5 = faster reaction, 1 to ignore filter
FILTER_JUMP_TOF = 9999        # ToF maximum jump (cm) allowed between readings

US_MAX_DISTANCE_FRONT = 2.0  # Ultrasonic front max read distance
US_MAX_DISTANCE_SIDE = 1.2   # Ultrasonic side max read distance

# ---------- Loop timing ----------
LOOP_DELAY = 0.005            # Delay between main loop iterations (seconds) 0.02
SENSOR_DELAY = 0.01          # Delay between sensor reads

#-----------------------------------------------------------------------------------------

# ---------- Plotting ----------
MAX_POINTS = 500               # Maximum number of points to store for plotting

RAD2DEG = 180.0 / 3.141592653589793

# --- I2C concurrency guard ---
I2C_LOCK = threading.Lock()

def dprint(*args, **kwargs): #Conditional print — only prints when DEBUG == 1.
    if DEBUG:
        print(*args, **kwargs)

# --- Headless-safe placeholders for GUI symbols ---
slider_vars = {}
btn_readings = btn_start = btn_stop = None
lbl_turns = lbl_laps = None
root = canvas = None

# ===============================
# LOAD VARIABLES FROM JSON (if exists)
# ===============================

def load_variables_from_json(path=CONFIG_FILE): # Override defaults with values from JSON if the file exists.
    if not os.path.exists(path):
        print("⚙️ No external config found — using built-in defaults.")
        return
    try:
         with open(path, "r") as f:
              data = json.load(f)
         count = 0
         for k, v in data.items():
             if k in globals():
                 globals()[k] = v
                 count += 1
         print(f"✅ Loaded {count} variables from {os.path.basename(path)}")
    except Exception as e:
         print(f"[ERROR] Config load failed: {e} — using built-in defaults.")

load_variables_from_json() # Call it right after defining defaults so everything below sees the overrides

if os.environ.get("DISPLAY", "") == "": # If there is no X server, force headless regardless of config
    USE_GUI = 0

def norm180(a: float) -> float: # Normalize angle to [-180, 180)
    return (a + 180.0) % 360.0 - 180.0

def snap90(a: float) -> float: # Nearest multiple of 90° (…,-180,-90,0,90,180,…)
    return round(a / 90.0) * 90.0

# --- Store immutable base values for scaling ---
BASE_SPEED_IDLE        = SPEED_IDLE
BASE_SPEED_STOPPED     = SPEED_STOPPED
BASE_SPEED_CRUISE      = SPEED_CRUISE
BASE_SPEED_TURN_INIT   = SPEED_TURN_INIT
BASE_SPEED_TURN        = SPEED_TURN
BASE_SPEED_POST_TURN   = SPEED_POST_TURN

BASE_MAX_CORRECTION     = MAX_CORRECTION
BASE_FRONT_TURN_TRIGGER = FRONT_TURN_TRIGGER
BASE_STOP_THRESHOLD     = STOP_THRESHOLD

# --- Global factors (automatically updated) ---
SPEED_ENV_FACTOR = 1.0
DIST_ENV_FACTOR  = 1.0  # for distance-related thresholds

# --- Helper functions to get "effective" (scaled) values ---
def eff_soft_margin(): #Return scaled soft margin distance (cm) for gentle steering corrections
    return int(SOFT_MARGIN * DIST_ENV_FACTOR)

def eff_max_correction(): #Return scaled maximum steering correction (degrees)
    return max(1, int(MAX_CORRECTION * DIST_ENV_FACTOR))

def eff_front_turn_trigger(): #Return scaled front distance threshold (cm) for initiating a turn
    return int(FRONT_TURN_TRIGGER * DIST_ENV_FACTOR)

def eff_stop_threshold(): #Return scaled front distance threshold (cm) for immediate stop
    return int(STOP_THRESHOLD * DIST_ENV_FACTOR)

# --- Scaled state speeds ---
def state_speed_value(state_name: str) -> int:
    base = {
        "IDLE":      SPEED_IDLE,
        "STOPPED":   SPEED_STOPPED,
        "CRUISE":    SPEED_CRUISE,
        "TURN_INIT": SPEED_TURN_INIT,
        "TURNING":   SPEED_TURN,
        "POST_TURN": SPEED_POST_TURN,
    }.get(state_name, SPEED_CRUISE)
    return int(base * SPEED_ENV_FACTOR)

def headless_toggle_pause(): #Toggle pause/resume of the driving loop in headless mode
    global paused_by_button, status_text
    if loop_event.is_set(): #Pause
        loop_event.clear()
        paused_by_button = True
        robot.stop_motor()
        robot.set_servo(SERVO_CENTER)
        RED_LED.off()     # paused
        GREEN_LED.on()
        status_text = "Paused (button)"
    else:  # RESUME
        paused_by_button = False
        loop_event.set()
        GREEN_LED.off()   # running
        RED_LED.on()
        status_text = "🚗 Resumed"

# ===============================
# HARDWARE SETUP
# ===============================

# Initialize Picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))  # Set resolution
picam2.start()

i2c = busio.I2C(board.SCL, board.SDA)

# ---------- Button & LEDs (gpiozero-based) ----------
START_BTN = Button(20, pull_up=True, bounce_time=0.03)
GREEN_LED = LED(19)
RED_LED   = LED(13)
# ✅ Turn RED on as soon as the program starts
RED_LED.on()

# ---------- Ultrasonic sensor pins ----------
TRIG_FRONT, ECHO_FRONT = 22, 23  # GPIO pins for front sensor
TRIG_LEFT, ECHO_LEFT = 27, 17    # GPIO pins for left sensor
TRIG_RIGHT, ECHO_RIGHT = 5, 6    # GPIO pins for right sensor

# ---------- Motor ----------
MOTOR_FWD = 1                 # PCA9685 channel for forward motor control
MOTOR_REV = 2                 # PCA9685 channel for reverse motor control

# ---------- Servo ----------
SERVO_CHANNEL = 0             # PCA9685 channel controlling steering servo
SERVO_CENTER = 90             # Neutral servo angle (degrees)
SERVO_MIN_ANGLE = 50          # Minimum physical servo angle (degrees)
SERVO_MAX_ANGLE = 130         # Maximum physical servo angle (degrees)
SERVO_PULSE_MIN = 1000        # Minimum PWM pulse width (microseconds)
SERVO_PULSE_MAX = 2000        # Maximum PWM pulse width (microseconds)
SERVO_PERIOD = 20000          # Servo PWM period (microseconds)

# ===============================
# SENSOR INITIALIZATION (ToF + Ultrasonic)
# ===============================

vl53_left = vl53_right = vl53_front = vl53_back = None
vl53_front_left = vl53_front_right = None
us_left = us_right = us_front = None

try:
    # --------- ToF sensors ----------
    if USE_TOF_SIDES or USE_TOF_FRONT or USE_TOF_CORNERS:
        print("Initializing VL53L0X ToF sensors...")
        # XSHUT pins setup (for powering sensors individually)
        xshut_left  = digitalio.DigitalInOut(board.D16)
        xshut_right = digitalio.DigitalInOut(board.D25)
        xshut_front = digitalio.DigitalInOut(board.D26)
        xshut_back  = digitalio.DigitalInOut(board.D8)
        xshut_front_left  = digitalio.DigitalInOut(board.D7)
        xshut_front_right = digitalio.DigitalInOut(board.D24)
        for xshut in [xshut_left, xshut_right, xshut_front, xshut_back,
                      xshut_front_left, xshut_front_right]:
            xshut.direction = digitalio.Direction.OUTPUT
            xshut.value = False  # power down
        time.sleep(0.1)

        # Initialize left ToF if needed
        if USE_TOF_SIDES:
            xshut_left.value = True
            time.sleep(0.05)
            vl53_left = adafruit_vl53l0x.VL53L0X(i2c)
            vl53_left.set_address(0x30)
            vl53_left.measurement_timing_budget = 20000
            vl53_left.start_continuous()
            print("✅ Left ToF sensor set to address 0x30")

        # Initialize right ToF if needed
        if USE_TOF_SIDES:
            xshut_right.value = True
            time.sleep(0.05)
            vl53_right = adafruit_vl53l0x.VL53L0X(i2c)
            vl53_right.set_address(0x31)
            vl53_right.measurement_timing_budget = 20000
            vl53_right.start_continuous()
            print("✅ Right ToF sensor set to address 0x31")

        # Initialize front ToF if needed
        if USE_TOF_FRONT:
            xshut_front.value = True
            time.sleep(0.05)
            vl53_front = adafruit_vl53l0x.VL53L0X(i2c)
            vl53_front.set_address(0x32)
            vl53_front.measurement_timing_budget = 20000
            vl53_front.start_continuous()
            print("✅ Front ToF sensor set to address 0x32")
            
        # Initialize corner/front-left ToF (optional)
        if USE_TOF_CORNERS:
            xshut_front_left.value = True
            time.sleep(0.05)
            vl53_front_left = adafruit_vl53l0x.VL53L0X(i2c)
            vl53_front_left.set_address(0x34)
            vl53_front_left.measurement_timing_budget = 20000
            vl53_front_left.start_continuous()
            print("✅ Front-Left ToF sensor set to address 0x34")
            
        # Initialize corner/front-right ToF (optional)
        if USE_TOF_CORNERS:
            xshut_front_right.value = True
            time.sleep(0.05)
            vl53_front_right = adafruit_vl53l0x.VL53L0X(i2c)
            vl53_front_right.set_address(0x35)
            vl53_front_right.measurement_timing_budget = 20000
            vl53_front_right.start_continuous()
            print("✅ Front-Right ToF sensor set to address 0x35")

        # Initialize back ToF (optional, if you want to use it)
        xshut_back.value = True
        time.sleep(0.05)
        vl53_back = adafruit_vl53l0x.VL53L0X(i2c)
        vl53_back.set_address(0x33)
        vl53_back.measurement_timing_budget = 20000
        vl53_back.start_continuous()
        print("✅ Back ToF sensor set to address 0x33")

except Exception as e:
    print(f"[ERROR] VL53L0X initialization failed: {e}")
    # fallback to ultrasonic if ToF fails
    USE_TOF_SIDES = 0
    USE_TOF_FRONT = 0
    USE_TOF_CORNERS = 0
    vl53_left = vl53_right = vl53_front = vl53_back = None
    vl53_front_left = vl53_front_right = None

time.sleep(0.2)

# --------- Ultrasonic sensors ----------
# Initialize ultrasonic sensors only if that sensor is set to ultrasonic
if not USE_TOF_FRONT:
    us_front = DistanceSensor(echo=ECHO_FRONT, trigger=TRIG_FRONT, max_distance=US_MAX_DISTANCE_FRONT, queue_len=US_QUEUE_LEN)
if not USE_TOF_SIDES:
    us_left  = DistanceSensor(echo=ECHO_LEFT,  trigger=TRIG_LEFT,  max_distance=US_MAX_DISTANCE_SIDE, queue_len=US_QUEUE_LEN)
    us_right = DistanceSensor(echo=ECHO_RIGHT, trigger=TRIG_RIGHT, max_distance=US_MAX_DISTANCE_SIDE, queue_len=US_QUEUE_LEN)

dprint("Sensors initialized:")
dprint(f"  Front: {'ToF' if USE_TOF_FRONT else 'Ultrasonic'}")
dprint(f"  Left : {'ToF' if USE_TOF_SIDES else 'Ultrasonic'}")
dprint(f"  Right: {'ToF' if USE_TOF_SIDES else 'Ultrasonic'}")
if USE_TOF_CORNERS:
    dprint("  Turn corners: ToF (front_left & front_right)")

pca = PCA9685(i2c)
pca.frequency = 50
mpu = adafruit_mpu6050.MPU6050(i2c)

# ---------- Calibrate gyro bias at startup ----------
dprint("Calibrating gyro...")
N = 500
bias = 0
for _ in range(N):
    with I2C_LOCK:
        bias += mpu.gyro[2]
    time.sleep(0.005)
bias /= N
dprint(f"Gyro bias: {bias}")

# ===============================
# CONTROL FLAGS & STORAGE
# ===============================

STATE_SPEED = {
    "IDLE": SPEED_IDLE,
    "STOPPED": SPEED_STOPPED,
    "CRUISE": SPEED_CRUISE,
    "TURN_INIT": SPEED_TURN_INIT,
    "TURNING": SPEED_TURN,
    "POST_TURN": SPEED_POST_TURN
}

# ---------- CONTROL FLAGS ----------
readings_event = Event()
loop_event = Event()
sensor_tick = Event()   # used to wake waits immediately
shutdown_event = Event()
status_text = "Idle"
turn_count = 0
lap_count = 0
stop_reason = None # Reason-aware stopping (USER / OBSTACLE / LAPS)
obstacle_wait_deadline = 0.0
# --- Headless START button toggle / debounce ---
BTN_DEBOUNCE_S = 0.30   # seconds
_btn_prev = 1           # pull-up idle state (button not pressed)
_btn_last_ts = 0.0
paused_by_button = False

# ---------- SENSOR DATA STORAGE ----------
time_data = deque(maxlen=MAX_POINTS)
front_data = deque(maxlen=MAX_POINTS)
left_data = deque(maxlen=MAX_POINTS)
right_data = deque(maxlen=MAX_POINTS)
front_left_data  = deque(maxlen=MAX_POINTS)
front_right_data = deque(maxlen=MAX_POINTS)
angle_data = deque(maxlen=MAX_POINTS)
state_data = deque(maxlen=MAX_POINTS)

# ===============================
# THREADED SENSOR READING
# ===============================
sensor_data = {
    "front": None,
    "left": None,
    "right": None,
    "front_left": None,
    "front_right": None,
}
sensor_lock = threading.Lock()

# ===============================
# ROBOT CONTROLLER
# ===============================

class RobotController:
    def __init__(self, pca):
        self.pca = pca
        self.front_history = deque(maxlen=N_READINGS)
        self.left_history = deque(maxlen=N_READINGS)
        self.right_history = deque(maxlen=N_READINGS)
        self.front_left_history = deque(maxlen=N_READINGS)
        self.front_right_history = deque(maxlen=N_READINGS)
        self.smooth_left = None
        self.smooth_right = None
        self.smooth_front = None
        self.smooth_front_left = None
        self.smooth_front_right = None
        self.d_front = None
        self.d_left = None
        self.d_right = None
        self.d_front_left = None
        self.d_front_right = None
        self.sensor_index = 0
        self.gyro_z_prev = 0
        self._servo_last_angle = SERVO_CENTER
        self._servo_last_ns = time.monotonic_ns()
        # -- Simple safe-straight latch --
        self.ss = 0          # 0 = inactive, +1 = left triggered, -1 = right triggered
        self.ss_base = None  # distance at trigger

    def ss_reset(self):
        self.ss = 0
        self.ss_base = None

    def set_filter_size(self, n):
        self.front_history = deque(list(self.front_history)[-n:], maxlen=n)
        self.left_history = deque(list(self.left_history)[-n:], maxlen=n)
        self.right_history = deque(list(self.right_history)[-n:], maxlen=n)
        self.front_left_history = deque(list(self.front_left_history)[-n:], maxlen=n)
        self.front_right_history = deque(list(self.front_right_history)[-n:], maxlen=n)
   
    def set_servo(self, angle):
        # clamp to physical limits
        target = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        # angle → pulse (µs)
        pulse = int(
            SERVO_PULSE_MIN
            + (SERVO_PULSE_MAX - SERVO_PULSE_MIN)
            * ((target - SERVO_MIN_ANGLE) / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE))
        )
        # write to PCA9685
        with I2C_LOCK:
            self.pca.channels[SERVO_CHANNEL].duty_cycle = int(pulse * 65535 / SERVO_PERIOD)
        return target

    def rotate_motor(self, speed):
        duty_cycle = int(min(max(abs(speed), 0), 100)/100*65535)
        with I2C_LOCK:
            if speed >= 0:
                self.pca.channels[MOTOR_FWD].duty_cycle = duty_cycle
                self.pca.channels[MOTOR_REV].duty_cycle = 0
            else:
                self.pca.channels[MOTOR_FWD].duty_cycle = 0
                self.pca.channels[MOTOR_REV].duty_cycle = duty_cycle

    def stop_motor(self):
        with I2C_LOCK:
            self.pca.channels[MOTOR_FWD].duty_cycle = 0
            self.pca.channels[MOTOR_REV].duty_cycle = 0

    def set_state_speed(self, state):
        # Set motor speed based on FSM state.
        speed = state_speed_value(state)
        self.rotate_motor(speed)

    def get_distance(self, sensor):
        #sensor: a gpiozero.DistanceSensor instance
        try:
            d = sensor.distance  # returns meters
            if d is None:
                return None
            return d * 100  # convert to cm
        except:
            return None

    def stable_filter(self, new_val, prev_val, alpha=FILTER_ALPHA, max_jump=FILTER_JUMP):
    #Reject large spikes and apply exponential moving average smoothing.
        if new_val is None:
            return prev_val  # keep previous if invalid
        if prev_val is not None and abs(new_val - prev_val) > max_jump:
            new_val = prev_val  # reject spike
        if prev_val is None:
            return new_val
        return alpha * new_val + (1 - alpha) * prev_val    

    def filtered_distance(self, sensor_obj, history, smooth_attr, sensor_type='us'):
        # Fallback to use when ToF is invalid or too far (use US side max, in cm)
        TOF_FALLBACK_CM = US_MAX_DISTANCE_SIDE * 100.0  # 1.2 m -> 120 cm
        try:
            if sensor_type == 'tof':
                with I2C_LOCK:
                    raw_mm = sensor_obj.range  # VL53L0X returns mm
                d = (raw_mm / 10.0) if raw_mm is not None else None  # -> cm
                invalid_far = (d is None) or (d >= 800.0)            # ≥ 8 m or None
                d_for_history = None if invalid_far else d           # keep filter clean
            else:
                d = sensor_obj.distance * 100.0   # meters -> cm
                invalid_far = False
                d_for_history = d
        except:
            d = None
            invalid_far = (sensor_type == 'tof')
            d_for_history = None
    
        # Update history with valid values only (for stable median/EMA)
        history.append(d_for_history)
        valid = [x for x in history if x is not None]
    
        # If ToF is invalid/too far, immediately report the side max distance (120 cm)
        if sensor_type == 'tof' and invalid_far:
            return TOF_FALLBACK_CM
    
        if not valid:
            # No valid history yet -> for ToF use side max, else keep your sentinel
            return TOF_FALLBACK_CM if sensor_type == 'tof' else 999
    
        median_val = np.median(valid)
        # avg_val = np.mean(valid)  # unused
        filtered_val = median_val
    
        prev_val = getattr(self, smooth_attr)
        alpha = FILTER_ALPHA_TOF if sensor_type == 'tof' else FILTER_ALPHA
        max_jump = FILTER_JUMP_TOF if sensor_type == 'tof' else FILTER_JUMP
        smoothed_val = self.stable_filter(filtered_val, prev_val, alpha, max_jump)
    
        setattr(self, smooth_attr, smoothed_val)
        return smoothed_val

    def safe_straight_control(self, d_left, d_right):
        # Treat 999/None as no data
        d_left  = None if (d_left  is None or d_left  >= 900) else d_left
        d_right = None if (d_right is None or d_right >= 900) else d_right

        m = eff_soft_margin()
        max_corr = eff_max_correction()

        # -------- ARM if inactive (inside margin by more than CORR_EPS) --------
        if self.ss == 0:
            if d_left is not None and d_left < (m - CORR_EPS):
                self.ss = +1; self.ss_base = d_left
            elif d_right is not None and d_right < (m - CORR_EPS):
                self.ss = -1; self.ss_base = d_right
            else:
                return SERVO_CENTER

        # -------- ACTIVE: correct only away from the triggering side --------
        d = d_left if self.ss > 0 else d_right

        # Release if sensor lost or clearly recovered beyond margin
        if d is None or d >= (m + CORR_EPS):
            self.ss_reset()
            return SERVO_CENTER

        # Small deadband near the margin to avoid micro-jitter
        err = m - d   # >0 when too close
        if err <= CORR_EPS:
            return SERVO_CENTER

        corr = self.ss * min(max_corr, CORRECTION_MULTIPLIER * err)
        angle = SERVO_CENTER + corr
        return max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))

    def turn_decision(self, d_left, d_right, d_fl=None, d_fr=None): #Prefer corner/front sensors for turn decision; fallback to sides if absent.
        l_val = d_fl if d_fl is not None and d_fl != 999 else d_left
        r_val = d_fr if d_fr is not None and d_fr != 999 else d_right
        left_open  = (l_val is None) or (l_val == 999) or (l_val > TURN_DECISION_THRESHOLD)
        right_open = (r_val is None) or (r_val == 999) or (r_val > TURN_DECISION_THRESHOLD)
    
        # Decide only when exactly one side is open; otherwise keep trying.
        if left_open ^ right_open:               #XOR: exactly one is True
            return "LEFT" if left_open else "RIGHT"
        return None # both open or both closed → keep trying

    def detect_obstacles():
        # Capture an image from the camera
        img_bgr = picam2.capture_array()  # Capture frame as a NumPy array
        
        # Convert the captured image to HSV
        imgHSV = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2HSV)  # Convert to HSV for color detection
        
        # Define HSV ranges for red and green colors
        RED_LO = np.array([0, 100, 80], dtype=np.uint8)
        RED_HI = np.array([10, 255, 200], dtype=np.uint8)
        GREEN_LO = np.array([65, 90, 90], dtype=np.uint8)
        GREEN_HI = np.array([80, 255, 150], dtype=np.uint8)
        
        # Create masks for red and green objects
        mask_red = cv2.inRange(imgHSV, RED_LO, RED_HI)
        mask_green = cv2.inRange(imgHSV, GREEN_LO, GREEN_HI)
        
        return mask_red, mask_green, img_bgr


    def get_largest_contour(mask, min_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < min_area:
            return None
        x, y, w, h = cv2.boundingRect(largest_contour)
        return largest_contour, (x, y), (x + w, y + h), w*h




robot = RobotController(pca)

def sensor_reader():
    global sensor_data
    while True:
        # Left sensor
        if vl53_left:  # ToF
            left = robot.filtered_distance(vl53_left, robot.left_history, "smooth_left", sensor_type='tof')
        elif us_left:  # ultrasonic
            left = robot.filtered_distance(us_left, robot.left_history, "smooth_left", sensor_type='us')
        else:
            left = None

        # Right sensor
        if vl53_right:
            right = robot.filtered_distance(vl53_right, robot.right_history, "smooth_right", sensor_type='tof')
        elif us_right:
            right = robot.filtered_distance(us_right, robot.right_history, "smooth_right", sensor_type='us')
        else:
            right = None

        # Front sensor
        if vl53_front:
            front = robot.filtered_distance(vl53_front, robot.front_history, "smooth_front", sensor_type='tof')
        elif us_front:
            front = robot.filtered_distance(us_front, robot.front_history, "smooth_front", sensor_type='us')
        else:
            front = None

        # Corner/front-left sensor
        if vl53_front_left:
            front_left = robot.filtered_distance(vl53_front_left, robot.front_left_history, "smooth_front_left", sensor_type='tof')
        else:
            front_left = None
        # Corner/front-right sensor
        if vl53_front_right:
            front_right = robot.filtered_distance(vl53_front_right, robot.front_right_history, "smooth_front_right", sensor_type='tof')
        else:
            front_right = None

        # Save readings in global dictionary
        with sensor_lock:
            sensor_data["front"] = front
            sensor_data["left"] = left
            sensor_data["right"] = right
            sensor_data["front_left"] = front_left
            sensor_data["front_right"] = front_right
            
        sensor_tick.set()  # Set the event
        #time.sleep(SENSOR_DELAY)
        sensor_tick.wait(SENSOR_DELAY)
        sensor_tick.clear()

# ===============================
# FINITE STATE MACHINE: robot_loop
# ===============================

class RobotState(Enum):
    IDLE = auto()
    CRUISE = auto()
    TURN_INIT = auto()
    TURNING = auto()
    POST_TURN = auto()
    STOPPED = auto()

locked_turn_direction = None # Keep a global locked_turn_direction variable to persist across runs (reset in stop_loop)

def robot_loop():   
    global status_text, turn_count, lap_count, locked_turn_direction, stop_reason, obstacle_wait_deadline
    global _btn_prev, _btn_last_ts, paused_by_button
    state = RobotState.IDLE
    direction = None
    yaw = 0.0
    turn_start_yaw = 0.0
    turn_start_time = 0.0
    post_turn_start = 0.0
    last_turn_time = -999
    last_ns = time.monotonic_ns()
    start_ns = last_ns
    turn_target_delta = 0.0  # relative yaw target for this turn (deg)
    robot.stop_motor()                 # Ensure robot stopped at start
    robot.set_servo(SERVO_CENTER)      # Ensure robot stopped at start

    while True:
        current_ns = time.monotonic_ns()
        dt = (current_ns - last_ns) * 1e-9   # seconds
        last_ns = current_ns
        current_time = current_ns * 1e-9     # float seconds, used by UI/timers

        sensor_tick.wait()  # Wait until the sensor data is updated
        sensor_tick.clear()  # Clear the event after reading

        # Now sensor data is available, process it
        with sensor_lock:
            d_front = sensor_data["front"]
            d_left = sensor_data["left"]
            d_right = sensor_data["right"]
            d_front_left = sensor_data["front_left"]
            d_front_right = sensor_data["front_right"]

        
        # Update robot attributes for plotting/logging and decision making
        robot.d_front = d_front
        robot.d_left = d_left
        robot.d_right = d_right
        robot.d_front_left = d_front_left
        robot.d_front_right = d_front_right

        # --- Headless START button edge detection (press to PAUSE/RESUME) ---
        if not USE_GUI:
            cur = 0 if START_BTN.is_pressed else 1
            if _btn_prev == 1 and cur == 0 and (current_time - _btn_last_ts) > BTN_DEBOUNCE_S:
                _btn_last_ts = current_time
                headless_toggle_pause()
            _btn_prev = cur

        if not readings_event.is_set():
            # if readings not started, keep motors stopped and idle
            robot.stop_motor()
            state = RobotState.IDLE
            status_text = "Idle"
            sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
            continue

        # ----------------------------
        # Sensor reads & gyro integration (always update while readings_flag)
        # ----------------------------
        with sensor_lock:
            d_front = sensor_data["front"]
            d_left = sensor_data["left"]
            d_right = sensor_data["right"]
            d_front_left = sensor_data["front_left"]
            d_front_right = sensor_data["front_right"]

        # Update robot attributes for plotting/logging and decision making
        robot.d_front = d_front
        robot.d_left = d_left
        robot.d_right = d_right
        robot.d_front_left = d_front_left
        robot.d_front_right = d_front_right

        # ---- Narrow corridor detector (sum of side distances) ----
        if not hasattr(robot_loop, "_narrow_mode"):
            robot_loop._narrow_mode = False
     
        # Use None-safe math (ignore invalid readings)
        l = robot.d_left  if (robot.d_left  is not None and robot.d_left  != 999) else None
        r = robot.d_right if (robot.d_right is not None and robot.d_right != 999) else None
        sum_lr = None if (l is None or r is None) else (l + r)
     
        enter_thresh = NARROW_SUM_THRESHOLD
        exit_thresh  = NARROW_SUM_THRESHOLD + NARROW_HYSTERESIS
     
        prev_mode = robot_loop._narrow_mode
        if sum_lr is not None:
            if not robot_loop._narrow_mode and sum_lr < enter_thresh:
                robot_loop._narrow_mode = True
            elif robot_loop._narrow_mode and sum_lr > exit_thresh:
                robot_loop._narrow_mode = False
     
        # Apply global factors
        global SPEED_ENV_FACTOR, DIST_ENV_FACTOR
        SPEED_ENV_FACTOR = NARROW_FACTOR_SPEED if robot_loop._narrow_mode else 1.0
        DIST_ENV_FACTOR  = NARROW_FACTOR_DIST  if robot_loop._narrow_mode else 1.0
     
        # Optional console feedback
        if robot_loop._narrow_mode != prev_mode:
            mode_txt = "NARROW mode ON" if robot_loop._narrow_mode else "NARROW mode OFF"
            dprint(f"[corridor] {mode_txt} (L+R = {sum_lr:.1f} cm)" if sum_lr is not None else f"[corridor] {mode_txt}")
             
        with I2C_LOCK:
            raw_gyro_z = mpu.gyro[2] - bias      # rad/s from MPU6050
            ALPHA = 0.8
            gyro_z_filtered = ALPHA * raw_gyro_z + (1 - ALPHA) * getattr(robot, 'gyro_z_prev', 0.0)
            robot.gyro_z_prev = gyro_z_filtered
            yaw += (gyro_z_filtered * RAD2DEG) * dt

        # Append to deques for plotting/logging
        elapsed_time = (current_ns - start_ns) * 1e-9
        time_data.append(elapsed_time)
        front_data.append(robot.d_front if robot.d_front is not None else 0)
        left_data.append(robot.d_left if robot.d_left is not None else 0)
        right_data.append(robot.d_right if robot.d_right is not None else 0)
        _fl = getattr(robot, "d_front_left", None)
        _fr = getattr(robot, "d_front_right", None)
        front_left_data.append(_fl if _fl is not None else 0)
        front_right_data.append(_fr if _fr is not None else 0)
        angle_data.append(yaw)
        state_data.append(state.name)
       
        # If loop_flag is still False, do NOT run FSM/motors yet—just update plots
        if not loop_event.is_set():
            robot.stop_motor()
            state = RobotState.IDLE
            status_text = "Paused" if paused_by_button else "Sensor readings started"
            sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
            continue

        # ----------------------------
        # FSM transitions & actions
        # ----------------------------
        if state == RobotState.IDLE:
            status_text = "Ready (readings started, loop not started)" if readings_event.is_set() else "Idle"
            robot.stop_motor()
            robot.set_servo(SERVO_CENTER)
            robot.ss_reset()
            if loop_event.is_set():
                state = RobotState.CRUISE
                status_text = "Driving (cruise)"
                for speed in range(0, SPEED_CRUISE + 1):
                    if not loop_event.is_set():
                        robot.stop_motor()
                        break
                    robot.rotate_motor(speed)
                    sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()

        elif state == RobotState.CRUISE:
            status_text = "Driving (cruise)"
            # Capture an image and detect obstacles
            mask_red, mask_green, img_bgr = detect_obstacles()


         # Check for red obstacles
            red_data = get_largest_contour(mask_red, min_area=MIN_AREA)
            if red_data:
                _, top_left, bottom_right, area = red_data
                # Trigger avoidance logic for red
                avoidance_mode = True
                avoid_direction = "left"  # Example direction for red
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -AVOID_SPEED)
                set_servo_angle(SERVO_CHANNEL, LEFT_FAR)
                continue  # Skip to next loop iteration to avoid obstacle
        
            # Check for green obstacles
            green_data = get_largest_contour(mask_green, min_area=MIN_AREA)
            if green_data:
                _, top_left, bottom_right, area = green_data
                # Trigger avoidance logic for green
                avoidance_mode = True
                avoid_direction = "right"  # Example direction for green
                set_motor_speed(MOTOR_FWD, MOTOR_REV, -AVOID_SPEED)
                set_servo_angle(SERVO_CHANNEL, RIGHT_FAR)
                continue  # Skip to next loop iteration to avoid obstacle


            # -------------------------------
            # Emergency stop: immediate
            # -------------------------------
            if robot.d_front is not None and robot.d_front < eff_stop_threshold():
                robot.stop_motor()
                robot.set_servo(SERVO_CENTER)     
                status_text = "Stopped! Obstacle ahead"
                dprint(status_text)
                state = RobotState.STOPPED

                stop_reason = "OBSTACLE"
                obstacle_wait_deadline = current_time + OBSTACLE_WAIT_TIME
                continue
        
            # -------------------------------
            # Check if we can trigger a turn
            # -------------------------------
            front_triggered = (robot.d_front is not None and robot.d_front < eff_front_turn_trigger())
            lockout_ok = (current_time - last_turn_time >= TURN_LOCKOUT)

            if front_triggered and lockout_ok and loop_event.is_set():
                # immediately enter TURN_INIT and drop speed
                state = RobotState.TURN_INIT
                status_text = "Approaching turn – waiting for side to open"
                continue
                  
            # -------------------------------
            # Normal cruise servo control
            # Safe straight control active only after the 1st turn
            # -------------------------------
            if turn_count >= 1:
                desired_servo_angle = robot.safe_straight_control(robot.d_left, robot.d_right)
            else:
                desired_servo_angle = SERVO_CENTER
            
            robot.set_servo(desired_servo_angle)
            robot.set_state_speed(state.name)        

        elif state == RobotState.TURN_INIT:
            # Stay slow and keep wheels mostly straight while we wait for a side to open.
            # If you prefer slight wall-following here, use safe_straight_control().
            status_text = "Turn init – waiting for open side"
            robot.set_state_speed(state.name)
            robot.ss_reset()
        
            # fail-safe: if front becomes safe again, return to cruise
            if robot.d_front is not None and robot.d_front >= eff_front_turn_trigger():
                robot.set_servo(SERVO_CENTER)
                state = RobotState.CRUISE
                status_text = "Driving (cruise)"
                # brief yield
                sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
                continue
         
            # keep the wheels centered (or gentle straight correction)
            if turn_count >= 1:
                 desired_servo_angle = robot.safe_straight_control(robot.d_left, robot.d_right)
            else:
                 # No correction before first turn: keep the wheels centered (or gentle straight correction)
                 desired_servo_angle = SERVO_CENTER
            robot.set_servo(desired_servo_angle)

            # Decide direction only when exactly one side is open
            proposed_direction = robot.turn_decision(
              robot.d_left, robot.d_right,
              d_fl=robot.d_front_left, d_fr=robot.d_front_right
            )
        
            # If neither or both are open, keep waiting at TURN_INIT speed
            if proposed_direction is None:
                sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
                continue
        
            # Turn lock mechanic (unchanged logic, just applied here)
            if LOCK_TURN_DIRECTION == 1:
                if locked_turn_direction is None:
                    locked_turn_direction = proposed_direction
                    dprint(f"Turn direction locked to {locked_turn_direction}")
                elif proposed_direction != locked_turn_direction:
                    # keep waiting at TURN_INIT speed for the locked direction to open
                    status_text = f"Ignoring {proposed_direction} (locked to {locked_turn_direction})"
                    robot.set_servo(desired_servo_angle)
                    sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
                    continue
                direction = locked_turn_direction
            else:
                direction = proposed_direction
        
            # Commit the turn now that one side is open and lock (if any) allows it
            dprint(f"🔄 Turn initiated {direction}. Left: {robot.d_left if robot.d_left is not None else -1:.1f} cm, Right: {robot.d_right if robot.d_right is not None else -1:.1f} cm")
            turn_start_yaw = yaw
            turn_start_time = current_time
            
            # how skewed are we vs the nearest corridor axis at entry?
            entry_skew = norm180(yaw - snap90(yaw))  # + = already rotated to the LEFT
            
            # base ±TARGET_TURN_ANGLE, then compensate by entry skew
            base = TARGET_TURN_ANGLE if direction == "LEFT" else -TARGET_TURN_ANGLE
            turn_target_delta = base - entry_skew
            
            # keep within your safety bounds
            turn_target_delta = max(-MAX_TURN_ANGLE, min(MAX_TURN_ANGLE, turn_target_delta))
            if abs(turn_target_delta) < MIN_TURN_ANGLE:
                turn_target_delta = MIN_TURN_ANGLE if turn_target_delta >= 0 else -MIN_TURN_ANGLE
            
            # start turning
            angle = TURN_ANGLE_LEFT if direction == "LEFT" else TURN_ANGLE_RIGHT
            robot.set_servo(angle)
            robot.set_state_speed("TURNING")
            state = RobotState.TURNING

        elif state == RobotState.TURNING:
            # active turning: monitor yaw & safety/time conditions to stop
            # compute turn angle relative to start
            turn_angle = yaw - turn_start_yaw  # +left, -right
            target_angle = turn_target_delta   # adjusted by entry skew
            robot.set_state_speed("TURNING")
            stop_condition = False
            robot.ss_reset()

            # Condition A: Turn angle
            if abs(turn_angle - target_angle) <= TURN_ANGLE_TOLERANCE:
                dprint("Stop Turn - Target Angle")
                stop_condition = True
            # Condition B: timeout
            if current_time - turn_start_time > TURN_TIMEOUT:
                dprint("Stop Turn - Max turn time")
                stop_condition = True
            # Condition C: Max turn angle
            if abs(turn_angle) >= MAX_TURN_ANGLE:
                dprint("Stop Turn - Max turn angle")
                stop_condition = True

            # If any stop condition met -> finish turn
            if stop_condition:
                robot.stop_motor()
                robot.set_servo(SERVO_CENTER)
                last_turn_time = current_time
                yaw = snap90(yaw)
                # update counters
                turn_count += 1
                if turn_count % 4 == 0:
                    lap_count += 1
                    dprint(f"✅ Lap completed! Total laps: {lap_count}")
                    # if reached max laps -> do final drive then stop (as in original logic)
                    if MAX_LAPS > 0 and lap_count >= MAX_LAPS:
                        # Drive forward a little before stopping
                        robot.set_servo(SERVO_CENTER)
                        robot.set_state_speed(state.name)
                        sensor_tick.wait(POST_LAP_DURATION); sensor_tick.clear()
                        robot.stop_motor()
                        stop_reason = "LAPS"
                        loop_event.clear()
                        sensor_tick.set()  # wake waits immediately
                        state = RobotState.STOPPED
                        status_text = f"Stopped - Max Laps ({MAX_LAPS}) reached"
                        continue

                # prepare post-turn
                post_turn_start = current_time
                state = RobotState.POST_TURN
                status_text = "Driving (post-turn)"

        elif state == RobotState.POST_TURN:
            # drive straight for short duration then return to CRUISE
            if current_time - post_turn_start < POST_TURN_DURATION:
                robot.set_servo(SERVO_CENTER)
                robot.set_state_speed(state.name)
                status_text = "Driving (post-turn)"
            else:
                state = RobotState.CRUISE
                status_text = "Driving (cruise)"
        
        elif state == RobotState.STOPPED:
            robot.stop_motor()
            robot.set_servo(SERVO_CENTER)
            robot.ss_reset()
        
            # --- Obstacle-driven stop: auto-retry after OBSTACLE_WAIT_TIME ---
            if stop_reason == "OBSTACLE":
                # nice status with countdown
                remaining = max(0.0, obstacle_wait_deadline - current_time)
                status_text = f"Stopped (obstacle) – retry in {remaining:.1f}s"
        
                # time to retry? re-check front distance
                if current_time >= obstacle_wait_deadline:
                    # quick re-sample already happens in the sensor thread;
                    # just use the latest filtered reading here:
                    d_front_now = robot.d_front
        
                    if d_front_now is not None and d_front_now >= eff_stop_threshold():
                        # clear → resume cruise
                        dprint("✅ Obstacle cleared — resuming")
                        status_text = "Driving (cruise)"
                        stop_reason = None
                        state = RobotState.CRUISE
                        robot.set_state_speed("CRUISE")
                        # small fall-through; next loop iteration will continue normally
                    else:
                        # still blocked → schedule another check in OBSTACLE_WAIT_TIME
                        obstacle_wait_deadline = current_time + OBSTACLE_WAIT_TIME
        
                # yield CPU while waiting
                sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
                continue
        
            # --- User or laps stop: stay stopped until user action (original behavior) ---
            status_text = "Stopped"
            sensor_tick.wait(LOOP_DELAY); sensor_tick.clear()
            continue

# ===============================
# GUI SECTION
# ===============================
def launch_gui(): #Initialize Tkinter GUI, matplotlib plots, sliders, and status indicators.
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.use("TkAgg")  # explicit backend for GUI mode
    import matplotlib.pyplot as plt
    import tkinter.filedialog as fd
    global btn_readings, btn_start, btn_stop, root, canvas
    global ax_front, ax_side, ax_angle
    global front_line, left_line, right_line, angle_line
    global status_circle, status_canvas, status_text_id, lbl_status
    global lbl_turns, lbl_laps
    global sliders_frame, slider_vars, btn_export

    from tkinter import TclError 
    import sys, os, signal  # add

    GUI_CLOSING = False
    after_ids = {"status": None, "plot": None}

    # -------------------------
    # Export to CSV
    # -------------------------
    def export_data_csv():
        if not USE_GUI:
            dprint("Headless: CSV export disabled.")
            return
        filename = f"viva_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        slider_values = {name: var.get() for name, var in slider_vars.items()}
    
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [
                "Time (s)", "Front (cm)", "Left (cm)", "Right (cm)", "Yaw (deg)",
                "State", "Turns", "Laps", "Servo Angle", "Speed", "Turning", "Direction"
            ]
            header += [name for name in slider_values.keys()]
            writer.writerow(header)
    
            for i in range(len(time_data)):
                row = [
                    round(time_data[i], 2),
                    round(front_data[i], 2),
                    round(left_data[i], 2),
                    round(right_data[i], 2),
                    round(angle_data[i], 2),
                    state_data[i],
                    turn_count,
                    lap_count,
                    globals().get("servo_angle", 0),  # latest steering
                    globals().get("SPEED_CRUISE", 0),
                    "YES" if "Turning" in state_data[i] else "NO",
                    globals().get("locked_turn_direction", "")
                ]
                row += [slider_values[name] for name in slider_values.keys()]
                writer.writerow(row)
    
        print(f"✅ Data exported to {filename}")
    
    # -------------------------
    # Save/Load Slider Config
    # -------------------------
    def save_sliders_json():
        if not USE_GUI:
            print("Headless: save_sliders_json disabled.")
            return
        import tkinter.filedialog as fd
        default_filename = f"sliders_config_{datetime.now().strftime('%Y%m%d')}.json"
        file_path = fd.asksaveasfilename(initialdir=BASE_DIR,
                                         initialfile=default_filename,
                                         defaultextension=".json",
                                         filetypes=[("JSON files", "*.json")],
                                         title="Save Slider Configuration")
        if file_path:
            data = {name: var.get() for name, var in slider_vars.items()}
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Slider values saved to {file_path}")
    
    def load_sliders_json():
        if not USE_GUI:
            print("Headless: load_sliders_json disabled.")
            return
        import tkinter.filedialog as fd
        file_path = fd.askopenfilename(initialdir=BASE_DIR,
                                       defaultextension=".json",
                                       filetypes=[("JSON files", "*.json")],
                                       title="Load Slider Configuration")
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                for name, value in data.items():
                    if name in slider_vars:
                        slider_vars[name].set(value)
                        globals()[name] = value
                print(f"Sliders restored from {file_path}")
            except Exception as e:
                print(f"Failed to load sliders: {e}")

    def gui_alive():
        try:
            return not GUI_CLOSING and root.winfo_exists()
        except Exception:
            return False

    def _cancel_afters():
        try:
            if after_ids.get("status"):
                root.after_cancel(after_ids["status"])
        except Exception:
            pass
        try:
            if after_ids.get("plot"):
                root.after_cancel(after_ids["plot"])
        except Exception:
            pass

    def _really_exit():
        # ultimate “get me out” in case something else is blocking
        try: root.quit()
        except: pass
        try: root.destroy()
        except: pass
        os._exit(0)


    root = tk.Tk()
    root.title("VivaLaVida Robot Control")

    slider_vars = {}

    # ---------- Matplotlib Figure ----------
    fig, (ax_front, ax_side, ax_angle) = plt.subplots(3, 1, figsize=(6, 8))
    fig.tight_layout(pad=3.0)

    ax_front.set_ylim(0, 350)
    ax_front.set_title("Front Sensor")
    ax_front.grid(True)
    front_line, = ax_front.plot([], [], color="blue")

    ax_side.set_ylim(-150, 150)
    ax_side.set_title("Left (+Y) vs Right (-Y) Sensors")
    ax_side.grid(True)
    ax_side.axhline(0, color="black")
    left_line,  = ax_side.plot([], [], color="#009E73", label="Left")
    right_line, = ax_side.plot([], [], color="#E69F00", label="Right (neg)")
    # New: front-corner ToFs drawn on the same axes
    fl_line,    = ax_side.plot([], [], linestyle="--", color="#58D1B1", label="Front-Left")
    fr_line,    = ax_side.plot([], [], linestyle="--", color="#F3C553", label="Front-Right (neg)")
    #ax_side.legend(loc="upper right", fontsize=8)
    ax_side.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.18)  # make room at the bottom


    ax_angle.set_ylim(-180, 180)
    ax_angle.set_title("Yaw Angle")
    ax_angle.grid(True)
    angle_line, = ax_angle.plot([], [], color="purple")

    # ---------- Buttons ----------
    btn_readings = tk.Button(root, text="Start Readings", command=start_readings,
                             width=20, height=2, bg="blue", fg="white")
    btn_start = tk.Button(root, text="Start Loop", command=start_loop,
                          width=20, height=2, bg="green", fg="white", state="disabled")
    btn_stop = tk.Button(root, text="Stop Loop", command=stop_loop,
                         width=20, height=2, bg="red", fg="white", state="disabled")
  
    # ---------- Status Labels ----------
    lbl_status = tk.Label(root, text="Idle", font=("Arial", 14))
    lbl_turns = tk.Label(root, text=f"Turns: {turn_count}", font=("Arial", 14))
    lbl_laps = tk.Label(root, text=f"Laps: {lap_count}", font=("Arial", 14))

    # ---------- Circular Status Indicator ----------
    status_canvas = tk.Canvas(root, width=100, height=100, highlightthickness=0, bg=root.cget("bg"))
    status_circle = status_canvas.create_oval(10, 10, 90, 90, fill="grey", outline="")
    status_text_id = status_canvas.create_text(50, 50, text="IDLE", fill="white", font=("Arial", 14, "bold"))

    # ---------- Sliders Frame ----------
    sliders_frame = tk.LabelFrame(root, text="Parameters", padx=10, pady=10)
    sliders_frame.grid(row=0, column=1, rowspan=8, sticky="ns", padx=10, pady=5)

    # ---------- Column 3 frame for actions ----------
    actions_frame = tk.LabelFrame(root, text="Actions", padx=10, pady=10)
    actions_frame.grid(row=0, column=2, rowspan=8, sticky="ns", padx=10, pady=5)

    # ---------- Two-column layout ----------
    btn_readings.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
    btn_start.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
    btn_stop.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
    lbl_status.grid(row=3, column=0, pady=5)
    lbl_turns.grid(row=4, column=0, pady=2)
    lbl_laps.grid(row=5, column=0, pady=2)
    status_canvas.grid(row=6, column=0, pady=5)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=7, column=0, sticky="nsew", padx=5, pady=5)

    root.grid_columnconfigure(0, weight=1)  # main buttons + plots
    root.grid_columnconfigure(1, weight=0)  # sliders
    root.grid_columnconfigure(2, weight=0)  # actions
    root.grid_rowconfigure(7, weight=1)     # plot row

    # ---------- Slider Definitions ----------
    slider_groups = {
        "Speeds": [
            ("Cruise Speed", "SPEED_CRUISE", 0, 100, "int"),
            ("Turn Init Speed", "SPEED_TURN_INIT", 0, 100, "int"),
            ("Turn Speed", "SPEED_TURN", 0, 100, "int"),            
            ("Post Turn Speed", "SPEED_POST_TURN", 0, 100, "int")
        ],
        "Driving & Safety, Wall following, Turns": [
            ("Turn angle Left", "TURN_ANGLE_LEFT", 55, 90, "int"),
            ("Turn angle Right", "TURN_ANGLE_RIGHT", 90, 125, "int"),
            ("Soft side Margin (cm)", "SOFT_MARGIN", 0, 75, "int"),
            ("Max Angle Correction dif at Soft Margin (o)", "MAX_CORRECTION", 0, 30, "int"),
            ("Stop Threshold (cm)", "STOP_THRESHOLD", 0, 100, "int"),
            ("Front Turn Trigger (cm)", "FRONT_TURN_TRIGGER", 50, 150, "int"),
            ("Turn Decision Threshold (Left/Right) (cm)", "TURN_DECISION_THRESHOLD", 0, 200, "int"),
            ("Turn Timeout (s)", "TURN_TIMEOUT", 0.1, 10, "float"),
            ("Max Turn Angle (o)", "MAX_TURN_ANGLE", 90, 120, "int"),
            ("Post Turn Duration (s)", "POST_TURN_DURATION", 0, 5, "float"),
            ("Turn Lockout (s)", "TURN_LOCKOUT", 0.1, 5, "float")
        ],
        "Other": [
            ("Sensor Filter N (Median)", "N_READINGS", 1, 10, "int"),
            ("Max Laps - 0 for infinite", "MAX_LAPS", 0, 20, "int"),
            ("Filter Smoothness (alpha)", "FILTER_ALPHA", 0.05, 0.9, "float"),
            ("Max Jump (cm)", "FILTER_JUMP", 5, 999, "int"),
        ]
    }

    # ---------- Create Sliders ----------
    for group_name, sliders in slider_groups.items():
        group_frame = tk.LabelFrame(sliders_frame, text=group_name, padx=5, pady=5)
        group_frame.pack(fill="x", pady=5)
        for label_text, var_name, vmin, vmax, vartype in sliders:
            frame = tk.Frame(group_frame)
            frame.pack(fill="x", pady=2)

            if vartype == "float":
                var = tk.DoubleVar(value=globals()[var_name])
                res = 0.1
            else:
                var = tk.IntVar(value=globals()[var_name])
                res = 1

            slider_vars[var_name] = var

            scale = tk.Scale(frame, from_=vmin, to=vmax, orient="horizontal",
                             variable=var, resolution=res)
            scale.pack(side="left", fill="x", expand=True)

            lbl_val = tk.Label(frame, text=f"{label_text}: {var.get()}", width=25, anchor="w")
            lbl_val.pack(side="right", padx=5)

            def make_callback(lbl, name, var, label_text):
                def callback(value, label_text=label_text):
                    globals()[name] = var.get()
                    if name == "MAX_LAPS" and var.get() == 0:
                        lbl.config(text=f"{label_text}: ∞")
                    else:
                        lbl.config(text=f"{label_text}: {var.get():.1f}" if isinstance(var.get(), float) else f"{label_text}: {var.get()}")
                return callback

            scale.config(command=make_callback(lbl_val, var_name, var, label_text))

    # ---------- Buttons under sliders ----------
    btn_export = tk.Button(actions_frame, text="Export CSV", command=export_data_csv,
                       width=20, height=2, bg="purple", fg="white")
    btn_export.pack(pady=10, fill="x")

    btn_save_sliders = tk.Button(actions_frame, text="Save Sliders", command=save_sliders_json,
                             width=20, height=2, bg="green", fg="white")
    btn_save_sliders.pack(pady=5, fill="x")

    btn_load_sliders = tk.Button(actions_frame, text="Load Sliders", command=load_sliders_json,
                             width=20, height=2, bg="blue", fg="white")
    btn_load_sliders.pack(pady=5, fill="x")

    # ===============================
    # GUI Update Functions (nested)
    # ===============================
    def status_color(state: str) -> str:
        if "Stopped" in state:
            return "red"
        elif "Turning" in state:
            return "yellow"
        elif "Driving" in state:
            return "green"
        elif "Loop Started" in state or "Sensor readings started" in state:
            return "blue"
        else:
            return "grey"

    def status_label(state: str) -> str:
        if "Stopped" in state:
            return "STOP"
        elif "Turning" in state:
            return "TURN"
        elif "Driving" in state:
            return "GO"
        elif "Loop Started" in state:
            return "LOOP"
        elif "Sensor readings started" in state:
            return "READ"
        else:
            return "IDLE"

    def update_status():
        if not gui_alive():
            return
        lbl_status.config(text=status_text)
        status_canvas.itemconfig(status_circle, fill=status_color(status_text))
        status_canvas.itemconfig(status_text_id, text=status_label(status_text))
        lbl_turns.config(text=f"Turns: {turn_count}")
        lbl_laps.config(text=f"Laps: {lap_count}" if slider_vars["MAX_LAPS"].get() > 0 else f"Laps: {lap_count}/∞")
        try:
            after_ids["status"] = root.after(200, update_status)
        except TclError:
            # window is gone; just stop
            return

    _prev_label_pos = {
        "front": None, "left": None, "right": None,
        "front_left": None, "front_right": None,
        "angle": None
    }

    def update_plot():
        if not gui_alive():
            return
        try:
            if time_data:
                # Update lines
                front_line.set_data(range(len(front_data)), front_data)
                left_line.set_data(range(len(left_data)), left_data)
                right_line.set_data(range(len(right_data)), [-v for v in right_data])
                fl_line.set_data(range(len(front_left_data)), front_left_data)
                fr_line.set_data(range(len(front_right_data)), [-v for v in front_right_data])
                angle_line.set_data(range(len(angle_data)), angle_data)
        
                # Set axis limits
                ax_front.set_xlim(0, MAX_POINTS)
                ax_side.set_xlim(0, MAX_POINTS)
                ax_angle.set_xlim(0, MAX_POINTS)
        
                # Remove previous text annotations
                for ax in [ax_front, ax_side, ax_angle]:
                    for t in ax.texts:
                        t.remove()
        
                # Helper for smoothing
                def smooth_move(prev, target, alpha=0.3):
                    if prev is None:
                        return target
                    return prev + alpha * (target - prev)
        
                # ----------------------------
                # FRONT SENSOR
                # ----------------------------
                if len(front_data) > 0:
                    x = len(front_data) - 1
                    y_target = front_data[-1]
                    y_prev = _prev_label_pos["front"]
                    y_smooth = smooth_move(y_prev, y_target)
                    _prev_label_pos["front"] = y_smooth
                    ax_front.text(x, y_smooth, f"{y_target:.1f} cm", color="blue",
                                fontsize=9, fontweight="bold", va="bottom", ha="left")
        
                # ----------------------------
                # LEFT SENSOR
                # ----------------------------
                if len(left_data) > 0:
                    x = len(left_data) - 1
                    y_target = left_data[-1]
                    y_prev = _prev_label_pos["left"]
                    y_smooth = smooth_move(y_prev, y_target)
                    _prev_label_pos["left"] = y_smooth
                    ax_side.text(x, y_smooth, f"L: {y_target:.1f} cm", color="#009E73",
                                fontsize=9, fontweight="bold", va="bottom", ha="left")
        
                # ----------------------------
                # RIGHT SENSOR
                # ----------------------------
                if len(right_data) > 0:
                    x = len(right_data) - 1
                    y_target = -right_data[-1]
                    y_prev = _prev_label_pos["right"]
                    y_smooth = smooth_move(y_prev, y_target)
                    _prev_label_pos["right"] = y_smooth
                    ax_side.text(x, y_smooth, f"R: {right_data[-1]:.1f} cm", color="#E69F00",
                                fontsize=9, fontweight="bold", va="bottom", ha="left")

        
                # ----------------------------
                # FRONT-LEFT SENSOR (corner)
                # ----------------------------
                if len(front_left_data) > 0:
                    x = len(front_left_data) - 1
                    y_target = front_left_data[-1]
                    y_prev = _prev_label_pos["front_left"]
                    y_smooth = smooth_move(y_prev, y_target)
                    _prev_label_pos["front_left"] = y_smooth
                    ax_side.text(x, y_smooth, f"FL: {y_target:.1f} cm", color="#58D1B1", fontsize=9, fontweight="bold", va="bottom", ha="left")
                # ----------------------------
                # FRONT-RIGHT SENSOR (corner)
                # ----------------------------
                if len(front_right_data) > 0:
                    x = len(front_right_data) - 1
                    y_target = -front_right_data[-1]
                    y_prev = _prev_label_pos["front_right"]
                    y_smooth = smooth_move(y_prev, y_target)
                    _prev_label_pos["front_right"] = y_smooth
                    ax_side.text(x, y_smooth, f"FR: {front_right_data[-1]:.1f} cm", color="#F3C553", fontsize=9, fontweight="bold", va="bottom", ha="left")

                # ----------------------------
                # YAW ANGLE
                # ----------------------------
                if len(angle_data) > 0:
                    x = len(angle_data) - 1
                    y_target = angle_data[-1]
                    y_prev = _prev_label_pos["angle"]
                    y_smooth = smooth_move(y_prev, y_target)
                    _prev_label_pos["angle"] = y_smooth
                    ax_angle.text(x, y_smooth, f"{y_target:.1f}°", color="purple",
                                fontsize=9, fontweight="bold", va="bottom", ha="left")
        
                # ----------------------------
                # Dynamic Y-axis scaling for yaw
                # ----------------------------
                if angle_data:
                    min_angle = min(angle_data)
                    max_angle = max(angle_data)
                    if max_angle - min_angle > 180:
                        ax_angle.set_ylim(-180, 180)
                    else:
                        ax_angle.set_ylim(min_angle - 10, max_angle + 10)

                canvas.draw()
        except TclError:
            return
        try:
            after_ids["plot"] = root.after(100, update_plot)
        except TclError:
            return

    def on_closing():
        nonlocal GUI_CLOSING
        GUI_CLOSING = True

        # stop scheduling anything new
        _cancel_afters()

        # tell loops/threads to stand down
        try: loop_event.clear()
        except: pass
        try: readings_event.clear()
        except: pass
        try: sensor_tick.set()
        except: pass

        # stop robot now
        try:
            robot.stop_motor()
            robot.set_servo(SERVO_CENTER)
        except: 
            pass

        # shut down ToF
        for sensor in [vl53_left, vl53_right, vl53_front, vl53_back,
                    vl53_front_left, vl53_front_right]:
            if sensor is not None:
                try:
                    with I2C_LOCK:
                        sensor.stop_continuous()
                except Exception as e:
                    dprint(f"Warning: could not stop sensor {sensor}: {e}")

        # close DistanceSensors
        for us in [us_front, us_left, us_right]:
            try:
                if us is not None:
                    us.close()
            except:
                pass

        # LEDs / button
        try:
            GREEN_LED.off(); RED_LED.off()
            GREEN_LED.close(); RED_LED.close(); START_BTN.close()
        except:
            pass

        # IMPORTANT: quit the Tk loop first, then destroy
        try: root.quit()
        except: pass

        # Some Tk/Matplotlib stacks need the canvas widget gone before destroy
        try:
            canvas.get_tk_widget().destroy()
        except:
            pass

        try: root.destroy()
        except:
            pass


        # When done, stop the Picamera
        picam2.stop()

        # Failsafe: if something is still hanging, hard-exit soon
        root.after(100, _really_exit)


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.bind("<Escape>", lambda e: on_closing())   
    update_status()
    update_plot()
    root.mainloop()

# -------------------------
# Start / Stop
# -------------------------

def start_readings():
    global status_text
    readings_event.set()
    status_text = "Sensor readings started"
    if USE_GUI:
        btn_readings.config(state="disabled")
        btn_start.config(state="normal")

    # Start background sensor thread once
    if not hasattr(start_readings, "sensor_thread_started"):
        threading.Thread(target=sensor_reader, daemon=True).start()
        start_readings.sensor_thread_started = True

    if not hasattr(start_readings, "loop_thread_started"):
        threading.Thread(target=robot_loop, daemon=True).start()
        start_readings.loop_thread_started = True

    sensor_tick.set()

def start_loop(): #Start the robot driving loop
    global status_text
    if not readings_event.is_set():
        print("Start sensor readings first!")
        return

    loop_event.set()
    status_text = "🚗 Loop Started"
    if USE_GUI:
        btn_start.config(state="disabled")
        btn_stop.config(state="normal")

def stop_loop(): #Stop the robot loop and motor
    global status_text, turn_count, lap_count, locked_turn_direction, stop_reason
    loop_event.clear()
    stop_reason = "USER"
    sensor_tick.set()  # wake any waits immediately
    robot.stop_motor()
    status_text = "Stopped"
    turn_count = 0
    lap_count = 0

    if USE_GUI:
        btn_start.config(state="normal")
        btn_stop.config(state="disabled")
        lbl_turns.config(text=f"Turns: {turn_count}")
        lbl_laps.config(text=f"Laps: {lap_count}")

    locked_turn_direction = None # Reset direction lock so a new session can re-choose

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print(f"Starting VivaLaVida Autonomous Drive - GUI mode: {USE_GUI}")
    
    # Force headless if no display variable is present
    if os.environ.get("DISPLAY", "") == "":
        USE_GUI = 0

    if USE_GUI:
        started = launch_gui()
        if not started:
            print("⚠️ GUI not available. Falling back to headless.")
            USE_GUI = 0

    if not USE_GUI: # Start the sensor reading thread
        sensor_thread = threading.Thread(target=sensor_reader, daemon=True)
        sensor_thread.start()
        print("Headless mode: waiting for START button...")
        RED_LED.on()     # turn on red when ready
        GREEN_LED.blink(on_time=0.5, off_time=0.5, background=True)
        #GREEN_LED.on()  # green = ready to press the button
        try: # Wait for button press (GPIO input goes LOW when pressed)
            while not START_BTN.is_pressed:
                time.sleep(0.05)
            print("✅ START button pressed! Beginning autonomous loop...")
            readings_event.set() #readings_flag = True
            GREEN_LED.blink(on_time=0.1, off_time=0.1, background=True)
            RED_LED.off() 

            while START_BTN.is_pressed:
                time.sleep(0.01)
            time.sleep(BTN_DEBOUNCE_S)  # use existing debounce constant
            
            #time.sleep(2)  # wait 2 seconds before starting loop
            #loop_event.set() #loop_flag = True
            # Warm-up: wait (max 2s) until front/left/right have at least one usable value
            deadline = time.time() + 2.0
            while time.time() < deadline:
                with sensor_lock:
                    f = sensor_data["front"]
                    l = sensor_data["left"]
                    r = sensor_data["right"]
                if all(v is not None and v != 999 for v in (f, l, r)):
                    break
                sensor_tick.set()          # nudge the sensor thread
                time.sleep(0.5)
                
            time.sleep(1.5)
            loop_event.set()

            GREEN_LED.on()  # running
            print("Starting main robot loop...")
            robot_loop() # Start robot loop in this thread (blocking)
        except KeyboardInterrupt:
            print("\n❌ Keyboard interrupt received. Stopping robot loop.")     
            try: loop_event.clear() # Stop loops/threads
            except: pass
            try: readings_event.clear()
            except: pass
            try: sensor_tick.set()
            except: pass
            try: robot.stop_motor() # Motors & servo to safe state
            except: pass
            try: robot.set_servo(SERVO_CENTER)
            except: pass  
            try: # Stop any ToF sensors cleanly (if enabled)
                for s in [vl53_left, vl53_right, vl53_front, vl53_back, vl53_front_left, vl53_front_right]:
                    if s is not None:
                        try:
                            with I2C_LOCK:
                                s.stop_continuous()
                        except Exception as e:
                            dprint(f"[KeyboardInterrupt] ToF stop warning: {e}")
            except: 
                pass   
            try: # Zero PCA9685 outputs
                with I2C_LOCK:
                    for ch in [MOTOR_FWD, MOTOR_REV, SERVO_CHANNEL]:
                        pca.channels[ch].duty_cycle = 0
            except:
                pass
            try:
                GREEN_LED.off()
                RED_LED.off()
                GREEN_LED.close()
                RED_LED.close()
                START_BTN.close()
            except:
                pass

# End
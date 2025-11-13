from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# Initialize I2C and PCA9685 once
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)

def set_servo_angle(channel, angle, min_us=500, max_us=2500, frequency=50):
    """
    Control a servo on a given channel.
    :param channel: PCA9685 channel (0–15)
    :param angle: Servo angle (0–180)
    """
    if angle < 0: angle = 0
    if angle > 180: angle = 180

    pca.frequency = frequency
    pulse_length = 1000000 / pca.frequency / 4096  # microseconds per tick
    pulse = min_us + (angle / 180.0) * (max_us - min_us)
    ticks = int(pulse / pulse_length)  # 0–4095

    # Scale 12-bit ticks (0–4095) to 16-bit duty cycle (0–65535)
    pca.channels[channel].duty_cycle = int(ticks / 4096 * 0xFFFF)



def set_motor_speed(channel_forward, channel_reverse, speed, frequency=50):
    """
    Control a DC motor using two channels.
    :param channel_forward: PCA9685 channel for forward
    :param channel_reverse: PCA9685 channel for reverse
    :param speed: -100 (reverse) to +100 (forward)
    """
    if speed > 100: speed = 100
    if speed < -100: speed = -100

    pca.frequency = frequency
    duty_cycle = int(abs(speed) / 100.0 * 0xFFFF)

    if speed > 0:
        pca.channels[channel_forward].duty_cycle = duty_cycle
        pca.channels[channel_reverse].duty_cycle = 0
    elif speed < 0:
        pca.channels[channel_forward].duty_cycle = 0
        pca.channels[channel_reverse].duty_cycle = duty_cycle
    else:
        pca.channels[channel_forward].duty_cycle = 0
        pca.channels[channel_reverse].duty_cycle = 0
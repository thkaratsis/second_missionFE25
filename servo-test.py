# ======= SIMPLE SERVO TEST (compatible with your main robot code) =======

import time
from pca9685_control import set_servo_angle   # same module as in your main code

# ---- SAME SERVO CONSTANTS AS YOUR MAIN SCRIPT ----
SERVO_CHANNEL = 0

CENTER_ANGLE = 90
LEFT_FAR     = 110
LEFT_NEAR    = 130
RIGHT_FAR    = 70
RIGHT_NEAR   = 50

MIN_ANGLE = 50    # physical clamp (same as in main script)
MAX_ANGLE = 130

def safe_set(angle):
    """Clamp angle to safe range and send to servo."""
    angle = max(MIN_ANGLE, min(MAX_ANGLE, int(angle)))
    print(f" -> Servo angle: {angle}°")
    set_servo_angle(SERVO_CHANNEL, angle)
    return angle

def sweep_test():
    print("\n[SWEEP] Center -> Left_NEAR -> Right_NEAR -> Center")
    seq = [CENTER_ANGLE, LEFT_NEAR, RIGHT_NEAR, CENTER_ANGLE]
    for a in seq:
        safe_set(a)
        time.sleep(1.0)

def fine_sweep():
    print("\n[FINE SWEEP] Slowly sweep from 50° to 130° and back.")
    # forward
    for a in range(MIN_ANGLE, MAX_ANGLE + 1, 5):
        safe_set(a)
        time.sleep(0.05)
    # backward
    for a in range(MAX_ANGLE, MIN_ANGLE - 1, -5):
        safe_set(a)
        time.sleep(0.05)
    safe_set(CENTER_ANGLE)

def interactive_loop():
    print(
        "\n=== SERVO INTERACTIVE TEST ===\n"
        "Commands:\n"
        "  c   -> center (90°)\n"
        "  ln  -> LEFT_NEAR  (130°)\n"
        "  lf  -> LEFT_FAR   (110°)\n"
        "  rn  -> RIGHT_NEAR (50°)\n"
        "  rf  -> RIGHT_FAR  (70°)\n"
        "  s   -> custom angle (0-180)\n"
        "  q   -> quit\n"
    )

    current = safe_set(CENTER_ANGLE)

    while True:
        cmd = input("Command [c/ln/lf/rn/rf/s/q]: ").strip().lower()

        if cmd == "q":
            print("Exiting test, centering servo...")
            safe_set(CENTER_ANGLE)
            break
        elif cmd == "c":
            current = safe_set(CENTER_ANGLE)
        elif cmd == "ln":
            current = safe_set(LEFT_NEAR)
        elif cmd == "lf":
            current = safe_set(LEFT_FAR)
        elif cmd == "rn":
            current = safe_set(RIGHT_NEAR)
        elif cmd == "rf":
            current = safe_set(RIGHT_FAR)
        elif cmd == "s":
            try:
                val = int(input("Enter angle (0-180, will be clamped to 50-130): "))
                current = safe_set(val)
            except ValueError:
                print("Invalid number.")
        else:
            print("Unknown command.")

def main():
    print("=== SERVO TEST START ===")
    print("Initializing to CENTER...")
    safe_set(CENTER_ANGLE)
    time.sleep(1.0)

    # Quick automatic tests
    sweep_test()
    fine_sweep()

    # Manual control
    interactive_loop()

    print("=== SERVO TEST END ===")

if __name__ == "__main__":
    main()

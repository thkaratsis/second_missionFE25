import cv2
import numpy as np
from picamera2 import Picamera2

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

frame = None

def mouse_callback(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        bgr = frame[y, x]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Clicked at ({x},{y}) HSV: {hsv}")

cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", mouse_callback)

while True:
    # Capture RGB frame
    frame_rgb = picam2.capture_array()
    
    # Convert RGB → BGR for OpenCV display and correct HSV
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break

picam2.stop()
cv2.destroyAllWindows()
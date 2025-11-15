# Camera masks + highlight boxes (red/green) and lines (orange/blue)
# Raspberry Pi 5 + Picamera2 + OpenCV
# ---------------------------------------------------------------

import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ---- Tunables ----
CAM_SIZE = (640, 480)   # (W, H)
MIN_AREA = 2000         # ignore tiny blobs

# HSV thresholds (tweak for venue lighting)
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

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": CAM_SIZE}))
    picam2.start()
    time.sleep(1.0)

    try:
        while True:
            # Picamera2 gives RGB; convert once to BGR for correct drawing colors
            img_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # HSV for masking
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            # Masks
            red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
            red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
            mask_red = cv2.bitwise_or(red1, red2)
            mask_green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
            mask_orange = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
            mask_blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)

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
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
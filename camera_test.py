import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array("main")
    cv2.imshow("TEST", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

picam2.stop()
cv2.destroyAllWindows()

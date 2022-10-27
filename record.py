"""
Record videos
"""

import time

import cv2

from hardware import VidCap

cap = VidCap(1, 640, 480, 10)

while True:
    cap.read()
    cap.write()
    cap.imshow()

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

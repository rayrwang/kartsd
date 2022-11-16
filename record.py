"""
Record videos
"""

import cv2

from hardware import VidCap

try:
    with open("record_session_n.txt", 'r') as f:
        session_n = int(f.read())
except FileNotFoundError:
    session_n = 0

with open("record_session_n.txt", "w") as f:
    f.write(f"{session_n+1}")

cap = VidCap(5, 640, 480, session_n, 5)

while True:
    cap.read()
    cap.write()
    cap.imshow()

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

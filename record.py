"""
Record video to train.csv
"""

import time

import cv2 as cv
import numpy as np

import hardware

cap, board, angle_region, angle_read, last = hardware.init_hardware(no_pygame=True)

start = time.perf_counter()
with open("train.csv", "a") as file:
    while time.perf_counter() - start < 10:
        _, img = cap.read(0)
        cv.imshow("", img)

        if cv.waitKey(1) == ord("f"):
            cv.destroyAllWindows() 
            cv.VideoCapture(0).release()
            break

        angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

        flat = img.reshape(36864)
        flat = flat.astype("int8")
        full = np.insert(flat, 0, degree)

        np.savetxt(file, [full], fmt="%.0f", delimiter=",", newline="")
        file.write("\n")

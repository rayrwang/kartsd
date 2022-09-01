"""
Record video to train.csv
"""

import time

import cv2
import numpy as np
import pygame as pg

import hardware

cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=100)

with open("train.csv", "a") as file:
    start = time.perf_counter()
    while time.perf_counter() - start < 10:
        if cv2.waitKey(1) == ord("f"):
            cv2.destroyAllWindows()
            cv2.VideoCapture(0).release()
            break

        angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

        for event in pg.event.get():
            if event.type == update:
                # Update display and record frame at same time
                _, img = cap.read(0)
                cv2.imshow("", img)

                flat = img.reshape(36864)
                flat = flat.astype("float16")
                full = np.insert(flat, 0, degree)

                np.savetxt(file, [full], fmt="%.2f", delimiter=",", newline="")
                file.write("\n")

                hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree, 0)

"""
Record video to train.csv
"""

import math

import cv2 as cv
import numpy as np
import pygame as pg

import hardware

cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=500)

start = time.perf_counter()
with open("train.csv", "a") as file:
    while True:
        if cv.waitKey(1) == ord("f"):
            cv.destroyAllWindows() 
            cv.VideoCapture(0).release()
            break

        angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

        for event in pg.event.get():
            if event.type == update:
                # Update display and record frame at same time
                _, img = cap.read(0)
                cv.imshow("", img)

                flat = img.reshape(36864)
                flat = flat.astype("int16")
                full = np.insert(flat, 0, degree)

                np.savetxt(file, [full], fmt="%.0f", delimiter=",", newline="")
                file.write("\n")

                hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree)

"""
Record video to train.csv
"""

import math

import cv2 as cv
import numpy as np
import pygame as pg

import hardware

cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=1000)

with open("train.csv", "a") as file:
    while True:
        _, img = cap.read(0)
        cv.imshow("", img)

        if cv.waitKey(1) == ord("f"):
            cv.destroyAllWindows() 
            cv.VideoCapture(0).release()
            break

        angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

        flat = img.reshape(36864)
        flat = flat.astype("int16")
        full = np.insert(flat, 0, degree)

        np.savetxt(file, [full], fmt="%.0f", delimiter=",", newline="")
        file.write("\n")

        for event in pg.event.get():
            if event.type == update:
                hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree)

"""
Record video to train.csv
"""

import time
import threading

import cv2
import numpy as np
import pygame as pg

import hardware


def capture(cap0, cap1, cap2):
    while True:
        global img0
        global img1
        global img2
        _, img0 = cap0.read(0)
        _, img1 = cap1.read(0)
        _, img2 = cap2.read(0)


cap0, cap1, cap2, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=500)

capture_thread = threading.Thread(target=capture, args=[cap0, cap1, cap2])
capture_thread.start()
time.sleep(5)
with open("train.csv", "a") as file:
    while True:
        if cv2.waitKey(1) == ord("f"):
            cv2.destroyAllWindows()
            cv2.VideoCapture(0).release()
            break

        angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

        for event in pg.event.get():
            if event.type == update:
                # Update display and record frame at same time
                cv2.imshow("0", img0)
                cv2.imshow("1", img1)
                cv2.imshow("2", img2)

                flat0 = img0.reshape(36864)
                flat0 = flat0.astype("float16")
                flat1 = img1.reshape(76032)
                flat1 = flat1.astype("float16")
                flat2 = img2.reshape(76032)
                flat2 = flat2.astype("float16")
                full = np.concatenate((flat0, flat1, flat2))
                # full = np.insert(full, 0, degree)

                np.savetxt(file, [full], fmt="%.0f", delimiter=",", newline="")
                file.write("\n")

                hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree, 0)

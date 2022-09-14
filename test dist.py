

import cv2
import numpy as np
import pygame as pg

import hardware

cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=33)

while True:
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

            edges_img = cv2.Canny(img, 100, 200, apertureSize=3)
            edges_img[:29] = 0
            cv2.imshow("b", edges_img)

            hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree, 0)

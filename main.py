"""
Main file for inference and control
"""

import pygame as pg
import cv2

import hardware

cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=200)
while True:
    _, img = cap.read(0)
    cv2.imshow("", img)

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

    for event in pg.event.get():
        if event.type == update:
            hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree)

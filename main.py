"""
Main file for inference and control
"""

import math

import pyfirmata as pf
import pygame as pg
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

import hardware


class Network(nn.Module):
    def __init__(self):
        super().__init__()


cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=200)
while True:
    _, img = cap.read(0)
    cv.imshow("", img)

    if cv.waitKey(1) == ord("f"):
        cv.destroyAllWindows() 
        cv.VideoCapture(0).release()
        break

    angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

    for event in pg.event.get():
        if event.type == update:
            hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree)

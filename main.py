import time
import math
import sys

import pyfirmata as pf
import pygame as pg
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from . import hardware


class Network(nn.Module):
    def __init__(self):
        super().__init__()


def turn(deg):
    """turn steering by deg degrees (+ = ccw)"""
    if deg > 0:
        board.digital[4].write(1)
    elif deg < 0:
        board.digital[4].write(0)

    for i in range(round(deg * 7 * 800 / 360)):
        board.digital[2].write(0)
        board.digital[2].write(1)

# Init pygame display
window = pg.display.set_mode((0, 0))
pg.init()

clock = pg.time.Clock()

update = pg.USEREVENT + 1
pg.time.set_timer(update, 200)

# Init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv.CAP_PROP_FPS, 36)
while True:
    _, img = cap.read(0)
    cv.imshow("", img)

    if cv.waitKey(1) == ord("f"):
        break

    hardware.update_angle()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()
        if event.type == update:
            hardware.display_update()

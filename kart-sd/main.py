import pyfirmata as pf
import time
import pygame as pg
import cv2 as cv
from picamera import PiCamera
from picamera.array import PiRGBArray
import math
import sys

import numpy as np
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()


def turn(deg):
    if deg > 0:
        board.digital[4].write(1)
    elif deg < 0:
        board.digital[4].write(0)

    for i in range(round(deg * 7 * 800 / 360)):
        board.digital[2].write(0)
        board.digital[2].write(1)


board = pf.Arduino('/dev/ttyACM0')
it = pf.util.Iterator(board)
it.start()
board.analog[0].enable_reporting()

window = pg.display.set_mode((0, 0))
pg.init()

clock = pg.time.Clock()

update = pg.USEREVENT + 1
pg.time.set_timer(update, 200)

angle_region = 1
font0 = pg.font.Font("Helvetica.ttf", 100)
font1 = pg.font.Font("Helvetica.ttf", 75)
font2 = pg.font.Font("Helvetica.ttf", 25)
angle_read = 0.5
last = 0.5

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    cv.imshow("", image)
    rawCapture.truncate(0)

    if cv.waitKey(1) == ord("f"):
        break

    last = angle_read
    angle_read = board.analog[0].read()

    if angle_read > 0.85 and last < 0.15:
        angle_region -= 1
        print(angle_read, last)
    if angle_read < 0.15 and last > 0.85:
        angle_region += 1
        print(angle_read, last)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()
        if event.type == update:
            # Display update

            window.fill((255, 255, 255))

            region_txt = font2.render(f"{angle_region}", False, (0, 0, 0))
            window.blit(region_txt, (0, 0))

            angle_read_txt = font2.render(f"{angle_read}", False, (0, 0, 0))
            window.blit(angle_read_txt, (0, 50))

            if not 0 <= angle_region <= 2:
                degree_txt = font0.render("Error", False, (0, 0, 0))
                dir_txt = font1.render("Error", False, (0, 0, 0))
            else:
                if angle_region == 0:
                    degree = -1 / 7 * (0.04 - (0.96 - angle_read) - 0.5) * 360 / 0.92
                elif angle_region == 1:
                    degree = -1 / 7 * (angle_read - 0.5) * 360 / 0.92
                elif angle_region == 2:
                    degree = -1 / 7 * (0.96 + (angle_read - 0.04) - 0.5) * 360 / 0.92

                degree_txt = font0.render(f"{math.fabs(degree) : .2f}", False, (0, 0, 0))

                if not -35 < degree < 50:
                    degree_txt = font0.render("Error", False, (0, 0, 0))
                    dir_txt = font1.render("Error", False, (0, 0, 0))
                elif degree > 0:
                    dir_txt = font1.render("Left", False, (0, 0, 0))
                elif degree <= 0:
                    dir_txt = font1.render("Right", False, (0, 0, 0))

            degree_rect = degree_txt.get_rect()
            degree_rect.center = (300, 512)
            window.blit(degree_txt, degree_rect)

            dir_rect = dir_txt.get_rect()
            dir_rect.center = (300, 400)
            window.blit(dir_txt, dir_rect)

            pg.display.update()

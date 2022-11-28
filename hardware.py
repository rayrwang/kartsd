"""
For interacting with hardware (cameras, Arduino ports, display)
"""

import math
import time

import pyfirmata as pf
import pygame as pg
import cv2


class VidCap:
    """
    Class to hold necessary objects and functionality for cameras
    """

    def __init__(self, n_cams, fw, fh, session_n=None, fps=None):
        self.session_n = session_n
        self.n_cams = n_cams

        for cam in range(n_cams):
            new_cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
            setattr(self, f"cap{cam}", new_cap)

            if (session_n, fps) is not (None, None):
                setattr(self, f"wr{cam}",
                        cv2.VideoWriter(f"{session_n}_{cam}.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (fw, fh)))

            cv2.namedWindow(f"{cam}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{cam}", 240, 180)

    def read(self):
        for cam in range(self.n_cams):
            getattr(self, f"cap{cam}").grab()
        for cam in range(self.n_cams):
            _, img = getattr(self, f"cap{cam}").retrieve()
            setattr(self, f"img{cam}", img)

    def write(self):
        for cam in range(self.n_cams):
            getattr(self, f"wr{cam}").write(getattr(self, f"img{cam}"))

    def imshow(self):
        for cam in range(self.n_cams):
            cv2.imshow(f"{cam}", getattr(self, f"img{cam}"))


class Board:
    """Arduino control"""
    def __init__(self, port):
        # Init Arduino
        self.board = pf.Arduino(port)
        self.it = pf.util.Iterator(self.board)
        self.it.start()
        self.board.analog[0].enable_reporting()

        # Angle input data 0.04 to 0.96, 0.5=center
        # Angle region handles data crossing boundary (e.g. 0.96 to 0.04)
        self.angle_region = 1
        self.angle_read = 0.5
        self.last = 0.5
        self.degree = 0

        self.turn_deg = 0

    def update_angle(self):
        while True:
            time.sleep(0.01)  # Bare loops seem to mess with program flow (?)
            self.last = self.angle_read
            self.angle_read = self.board.analog[0].read()

            # Handle crossing angle region boundary
            if self.angle_read > 0.80 and self.last < 0.20:
                self.angle_region -= 1
            if self.angle_read < 0.20 and self.last > 0.80:
                self.angle_region += 1

            if self.angle_region == 0:
                self.degree = -1 / 7 * (0.04 - (0.96 - self.angle_read) - 0.5) * 360 / 0.92
            elif self.angle_region == 1:
                self.degree = -1 / 7 * (self.angle_read - 0.5) * 360 / 0.92
            elif self.angle_region == 2:
                self.degree = -1 / 7 * (0.96 + (self.angle_read - 0.04) - 0.5) * 360 / 0.92

    def turn(self):
        """turn steering ccw (dir=True), cw"""
        while True:
            deg = self.turn_deg - self.degree
            if math.fabs(deg) > 0.5:
                if deg > 0:
                    self.board.digital[4].write(1)
                else:
                    self.board.digital[4].write(0)

                self.board.digital[2].write(0)
                self.board.digital[2].write(1)
            else:
                time.sleep(0.01)  # Else deg display won't update idk why


class Display:
    def __init__(self, window):
        # Init pygame display
        self.window = window

        # Init fonts
        self.font0 = pg.font.Font("Helvetica.ttf", 75)
        self.font1 = pg.font.Font("Helvetica.ttf", 50)
        self.font2 = pg.font.Font("Helvetica.ttf", 25)

    def update(self, board, pred):
        # Display update
        region_txt = self.font2.render(f"{board.angle_region}", False, (0, 0, 0))
        self.window.blit(region_txt, (0, 0))

        angle_read_txt = self.font2.render(f"{board.angle_read}", False, (0, 0, 0))
        self.window.blit(angle_read_txt, (0, 50))

        pred_txt = self.font2.render(f"{pred : .2f}", False, (0, 0, 0))
        self.window.blit(pred_txt, (0, 100))

        degree_txt = self.font0.render(f"{math.fabs(board.degree) : .2f}", False, (0, 0, 0))

        if board.degree > 0:
            dir_txt = self.font1.render("Left", False, (0, 0, 0))
        elif board.degree <= 0:
            dir_txt = self.font1.render("Right", False, (0, 0, 0))

        # Display text
        degree_rect = degree_txt.get_rect()
        degree_rect.center = (252.5, 560)
        self.window.blit(degree_txt, degree_rect)

        dir_rect = dir_txt.get_rect()
        dir_rect.center = (252.5, 500)
        self.window.blit(dir_txt, dir_rect)

        pg.display.update()

"""
For interacting with hardware (cameras, Arduino ports, display)
"""

import math

import pyfirmata as pf
import pygame as pg
import cv2


class VidCap:
    """
    Class to hold necessary objects and functionality for cameras
    """
    def __init__(self, session_n, n_cams, fw, fh, fps):
        self.session_n = session_n
        self.n_cams = n_cams

        for cam in range(n_cams):
            new_cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
            new_cap.set(cv2.CAP_PROP_FPS, fps)
            setattr(self, f"cap{cam}", new_cap)

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

    def update_angle(self):
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

    def turn(self, deg):
        """turn steering ccw (dir=True), cw"""
        if deg > 0:
            self.board.digital[4].write(1)
        else:
            self.board.digital[4].write(0)

        for i in range(min(1, round(math.fabs(deg * 4)))):
            self.board.digital[2].write(0)
            self.board.digital[2].write(1)


class Display:
    def __init__(self, update_msec):
        # Init pygame display
        self.window = pg.display.set_mode((0, 0))
        pg.init()
        self.clock = pg.time.Clock()

        self.update_event = pg.USEREVENT + 1
        pg.time.set_timer(self.update_event, update_msec)

        # Init fonts
        self.font0 = pg.font.Font("Helvetica.ttf", 100)
        self.font1 = pg.font.Font("Helvetica.ttf", 75)
        self.font2 = pg.font.Font("Helvetica.ttf", 25)

    def update(self, board, pred):
        # Display update
        self.window.fill((255, 255, 255))

        region_txt = self.font2.render(f"{board.angle_region}", False, (0, 0, 0))
        self.window.blit(region_txt, (0, 0))

        angle_read_txt = self.font2.render(f"{board.angle_read}", False, (0, 0, 0))
        self.window.blit(angle_read_txt, (0, 50))

        pred_txt = self.font2.render(f"{pred : .4f}", False, (0, 0, 0))
        self.window.blit(pred_txt, (0, 100))

        degree_txt = self.font0.render(f"{math.fabs(board.degree) : .2f}", False, (0, 0, 0))

        if board.degree > 0:
            dir_txt = self.font1.render("Left", False, (0, 0, 0))
        elif board.degree <= 0:
            dir_txt = self.font1.render("Right", False, (0, 0, 0))

        # Display text
        degree_rect = degree_txt.get_rect()
        degree_rect.center = (300, 512)
        self.window.blit(degree_txt, degree_rect)

        dir_rect = dir_txt.get_rect()
        dir_rect.center = (300, 400)
        self.window.blit(dir_txt, dir_rect)

        pg.display.update()

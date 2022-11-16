"""
Main file for inference and control
"""

import math

import numpy as np
import cv2
import pygame as pg
import torch

from networks import VSNet
from hardware import VidCap


window = pg.display.set_mode((505, 600))
pg.init()
car = pg.Surface((0.9/0.25*5, 1.5/0.25*5))
car.fill((0, 0, 0))

# Read from videos
cap = VidCap(5, 640, 480)

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/test/vs25.pth", map_location=device))
model.eval()
while True:
    # Get new image
    cap.read()
    for i in range(5):
        _, globals()[f"img{i}"] = getattr(cap, f"img{i}")
    cap.imshow()

    # Inference
    img0 = torch.from_numpy(img0[None, :].astype("float32"))
    img1 = torch.from_numpy(img1[None, :].astype("float32"))
    img2 = torch.from_numpy(img2[None, :].astype("float32"))
    img3 = torch.from_numpy(img3[None, :].astype("float32"))
    img4 = torch.from_numpy(img4[None, :].astype("float32"))
    img0 = torch.swapaxes(img0, 1, 3)
    img1 = torch.swapaxes(img1, 1, 3)
    img2 = torch.swapaxes(img2, 1, 3)
    img3 = torch.swapaxes(img3, 1, 3)
    img4 = torch.swapaxes(img4, 1, 3)
    img0 = torch.swapaxes(img0, 2, 3)
    img1 = torch.swapaxes(img1, 2, 3)
    img2 = torch.swapaxes(img2, 2, 3)
    img3 = torch.swapaxes(img3, 2, 3)
    img4 = torch.swapaxes(img4, 2, 3)
    img0 = img0.to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    img3 = img3.to(device)
    img4 = img4.to(device)

    yh = model(img0, img1, img2, img3, img4)
    drivable = yh[:, :12120].reshape(120, 101)
    edge = yh[:, 12120:].reshape(120, 101)

    # Display
    # vs_blur = cv2.GaussianBlur(vs, (9, 9), 0)
    window.fill((255, 255, 255))
    for n_y, (edge_row, drivable_row) in enumerate(zip(edge, drivable)):
        for n_x, (e, d) in enumerate(zip(edge_row, drivable_row)):
            px = pg.Surface((5, 5))
            d = 210 + 45 * d
            d = max(0, d)
            d = min(255, d)
            pg.draw.rect(px, (d, d, d), (0, 0, 5, 5))

            if e > 0.2:
                e = 255 - e * 255
                e = max(0, e)
                e = min(255, e)
                pg.draw.rect(px, (255, e, e), (0, 0, 5, 5))

            window.blit(px, (5 * n_x, 595 - (5 * n_y)))
    window.blit(car, car.get_rect(center=(252.5, 400 + (1.5/2 - 0.2)/0.25*5)))
    pg.display.update()

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

# import threading
#
# import pygame as pg
# import cv2
# import torch
#
# import hardware
# from networks import SteerNet, VSNet
#
# model = SteerNet()
# device = torch.device("cpu")
# model.load_state_dict(torch.load("models/light_sides.pth", map_location=device))
# model.eval()
#
# cap0, cap1, cap2, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=500)
# cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
# cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
# center = -1.5
# gain = 1
# while True:
#     _, img0 = cap0.read(0)
#     cv2.imshow("", img0)
#
#     if cv2.waitKey(1) == ord("f"):
#         cv2.destroyAllWindows()
#         cv2.VideoCapture(0).release()
#         break
#
#     img0 = img0.astype("float32")
#     x = torch.from_numpy(img0[None, :])
#     x = torch.swapaxes(x, 1, 3)
#     x = torch.swapaxes(x, 2, 3)
#     x = x.to(device)
#     yhat = gain*(model(x).item() - center) + center
#     angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)
#
#     turn_thread = threading.Thread(target=hardware.turn, args=(board, yhat-degree))
#     # Safeguard
#     if -30 < degree < 30:
#         turn_thread.start()
#
#     for event in pg.event.get():
#         if event.type == update:
#             hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree, yhat)

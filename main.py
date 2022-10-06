"""
Main file for inference and control
"""

import threading

import pygame as pg
import cv2
import torch

import hardware
from networks import SteerNet, VSNet

model = SteerNet()
device = torch.device("cpu")
model.load_state_dict(torch.load("models/light_sides.pth", map_location=device))
model.eval()

cap0, cap1, cap2, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=500)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
center = -1.5
gain = 1
while True:
    _, img0 = cap0.read(0)
    cv2.imshow("", img0)

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    img0 = img0.astype("float32")
    x = torch.from_numpy(img0[None, :])
    x = torch.swapaxes(x, 1, 3)
    x = torch.swapaxes(x, 2, 3)
    x = x.to(device)
    yhat = gain*(model(x).item() - center) + center
    angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

    turn_thread = threading.Thread(target=hardware.turn, args=(board, yhat-degree))
    # Safeguard
    if -30 < degree < 30:
        turn_thread.start()

    for event in pg.event.get():
        if event.type == update:
            hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree, yhat)

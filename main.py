"""
Main file for inference and control
"""

import pygame as pg
import cv2
import torch

import hardware
from network import Network

model = Network()
device = torch.device("cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

cap, board, angle_region, angle_read, last, window, font0, font1, font2, update = hardware.init_hardware(update_msec=200)
cal = 14
while True:
    _, img = cap.read(0)
    cv2.imshow("", img)

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    img = img.astype("float32")
    x = torch.from_numpy(img[None, :])
    x = torch.swapaxes(x, 1, 3)
    x = torch.swapaxes(x, 2, 3)
    x = x.to(device)
    yhat = model(x).item() - cal

    angle_region, angle_read, last, degree = hardware.update_angle(board, angle_region, angle_read, last)

    direction = True if yhat > degree else False
    # Safeguard
    if -20 < degree < 20:
        hardware.turn(board, direction)

    for event in pg.event.get():
        if event.type == update:
            hardware.update_display(window, font0, font1, font2, angle_region, angle_read, degree, yhat)

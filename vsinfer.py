import time
import math
import threading

import numpy as np
import cv2
import pygame as pg
import torch

from network import VSNet


def capture(cap0, cap1, cap2):
    while True:
        global img0
        global img1
        global img2
        _, img0 = cap0.read(0)
        _, img1 = cap1.read(0)
        _, img2 = cap2.read(0)


# window = pg.display.set_mode((505, 655))
# pg.init()

# Init video and vs displays
cap0 = cv2.VideoCapture(0)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
cap0.set(cv2.CAP_PROP_FPS, 30)

cap1 = cv2.VideoCapture(42)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 176)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
cap1.set(cv2.CAP_PROP_FPS, 30)

cap2 = cv2.VideoCapture(43)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 176)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
cap2.set(cv2.CAP_PROP_FPS, 30)

capture_thread = threading.Thread(target=capture, args=[cap0, cap1, cap2])
capture_thread.start()

# car = pg.Surface((35, 55))

# device = torch.device("cpu")
# model = VSNet().to(device)
# model.load_state_dict(torch.load("models/vs.pth", map_location=device))

# model.eval()
# norm1 = torch.nn.BatchNorm2d(3)
# norm1 = norm1.to(device)

time.sleep(5)
while True:
    cv2.imshow("1", img0)
    cv2.imshow("2", img1)
    cv2.imshow("3", img2)
    # img0 = img0[50:, :, :]
    # img1 = img1[85:, :, :]
    # img2 = img2[85:, :, :]
    # cv2.imshow("4", img0)
    # cv2.imshow("5", img1)
    # cv2.imshow("6", img2)

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    

    # # Test inference
    # img0 = img0.astype("float32")
    # img1 = img1.astype("float32")
    # img2 = img2.astype("float32")
    # x0 = torch.from_numpy(img0[None, :])
    # x0 = torch.swapaxes(x0, 1, 3)
    # x0 = torch.swapaxes(x0, 2, 3)
    # x0 = x0.to(device)

    # x1 = torch.from_numpy(img1[None, :])
    # x1 = torch.swapaxes(x1, 1, 3)
    # x1 = torch.swapaxes(x1, 2, 3)
    # x1 = x1.to(device)

    # x2 = torch.from_numpy(img2[None, :])
    # x2 = torch.swapaxes(x2, 1, 3)
    # x2 = torch.swapaxes(x2, 2, 3)
    # x2 = x2.to(device)
    # print(x0.dtype, x1.dtype, x2.dtype)
    # x0 = norm1(x0)
    # x1 = norm1(x1)
    # x2 = norm1(x2)
    # vs_pred = model(x0, x1, x2)

    # # Display vs
    # window.fill((255, 255, 255))
    # window.blit(car, (235, 600))
    # for n_y, y_row in enumerate(vs_pred.reshape(120, 101)):
    #     for n_x, x in enumerate(y_row):
    #         x = 255 - x * 255
    #         x = max(0, x)
    #         x = min(255, x)

    #         rect = pg.Surface((5, 5))
    #         pg.draw.rect(rect, (255, x, x), (0, 0, 5, 5))
    #         window.blit(rect, (5 * n_x, 595 - (5 * n_y)))
    # pg.display.update()

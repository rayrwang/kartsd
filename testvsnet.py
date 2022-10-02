import time
import math

import numpy as np
import cv2
import pygame as pg
import torch

from network import VSNet


window = pg.display.set_mode((505, 655))
pg.init()

# Load data
arr = np.loadtxt("rawvids/big1.csv", dtype="float16", delimiter=",", max_rows=None)
vid_arr = arr[:, :299520]
vid_arr = vid_arr.astype("uint8")

# Init video and vs displays
img_num = 0
prev_img_num = -1

cv2.namedWindow("1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1", 512, 384)
cv2.namedWindow("2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2", 352, 288)
cv2.namedWindow("3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3", 352, 288)

car = pg.Surface((35, 55))

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/vs.pth", map_location=device))

model.eval()
while True:
    if img_num != prev_img_num:
        images = vid_arr[img_num]
        img0 = images[:147456]
        img0 = img0.reshape(192, 256, 3)
        img1 = images[147456:223488]
        img1 = img1.reshape(144, 176, 3)
        img2 = images[223488:299520]
        img2 = img2.reshape(144, 176, 3)
        cv2.imshow("1", img0)
        cv2.imshow("2", img1)
        cv2.imshow("3", img2)
        img0 = img0[50:, :, :]
        img1 = img1[85:, :, :]
        img2 = img2[85:, :, :]

        # Test inference
        img0 = img0.astype("float32")
        img1 = img1.astype("float32")
        img2 = img2.astype("float32")
        x0 = torch.from_numpy(img0[None, :])
        x0 = torch.swapaxes(x0, 1, 3)
        x0 = torch.swapaxes(x0, 2, 3)
        x0 = x0.to(device)

        x1 = torch.from_numpy(img1[None, :])
        x1 = torch.swapaxes(x1, 1, 3)
        x1 = torch.swapaxes(x1, 2, 3)
        x1 = x1.to(device)

        x2 = torch.from_numpy(img2[None, :])
        x2 = torch.swapaxes(x2, 1, 3)
        x2 = torch.swapaxes(x2, 2, 3)
        x2 = x2.to(device)
        vs_pred = model(x0, x1, x2)

        # Display vs
        window.fill((255, 255, 255))
        window.blit(car, (235, 600))
        for n_y, y_row in enumerate(vs_pred.reshape(120, 101)):
            for n_x, x in enumerate(y_row):
                # if x == 1:
                x = 255 - x * 255
                x = max(0, x)
                x = min(255, x)

                rect = pg.Surface((5, 5))
                pg.draw.rect(rect, (255, x, x), (0, 0, 5, 5))
                window.blit(rect, (5 * n_x, 595 - (5 * n_y)))
        pg.display.update()

        prev_img_num = img_num

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    keys = pg.key.get_pressed()
    if keys[pg.K_a]:
        img_num -= 5
    if keys[pg.K_d]:
        img_num += 5

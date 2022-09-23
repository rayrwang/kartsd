import time
import math

import numpy as np
import cv2
import pygame as pg
import torch

from network import VSNet


window = pg.display.set_mode((810, 810))
pg.init()

# Load data
arr = np.loadtxt("rawvids/1.csv", dtype="float16", delimiter=",")
steer_arr = arr[:, 0]
vid_arr = np.delete(arr, 0, axis=1)
vid_arr = vid_arr.astype("uint8")

# arr = np.loadtxt("vs_train.csv", delimiter=",")
# vid_arr = arr[:, :36864]
# vid_arr = vid_arr.astype("uint8")
# vs_arr = arr[:, 36864:]

# Init video and vs displays
img_num = 0
prev_img_num = -1

cv2.namedWindow("a", cv2.WINDOW_NORMAL)
cv2.resizeWindow("a", 512, 384)

# cv2.namedWindow("b", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("b", 512, 384)

car = pg.Surface((70, 110))
pg.draw.rect(car, (0, 0, 0), (0, 0, 70, 110))

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/vs.pth", map_location=device))

model.eval()
while True:
    if img_num != prev_img_num:
        img = vid_arr[img_num]
        img = img.reshape(96, 128, 3)
        cv2.imshow("a", img)

        # Test inference
        img = img.astype("float32")
        x = torch.from_numpy(img[None, :])
        x = torch.swapaxes(x, 1, 3)
        x = torch.swapaxes(x, 2, 3)
        x = x.to(device)
        vs_pred = model(x)

        # Display vs
        window.fill((255, 255, 255))
        window.blit(car, (370, 700))
        for n_y, y_row in enumerate(vs_pred.reshape(70, 81)):
            for n_x, x in enumerate(y_row):
                # if x != 0:
                x = 255 - x*255
                x = max(0, x)
                x = min(255, x)

                rect = pg.Surface((10, 10))
                pg.draw.rect(rect, (255, x, x), (0, 0, 10, 10))
                window.blit(rect, (10*n_x, 690-(10*n_y)))
        pg.display.update()

        prev_img_num = img_num

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    # time.sleep(0.1)
    keys = pg.key.get_pressed()
    if keys[pg.K_a]:
        img_num -= 5
    if keys[pg.K_d]:
        img_num += 5

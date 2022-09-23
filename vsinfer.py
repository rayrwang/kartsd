import time
import math

import numpy as np
import cv2
import pygame as pg
import torch

from network import VSNet


window = pg.display.set_mode((810, 810))
pg.init()

# Init video and vs displays
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
cap.set(cv2.CAP_PROP_FPS, 30)

car = pg.Surface((70, 110))
pg.draw.rect(car, (0, 0, 0), (0, 0, 70, 110))

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/vs.pth", map_location=device))

model.eval()
while True:
    # steer = steer_arr[img_num]
    _, img = cap.read(0)
    cv2.imshow("", img)

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

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

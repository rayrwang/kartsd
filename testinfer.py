import math

import numpy as np
import cv2
import pygame as pg
import torch

from networks import VSNet


window = pg.display.set_mode((505, 600))
pg.init()
car = pg.Surface((0.9/0.25*5, 1.5/0.25*5))
car.fill((0, 0, 0))

# Init video displays
prev_img_num = -1
session_n = 1
img_num = 2000

# Read from videos
for i in range(5):
    globals()[f"cap{i}"] = cv2.VideoCapture(f"rawvids/{session_n}_{i}.avi")
    cv2.namedWindow(f"{i}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"{i}", 512, 384)

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/vs125.pth", map_location=device))
norm1 = torch.nn.BatchNorm2d(3)
norm1 = norm1.to(device)
while True:
    # Handle key pressed
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        img_num += 20
    elif keys[pg.K_a]:
        img_num -= 20
    elif keys[pg.K_f]:
        edge_img_changed = True

    if img_num != prev_img_num:
        # Get new image
        for i in range(5):
            globals()[f"cap{i}"].set(cv2.CAP_PROP_POS_FRAMES, img_num)
            _, globals()[f"img{i}"] = globals()[f"cap{i}"].read()
            cv2.imshow(f"{i}", globals()[f"img{i}"])

        prev_img_num = img_num

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
        img0 = norm1(img0)
        img1 = norm1(img1)
        img2 = norm1(img2)
        img3 = norm1(img3)
        img4 = norm1(img4)

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

            if e > 0.1:
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

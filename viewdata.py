import time
import math

import numpy as np
import cv2
import pygame as pg


window = pg.display.set_mode((810, 810))
pg.init()

# # Load data
# arr = np.loadtxt("Center - Copy.csv", dtype="float16", delimiter=",")
# steer_arr = arr[:, 0]
# vid_arr = np.delete(vid_arr, 0, axis=1)
# vid_arr = vid_arr.astype("uint8")

arr = np.loadtxt("vs_train.csv", delimiter=",")
vid_arr = arr[:, :36864]
vid_arr = vid_arr.astype("uint8")
vs_arr = arr[:, 36864:]

# Init video and vs displays
prev_img_num = -1
img_num = 0

cv2.namedWindow("a", cv2.WINDOW_NORMAL)
cv2.resizeWindow("a", 512, 384)

# cv2.namedWindow("b", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("b", 512, 384)

car = pg.Surface((70, 110))
pg.draw.rect(car, (0, 0, 0), (0, 0, 70, 110))
while True:
    if img_num != prev_img_num:
        # steer = steer_arr[img_num]
        img = vid_arr[img_num]
        img = img.reshape(96, 128, 3)
        cv2.imshow("a", img)

        # print(steer, img_num)

        # Display vs
        window.fill((255, 255, 255))
        window.blit(car, (370, 700))
        for n_y, y_row in enumerate(vs_arr[img_num].reshape(70, 81)):
            for n_x, x in enumerate(y_row):
                if x != 0:
                    rect = pg.Surface((10, 10))
                    pg.draw.rect(rect, (0, 0, 0), (0, 0, 10, 10))
                    window.blit(rect, (10*n_x, 690-(10*n_y)))
        pg.display.update()

        prev_img_num = img_num

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    time.sleep(0.05)
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        img_num += 1
    if keys[pg.K_a]:
        img_num -= 1

import time

import numpy as np
import cv2
import pygame as pg

window = pg.display.set_mode((100, 100))
pg.init()

vid_arr = np.loadtxt("train.csv", dtype="float16", delimiter=",")
steer_arr = vid_arr[:, 0]
vid_arr = np.delete(vid_arr, 0, axis=1)
vid_arr = vid_arr.astype("uint8")

prev_img_num = -1
img_num = 0

while True:
    if img_num != prev_img_num:
        steer = steer_arr[img_num]
        img = vid_arr[img_num]
        img = img.reshape(96, 128, 3)

        cv2.namedWindow("a", cv2.WINDOW_NORMAL)
        cv2.imshow("a", img)
        print(steer, img_num)

        prev_img_num = img_num

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

    time.sleep(0.1)
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        img_num += 1
    if keys[pg.K_a]:
        img_num -= 1

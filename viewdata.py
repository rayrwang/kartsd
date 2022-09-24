import time
import math

import numpy as np
import cv2
import pygame as pg

window = pg.display.set_mode((610, 810))
pg.init()

# Load data
arr = np.loadtxt("vstrainingdata/noshadows_clean_fixed.csv", delimiter=",")
vid_arr = arr[:, :36864]
vid_arr = vid_arr.astype("uint8")
vs_arr = arr[:, 36864:]

# Init video and vs displays
img_num = 0

cv2.namedWindow("a", cv2.WINDOW_NORMAL)
cv2.resizeWindow("a", 512, 384)

# cv2.namedWindow("b", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("b", 512, 384)

car = pg.Surface((70, 110))
pg.draw.rect(car, (0, 0, 0), (0, 0, 70, 110))

while True:
    # steer = steer_arr[img_num]
    img = vid_arr[img_num]
    img = img.reshape(96, 128, 3)
    cv2.imshow("a", img)

    # print(steer, img_num)
    print(img_num, arr.shape[0], vid_arr.shape[0], vs_arr.shape[0])

    # Display vs
    window.fill((255, 255, 255))
    window.blit(car, (270, 700))
    for n_y, y_row in enumerate(vs_arr[img_num].reshape(70, 61)):
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

    time.sleep(0.1)
    keys = pg.key.get_pressed()
    if keys[pg.K_w]:
        vid_arr = np.delete(vid_arr, img_num, 0)
        vs_arr = np.delete(vs_arr, img_num, 0)
    if keys[pg.K_a]:
        img_num -= 1
    if keys[pg.K_d]:
        img_num += 1

# with open("vstrainingdata/noshadows_clean.csv", "a") as file:
#     vid_arr = vid_arr.reshape(-1, 36864)
#     vs_arr = vs_arr.reshape(-1, 4270)
#     full = np.concatenate((vid_arr, vs_arr), axis=1)
#     np.savetxt(file, full, fmt="%.0f", delimiter=",")

import time
import math

import numpy as np
import cv2
import pygame as pg

window = pg.display.set_mode((505, 655))
pg.init()

# Load data
arr = np.loadtxt("vstrainingdata/vs_train_clean.csv", delimiter=",", dtype="float32", max_rows=10)
vid_arr = arr[:, :299520]
vid_arr = vid_arr.astype("uint8")
vs_arr = arr[:, 299520:]

vs_blur_arr = np.zeros((1, 120, 101))
for vs in vs_arr:
    # vs_blur = cv2.GaussianBlur(vs.reshape((120, 101)), (25, 25), 1)
    vs_blur = vs.reshape((120, 101))
    vs_blur_arr = np.append(vs_blur_arr, [vs_blur], 0)
vs_blur_arr = np.delete(vs_blur_arr, 0, 0)

# Init video and vs displays
img_num = 0

cv2.namedWindow("1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1", 512, 384)
cv2.namedWindow("2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2", 352, 288)
cv2.namedWindow("3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3", 352, 288)

car = pg.Surface((35, 55))

while True:
    try:
        images = vid_arr[img_num]
    except:
        break
    img0 = images[:147456]
    img0 = img0.reshape(192, 256, 3)
    img1 = images[147456:223488]
    img1 = img1.reshape(144, 176, 3)
    img2 = images[223488:]
    img2 = img2.reshape(144, 176, 3)
    cv2.imshow("1", img0)
    cv2.imshow("2", img1)
    cv2.imshow("3", img2)

    # print(steer, img_num)
    print(img_num, arr.shape[0], vid_arr.shape[0], vs_blur_arr.shape[0])

    # Display vs
    window.fill((255, 255, 255))
    window.blit(car, (235, 600))
    for n_y, y_row in enumerate(vs_blur_arr[img_num]):
        for n_x, x in enumerate(y_row):
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

    time.sleep(0.05)
    keys = pg.key.get_pressed()
    if keys[pg.K_w]:
        vid_arr = np.delete(vid_arr, img_num, 0)
        vs_blur_arr = np.delete(vs_blur_arr, img_num, 0)
    if keys[pg.K_a]:
        img_num -= 1
    if keys[pg.K_d]:
        img_num += 1


# with open("vstrainingdata/vs_train_clean.csv", "a") as file:
#     vs_blur_arr = vs_blur_arr.reshape(-1, 12120)
#     full = np.concatenate((vid_arr, vs_blur_arr), axis=1)
#     np.savetxt(file, full, fmt="%.4f", delimiter=",")

import time
import math

import numpy as np
import cv2
import pygame as pg

window = pg.display.set_mode((820, 830))
pg.init()

# Load data
vid_arr = np.loadtxt("Center.csv", dtype="float16", delimiter=",")
steer_arr = vid_arr[:, 0]
vid_arr = np.delete(vid_arr, 0, axis=1)
vid_arr = vid_arr.astype("uint8")

# Init video and vs displays
prev_img_num = -1
img_num = 0

cv2.namedWindow("a", cv2.WINDOW_NORMAL)
cv2.resizeWindow("a", 512, 384)

cv2.namedWindow("b", cv2.WINDOW_NORMAL)
cv2.resizeWindow("b", 512, 384)

car = pg.Surface((70, 110))
pg.draw.rect(car, (0, 0, 0), (0, 0, 70, 110))
while True:
    if img_num != prev_img_num:
        steer = steer_arr[img_num]
        img = vid_arr[img_num]
        img = img.reshape(96, 128, 3)
        cv2.imshow("a", img)

        edges_img = cv2.Canny(img, 100, 200, apertureSize=3)

        edges_img[:29] = 0
        cv2.imshow("b", edges_img)
        print(steer, img_num)

        # Compute physical x and y for pixels in edges
        vs = np.zeros((72, 82))

        for px_y, row in enumerate(edges_img[30:]):  # px_y : pixels below horizon
            for px_x, pos in enumerate(row):  # px_x : pixels from left (48 to center)
                if pos == 255:
                    att = math.pi/180 * 0.48*(px_y + 10)
                    azi = math.pi/180 * 0.48*(px_x - 64)
                    dist = 0.7 / math.tan(att)  # dist : distance on ground from camera to px location
                    x = dist * math.sin(azi)
                    y = dist * math.cos(azi)

                    vs_x = round((x + 5.125) / 0.125)  # Adjust x to move 0 from center to left
                    vs_y = round((y - 0.125) / 0.125)  # Pull y half a vs block down
                    vs[vs_y, vs_x] = 1

        # Display vs
        window.fill((255, 255, 255))
        window.blit(car, (375, 720))
        for n_y, y_row in enumerate(vs):
            for n_x, x in enumerate(y_row):
                if x == 1:
                    rect = pg.Surface((10, 10))
                    pg.draw.rect(rect, (0, 0, 0), (0, 0, 10, 10))
                    window.blit(rect, (10*n_x, 720-(10*n_y)))
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

import time

import numpy as np
import cv2
import pygame as pg

window = pg.display.set_mode((505, 600))
pg.init()
car = pg.Surface((0.9/0.25*5, 1.5/0.25*5))
car.fill((0, 0, 0))

img_num = 0

# Read from videos
data = np.load("vstrainingdata/vs_train_rough.npy")
data = data.astype("float32")
for i in range(5):
    globals()[f"img_arr{i}"] = data[:, i*640*480*3:(i+1)*640*480*3].astype("uint8")
    cv2.namedWindow(f"{i}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"{i}", 512, 384)
drivable_arr = data[:, 5*640*480*3:5*640*480*3 + 120*101]
edge_arr = data[:, 5*640*480*3 + 120*101:]

# Blur edges
blur_arr = np.empty(edge_arr.reshape(-1, 120, 101).shape)
blur_arr = cv2.GaussianBlur(edge_arr.reshape(-1, 120, 101).swapaxes(0, 2), (3, 3), 1).swapaxes(0, 2).reshape(-1, 12120)
blur_arr = np.minimum(3*blur_arr, 1)

drivable_arr = cv2.GaussianBlur(drivable_arr.reshape(-1, 120, 101).swapaxes(0, 2), (5, 5), 1).swapaxes(0, 2).reshape(-1, 12120)


while True:
    print(img_num, data.shape[0])
    # Handle key pressed
    time.sleep(0.05)
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        img_num += 1
    elif keys[pg.K_a]:
        img_num -= 1

    for i in range(5):
        globals()[f"img{i}"] = globals()[f"img_arr{i}"][img_num]
    drivable = drivable_arr[img_num]
    edge = blur_arr[img_num]

    img0 = img0.reshape(480, 640, 3)
    img1 = img1.reshape(480, 640, 3)
    img2 = img2.reshape(480, 640, 3)
    img3 = img3.reshape(480, 640, 3)
    img4 = img4.reshape(480, 640, 3)
    drivable = drivable.reshape(120, 101)
    edge = edge.reshape(120, 101)

    # Show images
    for i in range(5):
        cv2.imshow(f"{i}", globals()[f"img{i}"])

    # Display
    window.fill((255, 255, 255))
    for n_y, (edge_row, drivable_row) in enumerate(zip(edge, drivable)):
        for n_x, (e, d) in enumerate(zip(edge_row, drivable_row)):
            px = pg.Surface((5, 5))
            d = 210 + 45 * d
            d = max(0, d)
            d = min(255, d)
            pg.draw.rect(px, (d, d, d), (0, 0, 5, 5))

            if e > 0:
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

full = np.concatenate((img_arr0, img_arr1, img_arr2, img_arr3, img_arr4, drivable_arr, blur_arr), 1)
full = full[1:]
full = full.astype("float32")
np.save("vstrainingdata/vs_train_clean.npy", full.astype("float32"))

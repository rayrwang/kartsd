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
data = data.astype("uint8")
for i in range(5):
    globals()[f"img_arr{i}"] = data[:, i*640*480*3:(i+1)*640*480*3]
    cv2.namedWindow(f"{i}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"{i}", 512, 384)
drivable_arr = data[:, 5*640*480*3:5*640*480*3 + 120*101]
edge_arr = data[:, 5*640*480*3 + 120*101:]

while True:
    # Handle key pressed
    for event in pg.event.get():
        if event.type == pg.KEYDOWN:
            keys = pg.key.get_pressed()
            if keys[pg.K_d]:
                img_num += 1
            elif keys[pg.K_s]:
                img_num += 1
            elif keys[pg.K_a]:
                img_num -= 1
            elif keys[pg.K_f]:
                edge_img_changed = True

    for i in range(5):
        globals()[f"img{i}"] = globals()[f"img_arr{i}"][img_num]
    drivable = drivable_arr[img_num]
    edge = edge_arr[img_num]

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
    # vs_blur = cv2.GaussianBlur(vs, (9, 9), 0)
    window.fill((255, 255, 255))
    for n_y, (edge_row, drivable_row) in enumerate(zip(edge, drivable)):
        for n_x, (v, d) in enumerate(zip(edge_row, drivable_row)):
            rect = pg.Surface((5, 5))
            if d != 0:
                pg.draw.rect(rect, (210 + 45*d, 210 + 45*d, 210 + 45*d), (0, 0, 5, 5))
            elif v != 0:
                v = 255 - v * 255
                v = max(0, v)
                v = min(255, v)
                pg.draw.rect(rect, (255, v, v), (0, 0, 5, 5))
            else:
                pg.draw.rect(rect, (210, 210, 210), (0, 0, 5, 5))
            window.blit(rect, (5 * n_x, 595 - (5 * n_y)))
    window.blit(car, car.get_rect(center=(252.5, 400 + (1.5/2 - 0.2)/0.25*5)))
    pg.display.update()

    if cv2.waitKey(1) == ord("f"):
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        break

"""
Construct vs labels from images
"""

import math

import numpy as np
import cv2
import pygame as pg


def project_center(*points):
    for point in points:
        att = math.pi / 180 * 0.19 * (point[1] + 0)  # + difference between index at top of (used portion) of image and horizon
        azi = math.pi / 180 * 0.19 * (point[0] - 127.5)  # - index at center of image
        try:
            dist = 0.7 / math.tan(att)  # dist : distance on ground from camera to px location
        except:
            return
        x = dist * math.sin(azi)
        y = dist * math.cos(azi)

        vs_x = round((x + 6.3125) / 0.125)  # Adjust x to move 0 from center to left
        vs_y = round((y - 0.0625) / 0.125)  # Pull y half a vs block down
        if 0 <= vs_x < 101 and 0 <= vs_y < 120:
            vs[vs_y, vs_x] = 1


def project_left(*points):
    for point in points:
        att = math.pi / 180 * 0.35 * (point[1] + 0)  # + difference between index at top of (used portion) of image and horizon
        azi = math.pi / 180 * 0.35 * (point[0] - 87.5)  # - index at center of image
        try:
            dist = 0.805 / math.tan(att)  # dist : distance on ground from camera to px location
        except:
            return
        x_old = dist * math.sin(azi)
        y_old = dist * math.cos(azi)

        # Correct for camera position and orientation relative to center camera
        x = x_old*math.cos(45 * math.pi/180) - y_old*math.sin(45 * math.pi/180) - 0.04
        y = x_old*math.sin(45 * math.pi/180) + y_old*math.cos(45 * math.pi/180) - 0.05

        vs_x = round((x + 6.3125) / 0.125)  # Adjust x to move 0 from center to left
        vs_y = round((y - 0.0625) / 0.125)  # Pull y half a vs block down
        if 0 <= vs_x < 101 and 0 <= vs_y < 120:
            vs[vs_y, vs_x] = 1


def project_right(*points):
    for point in points:
        att = math.pi / 180 * 0.35 * (point[1] + 0)  # + difference between index at top of (used portion) of image and horizon
        azi = math.pi / 180 * 0.35 * (point[0] - 87.5)  # - index at center of image
        try:
            dist = 0.805 / math.tan(att)  # dist : distance on ground from camera to px location
        except:
            return
        x_old = dist * math.sin(azi)
        y_old = dist * math.cos(azi)

        # Correct for camera position and orientation relative to center camera
        x = x_old*math.cos(-45 * math.pi/180) - y_old*math.sin(-45 * math.pi/180) + 0.04
        y = x_old*math.sin(-45 * math.pi/180) + y_old*math.cos(-45 * math.pi/180) - 0.05

        vs_x = round((x + 6.3125) / 0.125)  # Adjust x to move 0 from center to left
        vs_y = round((y - 0.0625) / 0.125)  # Pull y half a vs block down
        if 0 <= vs_x < 101 and 0 <= vs_y < 120:
            vs[vs_y, vs_x] = 1


def draw_img0(event, x, y, *args, **kwargs):
    global l_down
    global r_down
    global edge_img_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        l_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_down = True
    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if l_down:
            edges_img0[y, x] = 255


def draw_img1(event, x, y, *args, **kwargs):
    global l_down
    global r_down
    global edge_img_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        l_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_down = True
    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if l_down:
            edges_img1[y, x] = 255


def draw_img2(event, x, y, *args, **kwargs):
    global l_down
    global r_down
    global edge_img_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        l_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_down = True
    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if l_down:
            edges_img2[y, x] = 255


def draw_edge0(event, x, y, *args, **kwargs):
    global l_down
    global r_down
    global edge_img_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        l_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_down = True
    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if r_down:
            edges_img0[:, :] = 0
        elif l_down:
            edges_img0[y - 5: y + 6, x - 5: x + 6] = 0


def draw_edge1(event, x, y, *args, **kwargs):
    global l_down
    global r_down
    global edge_img_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        l_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_down = True
    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if r_down:
            edges_img1[:, :] = 0
        elif l_down:
            edges_img1[y - 5: y + 6, x - 5: x + 6] = 0


def draw_edge2(event, x, y, *args, **kwargs):
    global l_down
    global r_down
    global edge_img_changed
    if event == cv2.EVENT_LBUTTONDOWN:
        l_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_down = True
    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if r_down:
            edges_img2[:, :] = 0
        elif l_down:
            edges_img2[y - 5: y + 6, x - 5: x + 6] = 0


window = pg.display.set_mode((505, 655))
pg.init()

# Init video and vs displays
prev_img_num = -1

# Labeling progress
# big1 done
# big3 69
# big4 70
# big5 84
# big9 48
start_img_num = 48

img_num = 0

# Load data
vid_arr = np.loadtxt("rawvids/big9.csv", dtype="float16", delimiter=",", skiprows=start_img_num, max_rows=100)
vid_arr = vid_arr.astype("uint8")

cv2.namedWindow("1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1", 512, 384)
cv2.namedWindow("2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2", 352, 288)
cv2.namedWindow("3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3", 352, 288)
cv2.namedWindow("4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("4", 512, 384)
cv2.namedWindow("5", cv2.WINDOW_NORMAL)
cv2.resizeWindow("5", 352, 288)
cv2.namedWindow("6", cv2.WINDOW_NORMAL)
cv2.resizeWindow("6", 352, 288)

cv2.setMouseCallback("1", draw_img0)
cv2.setMouseCallback("2", draw_img1)
cv2.setMouseCallback("3", draw_img2)
cv2.setMouseCallback("4", draw_edge0)
cv2.setMouseCallback("5", draw_edge1)
cv2.setMouseCallback("6", draw_edge2)

l_down = False
r_down = False
edge_img_changed = False

car = pg.Surface((35, 55))

vs = np.zeros((120, 101))
with open(r"vstrainingdata/vs_train_rough.csv", "a") as file:
    while True:
        save = False

        # Handle key pressed
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                keys = pg.key.get_pressed()
                if keys[pg.K_d]:
                    img_num += 1
                elif keys[pg.K_s]:
                    img_num += 1
                    save = True
                elif keys[pg.K_a]:
                    img_num -= 1
                elif keys[pg.K_f]:
                    edge_img_changed = True

        # Handle mouse click to flip pixels
        buttons = pg.mouse.get_pressed(num_buttons=3)
        x_coord, y_coord = pg.mouse.get_pos()

        x = round((x_coord - 2.5) / 5)
        y = round((y_coord - 2.5) / 5)

        if buttons[0]:
            vs[119 - y - 1: 119 - y + 2, x - 1: x + 2] = 1
        elif buttons[1]:
            vs[119 - y - 3: 119 - y + 4, x - 3: x + 4] = 0
        elif buttons[2]:
            vs[119 - y, x] = 1

        if img_num != prev_img_num:
            # Write previous completed image and vs to file
            if save:
                img0 = img0.reshape(147456)
                img1 = img1.reshape(76032)
                img2 = img2.reshape(76032)
                vs = vs.reshape(12120)
                full = np.concatenate((img0, img1, img2, vs))
                np.savetxt(file, [full], fmt="%.0f", delimiter=",")

            # Get new image
            images = vid_arr[img_num]
            img0 = images[:147456]
            img0 = img0.reshape(192, 256, 3)
            img1 = images[147456:223488]
            img1 = img1.reshape(144, 176, 3)
            img2 = images[223488:]
            img2 = img2.reshape(144, 176, 3)
            cv2.imshow("1", img0)
            cv2.imshow("2", img1)
            cv2.imshow("3", img2)

            # Compute edges
            edges_img0 = cv2.Canny(img0, 150000, 25000, apertureSize=7, L2gradient=True)
            cv2.imshow("4", edges_img0)
            edges_img1 = cv2.Canny(img1, 150000, 25000, apertureSize=7, L2gradient=True)
            cv2.imshow("5", edges_img1)
            edges_img2 = cv2.Canny(img2, 150000, 25000, apertureSize=7, L2gradient=True)
            cv2.imshow("6", edges_img2)
            print(start_img_num + img_num, start_img_num + vid_arr.shape[0])

        cv2.imshow("4", edges_img0)
        cv2.imshow("5", edges_img1)
        cv2.imshow("6", edges_img2)
        if img_num != prev_img_num or edge_img_changed:
            edge_img_changed = False
            prev_img_num = img_num

            # Compute physical x and y for pixels in edges
            vs = np.zeros((120, 101))  # rows, columns
            for px_y, row in enumerate(edges_img0[34:]):  # px_y : pixels below horizon
                for px_x, pos in enumerate(row):  # px_x : pixels from left (48 to center)
                    if pos == 255:
                        project_center((px_x, px_y),
                                       (px_x + 0.1, px_y), (px_x - 0.1, px_y),
                                       (px_x + 0.2, px_y), (px_x - 0.2, px_y),
                                       (px_x + 0.3, px_y), (px_x - 0.3, px_y),
                                       (px_x + 0.4, px_y), (px_x - 0.4, px_y),
                                       (px_x + 0.5, px_y), (px_x - 0.5, px_y),
                                       (px_x, px_y + 0.05), (px_x, px_y - 0.05),
                                       (px_x, px_y + 0.1), (px_x, px_y - 0.1),
                                       (px_x, px_y + 0.15), (px_x, px_y - 0.15),
                                       (px_x, px_y + 0.2), (px_x, px_y - 0.2),
                                       (px_x, px_y + 0.25), (px_x, px_y - 0.25),
                                       (px_x, px_y + 0.3), (px_x, px_y - 0.3),
                                       (px_x, px_y + 0.35), (px_x, px_y - 0.35),
                                       (px_x, px_y + 0.4), (px_x, px_y - 0.4),
                                       (px_x, px_y + 0.45), (px_x, px_y - 0.45),
                                       (px_x, px_y + 0.5), (px_x, px_y - 0.5),
                                       (px_x + 0.1, px_y + 0.1), (px_x - 0.1, px_y + 0.1),
                                       (px_x + 0.2, px_y + 0.2), (px_x - 0.2, px_y + 0.2),
                                       (px_x + 0.3, px_y + 0.3), (px_x - 0.3, px_y + 0.3),
                                       (px_x + 0.4, px_y + 0.4), (px_x - 0.4, px_y + 0.4),
                                       (px_x + 0.5, px_y + 0.5), (px_x - 0.5, px_y + 0.5),
                                       (px_x - 0.1, px_y - 0.1), (px_x + 0.1, px_y - 0.1),
                                       (px_x - 0.2, px_y - 0.2), (px_x + 0.2, px_y - 0.2),
                                       (px_x - 0.3, px_y - 0.3), (px_x + 0.3, px_y - 0.3),
                                       (px_x - 0.4, px_y - 0.4), (px_x + 0.4, px_y - 0.4),
                                       (px_x - 0.5, px_y - 0.5), (px_x + 0.5, px_y - 0.5))
            for px_y, row in enumerate(edges_img1[85:]):  # px_y : pixels below horizon
                for px_x, pos in enumerate(row):  # px_x : pixels from left (48 to center)
                    if pos == 255:
                        project_right((px_x, px_y),
                                      (px_x + 0.1, px_y), (px_x - 0.1, px_y),
                                      (px_x + 0.2, px_y), (px_x - 0.2, px_y),
                                      (px_x + 0.3, px_y), (px_x - 0.3, px_y),
                                      (px_x + 0.4, px_y), (px_x - 0.4, px_y),
                                      (px_x + 0.5, px_y), (px_x - 0.5, px_y),
                                      (px_x, px_y + 0.05), (px_x, px_y - 0.05),
                                      (px_x, px_y + 0.1), (px_x, px_y - 0.1),
                                      (px_x, px_y + 0.15), (px_x, px_y - 0.15),
                                      (px_x, px_y + 0.2), (px_x, px_y - 0.2),
                                      (px_x, px_y + 0.25), (px_x, px_y - 0.25),
                                      (px_x, px_y + 0.3), (px_x, px_y - 0.3),
                                      (px_x, px_y + 0.35), (px_x, px_y - 0.35),
                                      (px_x, px_y + 0.4), (px_x, px_y - 0.4),
                                      (px_x, px_y + 0.45), (px_x, px_y - 0.45),
                                      (px_x, px_y + 0.5), (px_x, px_y - 0.5),
                                      (px_x + 0.1, px_y + 0.1), (px_x - 0.1, px_y + 0.1),
                                      (px_x + 0.2, px_y + 0.2), (px_x - 0.2, px_y + 0.2),
                                      (px_x + 0.3, px_y + 0.3), (px_x - 0.3, px_y + 0.3),
                                      (px_x + 0.4, px_y + 0.4), (px_x - 0.4, px_y + 0.4),
                                      (px_x + 0.5, px_y + 0.5), (px_x - 0.5, px_y + 0.5),
                                      (px_x - 0.1, px_y - 0.1), (px_x + 0.1, px_y - 0.1),
                                      (px_x - 0.2, px_y - 0.2), (px_x + 0.2, px_y - 0.2),
                                      (px_x - 0.3, px_y - 0.3), (px_x + 0.3, px_y - 0.3),
                                      (px_x - 0.4, px_y - 0.4), (px_x + 0.4, px_y - 0.4),
                                      (px_x - 0.5, px_y - 0.5), (px_x + 0.5, px_y - 0.5))
            for px_y, row in enumerate(edges_img2[85:]):  # px_y : pixels below horizon
                for px_x, pos in enumerate(row):  # px_x : pixels from left (48 to center)
                    if pos == 255:
                        project_left((px_x, px_y),
                                     (px_x + 0.1, px_y), (px_x - 0.1, px_y),
                                     (px_x + 0.2, px_y), (px_x - 0.2, px_y),
                                     (px_x + 0.3, px_y), (px_x - 0.3, px_y),
                                     (px_x + 0.4, px_y), (px_x - 0.4, px_y),
                                     (px_x + 0.5, px_y), (px_x - 0.5, px_y),
                                     (px_x, px_y + 0.05), (px_x, px_y - 0.05),
                                     (px_x, px_y + 0.1), (px_x, px_y - 0.1),
                                     (px_x, px_y + 0.15), (px_x, px_y - 0.15),
                                     (px_x, px_y + 0.2), (px_x, px_y - 0.2),
                                     (px_x, px_y + 0.25), (px_x, px_y - 0.25),
                                     (px_x, px_y + 0.3), (px_x, px_y - 0.3),
                                     (px_x, px_y + 0.35), (px_x, px_y - 0.35),
                                     (px_x, px_y + 0.4), (px_x, px_y - 0.4),
                                     (px_x, px_y + 0.45), (px_x, px_y - 0.45),
                                     (px_x, px_y + 0.5), (px_x, px_y - 0.5),
                                     (px_x + 0.1, px_y + 0.1), (px_x - 0.1, px_y + 0.1),
                                     (px_x + 0.2, px_y + 0.2), (px_x - 0.2, px_y + 0.2),
                                     (px_x + 0.3, px_y + 0.3), (px_x - 0.3, px_y + 0.3),
                                     (px_x + 0.4, px_y + 0.4), (px_x - 0.4, px_y + 0.4),
                                     (px_x + 0.5, px_y + 0.5), (px_x - 0.5, px_y + 0.5),
                                     (px_x - 0.1, px_y - 0.1), (px_x + 0.1, px_y - 0.1),
                                     (px_x - 0.2, px_y - 0.2), (px_x + 0.2, px_y - 0.2),
                                     (px_x - 0.3, px_y - 0.3), (px_x + 0.3, px_y - 0.3),
                                     (px_x - 0.4, px_y - 0.4), (px_x + 0.4, px_y - 0.4),
                                     (px_x - 0.5, px_y - 0.5), (px_x + 0.5, px_y - 0.5))

        # Display vs
        # vs_blur = cv2.GaussianBlur(vs, (9, 9), 0)
        window.fill((255, 255, 255))
        window.blit(car, (235, 600))
        for n_y, y_row in enumerate(vs):
            for n_x, x in enumerate(y_row):
                x = 255 - x * 255
                x = max(0, x)
                x = min(255, x)

                rect = pg.Surface((5, 5))
                pg.draw.rect(rect, (255, x, x), (0, 0, 5, 5))
                window.blit(rect, (5 * n_x, 595 - (5 * n_y)))
        pg.display.update()

        if cv2.waitKey(1) == ord("f"):
            cv2.destroyAllWindows()
            cv2.VideoCapture(0).release()
            break

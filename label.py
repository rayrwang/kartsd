"""
Construct labels from images
"""

import math

import numpy as np
import cv2
import pygame as pg


def get_rot_mat(azi):
    """Rotate world according to player camera orientation"""
    mat = np.array([
        [math.cos(azi*math.pi/180), -math.sin(azi*math.pi/180)],
        [math.sin(azi*math.pi/180), math.cos(azi*math.pi/180)]
    ])
    return mat


def project(px_x, px_y, b0, g0=0, ch=0.785, fx=54, fy=41, w=640, h=480):
    """pixel position, camera angle (up-down, left-right; deg), camera height, field of view (deg), # pixels"""
    b0 = b0 * math.pi / 180
    g0 = g0 * math.pi / 180
    fx = fx * math.pi / 180
    fy = fy * math.pi / 180
    gamma = math.atan(math.tan(fx/2)*(px_x-(w/2)+0.5) / (w/2)) + g0  # Left-right angle
    beta = -math.atan(math.tan(fy/2)*(px_y-(h/2)+0.5) / (h/2)) + b0  # up-down angle

    y = ch / math.tan(-beta)
    x = math.sqrt(ch**2 + y**2) * math.tan(gamma)

    return x, y


def get_draw_on_img(img, img_type: str):
    def draw_on_img(event, x, y,  *args, **kwargs):
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
            if img_type == "img":
                if l_down:
                    globals()[img][y, x] = 255
            elif img_type == "edge":
                if l_down:
                    globals()[img][y - 5: y + 6, x - 5: x + 6] = 0
    return draw_on_img


window = pg.display.set_mode((505, 600))
pg.init()
car = pg.Surface((0.9/0.25*5, 1.5/0.25*5))
car.fill((0, 0, 0))

# Init video displays
prev_img_num = -1

# Labeling progress
# 0: 10 +20 2610
# 2: 10 +20 1210 2370
session_n = 2
img_num = 1210

# Camera calibrations
cal = {
    "beta0": -13,
    "beta1": -14,
    "beta2": -17,
    "beta3": -15,
    "beta4": -12,
    "gamma0": 0,
    "gamma1": 0,
    "gamma2": 0,
    "gamma3": 0,
    "gamma4": -10,
    "azi0": 95,
    "azi1": -85,
    "azi2": 0,
    "azi3": -45,
    "azi4": 37,
}

# Read from videos
for i in range(10):
    globals()[f"cap{i}"] = cv2.VideoCapture(f"rawvids/{session_n}_{i}.avi")
    cv2.namedWindow(f"{i}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"{i}", 512, 384)

# Set drawing on img events
for i in range(5):
    cv2.setMouseCallback(f"{i}", get_draw_on_img(f"edge_img{i}", "img"))
    cv2.setMouseCallback(f"{i+5}", get_draw_on_img(f"edge_img{i}", "edge"))

l_down = False
r_down = False
edge_img_changed = False

# Read old contents of file
try:
    data = np.load("vstrainingdata/vs_train_rough.npy")
    data = data.astype("uint8")
except FileNotFoundError:
    data = np.zeros((1, 5*640*480*3 + 2*120*101), dtype="uint8")
    open("vstrainingdata/vs_train_rough.npy", "x")
while True:
    print(img_num)
    save = False

    # Handle key pressed
    for event in pg.event.get():
        if event.type == pg.KEYDOWN:
            keys = pg.key.get_pressed()
            if keys[pg.K_d]:
                img_num += 20
            elif keys[pg.K_s]:
                img_num += 20
                save = True
            elif keys[pg.K_a]:
                img_num -= 20
            elif keys[pg.K_f]:
                edge_img_changed = True

    # Handle mouse click to flip pixels
    buttons = pg.mouse.get_pressed(num_buttons=3)
    x_coord, y_coord = pg.mouse.get_pos()

    x = round((x_coord - 2.5) / 5)
    y = round((y_coord - 2.5) / 5)

    if buttons[0]:  # Draw drivable area
        for i in range(119 - y - 2, 119 - y + 3):
            for j in range(x - 2, x + 3):
                if 0 <= i < 120 and 0 <= j < 101:
                    if edge[i, j] == 0:
                        drivable[i, j] = 1
    elif buttons[1]:  # Erase
        edge[119 - y - 1: 119 - y + 2, x - 1: x + 2] = 0
        drivable[119 - y - 1: 119 - y + 2, x - 1: x + 2] = 0
    elif buttons[2]:  # Draw road edges
        if drivable[119 - y, x] != 1:
            edge[119 - y, x] = 1

    if img_num != prev_img_num:
        # Write previous completed image and vs to file
        if save:
            img0 = img0.reshape(1, 640*480*3)
            img1 = img1.reshape(1, 640*480*3)
            img2 = img2.reshape(1, 640*480*3)
            img3 = img3.reshape(1, 640*480*3)
            img4 = img4.reshape(1, 640*480*3)
            drivable = drivable.reshape(1, 12120)
            edge = edge.reshape(1, 12120)
            full = np.concatenate((img0, img1, img2, img3, img4, drivable, edge), 1)
            full = full.astype("uint8")
            data = np.concatenate((data, full), 0)
            np.save("vstrainingdata/vs_train_rough.npy", data.astype("uint8"))

        # Get new image
        for i in range(5):
            globals()[f"cap{i}"].set(cv2.CAP_PROP_POS_FRAMES, img_num)
            _, globals()[f"img{i}"] = globals()[f"cap{i}"].read()
            cv2.imshow(f"{i}", globals()[f"img{i}"])

        # Init edges
        for i in range(5):
            globals()[F"edge_img{i}"] = np.zeros((480, 640), dtype="uint8")

    for i in range(5):
        cv2.imshow(f"{i+5}", globals()[f"edge_img{i}"])
    if img_num != prev_img_num or edge_img_changed:
        edge_img_changed = False
        prev_img_num = img_num

        # Compute physical x and y for pixels in edges
        edge = np.zeros((120, 101), dtype="uint8")  # rows, columns
        drivable = np.zeros((120, 101), dtype="uint8")
        for i in range(5):
            for px_y, row in enumerate(globals()[f"edge_img{i}"]):
                for px_x, pos in enumerate(row):
                    if pos == 255:
                        x, y = project(px_x, px_y, cal[f"beta{i}"], cal[f"gamma{i}"])
                        rotated = get_rot_mat(cal[f"azi{i}"]) @ np.array([[x], [y]])
                        # todo RELATVE POSITON
                        i_x = round(50 + float(rotated[0])/0.25)  # Grid size of 0.25m
                        i_y = round(float(rotated[1])/0.25 + 40)
                        if 0 <= i_x < 101 and 0 <= i_y < 120:
                            edge[i_y, i_x] = 1

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

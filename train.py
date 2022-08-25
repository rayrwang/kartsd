import numpy as np
import cv2 as cv

vid_arr = np.loadtxt("train.csv", dtype="int16", delimiter=",")
steer_arr = vid_arr[:, 0]
vid_arr = np.delete(vid_arr, 0, axis=1)
vid_arr = vid_arr.astype("uint8")

steer = steer_arr[0]
img = vid_arr[0]
img = img.reshape(96, 128, 3)

cv.imshow("", img)

while True:
    if cv.waitKey(1) == ord("f"):
        cv.destroyAllWindows() 
        cv.VideoCapture(0).release()
        break

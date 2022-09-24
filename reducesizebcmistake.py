import numpy as np

arr = np.loadtxt("vstrainingdata/noshadows_clean.csv", delimiter=",")
vid_arr = arr[:, :36864]
vs_arr = arr[:, 36864:]
vs_arr = vs_arr.reshape(-1, 70, 81)
vs_arr = np.delete(vs_arr, np.s_[61:81], 2)
with open("vstrainingdata/noshadows_clean_fixed.csv", "a") as file:
    vid_arr = vid_arr.reshape(-1, 36864)
    vs_arr = vs_arr.reshape(-1, 4270)
    full = np.concatenate((vid_arr, vs_arr), axis=1)
    np.savetxt(file, full, fmt="%.0f", delimiter=",")

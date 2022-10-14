import numpy as np

arr = np.loadtxt("rawvids/big6.csv", dtype="float16", delimiter=",", max_rows=None)
np.save("rawvids/big6.npy", arr)

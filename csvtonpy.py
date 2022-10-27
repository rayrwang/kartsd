"""
Convert .csv file to .npy

Run with filename without extension
"""

import sys

import numpy as np

arr = np.loadtxt(f"{sys.argv[1]}.csv", dtype="float16", delimiter=",")
np.save(f"{sys.argv[1]}.npy", arr)

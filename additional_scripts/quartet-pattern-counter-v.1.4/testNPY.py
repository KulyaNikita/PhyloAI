#!/usr/bin/env python3

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(precision=10)
np.set_printoptions(formatter={'float': '{: 0.16f}'.format})

argc = len(sys.argv)
if (argc != 2):
    print("Usage: testNPY.py npy-file\n")
    sys.exit(0);

filename = sys.argv[1]
v = np.load(filename)

print("Data type of object:       ", type(v))
print("Shape of numpy data:       ", v.shape)
print("Data type of numpy data:   ", v.dtype)

print(v)


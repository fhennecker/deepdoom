"""
Demo on how to open dumps outputed by doomy.py
"""

import numpy as np

positions = np.fromfile("ennemies.dump")
N = positions.shape[0]//2

print(N)

Y = positions.reshape(2, N).mean(axis=1)
X = np.fromfile("screens.dump", dtype=np.uint8).reshape(N, 125, 200, 3)


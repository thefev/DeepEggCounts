import cv2
import numpy as np
import data

X, Y, nsi = data.load_data()
i = 0
while i < 10:
    r = np.random.randint(X.shape[0])
    if np.sum(Y[r]) > 0:
        data.heat_map(X[r], Y[r])
        i += 1

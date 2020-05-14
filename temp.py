import cv2
import numpy as np
import matplotlib.pyplot as plt
import data

# X, Y, nsi = data.process_load_data_dir(directory='./egg_photos/test/')

# yolo = np.loadtxt('./test.txt')
# img = cv2.imread('./test.jpg')
# coor = data.yolo_to_xy(yolo, img.shape)
# dmap = data.generate_density_map(img, coor)
# data.heat_map(img, dmap)

X, Y, nsi = data.process_load_data_dir()
data.save_data_npy(X, Y, nsi)

# X, Y, nsi = data.load_data_npy()
# i = 0
# while i < 10:
#     r = np.random.randint(X.shape[0])
#     if np.sum(Y[r]) > 0:
#         data.heat_map(X[r], Y[r])
#         i += 1



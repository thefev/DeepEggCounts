import cv2
import numpy as np
import matplotlib.pyplot as plt
import data
# import keras
# from keras.models import *
# from keras.layers import *

# yolo = np.loadtxt('./test.txt')
# img = cv2.imread('./test.jpg')
# coor = data.yolo_to_xy(yolo, img.shape)
# dmap = data.generate_density_map(img, coor)
# data.heat_map(img, dmap)

# X, Y, nsi = data.process_load_data_dir()
# data.save_data_npy(X, Y, nsi)

X, Y, nsi = data.load_data_npy()
# data.sample_xy(20, X, Y)
data.filter_image(X[1780])

# X, Y, nsi = data.process_load_data_dir(e2e=True)
# data.save_data_npy(X, Y, nsi, e2e=True)
#
# X, Y, nsi = data.load_data_npy(e2e=True)
# for i in range(10):
#     r = np.random.randint(X.shape[0])
#     plt.figure()
#     plt.imshow(X[r])
#     print('Img:\t' + str(i+1) + '\tEggs:\t' + str(Y[r]))


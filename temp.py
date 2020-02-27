import keras
from keras.layers import Input, Conv2D, MaxPool2D, Concatenate
from keras.models import Model, Sequential
import cv2
import numpy as np

X = cv2.imread('egg_photos/DSC_2378_480p.JPG')
Y = cv2.imread('egg_photos/DSC_2378_480p_dmap.JPG')

model = Sequential()
model.add(Conv2D(64, 3, input_shape=X.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))


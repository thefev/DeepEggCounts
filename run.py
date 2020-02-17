# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:11:37 2020

@author: Kevin
"""

import numpy as np
from tensorflow import keras as K
import cv2

train_list = np.loadtxt('Egg_Photos/train.txt', dtype=str)
test_list = np.loadtxt('Egg_photos/test.txt', dtype=str)

train_size = len(train_list)
test_size = len(test_list)

img = cv2.imread(train_list[0])

train_X = np.zeros(train_size, img.shape)



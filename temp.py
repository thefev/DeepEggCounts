import cv2
import numpy as np
import data
img=cv2.imread("egg_photos/originals/DSC_2396.JPG")
imgr=data.split_image(img, "360p")
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:11:52 2020

@author: Kevin

Prompts user for image, then asks user to crop the section in which eggs exist.
Scales the cropped image to 480p format for further processing.
"""
import sys
import cv2
import numpy as np
from win32api import GetSystemMetrics
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import gaussian_filter


def mouse_crop(event, x, y, flags, param):
    """
    Grabs mouse coordinates from image display frame at commencement of cropping and end. Feeds to global coordinates
    for further use.
    """
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, cropped

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished

        ref_point = [(x_start, y_start), (x_end, y_end)]

        if len(ref_point) == 2:  # when two points were found, display cropped image
            cropped = True


def image_rescale(img, final_dim):
    """
    Takes in image and desired final dimensions. Returns rescaled image and the
    rescale factor for further calculations.

    Args:
        img:        input image to be rescaled
        final_dim:  final dimensions of image (width, height)

    Returns:
        img_rescaled
        rescale_factor:     rescale factor (r_f_x, r_f_y)
    """
    rescale_factor = (final_dim[0] / img.shape[1], final_dim[1] / img.shape[0])
    img_rescaled = cv2.resize(img, final_dim)
    return img_rescaled, rescale_factor


# ---------------------------------------------------------------------------------------------------------------------

cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
p480 = (640, 480)

root = tk.Tk()
root.withdraw()
valid_jpg_file = False
while not valid_jpg_file:
    file_path = filedialog.askopenfile()
    img_path = file_path.name
    if img_path.__contains__(".jpg") or img_path.__contains__(".JPG"):
        valid_jpg_file = True

data_path = img_path.rstrip("jpgJPG").rstrip('.') + ".txt"

image = cv2.imread(img_path)

# defining dimension in pixels based on current display size during cropping
#   and 480p for image processing - format: (width, height)
p_image = (image.shape[1], image.shape[0])
p_current_display = (GetSystemMetrics(0), GetSystemMetrics(1))

# ensuring linear rescaling that fits user's window
rf_crop_display = min(p_current_display[1] / image.shape[0], p_current_display[0] / image.shape[1])
p_crop_display = (int(rf_crop_display * image.shape[1]), int(rf_crop_display * image.shape[0]))
image_rescaled, rf_rescaled = image_rescale(image, p_crop_display)  # rescale to fit screen

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
while not cropped:
    image_copy = image_rescaled.copy()
    if not cropping:
        cv2.imshow("image", image_rescaled)
    elif cropping:
        cv2.rectangle(image_copy, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", image_copy)
    cv2.waitKey(1)
# close all open windows
cv2.destroyAllWindows()

# Grabs coordinates of cropped image corners, creates cropped image, and creates shifts coordinates to cropped image
# space.
cropped_corners = [(x_start, y_start), (x_end, y_end)]
image_cropped = image_rescaled[cropped_corners[0][1]:cropped_corners[1][1], cropped_corners[0][0]:cropped_corners[1][0]]

# Final downscaling to 480p format of both cropped image and coordinates for U-Net to process more easily.
image_480p, rf_480p = image_rescale(image_cropped, p480)

save_path = img_path.rstrip("jpgJPG").rstrip('.')
img_path_480p = save_path + "_480p.JPG"
cv2.imwrite(img_path_480p, image_480p)

sys.exit()

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


def yolo_to_xy(yolo_txt, img_dim):
    """
    Takes in coordinates in YOLO format. Converts to raw pixel # format

    Arg:
        yolo_txt:   data in YOLO format (class, x, y, w, h) format
                    *** x, y, w, h are all expressed as percentages
        img_dim:    img shape (y_len, x_len)

    Returns:
        coor_format:       np array of coordinates of eggs (x, y)
    """
    coor_format = yolo_txt[:, 1:3]  # extract (x, y) coordinate info
    coor_format = coor_format * (-1) + 1  # 180 degree rotation
    coor_format[:, 0] *= img_dim[1]  # transforming percentages to coordinates
    coor_format[:, 1] *= img_dim[0]
    coor_format = coor_format.astype(int)  # round to int
    return coor_format


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


def coor_rescale(original_coordinates, rescale_factor):
    """
    Rescales coordinates based of rescaling factor.

    Args:
        original_coordinates:   coordinates of eggs in (x, y) format
        rescale_factor:         rescale factor (r_f_x, r_f_y)

    Return:
        coordinates_rescaled:  coordinates in rescaled image space
    """
    coordinates_rescaled = np.zeros_like(original_coordinates)
    coordinates_rescaled[:, 0] = original_coordinates[:, 0] * rescale_factor[0]
    coordinates_rescaled[:, 1] = original_coordinates[:, 1] * rescale_factor[1]
    coordinates_rescaled = coordinates_rescaled.astype(int)
    return coordinates_rescaled


def coor_crop_shift(original_coordinates, cropped_image_corners):
    """
    Converts coordinates in original image space to coordinates in cropped
    image space.

    Args:
        original_coordinates:   egg coordinates of original image (x, y)
        cropped_image_corners:  cropped image coordinates [(x_start, y_start), (x_end, y_end)]

    Returns:
        cropped_coordinates:    coordinates in cropped image space (x, y)
    """
    cropped_coordinates = np.zeros_like(original_coordinates)
    min_x = min(cropped_image_corners[0][0], cropped_image_corners[1][0])
    min_y = min(cropped_image_corners[0][1], cropped_image_corners[1][1])
    cropped_coordinates[:, 0] = original_coordinates[:, 0] - min_x
    cropped_coordinates[:, 1] = original_coordinates[:, 1] - min_y
    return cropped_coordinates


def draw_points_on_image(image, coordinates):
    """
    Sanity check: draws dot on image at location of egg
    """
    paint = [0, 255, 0]
    for i in range(coordinates.shape[0]):
        image[coordinates[i, 1], coordinates[i, 0]] = paint
        image[coordinates[i, 1] + 1, coordinates[i, 0]] = paint
        image[coordinates[i, 1] - 1, coordinates[i, 0]] = paint
        image[coordinates[i, 1], coordinates[i, 0] + 1] = paint
        image[coordinates[i, 1], coordinates[i, 0] - 1] = paint

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None


def generate_density_map(img_path, coor_path):
    """
    Created on Mon Feb  3 13:30:13 2020

    @author: Kevin

    Generate a density map based on objects positions. Saves as file.

    Args:
        img_path:   location of image
        coor_path:  location of txt file containing coordinate info in (x, y) format
    Returns:
        None
    """

    assert img_path.strip('.JPG') == coor_path.strip('.txt'), 'img_path and coor_path files do not match'

    img = cv2.imread(img_path)
    coor = np.loadtxt(coor_path)
    coor = coor.astype(int)  # round to int
    # initialise density map of size image
    density_map = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # applying heat of 100 at location of eggs
    for i in range(coor.shape[0]):
        density_map[coor[i, 1], coor[i, 0]] += 100

    # apply Gaussian kernel to density map
    density_map = gaussian_filter(density_map, sigma=(1, 1), order=0)

    # save density map
    cv2.imwrite(img_path.strip('.JPG') + '_dmap.JPG', density_map)

    return None

# ---------------------------------------------------------------------------------------------------------------------


cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

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
data = np.loadtxt(data_path)
coor = yolo_to_xy(data, image.shape)

# defining dimension in pixels based on current display size during cropping
#   and 480p for image processing - format: (width, height)
p_image = (image.shape[1], image.shape[0])
p480 = (640, 480)
p_current_display = (GetSystemMetrics(0), GetSystemMetrics(1))

# ensuring linear rescaling that fits user's window
rf_crop_display = min(p_current_display[1] / image.shape[0], p_current_display[0] / image.shape[1])
p_crop_display = (int(rf_crop_display * image.shape[1]), int(rf_crop_display * image.shape[0]))
image_rescaled, rf_rescaled = image_rescale(image, p_crop_display)  # rescale to fit screen
coor_rescaled = coor_rescale(coor, rf_rescaled)  # updating coordinates of rescaled image

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
coor_cropped = coor_crop_shift(coor_rescaled, cropped_corners)

# sanity check: ensuring all coordinates are within bounds of cropped image space
assert min(coor_cropped[:, 0]) >= 0, "min(x) out of image"
assert max(coor_cropped[:, 0]) <= abs(x_end - x_start), "max(x) out of image"
assert min(coor_cropped[:, 1]) >= 0, "min(y) out of image"
assert max(coor_cropped[:, 1]) <= abs(y_end - y_start), "max(y) out of image"

# Final downscaling to 480p format of both cropped image and coordinates for U-Net to process more easily.
image_480p, rf_480p = image_rescale(image_cropped, p480)
coor_480p = coor_rescale(coor_cropped, rf_480p)

# Visual check that coordinates are at correct locations
draw_points_on_image(image_480p.copy(), coor_480p)

save_path = img_path.rstrip("jpgJPG").rstrip('.')
img_path_480p = save_path + "_480p.JPG"
coor_path_480p = save_path + "_480p.txt"
cv2.imwrite(img_path_480p, image_480p)
np.savetxt(coor_path_480p, coor_480p)
generate_density_map(img_path_480p, coor_path_480p)

sys.exit()

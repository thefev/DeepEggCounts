# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:11:52 2020

@author: Kevin

Prompts user for image, then asks user to crop the section in which eggs exist.
Scales the cropped image to 480p format for further processing.

Future modifications: added further resolutions for selection, e.g. 360p, 720p, 1080p, etc.
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

    Args:
        image:
        coordinates:

    Returns:
        None
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
    Generate a density map based on objects positions. Saves as file.

    Args:
        img_path (str):     location of image
        coor_path (str):    location of txt file containing coordinate info in (x, y) format

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


def average_egg_coefficient(dmap_path):
    """
    Sanity check. Takes in density map path. Infers coordinate data from that.
        Determines the amount of 'heat' per egg in the

    Arg:
        dmap_path (str):    location of density map (.JPG)

    Returns:
        aec (float):        average egg coefficient - the amount of 'heat' per egg
    """

    dmap = cv2.imread(dmap_path)
    data = np.loadtxt(dmap_path.strip('_dmap.JPG') + '.txt')
    eggs = data.shape[0]  # get number of eggs
    heat_sum = np.sum(dmap)  # integrating over training output (y)
    aec = heat_sum / eggs  # determining amount of 'heat' per egg

    print('Number of eggs:\t' + str(eggs))
    print('Average egg coefficient:\t' + str(aec))

    return aec


def cropped_box_shift(corners, desired_aspect_ratio):
    """
    Adjusts corner coordinates of a box to match desired aspect ratio so as not to distort the image. This is achieved
    by increasing either the height or width of the box coordinates to capture more of the image.

    Args:
        corners (list of (tuples)):     input box coordinates [(x_start, y_start), (x_end, y_end)]
        desired_aspect_ratio (float):   aspect ratio (width / height) of output box

    Returns:
        adjusted_corners (list of (tuples)):    adjusted [(x_start, y_start), (x_end, y_end)]
    """
    [(x1, y1), (x2, y2)] = corners
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    ar = w / h
    if ar >= desired_aspect_ratio:
        # input box too wide - increase height
        new_h = w / desired_aspect_ratio
        y1 = int(yc - (new_h / 2))
        y2 = int(yc + (new_h / 2))
    elif ar < desired_aspect_ratio:
        # input box too tall - increase width
        new_w = h * desired_aspect_ratio
        x1 = int(xc - (new_w / 2))
        x2 = int(xc + (new_w / 2))

    adjusted_corners = [(x1, y1), (x2, y2)]
    return adjusted_corners


def image_crop_and_scale(resolution="480p"):
    """
    Prompts user to crop image in the section in which eggs exist. Unless specified, crops to 640x480p.

    Args:
        image:
        resolution:

    Returns:

    """


def image_crop_scale_dmap(resolution="480p"):
    """

    Args:
        resolution:

    Returns:

    """
    cropping = False
    cropped = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    final_res = "480p"
    res = {"240p": (320, 240),
           "360p": (480, 360),
           "480p": (640, 480),
           "720p": (960, 720),
           "1080p": (1440, 1080)}
    resolution = res[final_res]
    aspect_ratio = resolution[0] / resolution[1]

    img_path = get_image_path()
    data_path = img_path.rstrip("jpgJPG").rstrip('.') + ".txt"

    image = cv2.imread(img_path)
    data = np.loadtxt(data_path)
    coor = yolo_to_xy(data, image.shape)

    # defining dimension in pixels based on current display size during cropping
    #   and 480p for image processing - format: (width, height)
    p_image = (image.shape[1], image.shape[0])
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
    cropped_corners = cropped_box_shift(cropped_corners, aspect_ratio)
    [(x_start, y_start), (x_end, y_end)] = cropped_corners
    image_cropped = image_rescaled[cropped_corners[0][1]:cropped_corners[1][1],
                    cropped_corners[0][0]:cropped_corners[1][0]]
    coor_cropped = coor_crop_shift(coor_rescaled, cropped_corners)

    # sanity check: ensuring all coordinates are within bounds of cropped image space
    assert min(coor_cropped[:, 0]) >= 0, "min(x) out of image"
    assert max(coor_cropped[:, 0]) <= abs(x_end - x_start), "max(x) out of image"
    assert min(coor_cropped[:, 1]) >= 0, "min(y) out of image"
    assert max(coor_cropped[:, 1]) <= abs(y_end - y_start), "max(y) out of image"

    # Final downscaling to 480p format of both cropped image and coordinates for U-Net to process more easily.
    image_final_res, rf_final_res = image_rescale(image_cropped, resolution)
    coor_final_res = coor_rescale(coor_cropped, rf_final_res)

    # Visual check that coordinates are at correct locations
    draw_points_on_image(image_final_res.copy(), coor_final_res)

    save_path = img_path.rstrip("jpgJPG").rstrip('.')
    img_path_save = save_path + "_" + final_res + ".JPG"
    coor_path_save = save_path + "_" + final_res + ".txt"
    cv2.imwrite(img_path_save, image_final_res)
    np.savetxt(coor_path_save, coor_final_res)
    generate_density_map(img_path_save, coor_path_save)


def get_image_path():
    """
    Prompts user to navigate to and select .jpg file to be read.
    Returns:
        image_path (str):   path of valid image location
    """
    root = tk.Tk()
    root.withdraw()
    valid_jpg_file = False
    while not valid_jpg_file:
        file_path = filedialog.askopenfile()
        image_path = file_path.name
        if image_path.__contains__(".jpg") or image_path.__contains__(".JPG"):
            valid_jpg_file = True

    return image_path


def load_data():
    """
    Loads in images and density map for training and validation

    Returns:    train and val X, Y set

    """
    files = ["Egg_photos / DSC_2378.JPG",
                "Egg_photos / DSC_2379.JPG",
                "Egg_photos / DSC_2380.JPG",
                "Egg_photos / DSC_2381.JPG",
                "Egg_photos / DSC_2382.JPG",
                "Egg_photos / DSC_2391.JPG",
                "Egg_photos / DSC_2392.JPG",
                "Egg_photos / DSC_2393.JPG",
                "Egg_photos / DSC_2394.JPG",
                "Egg_photos / DSC_2395.JPG"]
    X_size = cv2.imread(files[0].rstrip("jpgJPG").rstrip(".").replace(" ", "") + "_480p.JPG").shape
    Y_size = cv2.imread(files[0].rstrip("jpgJPG").rstrip(".").replace(" ", "") + "_480p_dmap.JPG").shape
    X = np.zeros((10, X_size[0], X_size[1], X_size[2]))
    Y = np.zeros((10, Y_size[0], Y_size[1], Y_size[2]))
    for i in range(len(files)):
        X[i] = cv2.imread(files[i].rstrip("jpgJPG").rstrip(".").replace(" ", "") + "_480p.JPG")
        Y[i] = cv2.imread(files[i].rstrip("jpgJPG").rstrip(".").replace(" ", "") + "_480p_dmap.JPG")

    X_train = X[:8]
    X_val = X[8:]
    Y_train = Y[:8]
    Y_val = Y[8:]
    return (X_train, Y_train), (X_val, Y_val)

# ---------------------------------------------------------------------------------------------------------------------


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:11:52 2020

@author: Kevin Yost

Various methods used for image pre- and post-processing.

Potential future work: alter generate_density_map to deal with dessicated eggs - read in class type, draw healthy eggs
    on first channel and dessicated on second channel. Will require modifying and re-training DeepEggCounts for two output
    classes.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from win32api import GetSystemMetrics
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from os.path import exists

res = {"240p": (320, 240),
       "360p": (480, 360),
       "480p": (640, 480),
       "720p": (960, 720),
       "1080p": (1440, 1080)}
cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


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


def yolo_to_xy(yolo_txt, img_shape: tuple):
    """
    Takes in coordinates in YOLO format. Converts to raw pixel # format

    Arg:
        yolo_txt (array):       data in YOLO format (class, x, y, w, h) format (# eggs, 5)
                                *** x, y, w, h are all expressed as percentages
        img_shape (tuple):      shape of image (h, w(, 3))

    Returns:
        coor_format (array of int):   np array of coordinates of eggs (x, y)
    """
    coor_format = yolo_txt[:, 1:3]  # extract (x, y) coordinate info
    coor_format[:, 0] *= img_shape[1]  # transforming percentages to coordinates
    coor_format[:, 1] *= img_shape[0]
    coor_format = coor_format.astype(int)  # round to int
    return coor_format


def image_rescale(img, final_dim: tuple):
    """
    Takes in image and desired final dimensions. Returns rescaled image and the
        rescale factor for further calculations.

    Args:
        img (array):                input image to be rescaled
        final_dim (tuple of int):   final dimensions of image (width, height)

    Returns:
        img_rescaled (array):               rescaled image
        rescale_factor (tuple of float):    rescale factor (r_f_x, r_f_y)
    """
    rescale_factor = (final_dim[0] / img.shape[1], final_dim[1] / img.shape[0])
    img_rescaled = cv2.resize(img, final_dim)
    return img_rescaled, rescale_factor


def coor_rescale(original_coordinates, rescale_factor: tuple):
    """
    Rescales coordinates based of rescaling factor.

    Args:
        original_coordinates (array):       coordinates of eggs (# eggs, 2)
        rescale_factor (tuple of float):    rescale factor (r_f_x, r_f_y)

    Return:
        coordinates_rescaled (array):   coordinates in rescaled image space
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
        original_coordinates (array):               egg coordinates of original image (x, y)
        cropped_image_corners (array of tuples):    cropped image coordinates [(x_start, y_start), (x_end, y_end)]

    Returns:
        cropped_coordinates (array):    coordinates in cropped image space (x, y)
    """
    cropped_coordinates = np.zeros([0, 2])
    min_x = min(cropped_image_corners[0][0], cropped_image_corners[1][0])
    max_x = max(cropped_image_corners[0][0], cropped_image_corners[1][0])
    min_y = min(cropped_image_corners[0][1], cropped_image_corners[1][1])
    max_y = max(cropped_image_corners[0][1], cropped_image_corners[1][1])
    for i in range(original_coordinates.shape[0]):
        if min_x <= original_coordinates[i, 0] < max_x and min_y < original_coordinates[i, 1] <= max_y:
            cropped_coordinates = np.append(cropped_coordinates, [[original_coordinates[i, 0] - min_x,
                                                                   original_coordinates[i, 1] - min_y]], axis=0)
    return cropped_coordinates


def draw_points_on_image(image, coordinates):
    """
    Sanity check: draws dot on image at location of egg

    Args:
        image (array):          image (h, w, 3)
        coordinates (array):    coordinates (# eggs, 2)

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

    plt.imshow(image)
    plt.show()

    return None


def generate_density_map(img, coor):
    """
    Generate a density map based on objects positions.

    Args:
        img (array):        image (h, w, 3)
        coor (array):       array containing egg coordinate info in xy format (# eggs, 2)

    Returns:
        density_map (array):    density map of inputted image (h, w, 1)
    """
    coor = coor.astype(int)  # round to int
    # initialise density map of size image - only 1 channel
    density_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    # applying heat of 100 at location of eggs
    if coor.size > 0:
        for i in range(coor.shape[0]):
            density_map[coor[i, 1], coor[i, 0]] += 100

    # apply Gaussian kernel to density map
    density_map = cv2.GaussianBlur(density_map, (5, 5), 0)  # kernel size of 5, standard deviation of 0

    density_map = density_map[..., np.newaxis]  # adding final filter axis
    return density_map


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


def image_crop_and_scale(resolution: str = "360p"):
    """
    Prompts user to crop image in the section in which eggs exist. Unless specified, crops to 360p. Saves cropped
        image as .JPG

    Args:
        resolution (str):   desired output image's resolution, default "360p"

    Returns:
        image_final_res (array):    cropped image (aspect ratio maintained)
    """
    global cropping, cropped, x_start, y_start, x_end, y_end
    cropping = False
    cropped = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    res_tuple = res[resolution]
    aspect_ratio = res_tuple[0] / res_tuple[1]

    img_path = get_image_path()
    image = cv2.imread(img_path)
    # check if image height > image width - if so, rotate
    if image.shape[0] > image.shape[1]:
        image = np.rot90(image)

    # retrieving current display size to ensure it fits the screen during cropping and by inputted resolution for
    # image processing - format: (width, height)
    p_current_display = (GetSystemMetrics(0), GetSystemMetrics(1))

    # determining rescale factor to fit image in screen while preserving aspect ratio of image
    rf_crop_display = min(p_current_display[1] / image.shape[0], p_current_display[0] / image.shape[1])
    # calculating the image dimensions in pixels which will fit on screen
    p_crop_display = (int(rf_crop_display * image.shape[1]), int(rf_crop_display * image.shape[0]))
    # gets a rescaled image which will fit on user's screen
    image_fit_screen, rf_fit_screen = image_rescale(image, p_crop_display)

    # generates a window in which the user is prompted to crop out the section they want analysed
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
    while not cropped:
        image_copy = image_fit_screen.copy()
        if not cropping:
            cv2.imshow("image", image_fit_screen)
        elif cropping:
            cv2.rectangle(image_copy, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", image_copy)
        cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()

    # grabs coordinates of cropped image corners
    cropped_corners = [(x_start, y_start), (x_end, y_end)]
    # shifts coordinates of the eggs to new image's coordinate space
    cropped_corners = cropped_box_shift(cropped_corners, aspect_ratio)
    # creates cropped image
    image_cropped = image_fit_screen[cropped_corners[0][1]:cropped_corners[1][1],
                                     cropped_corners[0][0]:cropped_corners[1][0]]

    # Final downscaling to 480p format of both cropped image and coordinates for U-Net to process more easily.
    image_final_res, rf_final_res = image_rescale(image_cropped, res_tuple)

    return image_final_res


def image_crop_scale_dmap(resolution: str = "480p", img_path: str = ""):
    """
    Prompts user to select an image to crop. Scales the image to the input resolution (extends smallest dimension
        between height and width to do so). Saves cropped image with the coordinates of eggs within that cropped image.
        Generates density map, and saves that too.

    Args:
        resolution:         desired resolution of final image
        img_path:           directory path of where image is at

    Returns:
        img_path:           path of the original image that user selected
        image_final_res:    cropped image
        coor_final_res:     coordinates of eggs within cropped image
        dmap:               density map of cropped image
    """
    global cropping, cropped, x_start, y_start, x_end, y_end
    cropping = False
    cropped = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    res_tuple = res[resolution]
    aspect_ratio = res_tuple[0] / res_tuple[1]

    # checks if img_path was given, if not, prompts user for path
    if not img_path:
        img_path = get_image_path()

    data_path = img_path[:-4] + ".txt"

    image = cv2.imread(img_path)
    data = np.loadtxt(data_path)
    coor = yolo_to_xy(data, image.shape)

    # get current display size to ensure image fits within user's current screen
    p_current_display = (GetSystemMetrics(0), GetSystemMetrics(1))
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
    image_final_res, rf_final_res = image_rescale(image_cropped, res_tuple)
    coor_final_res = coor_rescale(coor_cropped, rf_final_res)

    # Visual check that coordinates are at correct locations
    draw_points_on_image(image_final_res.copy(), coor_final_res)

    save_path = img_path[:-4]
    img_path_save = save_path + "_" + resolution + ".JPG"
    coor_path_save = save_path + "_" + resolution + ".txt"
    dmap_path_save = save_path + "_" + resolution + "_dmap.txt"
    cv2.imwrite(img_path_save, image_final_res)
    np.savetxt(coor_path_save, coor_final_res)

    dmap = generate_density_map(img_path_save, coor_path_save)
    np.savetxt(dmap_path_save, dmap)
    return img_path, image_final_res, coor_final_res, dmap


def image_down_res_dmap(resolution: str = "360p", img_path: str = ""):
    """
        Prompts user to select an image to crop. Downscales the image to the input resolution (extends smallest
            dimension between height and width to do so). Saves downscaled image with the coordinates of eggs within
            that downscaled image in (pixel_x, pixel_y) format. Generates density map, and saves that too.

        Args:
            resolution (str):   desired resolution of final image, default "360p"
            img_path (str):     directory path of where image is at, default ""

        Returns:
            img_path (str):             path of the original image that user selected
            image_final_res (array):    cropped image
            coor_final_res (array):     coordinates of eggs within cropped image (# eggs, 2)
            dmap (array):               density map of cropped image (h, w)
        """
    global cropping, cropped, x_start, y_start, x_end, y_end
    cropping = False
    cropped = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    res_tuple = res[resolution]

    # checks if img_path was given, if not, prompts user for path
    if not img_path:
        img_path = get_image_path()

    data_path = img_path[:-4] + ".txt"

    image = cv2.imread(img_path)
    coor = np.loadtxt(data_path)

    img_downscaled, rf_downscale = image_rescale(image, res_tuple)
    coor_downscaled = coor_rescale(coor, rf_downscale)

    # Visual check that coordinates are at correct locations
    draw_points_on_image(img_downscaled.copy(), coor_downscaled)

    save_path = img_path[:-8]
    img_path_save = save_path + resolution + ".JPG"
    coor_path_save = save_path + resolution + ".txt"
    dmap_path_save = save_path + resolution + "_dmap.txt"
    cv2.imwrite(img_path_save, img_downscaled)
    np.savetxt(coor_path_save, coor_downscaled)

    dmap = generate_density_map(img_downscaled, coor_downscaled)
    np.savetxt(dmap_path_save, dmap)
    return img_path, img_downscaled, coor_downscaled, dmap


def heat_map(x_img, y_img):
    """
    Sanity check to visually determine whether an image's density map matches up with ground truth.

    Args:
        x_img (array):  normal image of eggs (h, w, 3)
        y_img (array):  density map (h, w, 1)

    Returns:
        None
    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x_img)
    plt.subplot(1, 2, 2)
    plt.imshow(y_img[:, :, 0], cmap='plasma')
    return None


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


def process_load_data_dir(e2e: bool = False, directory: str = './egg_photos/originals', resolution: str = "360p"):
    """
    Given directory, loads images (X) and their corresponding density maps (Y), splits them into sub-images of defined
        resolution, filters out sub-images with no eggs in them to speed up training process, normalises the sub-images,
        return list of sub-images, their density maps, and an array to track how many sub-images were sampled from each
        full image.

    Args:
        e2e (bool):             end-to-end - will output Y as number of eggs instead of mask of Gaussian map
        directory (str):        location in directory where images and coordinates to be processed are location
        resolution (str):       resolution which sub-images should be, default: "360p"

    Returns:
        X (list of images):     training images (n examples, h, w, 3)
        Y (list of images):     density maps of training images (n examples, h, w, 1)
        num_sub_imgs (array):   array of number of sub-images that the initial image was split into
    """
    files = get_file_list(directory, ".JPG")
    np.random.shuffle(files)        # shuffling inputs to decrease bias in training
    res_tuple = res[resolution]     # default (480, 360)

    # down-scaling inputs (optional) - originals are 4912x7360, eggs are still discernible at 1/9th resolution (drf=3)
    down_res_factor = 3.
    X = np.empty((0, res_tuple[1], res_tuple[0], 3), dtype=np.uint8)
    if not e2e:
        Y = np.empty((0, res_tuple[1], res_tuple[0], 1), dtype=np.float32)
    elif e2e:
        Y = np.empty(0, dtype=np.float32)
    num_sub_imgs = np.empty(0, dtype=np.uint8)
    for f in tqdm(files):
        img = cv2.imread(f)
        yolo_coor = np.loadtxt(f[:-4] + ".txt")
        coor = yolo_to_xy(yolo_coor, img.shape[0:2])
        img_resc, rf = image_rescale(img, (int(img.shape[0]/down_res_factor), int(img.shape[1]/down_res_factor)))
        coor_resc = coor_rescale(coor, rf)

        # data augmenting each image
        images, coors = image_coor_flip_rotate(img_resc, coor_resc)
        for i in range(images.__len__()):
            sub_images, sub_dmaps = split_image_dmap(images[i], coors[i], resolution)

            # loading only sub-images with eggs
            n_with_eggs = 0
            for j in range(sub_dmaps.shape[0]):
                if np.sum(sub_dmaps[j]) > 0:
                    X = np.append(X, sub_images[j:j+1], axis=0)
                    if not e2e:
                        Y = np.append(Y, sub_dmaps[j:j+1], axis=0)
                    elif e2e:
                        Y = np.append(Y, np.sum(sub_dmaps[j]) / 100)
                    n_with_eggs += 1
            num_sub_imgs = np.append(num_sub_imgs, n_with_eggs)
    if e2e:
        Y = Y[..., np.newaxis]
    assert X.shape[0] == Y.shape[0], "Loaded data are not of equal length: X != Y"

    X = X.astype('float32')
    Y = Y.astype('float32')
    X /= 255.   # normalising inputs
    return X, Y, num_sub_imgs


def load_data_npy(e2e: bool = False):
    if not e2e:
        X = np.load('X.npy')
        Y = np.load('Y.npy')
        n_sub_imgs = np.load('nsi.npy')
    else:
        X = np.load('X_e2e.npy')
        Y = np.load('Y_e2e.npy')
        n_sub_imgs = np.load('nsi_e2e.npy')
    return X, Y, n_sub_imgs


def save_data_npy(X, Y, nsi, e2e: bool = False):
    if not e2e:
        np.save('X.npy', X)
        np.save('Y.npy', Y)
        np.save('nsi.npy', nsi)
    else:
        np.save('X_e2e.npy', X)
        np.save('Y_e2e.npy', Y)
        np.save('nsi_e2e.npy', nsi)


def sum_density_over_sub_images(Y, n_sub_images):
    """
    Sums density maps of sub-images to return the number of eggs within each full image
    Args:
        Y (array):                    density map of each sub-image (# sub-images, h, w, 1)
        n_sub_images (array of int):    array of number of sub-images within each full-image

    Returns:
        n_eggs (array of int):  number of eggs in each full image
    """
    n_eggs = np.zeros(n_sub_images.shape[0])
    ind_start = 0
    for i in range(n_sub_images.shape[0]):
        n_eggs[i] = int(np.round(np.sum(Y[ind_start:ind_start+n_sub_images[i]]) / 100))
        ind_start += n_sub_images[i]
    return n_eggs


def get_file_list(path: str, search_term: str):
    """
    Returns a list of files that matches search term within the directory provided.

    Args:
        path (str):             directory in which to search
        search_term (str):      search term

    Returns:
        files (list of str):    list of files
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if search_term.lower() in files or search_term.upper() in file:
                files.append(os.path.join(r, file))
    return files


def split_rot_image(image, resolution: str = "360p"):
    """
    Takes in larger image and converts it to multiple sub-image with minimal resolution loss.

    Args:
        image (array):      input image to be broken down (h, w, 3)
        resolution (str):   desired output sub-image resolution, default: "360p" = (480, 360)

    Returns:
        sub_images (array): array of shape (# sub-images, resolution height, resolution width, 3)
    """
    img_dim = image.shape  # (h, w, 3)
    res_dim = res[resolution]  # (w, h)

    # determining which sub-image orientation will yield minimal pixel lost in re-scaling
    n_img_h_by_res_w = int(np.floor(img_dim[0] / res_dim[0]))  # times sub-image width can fit in image's height
    n_img_w_by_res_h = int(np.floor(img_dim[1] / res_dim[1]))  # times sub-image height can fit in image's width
    n_img_h_by_res_h = int(np.floor(img_dim[0] / res_dim[1]))  # times sub-image height can fit in image's height
    n_img_w_by_res_w = int(np.floor(img_dim[1] / res_dim[0]))  # times sub-image width can fit in image's width
    n_sub_img_hw_wh = n_img_h_by_res_w * n_img_w_by_res_h  # number of sub-images if image is split by h/w & w/h
    n_sub_img_hw_hw = n_img_h_by_res_h * n_img_w_by_res_w  # number of sub-images if image is split by h/h & w/w

    if n_sub_img_hw_wh > n_sub_img_hw_hw:
        image = np.swapaxes(image, 0, 1)    # flip image
        n_sub_img_hw_hw = n_sub_img_hw_wh   # updating number of sub-images to return

    if n_sub_img_hw_hw <= 1:  # image cannot be broken down further
        sub_images = np.zeros(shape=(1, res_dim[1], res_dim[0], 3), dtype="uint8")
        sub_images[0], _ = image_rescale(image, res_dim)
        print("Image was re-scaled, but could not be broken down into sub-images.")
    else:
        sub_images = np.zeros(shape=(n_sub_img_hw_hw, res_dim[1], res_dim[0], 3), dtype="uint8")
        img_resc, _ = image_rescale(image, (res_dim[0] * n_img_w_by_res_w, res_dim[1] * n_img_h_by_res_h))
        for i in range(n_img_h_by_res_h):
            for j in range(n_img_w_by_res_w):
                sub_images[i * n_img_w_by_res_w + j] = img_resc[i * res_dim[1]:(i + 1) * res_dim[1],
                                                                j * res_dim[0]:(j + 1) * res_dim[0]]
    return sub_images


def split_rot_image_dmap(image, coor, resolution: str = "360p"):
    """
    Takes in larger image and converts it to multiple sub-image with minimal resolution loss.

    Args:
        image (array):        input image to be broken down (h, w, 3)
        coor (array):         array containing egg coordinate info in (x, y) format (# eggs, 2)
        resolution (str):       desired output sub-image resolution, default: "360p" = (480, 360)

    Returns:
        sub_images (array):   array of shape (# sub-images, (resolution), 3)
        sub_dmaps (array):    density maps of sub-images (# sub-images, resolution height, resolution width)
    """
    img_dim = image.shape  # (h, w, 3)
    res_dim = res[resolution]  # (w, h)

    # determining which sub-image orientation will yield minimal pixel lost in re-scaling
    n_img_h_by_res_w = int(np.floor(img_dim[0] / res_dim[0]))  # times sub-image width can fit in image's height
    n_img_w_by_res_h = int(np.floor(img_dim[1] / res_dim[1]))  # times sub-image height can fit in image's width
    n_img_h_by_res_h = int(np.floor(img_dim[0] / res_dim[1]))  # times sub-image height can fit in image's height
    n_img_w_by_res_w = int(np.floor(img_dim[1] / res_dim[0]))  # times sub-image width can fit in image's width
    n_sub_img_hw_wh = n_img_h_by_res_w * n_img_w_by_res_h  # number of sub-images if image is split by h/w & w/h
    n_sub_img_hw_hw = n_img_h_by_res_h * n_img_w_by_res_w  # number of sub-images if image is split by h/h & w/w

    if n_sub_img_hw_wh > n_sub_img_hw_hw:
        image = np.swapaxes(image, 0, 1)    # flip image
        temp = np.zeros_like(coor)
        temp[:, 0] = coor[:, 1]
        temp[:, 1] = coor[:, 0]
        coor = temp     # flipping coordinate axes
        n_sub_img_hw_hw = n_sub_img_hw_wh   # updating number of sub-images to return

    if n_sub_img_hw_hw <= 1:  # image cannot be broken down further
        sub_images = np.zeros(shape=(1, res_dim[1], res_dim[0], 3), dtype="uint8")
        sub_images[0], rf = image_rescale(image, res_dim)
        coor_resc = coor_rescale(coor, rf)
        sub_dmaps = np.zeros(shape=(1, res_dim[1], res_dim[0], 1), dtype=np.float32)
        sub_dmaps[0] = generate_density_map(sub_images[0], coor_resc)
        # print("Image was re-scaled, but could not be broken down into sub-images.")
    else:
        sub_images = np.zeros(shape=(n_sub_img_hw_hw, res_dim[1], res_dim[0], 3), dtype="uint8")
        img_resc, rf = image_rescale(image, (res_dim[0] * n_img_w_by_res_w, res_dim[1] * n_img_h_by_res_h))
        coor_resc = coor_rescale(coor, rf)
        dmap = generate_density_map(img_resc, coor_resc)
        sub_dmaps = np.zeros(shape=(n_sub_img_hw_hw, res_dim[1], res_dim[0], 1), dtype=np.float32)
        for i in range(n_img_h_by_res_h):
            for j in range(n_img_w_by_res_w):
                sub_images[i * n_img_w_by_res_w + j] = img_resc[i * res_dim[1]:(i + 1) * res_dim[1],
                                                                j * res_dim[0]:(j + 1) * res_dim[0]]
                sub_dmaps[i * n_img_w_by_res_w + j] = dmap[i * res_dim[1]:(i + 1) * res_dim[1],
                                                           j * res_dim[0]:(j + 1) * res_dim[0]]
    return sub_images, sub_dmaps


def split_image_dmap(image, coor, resolution: str = "360p"):
    """
    Takes in image and its egg coordinates and converts them into multiple sub-images and sub-density-maps.

    Args:
        image (array):      input image to be broken down (h, w, 3)
        coor (array):       egg coordinates info in (x, y) format (# eggs, 2)
        resolution (str):   desired output sub-image resolution, default: "360p" = (480, 360)

    Returns:
        sub_images (array):     array of shape (# sub-images, (resolution), 3)
        sub_dmaps (array):      density maps of sub-images (# sub-images, (resolution))
    """
    img_dim = image.shape       # (h, w, 3)
    res_dim = res[resolution]   # (w, h)

    n_sub_img_h = int(np.floor(img_dim[0] / res_dim[1]))  # times sub-image height can fit in image's height
    n_sub_img_w = int(np.floor(img_dim[1] / res_dim[0]))  # times sub-image width can fit in image's width
    n_sub_imgs = n_sub_img_h * n_sub_img_w  # number of sub-images if image is split by h/h & w/w

    if n_sub_imgs <= 1:  # image cannot be broken down further
        sub_images = np.zeros(shape=(1, res_dim[1], res_dim[0], 3), dtype="uint8")
        sub_images[0], rf = image_rescale(image, res_dim)
        coor_resc = coor_rescale(coor, rf)
        sub_dmaps = np.zeros(shape=(1, res_dim[1], res_dim[0], 1), dtype=np.float32)
        sub_dmaps[0] = generate_density_map(sub_images[0], coor_resc)
        # print("Image was re-scaled, but could not be broken down into sub-images.")
    else:
        sub_images = np.zeros(shape=(n_sub_imgs, res_dim[1], res_dim[0], 3), dtype="uint8")
        img_resc, rf = image_rescale(image, (res_dim[0] * n_sub_img_w, res_dim[1] * n_sub_img_h))
        coor_resc = coor_rescale(coor, rf)
        dmap = generate_density_map(img_resc, coor_resc)
        sub_dmaps = np.zeros(shape=(n_sub_imgs, res_dim[1], res_dim[0], 1), dtype=np.float32)
        for i in range(n_sub_img_h):
            for j in range(n_sub_img_w):
                sub_images[i * n_sub_img_w + j] = img_resc[i * res_dim[1]:(i + 1) * res_dim[1],
                                                           j * res_dim[0]:(j + 1) * res_dim[0]]
                sub_dmaps[i * n_sub_img_w + j] = dmap[i * res_dim[1]:(i + 1) * res_dim[1],
                                                      j * res_dim[0]:(j + 1) * res_dim[0]]
    return sub_images, sub_dmaps


def image_coor_flip_rotate(image, coor):
    """
    Takes in image and coordinate inputs, flips and rotates them in all possible (12) combinations, returns tuple of
        flipped and rotated images and coordinates. This function is for data augmentation purposes.

    Args:
        image (array):        input image to be flipped and rotated (h, w, 3)
        coor (array):         egg coordinates info of image in (x, y) format (# eggs, 2)

    Returns:
        images (list of array):   list of images that have gone through all possible flipping and rotations
        coors (list of array):    list of flipped and rotated coordinates corresponding to images

    Note:   images needed to be a tuple as its shape changes due to rotation. coors didn't need to be tuple and could
        have been implemented as an array, but I wanted to keep data types consistent with images.
    """
    assert image.shape.__len__() == 3, "Expected image to have length of 3 (h, w, 3), got " + str(image.shape.__len__())
    assert coor.shape.__len__() == 2, "Expected image to have length of 2 (# eggs, 2), got " + str(coor.shape.__len__())

    # flipping images and their corresponding coordinates
    # image_fliplr = np.fliplr(image)
    # coor_fliplr = coor.copy()   # copying so as not to alias
    # coor_fliplr[:, 0] = (-1) * coor_fliplr[:, 0] + image_fliplr.shape[1]    # flipping x's
    # image_flipud = np.flipud(image)
    # coor_flipud = coor.copy()
    # coor_flipud[:, 1] = (-1) * coor_fliplr[:, 1] + image_fliplr.shape[0]    # flipping y's

    # performing image and coordinate rotations
    images_norm, coors_norm = rotate_image_coor(image, coor)
    # images_lr, coors_lr = rotate_image_coor(image_fliplr, coor_fliplr)
    # images_ud, coors_ud = rotate_image_coor(image_flipud, coor_flipud)
    
    # combining all 12 possible flippings and rotations
    # images = images_norm + images_lr + images_ud
    # coors = coors_norm + coors_lr + coors_ud

    # not doing flipping as that is actually too much data to train on my poor laptop
    images = images_norm
    coors = coors_norm
    return images, coors


def rotate_image_coor(image, coor):
    """
    Rotates the image and coor input 90 degrees 4 times and returns

    Args:
        image (array):  image (h, w, 3)
        coor (array):   coordinates (# eggs, 2)

    Returns:
        images (list of array): list of images that have gone through all rotations
        coors (list of array):  list of rotated coordinates corresponding to images
    """
    #   image   ->  img90  ->   img180  ->  img270
    #   1   2       2   4       4   3       3   1
    #   3   4       1   3       2   1       4   2
    img90 = np.rot90(image)     # 90 degrees counter-clockwise rotation about (0, 0)
    img180 = np.rot90(img90)
    img270 = np.rot90(img180)

    c90 = coor_rot90(img90.shape, coor)
    c180 = coor_rot90(img180.shape, c90)
    c270 = coor_rot90(img270.shape, c180)

    images = [image, img90, img180, img270]
    coors = [coor, c90, c180, c270]

    return images, coors


def coor_rot90(curr_image_shape, prev_coor):
    """
    Rotates given coordinates 90 degrees counter-clockwise about (0, 0) w.r.t. given image dimensions

    Args:
        curr_image_shape (array):   shape of rotated image (h, w, 3)
        prev_coor (array):          un-rotated coordinates in xy format (# eggs, 2)

    Returns:
        coor_rotated (array):       rotated coordinates (# eggs, 2)
    """
    coor_rotated = np.zeros_like(prev_coor)
    coor_rotated[:, 0] = prev_coor[:, 1]                        # new_x = old_y
    coor_rotated[:, 1] = curr_image_shape[0] - prev_coor[:, 0]  # new_y = new_height - old_x
    return coor_rotated

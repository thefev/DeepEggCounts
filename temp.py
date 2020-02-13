import cv2
import numpy as np
import os.path
from os import path
from win32api import GetSystemMetrics


def mouse_crop(event, x, y, flags, param):
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

        if len(ref_point) == 2:  # when two points were found
            #            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            #            cv2.imshow("Cropped", roi)
            #            cv2.waitKey(3000)
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
    coor_format = yolo_txt[:, 1:3]      # extract (x, y) coordinate info
    coor_format[:, 0] *= img_dim[1]     # transforming percentages to coordinates
    coor_format[:, 1] *= img_dim[0]
    coor_format = coor_format.astype(int)   # round to int
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


def coor_rescale(coor, rf):
    """
    Rescales coordinates based of rescaling factor.
    
    Args: 
        coor:   coordinates of eggs in (x, y) format
        rf:     rescale factor (r_f_x, r_f_y)
        
    Return:
        coor_rescaled:  coordinates in rescaled image space
    """
    coor_rescaled = np.zeros_like(coor)
    coor_rescaled[:, 0] = coor[:, 0] * rf[0]
    coor_rescaled[:, 1] = coor[:, 1] * rf[1]
    coor_rescaled = coor_rescaled.astype(int)
    return coor_rescaled


def coor_crop_shift(coor, cropped_corners):
    """
    Converts coordinates in original image space to coordinates in cropped 
    image space.
    
    Args:
        coor:               egg coordinates of original image (x, y)
        cropped_corners:    cropped image coordinates
                            [(x_start, y_start), (x_end, y_end)]
    
    Returns:
        coor_cropped:       coordinates in cropped image space (x, y)
    """
    coor_cropped = coor
    coor_cropped[:, 0] -= cropped_corners[0][0]
    coor_cropped[:, 1] -= cropped_corners[0][1]
    return coor_cropped


cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

img_path = "Egg_Photos/DSC_2380.JPG"
image = cv2.imread(img_path)

## prompting user for image path
# path_exist = False
#
# while not path_exist:
#    img_path = input("img_path: ")
#    if path.exists(img_path):
#        path_exist = True
#
# image = cv2.imread(img_path)
data = np.loadtxt(img_path.rstrip("jpgJPG").rstrip('.') + ".txt")
coor = yolo_to_xy(data, image.shape)

# defining dimension in pixels based on current display size during cropping 
#   and 480p for image processing
p_image = image.shape
p480 = (640, 480)
p_current_display = (GetSystemMetrics(0), GetSystemMetrics(1))
# ensuring linear rescaling that fits user's window
rf_crop_display = min(p_current_display[1] / image.shape[0], p_current_display[0] / image.shape[1])
p_crop_display = (int(rf_crop_display * image.shape[1]), int(rf_crop_display * image.shape[0]))
image_rescaled, rf_rescaled = image_rescale(image, p_crop_display)  # rescale to fit screen
coor_rescaled = coor_rescale(coor, rf_rescaled)     # updating coordinates of rescaled image

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
while not cropped:
    i = image_rescaled.copy()
    if not cropping:
        cv2.imshow("image", image_rescaled)
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
    cv2.waitKey(1)
# close all open windows
cv2.destroyAllWindows()

cropped_corners = [(x_start, y_start), (x_end, y_end)]
image_cropped = image_rescaled[cropped_corners[0][1]:cropped_corners[1][1], cropped_corners[0][0]:cropped_corners[1][0]]
coor_cropped = coor_crop_shift(coor_rescaled, cropped_corners)


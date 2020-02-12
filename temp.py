import cv2
import numpy as np


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
        if cropping == True:
            x_end, y_end = x, y
            
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        
        refPoint = [(x_start, y_start), (x_end, y_end)]
        
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            cropped = True

cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = cv2.imread('Egg_Photos/DSC_2378.JPG')
if image.shape[0] > 1080:
    rescale_factor = 1080/image.shape[0]
    dsize = (int(image.shape[1] * rescale_factor), int(image.shape[0] * rescale_factor))
    image_rescaled = cv2.resize(image, dsize)

oriImage = image_rescaled.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while not cropped:
    
    i = image_rescaled.copy()
    
    if not cropping:
        cv2.imshow("image", image_rescaled)
        
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
        
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()

#refPoint_unscaled = [(int(x_start/rescale_factor), int(y_start/rescale_factor)), (int(x_end/rescale_factor), int(y_end/rescale_factor))]
#image_cropped = image[refPoint_unscaled[0][1]:refPoint_unscaled[1][1], refPoint_unscaled[0][0]:refPoint_unscaled[1][0]]

refPoint = [(x_start, y_start), (x_end, y_end)]
image_cropped = image_rescaled[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]





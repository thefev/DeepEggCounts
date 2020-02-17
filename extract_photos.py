# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:10:29 2020

@author: Kevin
"""


def extract_photos(img_path, txt_path):
    """Creates multiple sub-photos from image input at coordinates from txt
    
    Arguments:
        img_path -- location of image
        txt_path -- location of txt containing coordinates of objects in YOLO format (x, y, w, h)
    
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # load in text coordinates of objects and image
    txt = np.loadtxt(txt_path)
    img = Image.open(img_path)
    
    img_width, img_height = img.size
    
    txt[:, 1] *= img_width
    txt[:, 2] *= img_height
    txt[:, 3] *= img_width
    txt[:, 4] *= img_height
    
    img_cropped = img.crop((
            int(txt[0, 1] - txt[0, 3]/2),
            int(txt[0, 2] - txt[0, 4]/2),
            int(txt[0, 1] + txt[0, 3]/2),
            int(txt[0, 2] + txt[0, 4]/2) ))
    
    img_cropped.show()
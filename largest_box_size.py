# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:36:50 2020

@author: Kevin
"""

def largest_box_size(txt_path):
    """ Loads in text file with bounding box size and returns largest height and 
        width required for a bounding box
        
        Arguments:
        txt_path -- path within folder where txt file is
            -> txt file must be of YOLO format (None, 5)
                -> 5: classtype, x, y, width, height
        
        Return:
        largest_box_size -- (2, 1) array of largest bounding box size to capture all eggs
    
    """
    import numpy as np
    
    data = np.loadtxt(txt_path)
    
    largest_box_size = [np.max(data[:,3]), np.max(data[:,4])]
    
    return largest_box_size
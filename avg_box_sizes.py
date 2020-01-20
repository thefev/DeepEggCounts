# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:43:06 2020

@author: Kevin


"""

def avg_box_sizes(bb_txt_path):
    """Get average bounding box sizes for an image
    
    Arguments: 
    bb_txt_path -- path of bounding box .txt in YOLO format from EggCountNet folder with entries of shape [object class, x, y, width, height]
    
    Returns:
    avg_box_sizes -- (4, 2) width and height entries for 4 boxes 
    
    """

    import numpy as np
    
    # load file
    img_box_info = np.loadtxt(fname = bb_txt_path)
    assert img_box_info.shape[1]==5, "Input files dimensions not as expected ([None, 5])"
    entries = img_box_info.shape[0]
    
    # width-to-height ratio for each box
    wh_ratio = img_box_info[:,3] / img_box_info[:,4]    
    
    # initialise array for binning boxes of different dimensions
    boxes = np.array(np.zeros((4, 3)))
    
    # set binning conditions
    bin22 = 1 / np.tan(np.pi * 1 / 8)
    bin45 = 1 / np.tan(np.pi * 2 / 8)
    bin67 = 1 / np.tan(np.pi * 3 / 8)
    
    # assigning boxes into bins
    for i in range(entries):
        if wh_ratio[i] <= bin22:
            boxes[0][0:2] += img_box_info[i, 3:5]
            boxes[0][2] += 1
        elif wh_ratio[i] > bin22 and wh_ratio[i] <= bin45:
            boxes[1][0:2] += img_box_info[i, 3:5]
            boxes[1][2] += 1
        elif wh_ratio[i] > bin45 and wh_ratio[i] <= bin67:
            boxes[2][0:2] += img_box_info[i, 3:5]
            boxes[2][2] += 1
        elif wh_ratio[i] > bin67:
            boxes[3][0:2] += img_box_info[i, 3:5]
            boxes[3][2] += 1
            
    assert np.sum(boxes[:, 2]) == entries, "Some entries may have been missed or duplicated in binning process"
    print(boxes[:,2])
    # acquiring average box sizes
    avg_box_sizes = np.divide(boxes, (boxes[:, 2])[:, None])
    
    return avg_box_sizes
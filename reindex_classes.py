# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:16:52 2020

@author: Kevin
"""

def reindex_classes(txt_path):
    """Reindexes classes in txt files of YOLO config to shift down 15
    Initial config was for 15 + n classes, but we want to get rid of the 
    pre-defined classes, which leaves us with n classes to save computational 
    time while training and in production
    
    Argument:
    txt_path -- path of txt file containing class names and coordinates in YOLO 
        format
        
    Returns: nothing, but overrides existing file 
        *** PAY ATTENTION NOT TO RUN THIS ON A FILE TWICE ***
        I'm not going to bother writing a check to ensure positive int classes
    """
    
    import numpy as np
    
    txt = np.loadtxt(txt_path)
    txt[:, 0] -= 15
    np.savetxt(txt_path, txt)
    

def average_egg_coefficient(dmap_path):
    """
    Created on Tue Feb 11 14:52:00 2020
    
    @author: Kevin
    
    Sanity check. Takes in density map path. Infers coordinate data from that.
        Determines the amount of 'heat' per egg in the 
    
    Arg:
        dmap_path:  location of density map (.JPG)
    
    Returns:
        aec:        average egg coefficient - the amount of 'heat' per egg
    """
    
    import cv2
    import numpy as np
    
    dmap = cv2.imread(dmap_path)
    data = np.loadtxt(dmap_path.strip('_dmap.JPG')+'.txt')
    eggs = data.shape[0]        # get number of eggs
    heat_sum = np.sum(dmap)     # integrating over training output (y)
    aec = heat_sum/eggs         # determining amount of 'heat' per egg 
    
    print('Number of eggs:\t'+str(eggs))
    print('Average egg coefficient:\t'+str(aec))
    
    return aec
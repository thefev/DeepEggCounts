def generate_density_map(img_path, coor_path):
    """
    Created on Mon Feb  3 13:30:13 2020
    
    @author: Kevin
    
    Generate a density map based on objects positions. Saves as file.
    
    Args:
        img_path:   location of image
        coor_path:  location of txt file containing coordinate info in 
                    (class, x, y, w, h) format
    Returns:
        None
    """
    
    assert img_path.strip('.JPG') == coor_path.strip('.txt'), 'img_path and coor_path files do not match'
    
    import cv2
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    img = cv2.imread(img_path)
    data = np.loadtxt(coor_path)
    coor = data[:, 1:3] # extract coordinate info
    coor[:, 0] *= img.shape[0]  # transforming percentages to coordinates
    coor[:, 1] *= img.shape[1]  
    coor = coor.astype(int) # round to int
    # initialise density map of size image
    density_map = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    
    # applying heat of 100 at location of eggs
    for i in range(coor.shape[0]):
        density_map[coor[i,0], coor[i,1]] += 100
    
    # apply Gaussian kernel to density map
    density_map = gaussian_filter(density_map, sigma=(1, 1), order=0)
    
    # save density map
    cv2.imwrite(img_path.strip('.JPG') + '_dmap.JPG', density_map)
    
    return None
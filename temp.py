import cv2
import numpy as np
import data

img_path = "egg_photos/originals/DSC_3105.JPG"
img = cv2.imread(img_path)
coor_yolo = np.loadtxt(img_path[:-4] + ".txt")
coor = data.yolo_to_xy(coor_yolo, img.shape[0:2])
sub_imgs, dmaps = data.split_image_dmap(img, coor, "360p")
dmaps = dmaps[..., np.newaxis]
i = 0
while i < 10:
    r = np.random.randint(sub_imgs.shape[0])
    if np.sum(dmaps[r]) > 0:
        data.heat_map(sub_imgs[r], dmaps[r])
        i += 1

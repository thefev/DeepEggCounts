import cv2
import numpy as np
import data
img = cv2.imread("egg_photos/originals/DSC_2396.JPG")
coor_yolo = np.loadtxt("egg_photos/originals/DSC_2396.txt")
coor = data.yolo_to_xy(coor_yolo, img.shape[0:2])
sub_imgs, dmaps = data.split_image_dmap(img, coor, "360p")
dmaps = dmaps[..., np.newaxis]
# data.heat_map(sub_imgs[61], dmaps[61])

xt, yt, xv, yv = data.load_train_val_data()
yt = yt[..., np.newaxis]
yv = yv[..., np.newaxis]
data.heat_map(xt[148], yt[148])
data.heat_map(xv[81], yv[81])
for i in range(10):
    tr = np.random.randint(xt.shape[0])
    vr = np.random.randint(xv.shape[0])
    data.heat_map(xt[tr], yt[tr])
    data.heat_map(xv[vr], yv[vr])

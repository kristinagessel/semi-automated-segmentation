import os
from PIL import Image
import cv2
import numpy as np
import h5py

def create_subvolume(top_left_corner, width, height, z, start_slice, img_path, out_dir, dir_name):
    if not os.path.exists(out_dir + "/" + dir_name):
        os.mkdir(out_dir + "/" + dir_name)

    #This works:
    file = h5py.File(dir_name + "_test.hdf5", "w")
    dset = file.create_dataset("subvolume1_test", shape=(1, 250, 250), dtype=np.uint8)
    img_name = "0000.png"
    img = cv2.imread(img_path + "/" + img_name, 0)
    cropped_img = img[top_left_corner[1]:top_left_corner[1] + height, top_left_corner[0]:top_left_corner[0] + width]
    dset[0] = cropped_img
    cv2.imwrite(out_dir + "/" + dir_name + "/" + img_name, cropped_img)

    data = np.array(dset[0,:,:])

    im = Image.fromarray(data)
    im.show()

    #cv2.imwrite(str(z) + ".png", data)
    file.close()


output_dir = "/Volumes/Research/1. Research/Experiments/subvolumes"
path_to_imgs = "/Users/kristinagessel/Downloads"
#subvolume1 = [(22,1133), (272,1133), (272,1383), (22,1383)]
tl_corner = (22,1133)
read_path = "subvolume1.hdf5"


create_subvolume(tl_corner, 250, 250, 250, 0, path_to_imgs, output_dir, "subvolume1")
import glob
import cv2
import os
import h5py
import numpy as np
import argparse
from pathlib import Path

#TODO: Write metadata containing (x, y, z) start slice, etc.
def CreateSubvolume(top_left_corner, width, height, z, start_slice, img_path, out_dir, dir_name):
    out_path = out_dir +  "/" + dir_name + "_x" + str(top_left_corner[0]) + "_y" + str(top_left_corner[1]) + "_s" + str(start_slice)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    for slice in range(start_slice, z):
        img_name = str(slice).zfill(4) + ".tif"

        #Load as grayscale (0 as 2nd param allows this)
        img = cv2.imread(img_path + "/" + img_name, 0)

        #crop image to designated corner pts
        cropped_img = img[top_left_corner[1]:top_left_corner[1] + height, top_left_corner[0]:top_left_corner[0] + width]

        cv2.imwrite(out_path + "/" + img_name, cropped_img)
    return out_path

#Adapted to TIF based on a utility available with Januszewski et. al.'s Flood-Filling Network
def ToHDF5(img_dir, out_dir, ground_truth_mask_path):
    tif_files = glob.glob(img_dir + '/*.tif')
    tif_files.sort()
    images = [cv2.imread(i, 0) for i in tif_files]
    images = np.array(images)

    #current axes are (z, x, y). Shift them so they are (x, y, z).
    images = np.moveaxis(images, 0, -1)

    with h5py.File(out_dir + ".hdf5", 'w') as f:
        if ground_truth_mask_path == True:
            f.create_dataset('stack', data=images, compression='gzip', dtype='uint8')
        else:
            f.create_dataset('raw', data=images, compression='gzip', dtype='uint8')



parser = argparse.ArgumentParser(description="Create a subvolume with training data in HDF5 format.")
parser.add_argument("volume_path", type=str, help="Path to the volume from which to make a subvolume.")
parser.add_argument("output_path", type=str, help="Path to the output location.")
parser.add_argument("top_left_corner_x", type=int, help="x coordinate of the top left corner of the region of interest.")
parser.add_argument("top_left_corner_y", type=int, help="y coordinate of the top left corner of the region of interest.")
parser.add_argument("width", type=int, help="Width of subvolume.")
parser.add_argument("height", type=int, help="Height of subvolume.")
parser.add_argument("depth", type=int, help="Depth of subvolume.")
parser.add_argument("start_slice", type=int, help="Slice to start on.")
parser.add_argument("hdf5_name", type=str, help="Name of the resulting HDF5 file.")
parser.add_argument('-gt', "--is_ground_truth", help="Generating a ground truth subvolume?", action="store_true")
args = parser.parse_args()


out_path = CreateSubvolume((args.top_left_corner_x, args.top_left_corner_y), args.width, args.height, args.depth, args.start_slice, args.volume_path, args.output_path, args.hdf5_name)
ToHDF5(out_path, out_path, args.is_ground_truth)
#ReadHDF5(read_path, output_dir)
import glob
import cv2
import os
import h5py
import numpy as np
import argparse


def create_subvolume(top_left_corner, width, height, z, start_slice, img_path, out_dir, dir_name):
    if not os.path.exists(out_dir + "/" + dir_name):
        os.mkdir(out_dir + "/" + dir_name)

    for slice in range(start_slice, z):
        img_name = str(slice).zfill(4) + ".tif"

        #Load as grayscale (0 as 2nd param allows this)
        img = cv2.imread(img_path + "/" + img_name, 0)

        #crop image to designated corner pts
        cropped_img = img[top_left_corner[1]:top_left_corner[1] + height, top_left_corner[0]:top_left_corner[0] + width]

        cv2.imwrite(out_dir + "/" + dir_name + "/" + img_name, cropped_img)

#Adapted to TIF based on a utility available with Januszewski et. al.'s Flood-Filling Network
def to_hdf5(img_dir, out_dir, ground_truth_mask_path):
    tif_files = glob.glob(img_dir + '*.tif')
    tif_files.sort()
    images = [cv2.imread(i, 0) for i in tif_files]
    images = np.array(images)

    #current axes are (z, x, y). Shift them so they are (x, y, z).
    images = np.moveaxis(images, 0, -1)

    with h5py.File(out_dir + ".hdf5", 'w') as f:
        if ground_truth_mask_path:
            f.create_dataset('stack', data=images, compression='gzip', dtype='uint8')
        else:
            f.create_dataset('raw', data=images, compression='gzip', dtype='uint8')

#TODO: write out image is broken right now (openCV doesn't know how to make sense of the shifted axes (x, y, z))
def read_hdf5(path, out_dir):
    file = h5py.File(path, 'r')
    #set = file.keys()
    dset = file["raw"]

    test_save_dir = out_dir + "/testhdf5/"
    if not os.path.exists(test_save_dir):
        os.mkdir(test_save_dir)

    for z in range(0, 250):
        data = np.array(dset[:,:,z])
        img = data[:][:]
        cv2.imwrite(test_save_dir + str(z) + ".tif", img)



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
parser.add_argument("is_ground_truth", type=bool, default=False, help="Generating a ground truth subvolume?")
args = parser.parse_args()


create_subvolume((args.top_left_corner_x, args.top_left_corner_y), args.width, args.height, args.depth, args.start_slice, args.volume_path, args.output_path, args.hdf5_name)
to_hdf5(args.output_path + "/" + args.hdf5_name + "/", args.output_path + "/" + args.hdf5_name + "/" + args.hdf5_name, args.is_ground_truth)
#read_hdf5(read_path, output_dir)
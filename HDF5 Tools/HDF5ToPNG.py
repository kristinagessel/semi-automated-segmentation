import h5py
import os
import numpy as np
import cv2
import argparse

def ReadHDF5(path, out_dir):
    file = h5py.File(path, 'r')
    #set = file.keys()
    dset = file["seg"]

    test_save_dir = out_dir + "images/"
    if not os.path.exists(test_save_dir):
        os.mkdir(test_save_dir)

    for z in range(0, 250):
        data = np.array(dset[:,:,z], dtype=np.uint8)
        img = data[:][:]
        cv2.imwrite(test_save_dir + str(z) + ".png", img)



parser = argparse.ArgumentParser(description="Convert the contents of a HDF5 file into a stack of PNG images.")
parser.add_argument("hdf5_path", type=str, help="Full path to HDF5 file.")
parser.add_argument("output_path", type=str, help="Path to the output directory.")
args = parser.parse_args()

ReadHDF5(args.hdf5_path, args.output_path)
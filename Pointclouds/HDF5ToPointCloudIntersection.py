import h5py
import numpy as np
import cv2
import argparse


parser = argparse.ArgumentParser(description="Create a pointcloud openable in Meshlab that samples directly from the subvolume. (intersection)")
parser.add_argument("hdf5_path", type=str, help="Full path to HDF5 file.")
parser.add_argument("output_path", type=str, help="Path to the output directory.")
parser.add_argument("volume_path", type=str, help="Path to the directory containing the subvolume to sample color/gray values from.")
args = parser.parse_args()

#path to file:
path = args.hdf5_path
path_to_validation = args.volume_path

out_file = open(args.output_path, 'w')
f = h5py.File(path, 'r')
key_list = list(f.keys())
print(key_list)

dataset = f["seg"]
print(dataset.shape)
print(dataset.dtype)

num_color_dict = {}
pg_colors = []
obj_limit = 500

#TODO: this is broken, it's not sampling correctly from the volumes.
#record anything that's not 0 through all the layers (shape[2] is hopefully z, verify)
for z in range(0, dataset.shape[2]):
    slice_num = str(z).zfill(4)
    img = cv2.imread(path_to_validation + slice_num + ".tif")
    for y in range(0, dataset.shape[1]):
        for x in range(0, dataset.shape[0]):
            if dataset[y][x][z] != 0:
                #sample directly from volume
                color = img[y][x]
                out_file.write(str(x) + " " + str(y) + " " + str(z) + " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + "\n")
                #print(str(x) + " " + str(y) + " " + str(z) + " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + "\n")

f.close()
out_file.close()


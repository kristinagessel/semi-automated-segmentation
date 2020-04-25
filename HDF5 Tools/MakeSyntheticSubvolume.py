import argparse
import glob
import os

import numpy as np
import h5py

import cv2

'''
Given one hand-segmented ground truth slice, create a subvolume of specified size by shifting the image masks 1px left or right each slice.
'''

#Shift all pixels in the image right by n
def shift_right_n(img, n):
    shape = img.shape
    x_dims = shape[1]  # openCV is (y, x)
    y_dims = shape[0]
    im = img.copy()

    for iter in range(0, n):
        orig_copy = im.copy()
        for x in range(0, x_dims):
            for y in range(0, y_dims):
                # shift left or right depending on where we are
                # one side should be set all black, the pixels will 'move off' the image on the other side.
                # shift right
                # check we don't go out of image bounds; if we do, those pixels are gone.
                if x + 1 < x_dims:
                    im[y][x + 1] = orig_copy[y][x]
                if x == 0: #set the 'new' pixels to be black?
                    im[y][x] = 0
    return im

#Adapted to TIF based on a utility available with Januszewski et. al.'s Flood-Filling Network
def ToHDF5(img_dir, out_dir, is_gt):
    tif_files = glob.glob(img_dir + '/*.tif')
    tif_files.sort()
    images = [cv2.imread(i, 0) for i in tif_files]
    images = np.array(images)

    #current axes are (z, x, y). Shift them so they are (x, y, z).
    images = np.moveaxis(images, 0, -1)

    with h5py.File(os.path.join(out_dir, "ground_truth.hdf5"), 'w') as f:
        if is_gt:
            f.create_dataset('stack', data=images, compression='gzip', dtype='uint8')
        else:
            f.create_dataset('raw', data=images, compression='gzip', dtype='uint8')

parser = argparse.ArgumentParser(description="Make synthetic ground truth from a hand-annotated input image.")
parser.add_argument("annotation_path", type=str, help="Path to the annotated ground truth image.")
parser.add_argument("output_path", type=str, help="Path to the output location.")
parser.add_argument("depth", type=int, help="Depth of subvolume.")
parser.add_argument("shift_steps", type=int, help="How long to shift a direction.")
parser.add_argument('-gt', "--is_ground_truth", help="Generating a ground truth subvolume?", action="store_true")
args = parser.parse_args()

img = cv2.imread(args.annotation_path, 0)
im = img.copy()
shape = img.shape
x_dims = shape[1] #openCV is (y, x)
y_dims = shape[0]

shift_ctr = 0

for depth in range(0, args.depth):
    iter_copy = im.copy()
    if shift_ctr < args.shift_steps:
        im = shift_right_n(im, 1)
    else:
        #we're shifting 'back' left. Do some math based on the counter to see how far we've made it.
        disp_from_original = args.shift_steps - (shift_ctr - args.shift_steps)
        im = shift_right_n(img, disp_from_original)

    shift_ctr += 1
    cv2.imwrite(os.path.join(args.output_path, (str(depth).zfill(4) + ".tif")), im)
ToHDF5(args.output_path, args.output_path, args.is_ground_truth)

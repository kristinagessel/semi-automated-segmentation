import argparse
import ujson
import cv2
import glob
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description="Create a ground truth volume in HDF5 format.")
parser.add_argument("volume_path", type=str, help="Path to the volume from which to make a subvolume.")
parser.add_argument("output_path", type=str, help="Path to the output location.")
parser.add_argument("mask_pts_path", type=str, help="Path to instance segmentation text file (generated via InstanceFromAllFiles or TrainingMaskGenerator.)")
args = parser.parse_args()

f = open(args.mask_pts_path)
output = ujson.loads(f.read())

tif_files = glob.glob(args.volume_path + '*.tif')
tif_files.sort()

pg_colors = {}
assigned_colors = []

#assign colors
for page in output:
    # pick a (grayscale) color if this page hasn't been assigned a color
    if not page in pg_colors:
        color = np.random.choice(range(256))
        while color in assigned_colors:
            color = np.random.choice(range(256))
        pg_colors[page] = color
        assigned_colors.append(color)

#i is slice number
for i, img in enumerate(tif_files):
    im = cv2.imread(img, 0)
    img = np.zeros((im.shape[1], im.shape[0]), dtype="uint8")
    for page in output:
        #Check--pages will fall away as slices progress
        if str(i) in output[page]:
            pts = output[page][str(i)]
            for pt in pts:
                img[pt[1]][pt[0]] = pg_colors[page]
    #save the ground truth
    cv2.imwrite(args.output_path + "/" + str(i).zfill(4) + ".tif", img)

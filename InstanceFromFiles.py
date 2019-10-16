import os
import cv2
import glob
import json

import ujson

path_to_json_files = "/Volumes/Research/1. Research/Experiments/TrainingMasksAll/"
path_to_hi_res_images = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"

files = glob.glob(path_to_json_files + "/*.txt")
slices = glob.glob(path_to_hi_res_images + "/*.tif")

pg_pts = {}
for file in files:
    pg_file = os.path.basename(file)
    page = pg_file[:pg_file.find(".")]
    f = open(file)
    output = ujson.loads(f.read())  # get the dictionary
    f.close()
    pg_pts[page] = {}
    for slice in output:
        pg_pts[page][slice] = output[slice]
print("Finished loading.")
file = open(path_to_json_files + "instance_basic_avg/instance_pts.txt", "w")
ujson.dump(pg_pts, file, indent=1)

'''
#for each slice, read all the files... add appropriate points to the image, then close
for slice in slices:
    pg_pts = {}
    im = cv2.imread(path_to_hi_res_images + slice)
    for file in files:
        file = open(path_to_json_files + file)
        output = json.loads(file.read()) #get the dictionary
'''
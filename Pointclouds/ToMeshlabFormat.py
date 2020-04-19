import ujson
import glob
import os
import cv2
import argparse

'''
path_to_json_files = "/Volumes/Research/1. Research/Experiments/TrainingMasksAll/"
path_to_dense_files = "/Volumes/Research/1. Research/Experiments/meshlab/floodfill/"
path_to_raw_files = "/Volumes/Research/1. Research/Experiments/meshlab/vc_seg/"
path_to_raw_pointset_files = "/Volumes/Research/1. Research/Experiments/high_res_output/"
HI_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"


Read JSON
Convert from a text file (JSON) containing (x, y) coordinates for slices in a volume 
to a format readable by Meshlab, like (X Y Z R G B\n X Y Z R G B\n ...)

We have X and Y, Z increments by 1 each slice.
'''
def convertJSONToXYZ(path, out_path, volume_path):
    files = glob.glob(os.path.join(path, "*.txt"))

    for file in files:
        pg_file = os.path.basename(file)
        page = pg_file[:pg_file.find(".")]
        f = open(file)
        output = ujson.loads(f.read())  # get the dictionary
        f.close()

        out_file = open(os.path.join(out_path, str(page) + "_pointset.txt"), "w")
        z = 0 #TODO: change this to match the slice we're on.

        for slice in output:
            slice_num = slice.zfill(4)
            im = cv2.imread(volume_path + slice_num + ".tif")
            for pt in output[slice]:
                #we get a tuple (x, y)
                x = int(pt[0])
                y = int(pt[1])
                intensity = im[y][x]
                out_file.write(str(x) + " " + str(y) + " " + str(z) + " " + str(intensity[0]) + " " + str(intensity[1]) + " " + str(intensity[2]) + "\n")
            z += 1
        out_file.close()

parser = argparse.ArgumentParser(description="Create a subvolume with training data in HDF5 format.")
parser.add_argument("json_path", type=str, help="Full path to JSON file.")
parser.add_argument("output_path", type=str, help="Path to the output directory.")
parser.add_argument("volume_path", type=str, help="Path to the directory containing the volume to sample color/gray values from.")
args = parser.parse_args()

convertJSONToXYZ(args.json_path, args.output_path, args.volume_path)
#convertJSONToXYZ("/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/", "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/meshlab/", "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/")
#convertJSONToXYZ("/Volumes/Research/1. Research/Experiments/ExtrapolateMask/", "/Volumes/Research/1. Research/Experiments/meshlab/extrapolate_floodfill/")
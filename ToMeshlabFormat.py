import ujson
import glob
import os
import cv2

path_to_json_files = "/Volumes/Research/1. Research/Experiments/TrainingMasksAll/"
path_to_dense_files = "/Volumes/Research/1. Research/Experiments/meshlab/floodfill/"
path_to_raw_files = "/Volumes/Research/1. Research/Experiments/meshlab/vc_seg/"
path_to_raw_pointset_files = "/Volumes/Research/1. Research/Experiments/high_res_output/"
image_path = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"

HI_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"

'''
Read JSON
Convert to a format readable by Meshlab, like (X Y Z R G B\n X Y Z R G B\n ...)
We have X and Y, Z increments by 1 each slice.
'''
def convertJSONToXYZ(path, out_path):
    files = glob.glob(path + "*.txt")

    for file in files:
        pg_file = os.path.basename(file)
        page = pg_file[:pg_file.find(".")]
        f = open(file)
        output = ujson.loads(f.read())  # get the dictionary
        f.close()

        out_file = open(out_path + page + "_pointset.txt", "w")
        z = 0

        for slice in output:
            slice_num = slice.zfill(4)
            im = cv2.imread(image_path + slice_num + ".tif")
            for pt in output[slice]:
                #we get a tuple (x, y)
                x = int(pt[0])
                y = int(pt[1])
                intensity = im[y][x]
                out_file.write(str(x) + " " + str(y) + " " + str(z) + " " + str(intensity[0]) + " " + str(intensity[1]) + " " + str(intensity[2]) + "\n")
            z += 1
        out_file.close()

convertJSONToXYZ("/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/", "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/meshlab/")
#convertJSONToXYZ(path_to_raw_pointset_files, path_to_raw_files)
#convertJSONToXYZ(path_to_json_files, path_to_dense_files)
#convertJSONToXYZ("/Volumes/Research/1. Research/Experiments/ExtrapolateMask/", "/Volumes/Research/1. Research/Experiments/meshlab/extrapolate_floodfill/")
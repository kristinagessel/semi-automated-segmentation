import ujson
import glob
import os
import cv2
import argparse

'''
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

        for slice in output:
            slice_num = slice.zfill(4)
            im = cv2.imread(volume_path + slice_num + ".tif")
            for pt in output[slice]:
                #we get a tuple (x, y)
                x = int(pt[0])
                y = int(pt[1])
                intensity = im[y][x]
                out_file.write(str(x) + " " + str(y) + " " + str(slice) + " " + str(intensity[0]) + " " + str(intensity[1]) + " " + str(intensity[2]) + "\n")
        out_file.close()

parser = argparse.ArgumentParser(description="Convert a JSON set of points into something Meshlab can load-- XYZRGB newline format.")
parser.add_argument("json_path", type=str, help="Full path to JSON file.")
parser.add_argument("output_path", type=str, help="Path to the output directory.")
parser.add_argument("volume_path", type=str, help="Path to the directory containing the volume to sample color/gray values from.")
args = parser.parse_args()

convertJSONToXYZ(args.json_path, args.output_path, args.volume_path)
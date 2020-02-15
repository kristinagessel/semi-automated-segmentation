import os
import glob
import ujson
import argparse

'''
A quick script to just read in all the independent page JSON files, and output an instance file 
that has all the page points combined. When doing many pages, sometimes the dictionary
will get too big to hold in memory and TrainingMaskGenerator will get killed before the instance part can complete.
'''

parser = argparse.ArgumentParser(description="Read all independent JSON files in a directory, and output an instance file that combines them all.")
parser.add_argument("path_to_json_files", type=str, help="Path to the directory containing all the JSON files.")
parser.add_argument("instance_file_name", type=str, help="Name of the instance file.")
args = parser.parse_args()

path_to_json_files = "/Volumes/Research/1. Research/Experiments/TrainingMasksAll/instance/"
files = glob.glob(args.path_to_json_files + "*.txt")

pg_pts = {}
for file in files:
    pg_file = os.path.basename(file)

    #Use slice to just get the page name from the JSON file's name
    page = pg_file[:pg_file.find(".")]
    f = open(file)
    output = ujson.loads(f.read())  # get the dictionary
    f.close()
    pg_pts[page] = {}
    for slice in output:
        pg_pts[page][slice] = output[slice]
print("Finished loading JSON files.")
file = open(path_to_json_files + "/instance.txt", "w")
ujson.dump(pg_pts, file, indent=1)


import argparse
import glob
import os
import ujson

#load all pointsets in a directory
#put them into the same dictionary
#save the dictionary as a separate txt file



def merge_txt_files(path):
    pts = {}
    files = sorted(glob.glob(os.path.join(path, os.path.join("parts", "*.txt"))))
    for file in files:
        f = open(file)
        temp_pts = ujson.load(f)
        for pt in temp_pts:
            pts[pt] = temp_pts[pt]
    complete_file = open(os.path.join(path, "complete.txt"), 'w')
    complete_file.write(ujson.dumps(pts))

def merge_skeleton_files(path):
    pts = {}
    files = sorted(glob.glob(os.path.join(path, os.path.join("skeleton", "*.txt"))))
    for file in files:
        f = open(file)
        temp_pts = ujson.load(f)
        for pt in temp_pts:
            pts[pt] = temp_pts[pt]
    complete_file = open(os.path.join(path, "complete-skeleton.txt"), 'w')
    complete_file.write(ujson.dumps(pts))


parser = argparse.ArgumentParser(description="Merge multiple pointsets belonging to the same page into one (JSON) file.")
parser.add_argument("pointset_path", type=str, help="Full path to the pointset directory with all partial files.")
args = parser.parse_args()

merge_txt_files(args.pointset_path)
merge_skeleton_files(args.pointset_path)
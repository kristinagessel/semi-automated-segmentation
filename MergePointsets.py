import glob
import os
import ujson

#load all pointsets in a directory
#put them into the same dictionary
#save the dictionary as a separate txt file



def merge_txt_files(path):
    pts = {}
    files = glob.glob(os.path.join(path, "*.txt"))
    for file in files:
        f = open(file)
        temp_pts = ujson.load(f)
        for pt in temp_pts:
            pts[pt] = temp_pts[pt]
    complete_file = open(os.path.join(path, "complete.txt"), 'w')
    complete_file.write(ujson.dumps(pts))


merge_txt_files("/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/?/new_pruning/all_txt")
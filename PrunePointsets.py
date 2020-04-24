import argparse
import glob
import os

import ujson


def prune_pointset(path, bound):
    pts = {}
    new_dict = {}
    files = glob.glob(path)
    for file in files:
        f = open(file)
        temp_pts = ujson.load(f)
        new_dict = {k:temp_pts[k] for k in temp_pts if int(k) <= int(bound) and k in temp_pts}
    return new_dict

parser = argparse.ArgumentParser(description="Prune pointsets to a specified slice number to eliminate slices with errors.")
parser.add_argument("pointset_path", type=str, help="Full path to the pointset")
parser.add_argument("save_path", type=str, help="Path to the output directory.")
parser.add_argument("upper_bound", type=int, help="Upper bound of the pointset to keep.")
args = parser.parse_args()

dict = prune_pointset(args.pointset_path, args.upper_bound)
pruned_file = open(args.save_path, 'w+')
ujson.dump(dict, pruned_file, indent=1)

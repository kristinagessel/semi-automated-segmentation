import argparse
import glob
import os
import re
import VCTools.VCPSReader as vr

import ujson


def prune_pointset(path, bound):
    pts = {}
    new_dict = {}
    file = glob.glob(path)[0]
    f = open(file)
    temp_pts = ujson.load(f)
    new_dict = {k:temp_pts[k] for k in temp_pts if int(k) <= int(bound) and k in temp_pts}
    return new_dict

def auto_prune_pointset(path, is_skeleton):
    for dir in os.listdir(path):
        #get the number in "toxx?x?x?"
        regex = re.compile(r'\d+')
        print(dir)
        numbers = regex.findall(dir)
        if(not len(numbers) == 0 and dir.find(".txt") == -1):
            #try:
            file_path = None
            slice_num = numbers[1]  # Always the second set of numbers...
            if not is_skeleton:
                file_path = glob.glob(os.path.join(os.path.join(path, dir), slice_num + ".txt"))
                if len(file_path) == 0:
                    file_path = glob.glob(os.path.join(os.path.join(path, dir), "mask_pointset.vcps"))[0]
                    pointset = vr.VCPSReader(file_path).process_unordered_VCPS_file({})
                    file = open(os.path.join(os.path.join(path, dir), slice_num + "_mask.txt"), "w")
                    file.write(ujson.dumps(pointset, indent=1))
                    file.close()
                    file_path = os.path.join(os.path.join(path, dir), slice_num + "_mask.txt")
                else:
                    file_path = file_path[0]
            else:
                file_path = glob.glob(os.path.join(os.path.join(path, dir), r'*skeleton*.txt'))
                if len(file_path) == 0:
                    file_path = glob.glob(os.path.join(os.path.join(path, dir), "pointset.vcps"))[0]
                    pointset = vr.VCPSReader(file_path).process_unordered_VCPS_file({})
                    file = open(os.path.join(os.path.join(path, dir), slice_num + "_skeleton.txt"), "w")
                    file.write(ujson.dumps(pointset, indent=1))
                    file.close()
                    file_path = os.path.join(os.path.join(path, dir), slice_num + "_skeleton.txt")
                else:
                    file_path = file_path[0]
                pruned = prune_pointset(file_path, slice_num)
                pruned_file = open(os.path.join(args.save_path, slice_num + ".txt"), 'w+')
                ujson.dump(pruned, pruned_file, indent=1)
            #except:
            #    print("Error with directory " + dir)


parser = argparse.ArgumentParser(description="Prune pointsets to a specified slice number to eliminate slices with errors.")
parser.add_argument("pointset_path", type=str, help="Full path to the pointset")
parser.add_argument("save_path", type=str, help="Path to the output directory.")
parser.add_argument("--skeleton", action="store_true", help="Prune skeleton txt files instead of mask files.")
args = parser.parse_args()

#dict = prune_pointset(args.pointset_path, args.upper_bound)
dict = auto_prune_pointset(args.pointset_path, args.skeleton)




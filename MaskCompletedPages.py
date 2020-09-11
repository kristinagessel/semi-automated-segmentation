import argparse
import os
import ujson

parser = argparse.ArgumentParser(description="Merge all vcps/.txt files for a given page, and save each individual page independently.")
parser.add_argument("path_to_pg_dir", type=str, help="Path to the directory containing all the page directories.")
parser.add_argument("instance_file_path", type=str, help="Name of the instance file.")
args = parser.parse_args()

dirs = [ item for item in os.listdir(args.path_to_pg_dir) if os.path.isdir(os.path.join(args.path_to_pg_dir, item))]
page_dirs = []

#TODO: WE ONLY CARE ABOUT THE VERY FIRST SLICE FOR THIS USE CASE AND SKELETONS ARE FINE, DON'T NEED MASKS
pg_masks = {}

#Keep only numeric directories
for dir in dirs:
    try:
        int(dir)
        page_dirs.append(dir)
        pg_masks[dir] = {}
    except:
        continue

#Load all pointsets into one dictionary file
for dir in page_dirs:
    print(dir)
    #look for the 'pointclouds' directory
    path = os.path.join(args.path_to_pg_dir, dir)
    pointset_path = os.path.join(path, "txt")
    #find complete_pointset (this is the merged file)
    complete_pointset = os.path.join(pointset_path, "complete-skeleton.txt")
    f = open(complete_pointset)
    #read whole file
    output = ujson.loads(f.read())
    #keys = output.keys()
    first_key = next(iter(output))
    pg_masks[dir] = output[first_key]
    f.close()

file = open(os.path.join(args.instance_file_path, "instance.txt"), "w")
ujson.dump(pg_masks, file, indent=1)



#types = ('*.txt', '*.vcps')
#found_files = []
#for files in types:
#    found_files.extend(glob.glob(os.path.join(args.path_to_json_files, files)))
import VCPSReader as vr
import json
import os
import cv2
import re

PATH_TO_OUTPUT_DIR = "/Volumes/Research/1. Research/1. Iowa Examples/VC Renders/"#"/Volumes/Research/1. Research/Experiments/"
PATH_TO_VOLPKG = "/Volumes/Research/1. Research/"

HI_RES_PATH = PATH_TO_VOLPKG + "MS910.volpkg/volumes/20180122092342"
LOW_RES_PATH = PATH_TO_VOLPKG + "MS910.volpkg/volumes/20180220092522"
PATH_TO_HI_RES_WORK_DONE = "/Volumes/Research/1. Research/1. Iowa Examples/VC Renders/" #PATH_TO_VOLPKG + "MS910.volpkg/work-done/hi-res/"
PATH_TO_LOW_RES_WORK_DONE = PATH_TO_VOLPKG + "MS910.volpkg/work-done/low-res/"

# If you want to read points from a json file instead of parse all over again:
path_to_high_res_json = "/Users/kristinagessel/Desktop/ProjectExperiments/VCPoints/high_res_output/"
path_to_low_res_json = "/Users/kristinagessel/Desktop/ProjectExperiments/VCPoints/low_res_output/"
READ_JSON = False

INCLUDE_TEARS = True

class Extractor:
    def __init__(self):
        self.path_to_output_dir = PATH_TO_OUTPUT_DIR

    #Write to text file in JSON format for simplicity
    def write_to_txt_file(self, points, name):
        file = open(name + "_output.txt", "w")
        file.write(json.dumps(points, indent=1))

    #img_path: path to the image of this slice
    #slice_num: what slice number is this?
    #points: array of all the points (in tuple format) at that slice number
    def draw_on_img(self, img_path, slice_num, points, page, include_tears):
        im = cv2.imread(img_path)
        radius = 2

        for point in points:
            x = int(point[0])
            y = int(point[1])
            if not include_tears:
                pixel = im[y][x] #pixel access is BACKWARDS (y, x)
                if(pixel[0] > 30): #check for tears TODO: improve this
                    cv2.circle(im, (x, y), radius, 255, -1)
            else:
                cv2.circle(im, (x, y), radius, 255, -1)
        if not os.path.exists(self.path_to_output_dir + "DrawnImages/" + page + "/"):
            os.mkdir(self.path_to_output_dir + "DrawnImages/" + page + "/")
        cv2.imwrite(self.path_to_output_dir + "DrawnImages/" + page + "/" + str(slice_num) + ".tif", im)


    def is_int(self, var):
        try:
            int(var)
            return True
        except ValueError:
            return False

    def find_all_segmentations(self, path_to_segs):
        seg_dict = {}
        page_dirs = []

        #Get all the work done in low/high-res (split into pages)
        page_regex = re.compile("^[\d+]+(-\d+)?")
        for filename in os.listdir(path_to_segs):
            if page_regex.match(filename):
                page_dirs.append(filename)
        #for dirpath, dirnames, filenames in os.walk(path_to_segs):
            #folders have a naming convention -- usually either all ints (high res) or sometimes int "-" int in low res
        #    page_dirs = page_dirs + ([dirname for dirname in dirnames if page_regex.match(dirname)])

        for page in page_dirs:
            #add to the dictionary for each page found in the segmentation folder
            seg_dict = self.find_all_segmentations_for_pg(path_to_segs + page, page, seg_dict)
        return  seg_dict

    def find_all_segmentations_for_pg(self, path_to_pg, pg_num, seg_dict):
        segmentation_dirs = []

        #now get into the page directories and pick out all the segmentations that exist for each page
        #save in a dictionary as (page num: seg_1, seg_2, ...)
        seg_regex = re.compile("\d{14}")  # search for segmentation directories YYYYMMDDHHMMSS format
        #TODO: redundant:
        for dirpath, dirnames, filenames in os.walk(path_to_pg):
            segmentation_dirs = segmentation_dirs + ([dirname for dirname in dirnames if seg_regex.match(dirname)])

        for dir in segmentation_dirs:
            if pg_num in seg_dict:
                seg_dict[pg_num].append(dir)
            else:
                seg_dict[pg_num] = []
                seg_dict[pg_num].append(dir)
        return seg_dict

    def condense_segmentations(self, page, list_of_segs, path_to_segs):
        output = {} #will hold the merged version of all the individual pointsets
        for seg in list_of_segs:
            output = vr.VCPSReader(path_to_segs + seg + "/pointset.vcps").process_VCPS_file(output)
        return output


    def get_path_to_slice(self, slice_num, path_to_directory):
        slice_num = int(slice_num) #cast to int to match --handles the oddball cases where sometimes it's xxx.yyy where y > 0...
        for file in os.listdir(path_to_directory):
            filename = file[:file.find(".")]
            if self.is_int(filename): #handle the meta.json present in this folder
                filename = int(filename)
                if filename == slice_num:
                    return path_to_directory + "/" + str(file)

#-------
'''
def main():
    ex = Extractor()

    # page num : seg num
    pages = {
        "MS910": {
            "1": ["20191114132257", 0],  # segmentation #, start slice
            "2": ["20191125215208", 0],
            "3": ["20191114133552", 0],
            "11": ["20191123203022", 0],
            "?": ["20191126112158", 0],
            "??": ["20191126122204", 0],
            "???": ["20191126132825", 0]
        },
        "Paris59": {
            "test": ["20191203113358", 430],  # segmentation #, start slice
            "stephen1" : ["20190808124935", 725]
        }
    }

    object = "Paris59"
    page = "stephen1"
    segmentation_number = pages[object][page][0]

    paths = {
        "MS910": {
            "high-res": "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/",
            "low-res": "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180220092522/",
            "pointset": "/Volumes/Research/1. Research/MS910.volpkg/paths/" + segmentation_number,
            "save": "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/"
        },
        "Paris59": {
            "high-res": "/Volumes/Research/1. Research/Herculaneum/PHercParisObject59/PHercParisObjet59.volpkg/volumes/20190910090730/",
            "pointset": "/Volumes/Research/1. Research/Herculaneum/PHercParisObject59/PHercParisObjet59.volpkg/paths/" + segmentation_number,
            "save": "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/Paris59/"
        }
    }

    start_slice = pages[object][page][1]
    num_iterations = 15

    volume_path = paths[object]["high-res"]
    pointset_path = paths[object]["pointset"]
    save_path = paths[object]["save"]

    output_dictionary = {}
    output_dictionary = vr.VCPSReader(pointset_path + "/pointset.vcps").process_VCPS_file(output_dictionary)
    points_of_interest = output_dictionary[start_slice]
    ex.write_to_txt_file(output_dictionary, page)
    ex.draw_on_img(ex.get_path_to_slice(start_slice, volume_path), start_slice, points_of_interest, page, INCLUDE_TEARS)
'''

def main():
    path_to_volumetric_data = HI_RES_PATH
    path_to_work_done = PATH_TO_HI_RES_WORK_DONE

    ex = Extractor()

    seg_dict = ex.find_all_segmentations(path_to_work_done)

    if READ_JSON:
        print("Reading JSON...")
        for page in seg_dict:
            file = open(path_to_high_res_json + page + "_output.txt")
            output_dictionary = json.loads(file.read())

            for slice_num in output_dictionary:  # for key in dictionary
                points_of_interest = output_dictionary[slice_num]  # Look for the slice num in the dictionary
                ex.draw_on_img(ex.get_path_to_slice(slice_num, path_to_volumetric_data), slice_num, points_of_interest, page, INCLUDE_TEARS)
    else:
        for page in seg_dict:
            output_dictionary = ex.condense_segmentations(page, seg_dict[page], path_to_work_done+page+"/")
            ex.write_to_txt_file(output_dictionary, page)

            for slice_num in output_dictionary:  # for key in dictionary
                points_of_interest = output_dictionary[slice_num]  # Look for the slice num in the dictionary
                ex.draw_on_img(ex.get_path_to_slice(slice_num, path_to_volumetric_data), slice_num, points_of_interest, page, INCLUDE_TEARS)

main()

'''
Take the dictionary from JsonReader and use it to do a
flood fill threshold thing to make masks for each page that we have work done for
'''
import json

import cv2
import os
import numpy as np
import math

from Extractor import Extractor
from JsonReader import JsonReader
import matplotlib.pyplot as plt


class TrainingMaskGenerator:
    def __init__(self):
        self.HI_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"
        self.LOW_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180220092522/"
        self.path_to_high_res_json = "/Users/kristinagessel/Desktop/ProjectExperiments/high_res_output/"
        self.path_to_low_res_json = "/Users/kristinagessel/Desktop/ProjectExperiments/low_res_output/"
        self.img_path = self.HI_RES_PATH
        self.low_tolerance = 65
        self.high_tolerance = 255 #we don't want to pick up the minerals which show up as a bright white

    def generate_mask_for_pg(self, page_num):
        reader = JsonReader()
        output = reader.read(self.path_to_high_res_json, page_num, "_output.txt")
        pg_seg_pixels = self.do_floodfill(output, page_num)#filtered_output, page_num)
        #TODO: later use the pg_seg_pixels returned by do_floodfill to stitch all the pages together in one slice image?

        #TODO: eventually... to do instance segmentation we need polygons, not just points. When we get to that point, make the polygons with ONLY directly connected pixels. There are some weird straggler points not connected
        #to the main group of points
        return pg_seg_pixels

    #floodfill for a whole page
    def do_floodfill(self, seg_pts, page):
        iter = 0
        seg_pixels = {}

        for slice in seg_pts:
            slice_num = slice.zfill(4)
            stack = []
            visited = []
            im = cv2.imread(self.img_path + slice_num + ".tif")
            for elem in seg_pts[slice]:
                x = int(elem[0])
                y = int(elem[1])
                pixel = im[y][x][0]
                if(pixel > 30): #check for tears TODO: better filter params?
                    stack.append(((x, y), (x, y))) #append a tuple of ((point), (origin point)) to keep track of how far we are from the original point

            start_pts = stack.copy()
            height, width, channels = im.shape

            avg = int(self.calc_avg_pg_width(start_pts, im)) #avg width might change some in different slices?

            while stack:
                point = stack.pop()
                visited.append(point[0]) #only the point itself, don't care about the parent it came from
                x = int(point[0][0]) # point of interest's x (not origin point's x)
                y = int(point[0][1])
                im[y][x] = (255, 0, 0) #make the visited point blue
                valid_neighbors= self.floodfill_check_neighbors(im, point, height, width, avg)
                for pt in valid_neighbors:
                    if pt not in visited and (tuple(pt), tuple(point[1])) not in stack:
                        stack.append((tuple(pt), tuple(point[1]))) #append a tuple of form ((point), (origin point for parent point))

            seg_pixels[slice] = visited #put the list of all the pixels that make up the page into the dictionary to be returned at the end so we can use them all together if we want
            if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page):
                os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page)
            cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page + "/" + str(slice) + "_mask" + "_avg=" + str(avg) + ".tif",
                im)
        return seg_pixels


    #store the 'distance' in the stack too?
    def calc_avg_pg_width(self, pts, im):
        pt_counts = []
        for pt in pts:
            x_pos = pt[0][0]
            y_pos = pt[0][1]
            #from this pt, go left and right from the pixel. Get this length.
            grey_val = im[y_pos][x_pos][0]  # pixel access is BACKWARDS--(y,x)
            length_ctr = 1

            #go left
            while grey_val > self.low_tolerance and grey_val < self.high_tolerance:
                x_pos -= 1
                grey_val = im[y_pos][x_pos][0]
                length_ctr += 1

            #go right
            while grey_val > self.low_tolerance and grey_val < self.high_tolerance:
                x_pos += 1
                grey_val = im[y_pos][x_pos][0]
                length_ctr += 1
            pt_counts.append(length_ctr)
        pt_counts.sort()
        #pt_counts = pt_counts[2:len(pt_counts)-6] #TODO: do we need to throw out outliers?

        if math.isnan(np.average(pt_counts)):
            return 0
        return np.average(pt_counts)



    #TODO: could set a limit for how far the flood fill can reach out to prevent flood fill from crossing into a connected neighboring page?
    # (thicknesses vary between pages because it's vellum, but the same page is generally the same thickness except where tears are)
    # That way if some are extra long, we can throw them out and then give the flood fill a limit of how much it can expand by the average
    # Maybe even calculate the distance between each point so we can set a sort of oval boundary
    def floodfill_check_neighbors(self, im, pixel, height, width, avg_width):
        valid_neighbors = []
        x = [-1, 0, 1]
        y = [-1, 0, 1]
        x_pos = pixel[0][0]
        y_pos = pixel[0][1]
        for i in y: #height
            for j in x: #width
                if (y_pos != 0 or x_pos != 0) and x_pos + j < width-1 and y_pos + i < height-1 and self.calculate_distance_from_origin(pixel[0], pixel[1]) <= avg_width/2: #Don't want the center pixel or any out of bounds
                    grey_val = im[y_pos + i][x_pos + j][0] #pixel access is BACKWARDS--(y,x)
                    if grey_val > self.low_tolerance and grey_val < self.high_tolerance:
                        valid_neighbors.append((x_pos + j, y_pos + i))
        return valid_neighbors


    #what distance metric?
    #for now, try just distance in the x direction?
    #TODO: maybe later make a 'rectangle' of eligible space where x bound is average width and y bound is the y coord of the next point...?
    #TODO: try Euclidean?
    def calculate_distance_from_origin(self, point, origin):
        return abs(point[0] - origin[0])

    def create_semantic_training_set(self, page_points):
        master = {}
        #merge all the dictionaries we have
        for page in page_points:
            for slice in page_points[page]:
                if slice in master:
                    master[slice] = page_points[page][slice]
                else:
                    master[slice] = []
                    master[slice].append(page_points[page][slice])
        for slice in master:
            slice_num = slice.zfill(4)
            im = cv2.imread(self.img_path + slice_num + ".tif")
            for pt in master[slice]:
                x = int(pt[0]) # point of interest's x (not origin point's x)
                y = int(pt[1])
                im[y][x] = (255, 0, 0) #make the visited point blue
            if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/total"):
                os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/total")
            cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/total/" + str(
            slice) + "/_semantic_mask" + ".tif",
                    im)



def main():
    PATH_TO_HI_RES_WORK_DONE = "/Volumes/Research/1. Research/MS910.volpkg/work-done/hi-res/"
    PATH_TO_LOW_RES_WORK_DONE = "/Volumes/Research/1. Research/MS910.volpkg/work-done/low-res/"
    path_to_work_done = PATH_TO_HI_RES_WORK_DONE
    page_segs = {}

    gen = TrainingMaskGenerator()
    ex = Extractor()
    seg_dict = ex.find_all_segmentations(path_to_work_done)

    page_segs["1"] = gen.generate_mask_for_pg("1")
    page_segs["15"] = gen.generate_mask_for_pg("15")
    #for page in seg_dict:
    #    page_segs[page] = gen.generate_mask_for_pg(page)

    gen.create_semantic_training_set(page_segs)

main()
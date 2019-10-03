import json

import cv2
import os
import numpy as np
import math

from DrawPoints import Extractor
from JsonReader import JsonReader
import matplotlib.pyplot as plt
from scipy.spatial import distance


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
        pg_seg_pixels, filtered_pts_dict = self.do_floodfill(output, page_num)
        #TODO: eventually... to do instance segmentation we might need polygons, not just points. When we get to that point, make the polygons with ONLY directly connected pixels. There are some weird straggler points that aren't connected in some cases
        return pg_seg_pixels, filtered_pts_dict

    #floodfill for a whole page
    def do_floodfill(self, seg_pts, page):
        seg_pixels = {}
        start_pt_dict = {}

        for slice in seg_pts:
            slice_num = slice.zfill(4)
            stack = []
            visited = []
            start_pts = []
            im = cv2.imread(self.img_path + slice_num + ".tif")
            for elem in seg_pts[slice]:
                x = int(elem[0])
                y = int(elem[1])
                pixel = im[y][x][0]
                if(pixel > self.low_tolerance and pixel < self.high_tolerance): #check for tears and bright spots (minerals?)
                    stack.append(((x, y), (x, y))) #append a tuple of ((point), (origin point)) to keep track of how far we are from the original point
                    start_pts.append((x,y))
                    if slice not in start_pt_dict:
                        start_pt_dict[slice] = []
                    start_pt_dict[slice].append((x,y))

            #start_pts = stack.copy()
            height, width, channel = im.shape

            avg = int(self.calc_avg_pg_width(start_pts, im)) #avg width might change some in different slices?

            while stack:
                point = stack.pop()
                visited.append(point[0])
                x = int(point[0][0]) # point of interest's x (not origin point's x)
                y = int(point[0][1])
                im[y][x] = (255, 0, 0) #make the visited point blue
                valid_neighbors = self.floodfill_check_neighbors(im, point, height, width, avg)
                for pt in valid_neighbors:
                    if pt not in visited and tuple(pt) not in (i[0] for i in stack):
                        stack.append((tuple(pt), tuple(point[1]))) #append a tuple of form ((point), (parent's origin point))

            seg_pixels[slice] = visited #put the list of all the pixels that make up the page into the dictionary to be returned at the end so we can use them all together if we want
            if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page):
                os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page)
            cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page + "/" + str(slice) + "_mask" + "_avg=" + str(avg) + ".tif",
                im)
        return seg_pixels, start_pt_dict


    #store the 'distance' in the stack too?
    def calc_avg_pg_width(self, pts, im):
        pt_counts = []
        for pt in pts:
            x_pos = pt[0]
            y_pos = pt[1]
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
        #pt_counts.sort()
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
                if (y_pos != 0 or x_pos != 0) and x_pos + j < width-1 and y_pos + i < height-1 and self.calculate_distance_from_origin(pixel[0], pixel[1]) <= math.ceil(avg_width/2): #Don't want the center pixel or any out of bounds
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
                if slice not in master:
                    master[slice] = []
                for pt in page_points[page][slice]:
                    master[slice].append(pt)
        for slice in master:
            slice_num = slice.zfill(4)
            im = cv2.imread(self.img_path + slice_num + ".tif")
            for pt in master[slice]:
                x = int(pt[0])
                y = int(pt[1])
                im[y][x] = (255, 0, 0)
            if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/semantic_basic_avg"):
                os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/semantic_basic_avg")
            cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/semantic_basic_avg/" + str(slice) + "_semantic_mask" + ".tif", im)
        return master

    def create_instance_training_set(self, page_points, orig_pts):
        master = {}
        pg_colors = {}
        pg_ctr = 0
        #need to keep page specific data separate, but condense under the same slice
        for page in page_points:
            # select a random color for this page
            color = tuple(np.random.choice(range(256), size=3))
            if color not in pg_colors:  # we want a brand new color for each page
                pg_colors[page] = color
            else:
                while color in pg_colors:
                    color = tuple(np.random.choice(range(256), size=3) * 256)

            #put all the points into the master dictionary in the format we want
            for slice in page_points[page]:
                if slice not in master:
                    master[slice] = {}
                if page not in master[slice]: # slice: page: [pts]
                    master[slice][page] = []
                for pt in page_points[page][slice]:
                    if pt not in master[slice][page]: #TODO: looks like there are some duplicates in page_points, look into where that is happening
                        master[slice][page].append(pt)
                    else:
                        print("duplicate")
        for slice in master:
            slice_num = slice.zfill(4)
            im = cv2.imread(self.img_path + slice_num + ".tif")
            for page in master[slice]:
                color = pg_colors[page]
                for pt in master[slice][page]:
                    x = pt[0]
                    y = pt[1]

                    if tuple(im[y][x]) in pg_colors.values() and tuple(im[y][x]) != pg_colors[page]: #if it's a color we picked for a previous page, we have a 'merge conflict'. Decide which page this pixel belongs to by checking which page has the nearest point.
                        current_pg = page
                        conflict_pg = self.find_conflict_pg(tuple(im[y][x]), pg_colors)
                        current_pg_pt, conflict_pg_pt = self.find_closest_pt(current_pg, conflict_pg, (x, y), orig_pts, slice)
                        print("instance seg merge conflict between pages ", page, " and ", conflict_pg, ": ", x, ", ", y, " slice ", slice)
                        closest_color = self.compare_distances_to_px(current_pg_pt, color, conflict_pg_pt, tuple(im[y][x]), (x, y))
                        im[y][x] = closest_color
                        if closest_color == pg_colors[page]: #if we did change the color, reflect that in the master list
                            master[slice][conflict_pg].remove(pt)
                            master[slice][current_pg].append(pt)
                    else:
                        im[y][x] = color
            if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/instance_basic_avg"):
                os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/instance_basic_avg")
            cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/instance_basic_avg/" + str(slice) + "_instance_mask" + ".tif", im)
            #TODO: when we actually want to use this, we will have to find a way to create polygons out of these points
        return master


    #Find each page's closest point to the conflicted pixel
    def find_closest_pt(self, curr_pg, conflict_pg, pixel_loc, orig_pts, slice):
        curr_pg_pts = orig_pts[curr_pg][slice]
        conflict_pg_pts = orig_pts[conflict_pg][slice]
        pixel_loc = np.array(pixel_loc)
        curr_pg_pts = np.asarray(curr_pg_pts)
        conflict_pg_pts = np.asarray(conflict_pg_pts)

        closest_index_curr_pg = distance.cdist([pixel_loc], curr_pg_pts).argmin()
        closest_index_conflict_pg = distance.cdist([pixel_loc], conflict_pg_pts).argmin()

        return curr_pg_pts[closest_index_curr_pg], conflict_pg_pts[closest_index_conflict_pg]

    '''
    Calculate the Euclidean distance between the nearest candidate point for both pages and see which one is closer.
    (If they're equal, just leave it alone...)
    '''
    def compare_distances_to_px(self, pg1_pt, pg1_color, pg2_pt, pg2_color, pixel_pt):
        pg1_x_delta = abs(pixel_pt[0] - int(pg1_pt[0]))
        pg1_y_delta = abs(pixel_pt[1] - int(pg1_pt[1]))
        pg1_distance = math.sqrt(pg1_x_delta ** 2 + pg1_y_delta ** 2)

        pg2_x_delta = abs(pixel_pt[0] - int(pg2_pt[0]))
        pg2_y_delta = abs(pixel_pt[1] - int(pg2_pt[1]))
        pg2_distance = math.sqrt(pg2_x_delta ** 2 + pg2_y_delta ** 2)

        if pg1_distance < pg2_distance:
            return pg1_color
        else:
            return pg2_color


    def find_conflict_pg(self, color, pg_colors):
        for pg in pg_colors:
            if pg_colors[pg] == color:
                return pg
        return "unknown(error)"

def main():
    PATH_TO_HI_RES_WORK_DONE = "/Volumes/Research/1. Research/MS910.volpkg/work-done/hi-res/"
    PATH_TO_LOW_RES_WORK_DONE = "/Volumes/Research/1. Research/MS910.volpkg/work-done/low-res/"
    path_to_work_done = PATH_TO_HI_RES_WORK_DONE
    page_segs = {}
    filtered_pts = {}

    gen = TrainingMaskGenerator()

    #page_segs["1"], filtered_pts["1"] = gen.generate_mask_for_pg("1")
    page_segs["2"], filtered_pts["2"] = gen.generate_mask_for_pg("2")
    page_segs["3"], filtered_pts["3"] = gen.generate_mask_for_pg("3")
    #page_segs["4"], filtered_pts["4"] = gen.generate_mask_for_pg("4")
    #page_segs["5"], filtered_pts["5"] = gen.generate_mask_for_pg("5")
    #page_segs["15"], filtered_pts["15"] = gen.generate_mask_for_pg("15")
    #page_segs["16"], filtered_pts["16"] = gen.generate_mask_for_pg("16")


    #The following pages are incomplete:
    #page_segs["17"], filtered_pts["17"] = gen.generate_mask_for_pg("17")
    #page_segs["18"], filtered_pts["18"] = gen.generate_mask_for_pg("18")
    #for page in seg_dict:
    #    page_segs[page] = gen.generate_mask_for_pg(page)

    semantic_master = gen.create_semantic_training_set(page_segs)
    file = open("/Volumes/Research/1. Research/Experiments/TrainingMasks/semantic_basic_avg/semantic_pts.txt", "w")
    file.write(json.dumps(semantic_master, indent=1))

    instance_master = gen.create_instance_training_set(page_segs, filtered_pts)
    file = open("/Volumes/Research/1. Research/Experiments/TrainingMasks/instance_basic_avg/instance_pts.txt", "w")
    file.write(json.dumps(instance_master, indent=1))

main()
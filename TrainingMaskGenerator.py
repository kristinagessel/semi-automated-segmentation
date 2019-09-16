'''
Take the dictionary from JsonReader and use it to do a
flood fill threshold thing to make masks for each page that we have work done for
'''
import cv2
import os
import numpy as np
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
        filtered_output = self.filter_output(output)
        pg_seg_pixels = self.do_floodfill(filtered_output, page_num)

    def filter_output(self, output):
        filtered_output = {}
        for slice in output:
            slice_num = slice.zfill(4)
            im = cv2.imread(self.img_path + slice_num + ".tif", 0)
            for point in output[slice]:
                x = int(point[0])
                y = int(point[1])
                pixel = im[y][x]
                if(pixel > 30): #check for tears TODO: better filter params?
                    if slice not in filtered_output:
                        filtered_output[slice] = []
                    filtered_output[slice].append((x,y))
        return filtered_output

    #floodfill for a whole page
    def do_floodfill(self, filtered_output, page):
        iter = 0
        seg_pixels = {}

        for slice in filtered_output:
            slice_num = slice.zfill(4)
            im = cv2.imread(self.img_path + slice_num + ".tif")
            stack = filtered_output[slice].copy()
            start_pts = filtered_output[slice].copy()
            visited = []
            height, width, channels = im.shape

            avg = self.calc_avg_pg_width(start_pts, im) #avg width might change some in different slices?

            while stack: #"while stack is not empty"
                point = stack.pop()
                visited.append(point)
                x = int(point[0])
                y = int(point[1])
                im[y][x] = (255, 0, 0) #make the visited point blue
                valid_neighbors= self.floodfill_check_neighbors(im, point, height, width, avg, start_pts)
                for pt in valid_neighbors:
                    if pt not in visited and pt not in stack:
                        stack.append(pt)

                iter += 1
                if iter % 500 == 0:
                    if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page + "debug"):
                        os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page + "debug")
                    cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page + "debug/" + str(
                            slice) + "_" + str(iter) + ".tif", im)

            seg_pixels[slice] = visited #put the list of all the pixels that make up the page into the dictionary to be returned at the end so we can use them all together if we want
            #Save the final mask
            if not os.path.exists("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page):
                os.mkdir("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page)
            cv2.imwrite("/Volumes/Research/1. Research/Experiments/TrainingMasks/" + page + "/" + str(slice) + "_mask.tif",
                im)
        return seg_pixels


    #TODO: could set a limit for how far the flood fill can reach out to prevent flood fill from crossing into a connected neighboring page?
    #TODO: *****calculate the AVERAGE width of a single page by checking the width in a consistent way from each point.
    # (thicknesses vary between pages because it's vellum, but the same page is generally the same thickness except where tears are)
    # That way if some are extra long, we can throw them out and then give the flood fill a limit of how much it can expand by the average
    # Maybe even calculate the distance between each point so we can set a sort of oval boundary
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
        #TODO: do we need to throw out outliers?
        return np.average(pt_counts)



    #TODO: use avg width to limit how far we are checking for neighbors so we don't fill more pages where pages overlap
    #TODO: compare avg to start_pts? if current x > nearest start_pt + avg, stop? (buggy I think)
    def floodfill_check_neighbors(self, im, pixel, height, width, avg_width, start_pts):
        valid_neighbors = []
        x = [-1, 0, 1]
        y = [-1, 0, 1]
        x_pos = pixel[0]
        y_pos = pixel[1]
        for i in y: #height
            for j in x: #width
                if (y_pos != 0 or x_pos != 0) and x_pos + j < width-1 and y_pos + i < height-1: #Don't want the center pixel or any out of bounds
                    grey_val = im[y_pos + i][x_pos + j][0] #pixel access is BACKWARDS--(y,x)
                    if grey_val > self.low_tolerance and grey_val < self.high_tolerance:
                        valid_neighbors.append((x_pos + j, y_pos + i))
        return valid_neighbors

def main():
    gen = TrainingMaskGenerator()
    gen.generate_mask_for_pg("15")

main()
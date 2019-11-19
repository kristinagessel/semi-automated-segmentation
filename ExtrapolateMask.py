import VCPSReader as vr
import ujson
import cv2
import os
import math
import numpy as np
'''
Given an original slice and an original pointset tracing a single page through the slice, extrapolate in 3D to subsequent slices.
3D floodfill as a start?
Use Voronoi diagrams to find the skeleton in the center of the page, then seed those points on the next slice (if they are on a page?)
Might want a faster way to skeletonize, or perhaps a different approach altogether?
'''
PATH_TO_VOLUME = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"
PATH_TO_POINTSETS = ""
PATH_TO_SAVE_LOCATION = "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/"

'''
Scenario:
1. User uses VC to identify what page they want. (Draw the segmentation line on one(?) slice, hopefully close to the center of the page)
2. User runs this code with: path to the volume of images, path to the segmentation's pointsets.vcps, the page number (so it can be recorded in the output), the path to the save location, and the slice the user started with
3. Do floodfill on this slice.
4. Seed next slice thanks to the prior floodfill. (How?) (First, plain threshold to invalidate points that were valid on prior slice but now are black space...) 
    (OR skeletonize the flood filled slice and seed the next slice with those skeleton points?)
5. Repeat 3 & 4 a # of times
'''
class MaskExtrapolator:
    def __init__(self, vol_path, path_to_pointsets, page, save_path, start_slice, num_iterations):
        self.low_tolerance = 65
        self.high_tolerance = 255 #we don't want to pick up the minerals which show up as a bright white

        self.img_path = vol_path
        self.save_path = save_path
        self.path_to_volume = vol_path
        self.page = page

        self.all_checked_pixels = {}
        self.set_pixels = {}

        #Read in the start points
        self.orig_pts = self.read_points(path_to_pointsets)
        self.pts = self.orig_pts[start_slice].copy()

        #Find the slice we'll start with (the first one if multiple were provided)
        self.start_slice = list(self.orig_pts.keys())[0] #the slice for which we have the initial pointset
        self.current_slice = list(self.orig_pts.keys())[0]

        self.slice = start_slice

        self.flood_fill_data = {}

        print("Doing flood fill for 100 slices...")
        #Do flood fill for this slice
        #10 times for now to test
        for i in range(num_iterations):
            self.flood_fill_data[self.slice], self.pts, self.slice = self.do_2d_floodfill(self.pts, page, self.slice)

        #Save the final result
        file = open(self.save_path + page + ".txt", "w")
        ujson.dump(self.flood_fill_data, file, indent=1)

    '''
    Read pointsets.vcps and put the contents into a dictionary.
    '''
    def read_points(self, path_to_pointsets):
        start = vr.VCPSReader(path_to_pointsets + "/pointset.vcps").process_VCPS_file({})
        return start

    '''
    3D Flood Fill
    Run flood fill on a 'cube' (3x3x2 cube?) containing current slice and next slice.
    In this way we won't necessarily proceed in a linear way through the slices.
    Is this practical for this use case? (I'm sure it's good for the masks, not sure about segmenting ahead for a user...)
    '''
    def do_3d_floodfill(self, skeleton_pts, page):
        return 0 #TODO

    '''
    2D Flood Fill
    Below functions are for the 2D flood fill case.
    The way I envision this maybe working is:
    1. Flood fill current slice as we are now.
    2. When it's time to move to the next slice, seed that slice using the filled slice as a reference (since we don't have manual seg for it anymore)
        a. How? Seed using a skeleton of the points? (Voronoi, etc.?)
    3. Do this for 10 or so slices for now to see how it does.
    '''
    def do_2d_floodfill(self, skeleton_pts, page, slice):
        slice_num = str(slice).zfill(4)
        stack = []
        visited = []
        start_pts = []
        im = cv2.imread(self.img_path + slice_num + ".tif") #open the image of this slice

        #Filter Step: Remove points from the prior page skeleton that are now on too dark/light voxels according to the threshold.
        for elem in skeleton_pts:
            x = int(elem[0])
            y = int(elem[1])
            pixel = im[y][x][0]

            if (pixel > self.low_tolerance and pixel < self.high_tolerance):  # check for tears and bright spots (minerals?)
                stack.append([(x, y), (x, y)])  # append a tuple of ((point), (origin point)) to keep track of how far we are from the original point
                start_pts.append((x, y))

        height, width, channel = im.shape #grab the image properties for bounds checking later

        avg = int(self.calc_avg_pg_width(start_pts, im))

        #Floodfill Step: Fill all connected points that pass the threshold checks and are in the bounds specified by avg width of the page.
        while stack:
            point = stack.pop()
            visited.append(point[0])
            x = int(point[0][0])  # point of interest's x (not origin point's x)
            y = int(point[0][1])
            im[y][x] = (255, 0, 0)
            valid_neighbors = self.floodfill_check_neighbors(im, point, height, width, avg)
            for pt in valid_neighbors:
                if pt not in visited and tuple(pt) not in (i[0] for i in stack):
                    stack.append(
                        (tuple(pt), tuple(point[1])))  # append a tuple of form ((point), (parent's origin point))

        if not os.path.exists(self.save_path + page):
            os.mkdir(self.save_path + page)
        cv2.imwrite(self.save_path + page + "/" + str(slice) + "_mask" + "_avg=" + str(avg) + ".tif", im)

        skeleton = self.skeletonize(visited)
        return visited, skeleton, slice+1

    '''
    For 2D floodfill:
    Given the flood-filled current slice, seed the next slice.
    Produce the skeleton of the filled slice, as these will be passed in as seed points in the next round of floodfill.
    '''
    def skeletonize(self, points):
        return points  # TODO

    '''
    Use Voronoi diagrams to find the skeleton in the center of the page.
    '''
    def get_voronoi_skeleton(self, slice_pts):
        return 0  # TODO

    #Taken from TrainingMaskGenerator's implementation
    def floodfill_check_neighbors(self, im, pixel, height, width, avg_width):
        valid_neighbors = []
        x = [-1, 0, 1]
        y = [-1, 0, 1]
        x_pos = pixel[0][0]
        y_pos = pixel[0][1]
        for i in y: #height
            for j in x: #width
                if (y_pos != 0 or x_pos != 0) and x_pos + j < width-1 and y_pos + i < height-1 and self.calculate_distance_from_origin(pixel[0], pixel[1]) <= math.ceil(avg_width):# #Don't want the center pixel or any out of bounds
                    grey_val = im[y_pos + i][x_pos + j][0]
                    if grey_val > self.low_tolerance and grey_val < self.high_tolerance:
                        valid_neighbors.append((x_pos + j, y_pos + i))
        return valid_neighbors

    # Taken from TrainingMaskGenerator's implementation
    #what distance metric? Right now--Euclidean
    def calculate_distance_from_origin(self, point, origin):
        delta_x = abs(point[0] - origin[0])
        delta_y = abs(point[1] - origin[1])
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        return distance

    #Taken from TrainingMaskGenerator's implementation
    #TODO: improve on this. The complex shape of the pages means this doesn't work too well.
    def calc_avg_pg_width(self, pts, im):
        pt_counts = []
        for pt in pts:
            x_pos = pt[0]
            y_pos = pt[1]
            #from this pt, go left and right from the pixel. Get this length.
            grey_val = im[y_pos][x_pos][0]
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

        if math.isnan(np.average(pt_counts)):
            return 0
        return np.average(pt_counts)

#---------------------------------------------------

HI_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"
#LOW_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180220092522/"
segmentation_number = "20191114132257" #"20191114133552"
pointset_path = "/Volumes/Research/1. Research/MS910.volpkg/paths/" + segmentation_number
page = "1" #"3"
save_path = "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/"
start_slice = 0
num_iterations = 150

ex = MaskExtrapolator(HI_RES_PATH, pointset_path, page, save_path, start_slice, num_iterations)
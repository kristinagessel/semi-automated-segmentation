import argparse
import glob
import json

import VCPSReader as vr
import ujson
import cv2
import os
import math
import numpy as np
import queue

#import skimage.morphology as skm
import matplotlib.pyplot as plt

'''
Given an original slice and an original pointset tracing a single page through the slice, extrapolate in 3D to subsequent slices.
3D floodfill as a start?
Use Voronoi diagrams to find the skeleton in the center of the page, then seed those points on the next slice (if they are on a page?)
Might want a faster way to skeletonize, or perhaps a different approach altogether?
'''

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
    def __init__(self, vol_path, path_to_pointsets, page, save_path, start_slice, num_iterations, load_path, pseudo):
        self.low_tolerance = 55
        self.high_tolerance = 255 #we don't want to pick up the minerals which show up as a bright white

        self.img_path = vol_path
        self.save_path = save_path
        self.path_to_volume = vol_path
        self.page = page
        self.all_checked_voxels = {}
        self.set_voxels = {}
        #Read in the start points
        if pseudo:
            self.orig_pts = self.load_txt_pointset(path_to_pointsets, start_slice) #Must be in JSON format
        else:
            self.orig_pts = self.read_points(path_to_pointsets)

        self.fill_pts = self.orig_pts[start_slice].copy()

        #Find the slice we'll start with (the first one if multiple were provided)
        self.start_slice = list(self.orig_pts.keys())[0] #the slice for which we have the initial pointset
        self.current_slice = list(self.orig_pts.keys())[0]

        self.flood_fill_data = {}
        self.skeleton_data = {}

        if load_path != None:
            self.flood_fill_data = self.load_existing_pointset(load_path)
            while str(self.start_slice) in self.flood_fill_data:
                self.start_slice += 1
            #start from here -- if this pointset has gone deeper than the start slice, then update the start slice and start from the deepest part.
            # Otherwise, start from the start slice but keep the old points too


            orig_im = cv2.imread(self.img_path + str(self.start_slice).zfill(4) + ".tif")  # open the image of this slice (required by skeletonize)
            self.fill_pts, img = self.skeletonize(self.flood_fill_data[str(self.start_slice-1)], orig_im)
            # Save an image showing the skeleton itself for debugging purposes
            if not os.path.exists(self.save_path + page):
                os.mkdir(self.save_path + page)
            cv2.imwrite(self.save_path + page + "/" + str(self.start_slice-1) + "_skeleton" + ".tif", img)

        self.slice = self.start_slice

        print("Starting from ", self.start_slice)


        print("Doing flood fill for ", num_iterations, " slices...")
        #Do flood fill for this slice
        #10 times for now to test
        for i in range(num_iterations):

            #Save periodically
            if num_iterations % 50 == 0:
                file = open(self.save_path + page + " " + str(num_iterations) + ".txt", "w")
                ujson.dump(self.flood_fill_data, file, indent=1)
                file.close()

            self.flood_fill_data[self.slice], self.fill_pts, self.slice = self.do_2d_floodfill(self.fill_pts, page, self.slice)
            self.skeleton_data[self.slice] = self.fill_pts

        #Save the final result
        file = open(self.save_path + page + ".txt", "w")
        ujson.dump(self.flood_fill_data, file, indent=1)
        file.close()
        #Save the skeleton in a separate file
        file = open(self.save_path + page + "_skeleton.txt", "w")
        ujson.dump(self.flood_fill_data, file, indent=1)
        file.close()

    def load_txt_pointset(self, path_to_points, slice):
        start_pts = {}
        start_pts[slice] = []

        file = open(path_to_points)
        for line in file:
            line = line.split(',')
            point = tuple((int(line[0].rstrip()), int(line[1].rstrip())))
            start_pts[slice].append(point) #split on comma--x is first, y is second. Append as a tuple (x, y)
        return start_pts

    '''
    Read pointsets.vcps and put the contents into a dictionary.
    '''
    def read_points(self, path_to_pointsets):
        start = vr.VCPSReader(path_to_pointsets + "/pointset.vcps").process_VCPS_file({})
        return start

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
        path = (self.img_path + slice_num + ".tif")
        orig_im = cv2.imread(path)  # open the image of this slice
        im = orig_im.copy()

        #Filter Step: Remove points from the prior page skeleton that are now on too dark/light voxels according to the threshold.
        for elem in skeleton_pts:
            x = int(elem[0])
            y = int(elem[1])
            voxel = im[y][x][0]

            if (voxel > self.low_tolerance and voxel < self.high_tolerance):  # check for tears and bright spots (minerals?)
                stack.append([(x, y), (x, y)])  # append a tuple of ((point), (origin point)) to keep track of how far we are from the original point
                if x not in self.set_voxels:
                    self.set_voxels[x] = {}
                start_pts.append((x, y))

        height, width, channel = im.shape #grab the image properties for bounds checking later

        width_bound = int(self.calc_med_pg_width(start_pts, im))

        #Floodfill Step: Fill all connected points that pass the threshold checks and are in the bounds specified by avg width of the page.
        while stack:
            point = stack.pop()
            visited.append(point[0])
            x = int(point[0][0])  # point of interest's x (not origin point's x)
            y = int(point[0][1])
            im[y][x] = (255, 0, 0)
            valid_neighbors = self.floodfill_check_neighbors(im, point, height, width, width_bound)
            for pt in valid_neighbors:
                if pt not in visited and tuple(pt) not in (i[0] for i in stack):
                    stack.append((tuple(pt), tuple(point[1])))  # append a tuple of form ((point), (parent's origin point))

        #Save the masked image for viewing purposes later
        if not os.path.exists(self.save_path + page):
            os.mkdir(self.save_path + page)
        cv2.imwrite(self.save_path + page + "/" + str(slice) + "_mask" + "_avg=" + str(width_bound) + "_threshold=" + str(self.low_tolerance) +  ".tif", im)
        skeleton, img = self.skeletonize(visited, orig_im.copy())

        #Save an image showing the skeleton itself for debugging purposes
        if not os.path.exists(self.save_path + page):
            os.mkdir(self.save_path + page)
        cv2.imwrite(self.save_path + page + "/" + str(slice) + "_skeleton" + ".tif", img)
        return visited, skeleton, slice+1

    '''
    Given the flood-filled current slice, seed the next slice.
    Produce the skeleton of the filled slice, as these skeleton points will be passed in as seed points in the next round of floodfill.
    Inputs: 
        img: image on which to draw the thinned version for testing
        points: all the points making up the mask after flood fill
    '''
    def skeletonize(self, points, img):
        points = self.fill_holes_in_mask(points, img)
        points = self.opencv_distance_transform(points, img) #could I do this in a sliding window, and take a number below the max value in the window? ('local' distance transform?)
        skeleton = self.thin_cloud_continuous(points)
        skeleton = self.prune_skeleton(skeleton)
        for vx in skeleton:
            img[int(vx[1])][int(vx[0])] = (0, 255, 0)
        return skeleton, img

    #Fill the holes in the mask with a closing operation.
    def fill_holes_in_mask(self, mask_pts, img):
        filled_mask = []
        im = self.make_binary_img(mask_pts, img)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
        x_dims = closing.shape[1]
        y_dims = closing.shape[0]
        for y in range(0, y_dims):
            for x in range(0, x_dims):
                if closing[y][x] > 0:
                    filled_mask.append(tuple((x, y)))
        return filled_mask

    '''
    Do some cleanup on the skeleton -- remove stray, meaningless branches and smooth the skeleton.
    '''
    def prune_skeleton(self, skeleton):
        pruned_skeleton = self.remove_shortest_branches(skeleton)
        return pruned_skeleton

    '''
    The skeleton has spurs. These are generally short. When spurs occur, an 'intersection' is created where we could step forward 2+ different ways (a pixel is connected 
    to 3 or more pixels). Can I detect these intersections and then prune the shortest branch reachable from these intersections iteratively until all intersections only have
    2 pixels connected?
    '''
    def remove_shortest_branches(self, skeleton):
        skeleton = self.do_bfs_find_intersections_and_prune(skeleton, 6)
        return skeleton

    #Do breadth-first search along a skeleton, recording intersections where the move options are greater than 2 (the px we came from and the next px)
    def do_bfs_find_intersections_and_prune(self, skeleton, length_threshold):
        orig_skeleton = skeleton.copy()
        visited = []
        is_intersection = []
        end_pts = []
        q = queue.Queue()

        while len(visited) != len(orig_skeleton):
            skeleton = list(set(skeleton) - set(visited)) #list subtraction... Remove the points we have already visited.
            # Find tuple containing min y --this is typically going to be the most extreme point. (Check for disconnected components)
            min_y = min(skeleton, key=lambda t: t[1])
            q.put(min_y)
            while not q.empty():
                x = [-1, 0, 1]
                y = [-1, 0, 1]
                pt = q.get()
                if pt not in visited:
                    visited.append(pt)

                    #check neighbors
                    option_ctr = 0
                    for i in y:  # height
                        for j in x:  # width
                                x_ck = pt[0] + j
                                y_ck = pt[1] + i
                                pt_ck = tuple((x_ck, y_ck))
                                #If it's a novel point we haven't visited, it's an option to move forward.
                                if pt_ck in skeleton:
                                    option_ctr+=1
                                    if pt_ck not in visited:
                                        q.put(pt_ck)
                    if option_ctr > 3: #If it's not a straight linear path, -p- or p-, it branches and may have spurs we want to prune. Add this point to a list to check intersection length later.
                        is_intersection.append(pt)
                    if option_ctr == 1: #If we can only travel one direction, it is a dead end.
                        end_pts.append(pt)
            #If the queue is empty, we have explored that entire connected component.
        print("Found " + str(len(is_intersection)) + " intersections.")
        pruned_skeleton = self.prune_shortest_branches(orig_skeleton, is_intersection, end_pts, length_threshold)
        return pruned_skeleton  # TODO

    #could modify to take a length threshold, and if search goes more pixels than that threshold, abort because it's too long.
    def do_bfs_find_length(self, start, parent, skeleton, length_threshold):
        #search in a direction from pt away from origin through the skeleton. return the final length of the path.
        length = 0
        visited = []
        path = []
        q = queue.Queue()
        visited.append(parent)
        q.put(start)
        while(not q.empty()):
            x = [-1, 0, 1]
            y = [-1, 0, 1]
            pt = q.get()
            if pt not in visited:
                visited.append(pt)
                #if pt != start:
                path.append(pt)
                length += 1

                if length > length_threshold:
                    return None, None

                # check neighbors
                for i in y:  # height
                    for j in x:  # width
                        x_ck = pt[0] + j
                        y_ck = pt[1] + i
                        pt_ck = tuple((x_ck, y_ck))
                        # If it's a novel point we haven't visited, it's an option to move forward.
                        if pt_ck in skeleton:
                            if pt_ck not in visited:
                                q.put(pt_ck)
        return length, path

    def prune_shortest_branches(self, skeleton, intersection_list, end_pts, length_threshold):
        x = [-1, 0, 1]
        y = [-1, 0, 1]
        for pt in intersection_list:
            pt_x = pt[0]
            pt_y = pt[1]
            for i in y:  # height
                for j in x:  # width
                    x_ck = pt[0] + j
                    y_ck = pt[1] + i
                    pt_ck = tuple((x_ck, y_ck))
                    if pt not in end_pts and pt_ck in skeleton:
                        #go in a direction away from the original point, measure the length until you hit a dead end.
                        length, pt_path = self.do_bfs_find_length(pt_ck, pt, skeleton, length_threshold)
                        if length != None and pt_path != None:
                            print("Pruning " + str(len(pt_path)) + " points.")
                            skeleton = list(set(skeleton) - set(pt_path)) #remove all the points that make up this too-short path.
        return skeleton

    '''
    Given a skeleton with noisy spurs and loops:
        Make a sliding window of a certain dimension
        Fit this sliding window over a subset of all the points
        Fit a third order polynomial to these points
        Use that polynomial instead of the points for our skeleton -- should not follow the spurs since they start out small 
            and the main skeleton body will outweigh them.
    '''
    def prune_with_sliding_window(self, points, img):
        #Slide a large-ish window over the entire image. When points are in view, call numpy's polyfit to create a 3rd order
        #polynomial curve that we replace the skeleton pts with
        pruned_skeleton = []
        window_dims = 7 #square window: window_dims * window_dims pixels inside
        min_x = min(points)[0] #TODO: find the min bound based on the skeleton to constrain the sliding window a little more
        min_y = min(points)[1] #TODO: find the min bound based on the skeleton
        max_x = img.shape[1]
        max_y = img.shape[0]

        for x in range(min_x-1, max_x - window_dims, int(window_dims/2)): #start, end, step (all positions of the top left corner of the sliding window)
            for y in range(min_y-1, max_y - window_dims, int(window_dims/2)):
                points_in_window = self.find_points_in_bounds(x, y, window_dims, points)
                x_vals = []
                y_vals = []
                #split the points into x and y coordinate lists:
                for pt in points_in_window:
                    pt_x = pt[0]
                    pt_y = pt[1]
                    x_vals.append(pt_x)
                    y_vals.append(pt_y)
                #call numpy.polyfit with degree 3
                if len(points_in_window) > 0:
                    coefficients = np.polynomial.polynomial.polyfit(x_vals, y_vals, 3)
                    #print("Obtained coefficients.")
                    #now, we can make a function of the form y = c0 + c1*x + c2*x^2 + c3*x^3 that fits the points in the window the most closely
                    #pass the original x values in to get the right fit polynomial.
                    #TODO: try going from the smallest value of x to the largest value of x with a small step value.
                    new_x_vals = []
                    for xv in range(min(x_vals), max(x_vals), 1):
                        new_x_vals.append(xv)
                    fit = np.polynomial.polynomial.polyval(new_x_vals, coefficients)
                    for pos in range(0, len(new_x_vals)):
                        pruned_skeleton.append(tuple((new_x_vals[pos], fit[pos])))
                    #plt.plot(x_vals, fit)
                    #plt.show()
                    #print("Done with this window.")
        return pruned_skeleton #TODO: complete implementation

    #Input: x of top-left corner, y of top-left corner, dimensions, and all the points in the skeleton.
    def find_points_in_bounds(self, x, y, dims, points):
        pts_in_bounds = []
        for pt in points:
            pt_x = pt[0]
            pt_y = pt[1]
            max_x = x + dims
            max_y = y + dims
            if pt_x < max_x and pt_x > x:
                if pt_y < max_y and pt_y > y:
                    pts_in_bounds.append(pt)
        return pts_in_bounds

    def make_binary_img(self, points, img):
        shape = img.shape
        x_dims = shape[1]
        y_dims = shape[0]
        im = np.zeros(shape=[y_dims, x_dims], dtype=np.uint8)
        #make a binary image from the mask:
        for pt in points:
            x = pt[0]
            y = pt[1]
            im[y][x] = 1
        return im

    #Doesn't seem to work as well on the low-res
    def opencv_distance_transform(self, points, img):
        im = self.make_binary_img(points, img)

        dist = cv2.distanceTransform(im, cv2.DIST_L2, 0)

        trimmed_mask = []
        x_dims = dist.shape[1]
        y_dims = dist.shape[0]
        for y in range(0, y_dims):
            for x in range(0, x_dims):
                if dist[y][x] > 1:
                    trimmed_mask.append(tuple((x, y)))
        return trimmed_mask


    '''Morphological Thinning
    Thin cloud to get a skeleton and produce a continuous skeleton.
    
    This is the basic implementation given in section 8.6.2 of "Computer Vision", 5th Edition, by E.R. Davies
    On average, it takes 5 minutes to segment a layer. Not good.
    
    *****TODO: perhaps try to instead put points into a dictionary like: dict[x][y] for faster lookup? 
    (Then you just check if x and y exist in dictionary. Maybe it's a little faster?)
    '''
    def thin_cloud_continuous(self, points):
        skeleton = points.copy()

        while(True):
            skeleton, thinned_n = self.strip_north_pts(skeleton)
            skeleton, thinned_s = self.strip_south_pts(skeleton)
            skeleton, thinned_e = self.strip_east_pts(skeleton)
            skeleton, thinned_w = self.strip_west_pts(skeleton)
            #If no thinning occurred in this last iteration, we are finished.
            if not(thinned_n or thinned_s or thinned_e or thinned_w):
                break

        return skeleton

    def strip_north_pts(self, mask):
        thinned = False
        points_to_remove = []
        for point in mask:
            #get neighbors we care about: (x, y+1) and (x, y-1)
            x = point[0]
            y = point[1]
            sigma, chi = self.calculate_params(point, mask)
            #check if chi == 2 and sigma != 1:
            #(implied that center pixel is active if it's in the mask, so no need to check)
            if chi == 2 and sigma != 1:
                if tuple((x, y+1)) not in mask: #i.e. it's 0
                        if tuple((x, y-1)) in mask: #i.e. it's 1
                                #remove the pixel from the mask
                                points_to_remove.append(point)
                                thinned = True
        print("Removing ", len(points_to_remove), " points.")
        for point in points_to_remove:
            mask.remove(point)
        return mask, thinned

    def strip_south_pts(self, mask):
        thinned = False
        points_to_remove = []
        for point in mask:
            #get neighbors we care about: (x, y+1) and (x, y-1)
            x = point[0]
            y = point[1]
            sigma, chi = self.calculate_params(point, mask)
            #check if chi == 2 and sigma != 1:
            #(implied that center pixel is active if it's in the mask, so no need to check)
            if chi == 2 and sigma != 1:
                if tuple((x, y-1)) not in mask:
                    if tuple((x, y+1)) in mask:
                        #remove the pixel from the mask
                        points_to_remove.append(point)
                        thinned = True
        print("Removing ", len(points_to_remove), " points.")
        for point in points_to_remove:
            mask.remove(point)
        return mask, thinned

    def strip_east_pts(self, mask):
        thinned = False
        points_to_remove = []
        for point in mask:
            # get neighbors we care about: (x, y+1) and (x, y-1)
            x = point[0]
            y = point[1]
            sigma, chi = self.calculate_params(point, mask)
            # check if chi == 2 and sigma != 1:
            # (implied that center pixel is active if it's in the mask, so no need to check)
            if chi == 2 and sigma != 1:
                if tuple((x + 1, y)) not in mask:
                    if tuple((x - 1, y)) in mask:
                        # remove the pixel from the mask
                        points_to_remove.append(point)
                        thinned = True
        print("Removing ", len(points_to_remove), " points.")
        for point in points_to_remove:
            mask.remove(point)
        return mask, thinned

    def strip_west_pts(self, mask):
        thinned = False
        points_to_remove = []
        for point in mask:
            # get neighbors we care about: (x, y+1) and (x, y-1)
            x = point[0]
            y = point[1]
            sigma, chi = self.calculate_params(point, mask)
            # check if chi == 2 and sigma != 1:
            # (implied that center pixel is active if it's in the mask, so no need to check)
            if chi == 2 and sigma != 1:
                if tuple((x - 1, y)) not in mask:
                    if tuple((x + 1, y)) in mask:
                        # remove the pixel from the mask
                        points_to_remove.append(point)
                        thinned = True
        print("Removing ", len(points_to_remove), " points.")
        for point in points_to_remove:
            mask.remove(point)
        return mask, thinned

    def calculate_params(self, point, mask):
        x = point[0]
        y = point[1]
        A1 = int(tuple((x, y + 1)) in mask)
        A2 = int(tuple((x + 1, y + 1)) in mask)
        A3 = int(tuple((x + 1, y)) in mask)
        A4 = int(tuple((x + 1, y - 1)) in mask)
        A5 = int(tuple((x, y - 1)) in mask)
        A6 = int(tuple((x - 1, y - 1)) in mask)
        A7 = int(tuple((x - 1, y)) in mask)
        A8 = int(tuple((x - 1, y + 1)) in mask)

        # calculate chi (crossing number) and sigma (number of active neighbors)
        # No need to cast to int?
        chi = (A1 != A3) + (A3 != A5) + (A5 != A7) + int(A7 != A1) + (2 * (A2 > A1) and (A2 > A3)) + (
                    (A4 > A3) and (A4 > A5)) + ((A6 > A5) and (A6 > A7)) + ((A8 > A7) and (A8 > A1))
        sigma = self.calculate_sigma(point, mask)
        return sigma, chi

    def calculate_sigma(self, point, mask):
        row = [0, 1, -1]
        col = [0, 1, -1]
        pt_x = point[0]
        pt_y = point[1]
        ctr = 0
        for x in row:
            for y in col:
                if x != 0 or y != 0: #ignore (0,0)
                    if tuple((pt_x + x, pt_y + y)) in mask:
                        ctr += 1
        return ctr

    #Taken from TrainingMaskGenerator's implementation
    def floodfill_check_neighbors(self, im, voxel, height, width, width_bound):
        valid_neighbors = []
        x = [-1, 0, 1]
        y = [-1, 0, 1]
        x_pos = voxel[0][0]
        y_pos = voxel[0][1]
        for i in y: #height
            for j in x: #width
                if (i != 0 or j != 0) and x_pos + j < width-1 and y_pos + i < height-1 and self.calculate_distance_from_origin(voxel[0], voxel[1]) <= math.ceil(width_bound):# #Don't want the center pixel or any out of bounds
                    grey_val = im[y_pos + i][x_pos + j][0]
                    if grey_val > self.low_tolerance: #and grey_val < self.high_tolerance:
                        valid_neighbors.append((x_pos + j, y_pos + i))
        return valid_neighbors

    # Taken from TrainingMaskGenerator's implementation
    #what distance metric? Right now--Euclidean
    def calculate_distance_from_origin(self, point, origin):
        return self.euclidean_dist(point, origin)

    #Median of the width might be a more helpful metric?
    #TODO: improve on this. The complex shape of the pages means this doesn't work too well.
    def calc_med_pg_width(self, pts, im):
        pt_counts = []
        for pt in pts:
            x_pos = pt[0]
            y_pos = pt[1]
            #from this pt, go left and right from the pixel. Get this length.
            grey_val = im[y_pos][x_pos][0]
            length_ctr = 1

            #go left
            while grey_val > self.low_tolerance:# and grey_val < self.high_tolerance:
                x_pos -= 1
                grey_val = im[y_pos][x_pos][0]
                length_ctr += 1

            #go right
            while grey_val > self.low_tolerance:# and grey_val < self.high_tolerance:
                x_pos += 1
                grey_val = im[y_pos][x_pos][0]
                length_ctr += 1
            pt_counts.append(length_ctr)

        if math.isnan(np.median(pt_counts)):
            return 0
        return np.median(pt_counts)

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
            while grey_val > self.low_tolerance:# and grey_val < self.high_tolerance:
                x_pos -= 1
                grey_val = im[y_pos][x_pos][0]
                length_ctr += 1

            #go right
            while grey_val > self.low_tolerance:# and grey_val < self.high_tolerance:
                x_pos += 1
                grey_val = im[y_pos][x_pos][0]
                length_ctr += 1
            pt_counts.append(length_ctr)

        if math.isnan(np.average(pt_counts)):
            return 0
        return np.average(pt_counts)

    def load_existing_pointset(self, path):
        mask = glob.glob(path + "*[0-9].txt")
        file = open(mask[0])
        points = json.loads(file.read())
        file.close()
        return points

#---------------------------------------------------

#page num : [seg #, start slice]
pages = {
    "MS910": { #Note: grey tolerance 65 works well
    "1" : ["20191114132257", 0],  #segmentation #, start slice
    "2" : ["20191125215208", 0],
    "3" : ["20191114133552", 0],
    "11" : ["20191123203022", 0],
    "?" : ["20191126112158", 0],
    "?600" : ["20200329223634", 600],
    "?1100" : ["20200402121109", 1100],
    "??" : ["20191126122204", 0],
    "???" : ["20191126132825", 0],
    "????" : ["20200307101907", 10],
    "?????" : ["20200309085522", 0],
    "lr1" : ["20200217180742", 190] #low res
    },
    "Paris59": { #Note: grey tolerance 35 works well
        "test" : ["20191204123934", 430], #segmentation #, start slice
        "test2" : ["20191204125435", 100], #segmentation #, start slice
        "test3" : ["20191204135310", 100],
        "test4" : ["20191206191654", 300],
        "test5" : ["20191206192523", 500],
        "2ndlayer" : ["2ndlayer", 365]
    }
}

parser = argparse.ArgumentParser(description="Perform semi-automated segmentation with flood-filling and skeletonization")
parser.add_argument("pointset_path", type=str, help="Full path to the pointset")
parser.add_argument("save_path", type=str, help="Path to the output directory.")
parser.add_argument("page_name", type=str, help="Name of the page.")
parser.add_argument("volume_path", type=str, help="Path to the directory containing the volume.")
parser.add_argument("start_slice", type=int, help="Slice to begin segmenting on (must correspond with the pointset's slice.)")
parser.add_argument("num_iterations", type=int, help="Number of slices to segment.")
parser.add_argument("load_path", type=str, help="Path to existing point set (if applicable, otherwise leave empty.)", nargs='?', default=None)
parser.add_argument("-pseudo", "--is_pseudo_pointset", action="store_true",  help="Set this flag if this is synthetic training data.")
args = parser.parse_args()
'''
object = "MS910"
page = "?"
segmentation_number = pages[object][page][0]

paths = {
    "MS910": {
        "high-res" : "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/",
        "low-res" : "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180220092522/",
        "pointset" : "/Volumes/Research/1. Research/MS910.volpkg/paths/" + segmentation_number,
        "save" : "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/MS910/"
    },
    "Paris59": {
        "high-res" : "/Volumes/Research/1. Research/Herculaneum/PHercParisObject59/PHercParisObjet59.volpkg/volumes/20190910090730/",
        "pointset" : "/Volumes/Research/1. Research/Herculaneum/PHercParisObject59/PHercParisObjet59.volpkg/paths/" + segmentation_number,
        "save" : "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/Paris59/"
    }
}


start_slice = pages[object][page][1]
num_iterations = 300

volume_path = paths[object]["high-res"]
pointset_path = paths[object]["pointset"]
save_path = paths[object]["save"]
'''

ex = MaskExtrapolator(args.volume_path, args.pointset_path, args.page_name, args.save_path, args.start_slice, args.num_iterations, args.load_path, args.is_pseudo_pointset)
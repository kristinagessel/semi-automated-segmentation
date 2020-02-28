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
        self.low_tolerance = 45#65
        #self.high_tolerance = 255 #we don't want to pick up the minerals which show up as a bright white

        self.img_path = vol_path
        self.save_path = save_path
        self.path_to_volume = vol_path
        self.page = page

        self.all_checked_voxels = {}
        self.set_voxels = {}

        #Read in the start points
        self.orig_pts = self.read_points(path_to_pointsets)
        self.fill_pts = self.orig_pts[start_slice].copy()

        #Find the slice we'll start with (the first one if multiple were provided)
        self.start_slice = list(self.orig_pts.keys())[0] #the slice for which we have the initial pointset
        self.current_slice = list(self.orig_pts.keys())[0]

        self.slice = start_slice

        self.flood_fill_data = {}
        self.skeleton_data = {}

        print("Doing flood fill for ", num_iterations, " slices...")
        #Do flood fill for this slice
        #10 times for now to test
        for i in range(num_iterations):
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
        orig_im = cv2.imread(self.img_path + slice_num + ".tif")  # open the image of this slice
        im = orig_im.copy()

        #Filter Step: Remove points from the prior page skeleton that are now on too dark/light voxels according to the threshold.
        for elem in skeleton_pts:
            x = int(elem[0])
            y = int(elem[1])
            voxel = im[y][x][0]

            if (voxel > self.low_tolerance):# and voxel < self.high_tolerance):  # check for tears and bright spots (minerals?)
                stack.append([(x, y), (x, y)])  # append a tuple of ((point), (origin point)) to keep track of how far we are from the original point
                if x not in self.set_voxels:
                    self.set_voxels[x] = {}
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
                    stack.append((tuple(pt), tuple(point[1])))  # append a tuple of form ((point), (parent's origin point))

        #Save the masked image for viewing purposes later
        if not os.path.exists(self.save_path + page):
            os.mkdir(self.save_path + page)
        cv2.imwrite(self.save_path + page + "/" + str(slice) + "_mask" + "_avg=" + str(avg) + "_threshold=" + str(self.low_tolerance) +  ".tif", im)

        skeleton, img = self.skeletonize(visited, orig_im.copy())

        #Save an image showing the skeleton itself for debugging purposes
        if not os.path.exists(self.save_path + page):
            os.mkdir(self.save_path + page)
        cv2.imwrite(self.save_path + page + "/" + str(slice) + "_skeleton" + ".tif", img)
        return visited, skeleton, slice+1

    '''
    For 2D floodfill:
    Given the flood-filled current slice, seed the next slice.
    Produce the skeleton of the filled slice, as these will be passed in as seed points in the next round of floodfill.
    Inputs: 
        img: image on which to draw the thinned version for testing
        points: all the points making up the mask after flood fill
    '''
    def skeletonize(self, points, img):
        #skeleton, img = self.thin_cloud(points, img)
        #skeleton, img = self.do_a_star(img)
        #skeleton = self.thin_cloud_zhang_suen(points)
        #skeleton = self.thin_cloud_continuous(points)
        skeleton = self.parallel_thin(points, img)


        #skeleton = self.prune_skeleton(skeleton)

        for vx in skeleton:
            img[int(vx[1])][int(vx[0])] = (0, 255, 0)
        return skeleton, img

    '''
    "A Fast Parallel Thinning Algorithm for the Binary Image Skeletonization"
    Deng, Iyengar, Brener
    Name: OPATA8
    Supposed to solve the edge cases that foil Zhang Suen; does an asymmetrical thinning approach.
    NOT fast at all.
    '''
    #TODO: pass in the image and check it directly rather than checking the whole list, might speed things up
    def parallel_thin(self, mask, img):
        skeleton = mask.copy()
        #Define the 14 thinning patterns:
        pruning = True
        marked_pts = []
        while pruning:
            pruning = False
            for point in skeleton:
                #get neighbors we care about: (x, y+1) and (x, y-1)
                x = point[0]
                y = point[1]

                p0 = int(tuple((x, y + 1)) in skeleton)
                p1 = int(tuple((x+1, y+1)) in skeleton)
                p2 = int(tuple((x + 1, y)) in skeleton)
                p3 = int(tuple((x+1, y-1)) in skeleton)
                p4 = int(tuple((x, y - 1)) in skeleton)
                p5 = int(tuple((x-1, y-1)) in skeleton)
                p6 = int(tuple((x-1, y)) in skeleton)
                p7 = int(tuple((x-1, y+1)) in skeleton)
                p8 = int(tuple((x+2, y)) in skeleton)
                p9 = int(tuple((x, y-2)) in skeleton)

                #Wu and Tsai's 14 thinning patterns; base of this method
                a = p0 and not p2 and (p1 or p3) and p4 and p5 and p6 and p7
                b = p0 and p1 and p2 and (p3 or p5) and not p4 and p6 and p7
                c = p0 and p1 and p2 and p3 and p4 and (p5 or p7) and not p6 and p8
                d = p2 and p3 and p4 and p5 and p6 and (p1 or p7) and not p0 and p9
                e = not p0 and not p1 and not p2 and p4 and p6
                f = p0 and p1 and p2 and not p4 and not p5 and not p6
                g = p0 and p2 and not p1 and not p3 and not p4 and not p5 and not p6 and not p7
                h = p0 and not p2 and not p3 and not p4 and p6
                i = p2 and p3 and p4 and not p0 and not p6 and not p7
                j = not p0 and not p1 and p2 and not p3 and p4 and not p5 and not p6 and not p7
                k = not p0 and not p1 and not p2 and p3 and p4 and p5 and not p6 and not p7
                l = not p0 and not p1 and not p2 and not p3 and not p4 and p5 and p6 and p7
                m = p0 and p1 and not p2 and not p3 and not p4 and not p5 and not p6 and p7
                n = not p0 and p1 and p2 and p3 and not p4 and not p5 and not p6 and not p7

                #If any of the above thinning patterns match this pixel and its neighbors, prune the pixel.
                if(a or b or c or d or e or f or g or h or i or j or k or l or m or n):
                    skeleton.remove(point)
                    pruning = True

                #These alone do not eliminate concave points.
                #TODO: add the changes the authors made to recognize concave areas

        return skeleton



    '''
    Important note: A* will only work if it can find a path from the start point to the end point. 
    This A* implementation searches along the actual image, not the mask, to try and improve the chances of finding a path (the threshold is dropped to make it less picky.)
    Sometimes this fails quite spectacularly, though. A* won't find a path if a complete tear in the page exists.
    '''
    def do_a_star(self, img):
        #Get all the set voxels in a form where we just have their (x,y) position
        set_voxels = []
        for key in self.set_voxels.keys():
            for key2 in self.set_voxels[key].keys():
                set_voxels.append((key, key2))
                #(x, y)

        #Find tuple containing min y
        min_y = min(set_voxels, key=lambda t: t[1])

        #Find tuple containing max y
        max_y = max(set_voxels, key=lambda t: t[1])

        current_pos = (min_y, min_y, 0, 0)
        skeleton = []
        visited_voxels = []
        list_of_moves = []
        list_of_moves.append((min_y, min_y, 0, 0)) #(current location, parent location, g, f)
        #fixed cost of 1 for every move--no move is weighted more than another
        while current_pos[0] != max_y: #TODO: does this compare value or the actual object? check
            #if len(list_of_moves) > 0: # if a possible move exists and it isn't completely exhausted
            current_pos = list_of_moves.pop(0) #Take the top move and do it
            #else: #If we're out of moves and haven't reached the goal, we're stuck. There's a tear or something.
            #    return skeleton, img
            visited_voxels.append(current_pos)
            x = [-1, 0, 1]
            y = [-1, 0, 1]
            for i in y:  # height
                for j in x:  # width
                    if (i != 0 or j != 0) and (img[int(current_pos[0][1] + i)][int(current_pos[0][0] + j)][0] > self.low_tolerance-15): #check that the neighbor is a valid grey level to be set
                        tmp_pos = tuple(((current_pos[0][0] + j, current_pos[0][1] + i), current_pos[0], current_pos[2]+1, self.calculate_f(max_y, (current_pos[0][0] + j, current_pos[0][1] + i), current_pos)))
                        if tmp_pos not in visited_voxels:
                            if self.dest_not_visited(tmp_pos[0], visited_voxels) or tmp_pos[3] < self.get_f_of_existing(tmp_pos, visited_voxels):
                                if not tmp_pos[3] >= self.get_f_of_existing(tmp_pos, list_of_moves) or self.get_f_of_existing(tmp_pos, list_of_moves) == -1:
                                    list_of_moves.append(tmp_pos)
            list_of_moves.sort(key=lambda x:x[3]) #Sort by f value so we pick the smallest f every time
        skeleton.append(current_pos[0])
        skeleton = self.find_shortest_path(skeleton, visited_voxels, max_y, min_y)
        return skeleton, img

    '''
    A* UTILITIES
    '''
    #Travel from the goal to the start, generating the shortest path found by the algorithm.
    def find_shortest_path(self, skeleton, node_relationships, goal, start):
        current_location = goal
        pathComplete = False
        while (pathComplete is False):
            # find current_location in the list of tuples as a destination
            # add the 'source' to the shortest_path
            for elem in node_relationships:
                if elem[0] == current_location:
                    skeleton.append(elem[1])
                    current_location = elem[1]
            if start in skeleton:  # If we've added the start location to the shortest path, we've got a path.
                pathComplete = True
        return skeleton

    def get_f_of_existing(self, pos_tuple, list):
        for elem in list:
            if (pos_tuple[0] == elem[0]):
                return elem[3]
        return -1  # Didn't find it

    def dest_not_visited(self, dest, list):
        for elem in list:
            if (dest == elem[0]):
                return False
        return True

    def calculate_f(self, dest, new_loc, current_loc_tuple):
        h = self.euclidean_dist(new_loc, dest)
        g = current_loc_tuple[2]+1
        f = g + h
        return f

    '''
    Calculate the euclidean distance between 2 points
    '''
    def euclidean_dist(self, src, dest):
        delta_x = abs(src[0] - dest[0])
        delta_y = abs(src[1] - dest[1])
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        return distance
    '''
    END A* UTILITIES
    '''

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

            A1 = int(tuple((x, y + 1)) in mask)
            A2 = int(tuple((x+1, y+1)) in mask)
            A3 = int(tuple((x + 1, y)) in mask)
            A4 = int(tuple((x+1, y-1)) in mask)
            A5 = int(tuple((x, y - 1)) in mask)
            A6 = int(tuple((x-1, y-1)) in mask)
            A7 = int(tuple((x-1, y)) in mask)
            A8 = int(tuple((x-1, y+1)) in mask)

            #calculate chi (crossing number) and sigma (number of active neighbors)
            #No need to cast to int, Python can handle this
            chi = (A1 != A3) + (A3 != A5) + (A5 != A7) + int(A7 != A1) + (2 * (A2 > A1) and (A2 > A3)) + ((A4 > A3) and (A4 > A5)) + ((A6 > A5) and (A6 > A7)) + ((A8 > A7) and (A8 > A1))
            sigma = self.calculate_sigma(point, mask)

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

            A1 = int(tuple((x, y + 1)) in mask)
            A2 = int(tuple((x+1, y+1)) in mask)
            A3 = int(tuple((x + 1, y)) in mask)
            A4 = int(tuple((x+1, y-1)) in mask)
            A5 = int(tuple((x, y - 1)) in mask)
            A6 = int(tuple((x-1, y-1)) in mask)
            A7 = int(tuple((x-1, y)) in mask)
            A8 = int(tuple((x-1, y+1)) in mask)

            #calculate chi (crossing number) and sigma (number of active neighbors)
            #No need to cast to int?
            chi = (A1 != A3) + (A3 != A5) + (A5 != A7) + int(A7 != A1) + (2 * (A2 > A1) and (A2 > A3)) + ((A4 > A3) and (A4 > A5)) + ((A6 > A5) and (A6 > A7)) + ((A8 > A7) and (A8 > A1))
            sigma = self.calculate_sigma(point, mask)

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

            A1 = int(tuple((x, y + 1)) in mask)
            A2 = int(tuple((x+1, y+1)) in mask)
            A3 = int(tuple((x + 1, y)) in mask)
            A4 = int(tuple((x+1, y-1)) in mask)
            A5 = int(tuple((x, y - 1)) in mask)
            A6 = int(tuple((x-1, y-1)) in mask)
            A7 = int(tuple((x-1, y)) in mask)
            A8 = int(tuple((x-1, y+1)) in mask)

            #calculate chi (crossing number) and sigma (number of active neighbors)
            #No need to cast to int?
            chi = (A1 != A3) + (A3 != A5) + (A5 != A7) + int(A7 != A1) + (2 * (A2 > A1) and (A2 > A3)) + ((A4 > A3) and (A4 > A5)) + ((A6 > A5) and (A6 > A7)) + ((A8 > A7) and (A8 > A1))
            sigma = self.calculate_sigma(point, mask)

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

            A1 = int(tuple((x, y + 1)) in mask)
            A2 = int(tuple((x+1, y+1)) in mask)
            A3 = int(tuple((x + 1, y)) in mask)
            A4 = int(tuple((x+1, y-1)) in mask)
            A5 = int(tuple((x, y - 1)) in mask)
            A6 = int(tuple((x-1, y-1)) in mask)
            A7 = int(tuple((x-1, y)) in mask)
            A8 = int(tuple((x-1, y+1)) in mask)

            #calculate chi (crossing number) and sigma (number of active neighbors)
            #No need to cast to int?
            chi = (A1 != A3) + (A3 != A5) + (A5 != A7) + int(A7 != A1) + (2 * (A2 > A1) and (A2 > A3)) + ((A4 > A3) and (A4 > A5)) + ((A6 > A5) and (A6 > A7)) + ((A8 > A7) and (A8 > A1))
            sigma = self.calculate_sigma(point, mask)

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

    '''
    Do some cleanup on the skeleton -- remove stray, meaningless branches and smooth the skeleton.
    '''
    def prune_skeleton(self, skeleton):
        #eliminate 'stray' points that slipped by the skeletonization algorithm
        return skeleton #TODO

    '''
    Create a skeleton by thinning the point cloud mask that we have.
    "A Fast Parallel Algorithm for Thinning Digital Patterns" (Zhang, Suen)
    '''
    def thin_cloud_zhang_suen(self, points):
        #Need: all the mask points--can get those from self.fill_pts
        iteration_ctr = 0
        # Get a copy of the current points that are part of the mask. We will remove points from skeleton to skeletonize the mask.
        skeleton = points.copy()
        x = [0, 1, 1, 1, 0, -1, -1, -1]
        y = [1, 1, 0, -1, -1, -1, 0, 1]

        while True:
            print("Iteration ", iteration_ctr)
            iteration_ctr += 1
            eliminated_pt_ctr = 0
            points_to_remove = []

            #Subiteration 1: (Remove SE boundary and NW corner points)
            for point in skeleton:
                x_pos = point[0]
                y_pos = point[1]
                neighbors = []

                #start from displacement (0, 1) and proceed counter-clockwise.
                for disp in range(0, len(y)):  # height
                        if tuple((x_pos + x[disp], y_pos + y[disp])) in skeleton:
                            # This neighbor is live and part of the mask
                            neighbors.append(1)
                        else:
                            neighbors.append(0)

                #Point is a candidate for deletion if it has <= 6 and >= 2 neighbors.
                num_live_neighbors = self.get_num_live_neighbors(neighbors)

                #Point is a candidate for deletion if there is one and only one 01 pattern in the ordered set of neighboring points
                num_zero_one_patterns = self.get_zero_one_patterns(neighbors)

                # P2 * P4 * P6 = 0
                if not (tuple((x_pos, y_pos+1)) in skeleton) or not (tuple((x_pos+1, y_pos)) in skeleton) or not (tuple((x_pos, y_pos-1)) in skeleton):
                    # P4 * P6 * P8 = 0
                    if not (tuple((x_pos+1, y_pos)) in skeleton) or not (tuple((x_pos, y_pos-1)) in skeleton) or not (tuple((x_pos-1, y_pos)) in skeleton):
                        if num_live_neighbors <= 6 and num_live_neighbors >= 2 and num_zero_one_patterns == 1:
                            #if point satisfies all of these it should be eliminated; add it to a list to be eliminated.
                            points_to_remove.append(point)
                            eliminated_pt_ctr += 1
            if eliminated_pt_ctr == 0:
                break
            eliminated_pt_ctr = 0

            for point in points_to_remove:
                skeleton.remove(point)
            print("Removed ", len(points_to_remove), " points in this subiteration.")
            points_to_remove = []

            #Subiteration 2: (Remove NW boundary and SE corner points)
            for point in skeleton:
                x_pos = point[0]
                y_pos = point[1]
                neighbors = []

                #start from displacement (0, 1) and proceed counter-clockwise
                for disp in range(0, len(y)):  # height
                        if tuple((x_pos + x[disp], y_pos + y[disp])) in skeleton:
                            # This neighbor is live and part of the mask
                            neighbors.append(1)
                        else:
                            neighbors.append(0)

                #Point is a candidate for deletion if it has <= 6 and >= 2 neighbors.
                num_live_neighbors = self.get_num_live_neighbors(neighbors)

                #Point is a candidate for deletion if there is one and only one 01 pattern in the ordered set of neighboring points
                num_zero_one_patterns = self.get_zero_one_patterns(neighbors)

                # P2 * P4 * P8 = 0
                if not (tuple((x_pos, y_pos+1)) in skeleton) or not (tuple((x_pos+1, y_pos)) in skeleton) or not (tuple((x_pos-1, y_pos)) in skeleton):
                    # P2 * P6 * P8 = 0
                    if not(tuple((x_pos, y_pos+1)) in skeleton) or not (tuple((x_pos, y_pos-1)) in skeleton) or not (tuple((x_pos-1, y_pos)) in skeleton):
                        if num_live_neighbors <= 6 and num_live_neighbors >= 2 and num_zero_one_patterns == 1:
                            #if point satisfies all of these it should be eliminated; add it to a list to be eliminated.
                            points_to_remove.append(point)
                            eliminated_pt_ctr += 1
            if eliminated_pt_ctr == 0:
                break

            for point in points_to_remove:
                skeleton.remove(point)
            print("Removed ", len(points_to_remove), " points in this subiteration.")

        #TODO: prune the skeleton
        skeleton = self.prune_skeleton(skeleton)
        return skeleton

    '''
    Get how many 01 patterns (not live, live) exist in the ordered set of neighbors, starting from the center top and moving clockwise
    (Zhang Suen)
    '''
    def get_zero_one_patterns(self, neighbors):
        zero_one_ctr = 0
        for elem in range(0, len(neighbors)):
            if neighbors[elem] == 0 and elem+1 < len(neighbors):
                if neighbors[elem+1] == 1:
                    zero_one_ctr += 1
        return zero_one_ctr

    '''
    How many neighbors does a given point (that is part of the mask) have which are also part of the mask?
    (Zhang Suen)
    '''
    def get_num_live_neighbors(self, neighbor_list):
        neighbor_ctr = 0
        for elem in neighbor_list:
            #If it's a 'live' neighbor (value 1) count it
            if elem == 1:
                neighbor_ctr += 1
        return neighbor_ctr

    #Taken from TrainingMaskGenerator's implementation
    def floodfill_check_neighbors(self, im, voxel, height, width, avg_width):
        valid_neighbors = []
        x = [-1, 0, 1]
        y = [-1, 0, 1]
        x_pos = voxel[0][0]
        y_pos = voxel[0][1]
        for i in y: #height
            for j in x: #width
                if (i != 0 or j != 0) and x_pos + j < width-1 and y_pos + i < height-1 and self.calculate_distance_from_origin(voxel[0], voxel[1]) <= math.ceil(avg_width):# #Don't want the center pixel or any out of bounds
                    grey_val = im[y_pos + i][x_pos + j][0]
                    if grey_val > self.low_tolerance: #and grey_val < self.high_tolerance:
                        valid_neighbors.append((x_pos + j, y_pos + i))
        return valid_neighbors

    # Taken from TrainingMaskGenerator's implementation
    #what distance metric? Right now--Euclidean
    def calculate_distance_from_origin(self, point, origin):
        return self.euclidean_dist(point, origin)

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

#---------------------------------------------------

#page num : [seg #, start slice]
pages = {
    "MS910": { #Note: grey tolerance 65 works well
    "1" : ["20191114132257", 0],  #segmentation #, start slice
    "2" : ["20191125215208", 0],
    "3" : ["20191114133552", 0],
    "11" : ["20191123203022", 0],
    "?" : ["20191126112158", 0],
    "??" : ["20191126122204", 0],
    "???" : ["20191126132825", 0],
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

object = "MS910"
page = "lr1"
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

volume_path = paths[object]["low-res"]
pointset_path = paths[object]["pointset"]
save_path = paths[object]["save"]

ex = MaskExtrapolator(volume_path, pointset_path, page, save_path, start_slice, num_iterations)
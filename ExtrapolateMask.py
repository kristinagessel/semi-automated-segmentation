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

class Voxel:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_set = False
        self.neighbors = []
        '''
        Store neighbors' x, y coordinates. Not full references, because those get messy.
        Do we need to know in what direction the neighbor is from this original voxel?
        '''

    def set_neighbor(self, neighbor_x, neighbor_y):
        if (neighbor_x, neighbor_y) not in self.neighbors:
            self.neighbors.append((neighbor_x, neighbor_y))

    def set(self):
        self.is_set = True

    def unset(self):
        self.is_set = False

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
        #self.high_tolerance = 255 #we don't want to pick up the minerals which show up as a bright white

        self.img_path = vol_path
        self.save_path = save_path
        self.path_to_volume = vol_path
        self.page = page

        self.all_checked_voxels = {}
        self.set_voxels = {}

        #Read in the start points
        self.orig_pts = self.read_points(path_to_pointsets)
        self.pts = self.orig_pts[start_slice].copy()

        #Find the slice we'll start with (the first one if multiple were provided)
        self.start_slice = list(self.orig_pts.keys())[0] #the slice for which we have the initial pointset
        self.current_slice = list(self.orig_pts.keys())[0]

        self.slice = start_slice

        self.flood_fill_data = {}

        print("Doing flood fill for 10 slices...")
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
        orig_im = cv2.imread(self.img_path + slice_num + ".tif")  # open the image of this slice
        im = orig_im.copy()
        self.set_voxels = {} #Re-initialize them to clear out prior slice data
        self.all_checked_voxels = {}

        #Filter Step: Remove points from the prior page skeleton that are now on too dark/light voxels according to the threshold.
        for elem in skeleton_pts:
            x = int(elem[0])
            y = int(elem[1])
            voxel = im[y][x][0]
            if x not in self.all_checked_voxels:
                self.all_checked_voxels[x] = {}
            self.all_checked_voxels[x][y] = Voxel(x, y)

            if (voxel > self.low_tolerance):# and voxel < self.high_tolerance):  # check for tears and bright spots (minerals?)
                stack.append([(x, y), (x, y)])  # append a tuple of ((point), (origin point)) to keep track of how far we are from the original point
                if x not in self.set_voxels:
                    self.set_voxels[x] = {}
                self.all_checked_voxels[x][y].set() # Set as a valid page
                self.set_voxels[x][y] = self.all_checked_voxels[x][y]
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

        if not os.path.exists(self.save_path + page):
            os.mkdir(self.save_path + page)
        cv2.imwrite(self.save_path + page + "/" + str(slice) + "_mask" + "_avg=" + str(avg) + ".tif", im)

        skeleton, img = self.skeletonize(visited, orig_im.copy())

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
    '''
    def skeletonize(self, points, img):
        #skeleton, img = self.thin_cloud(15, img)
        #skeleton, img = self.do_bfs(self.set_voxels, img)
        #TODO: first clean the point cloud to eliminate non-connected single voxels (so A* doesn't pick those as start/end points)
        skeleton, img = self.do_a_star(self.set_voxels, img)
        for vx in skeleton:
            img[vx[1]][vx[0]] = (0, 255, 0)
        #First: get all the active/set points that are not completely surrounded by active/set neighbors.
        #un-set those points.
        #add their neighbors to the next round's 'stack'?
        #repeat for a specified depth, or until every remaining pixel has two or fewer set/active neighbors?
        #remove remaining pixels with no set/active neighbors?

        return skeleton, img

    def do_bfs(self, point_cloud, img):
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

        #Do BFS using set_voxels as the traversable points from min y until we reach max y
        #return this path as the skeleton
        #TODO: or A*? (Euclidean heuristic?) or just Greedy BFS since there's not much to confuse it?

        return point_cloud, img #TODO: fix later
        #TODO: what if the path is broken... try to maximize y?

    #TODO: handle breaks in the page
    def do_a_star(self, point_cloud, img):
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
            current_pos = list_of_moves.pop(0) #Take the top move and do it
            visited_voxels.append(current_pos)
            x = [-1, 0, 1]
            y = [-1, 0, 1]
            for i in y:  # height
                for j in x:  # width
                    if (i != 0 or j != 0) and (img[int(current_pos[0][1] + i)][int(current_pos[0][0] + j)][0] > self.low_tolerance): #check that the neighbor is a valid grey level to be set
                        tmp_pos = tuple(((current_pos[0][0] + j, current_pos[0][1] + i), current_pos[0], current_pos[2]+1, self.calculate_f(max_y, (current_pos[0][0] + j, current_pos[0][1] + i), current_pos)))
                        if tmp_pos not in visited_voxels:
                            if self.dest_not_visited(tmp_pos[0], visited_voxels) or tmp_pos[3] < self.get_f_of_existing(tmp_pos, visited_voxels):
                                if not tmp_pos[3] >= self.get_f_of_existing(tmp_pos, list_of_moves) or self.get_f_of_existing(tmp_pos, list_of_moves) == -1:
                                    list_of_moves.append(tmp_pos)
            list_of_moves.sort(key=lambda x:x[3]) #Sort by f value so we pick the smallest f every time
        #Do a* with Euclidean distance calculations
        skeleton.append(current_pos[0])
        skeleton = self.find_shortest_path(skeleton, visited_voxels, max_y, min_y)
        return skeleton, img

    '''
    A* UTILITIES
    '''

    #Travel from the goal to the start, generating the shortest path found by the algorithm.
    def find_shortest_path(self, skeleton, node_relationships, goal, start):
        current_location = goal
        pathComplete = False  # flag
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
        h = self.get_euclidean_dist_from_goal(new_loc, dest)
        g = current_loc_tuple[2]+1
        f = g + h
        return f

    '''
    Calculate the euclidean distance between 2 points
    '''
    def get_euclidean_dist_from_goal(self, src, dest):
        delta_x = abs(src[0] - dest[0])
        delta_y = abs(src[1] - dest[1])
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        return distance
    '''
    END A* UTILITIES
    '''

    '''
    This does not work well.
    Utilize all_checked_voxels and set_voxels dictionaries to determine the outermost points.
    Unset those outermost points, and keep doing that to the new outermost points until you reach a desired number of iterations/thickness.
    Send an 'unset' command to all voxels who have at least one voxel neighbor that is not set as part of the page. Then do it again, and again until ^
    Inputs: 
        number of iterations to thin the cloud
        the image to draw the intermediates on
    '''
    def thin_cloud(self, iterations, img):
        #get the difference between all_checked_voxels and set_voxels?
        complete_set = []
        for key in self.all_checked_voxels.keys():
            for key2 in self.all_checked_voxels[key].keys():
                complete_set.append((key, key2))

        set_set = []
        for key in self.set_voxels.keys():
            for key2 in self.set_voxels[key].keys():
                set_set.append((key, key2))
        #For now, these are just all 'non-page' voxels. No thinning done yet
        inactive_bound_voxels = set(complete_set) - set(set_set)

        voxels_to_deactivate = []
        for i in range(0, iterations):
            #Now we have the voxels that we checked but they weren't within the grey threshold. Find the active voxels who have any of these as a neighbor. These will be our outermost boundary voxels. (hopefully)
            for x in self.set_voxels:
                for y in self.set_voxels[x]:
                    voxel = self.set_voxels[x][y]
                    neighbors = set(voxel.neighbors)
                    inactive_neighbors = inactive_bound_voxels.intersection(neighbors)
                    if len(inactive_neighbors) > 0: #if any inactive neighbors exist, we want to de-activate this voxel.
                        voxels_to_deactivate.append((x, y)) #TODO: do we want to deactivate them after the fact and loop through again? Or?
            for vx in voxels_to_deactivate:
                inactive_bound_voxels.add((vx[0], vx[1]))
        #get all the set voxels in a format that's going to be nicer to work with (and now that we've deactivated some voxels, we can't re-use what we did above)
        active_voxels = set(set_set) - set(voxels_to_deactivate)
        print("Removed ", len(set_set) - len(active_voxels), " voxels. Total remaining: ", len(active_voxels))
        return active_voxels, img


    '''
    Use Voronoi diagrams to find the skeleton in the center of the page.
    '''
    def get_voronoi_skeleton(self, slice_pts):
        return 0  # TODO

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

                    #TODO: below line does not work
                    #if (x_pos + j) not in self.all_checked_pixels or (y_pos + i) not in self.all_checked_pixels[x_pos + j]: #If this exact point has not been checked before, check it.
                    if (x_pos + j) not in self.all_checked_voxels:
                        self.all_checked_voxels[x_pos + j] = {}
                    self.all_checked_voxels[x_pos + j][y_pos + i] = Voxel(x_pos + j, y_pos + i)

                    if grey_val > self.low_tolerance: #and grey_val < self.high_tolerance:
                        if (x_pos + j) not in self.set_voxels:
                            self.set_voxels[x_pos + j] = {}
                        self.all_checked_voxels[x_pos + j][y_pos + i].set() #since it fits in the tolerances, this pixel will be active.
                        self.set_voxels[x_pos + j][y_pos + i] = self.all_checked_voxels[x_pos + j][y_pos + i]
                        valid_neighbors.append((x_pos + j, y_pos + i))

                        #At this indentation level, there should exist voxels (on the outside of the cloud) that do not have 8 neighbors
                #At this indentation level, all voxels should have 8 neighbors
                if i != 0 or j != 0:
                    self.set_voxels[x_pos][y_pos].set_neighbor(x_pos + j, y_pos + i) #TODO: set x, y RELATIVE to the point so they will be easier to access. Point to the actual Pixel object too. Use all_checked_pixels because a point is not guaranteed to be on the page.
        return valid_neighbors

    # Taken from TrainingMaskGenerator's implementation
    #what distance metric? Right now--Euclidean
    def calculate_distance_from_origin(self, point, origin):
        return self.get_euclidean_dist_from_goal(point, origin)
        '''delta_x = abs(point[0] - origin[0])
        delta_y = abs(point[1] - origin[1])
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        return distance'''

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

#page num : seg num
pages = {
    "1" : "20191114132257",
    "2" : "20191125215208",
    "3" : "20191114133552",
    "11" : "20191123203022"
}
HI_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180122092342/"
#LOW_RES_PATH = "/Volumes/Research/1. Research/MS910.volpkg/volumes/20180220092522/"
page = "2"
segmentation_number = pages[page]
pointset_path = "/Volumes/Research/1. Research/MS910.volpkg/paths/" + segmentation_number
save_path = "/Volumes/Research/1. Research/Experiments/ExtrapolateMask/"
start_slice = 0
num_iterations = 30


ex = MaskExtrapolator(HI_RES_PATH, pointset_path, page, save_path, start_slice, num_iterations)
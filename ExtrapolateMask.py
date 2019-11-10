import VCPSReader as vr
import ujson
'''
Given an original slice and an original pointset tracing a single page through the slice, extrapolate in 3D to subsequent slices.
3D floodfill as a start?
Use Voronoi diagrams to find the skeleton in the center of the page, then seed those points on the next slice (if they are on a page?)
Might want a faster way to skeletonize, or perhaps a different approach altogether?
'''
class MaskExtrapolator:
    def __init__(self, vol_path, path_to_pointsets, page):
        self.path_to_volume = vol_path
        self.page = page

        #Read in the start points
        self.orig_pts = self.read_points(path_to_pointsets)
        self.pts = self.orig_pts.copy()

        #Find the slice we'll start with (the first one if multiple were provided)
        self.start_slice = list(self.orig_pts.keys())[0] #the slice for which we have the initial pointset
        self.current_slice = list(self.orig_pts.keys())[0]

        #Do flood fill for this slice

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
    def do_3d_floodfill(self):
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
    def do_2d_floodfill(self):
        return 0 #TODO

    '''
    For 2D floodfill:
    Given the flood-filled current slice, seed the next slice.
    '''
    def seed_next_slice(self):
        return 0  # TODO

    '''
    Use Voronoi diagrams to find the skeleton in the center of the page.
    '''
    def get_voronoi_skeleton(self, slice_pts):
        return 0  # TODO
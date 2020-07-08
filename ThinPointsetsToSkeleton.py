import argparse
import glob
import os

import ujson


def thin_pts(path):
    newpath = os.path.join(path, os.path.join("parts", os.path.join("incomplete", "*.txt")))
    files = sorted(glob.glob(newpath))
    for file in files:
        print(file)
        thinned_pts = {}
        f = open(file)
        temp_pts = ujson.load(f)
        # TODO: apply the thinning algorithm for each mask of each slice. Save the skeleton.
        for slice in temp_pts:
            for n, pt in enumerate(temp_pts[slice]):
                temp_pts[slice][n] = tuple(pt)
            skeleton = thin_cloud_continuous(temp_pts[slice])
            thinned_pts[slice] = skeleton
        #Every finished file will be saved.
        filename = os.path.splitext(file)[0]
        part_file = open(os.path.join(path, filename + "_skeleton.txt"), 'w')
        part_file.write(ujson.dumps(thinned_pts))
        f.close()





'''Morphological Thinning
Thin cloud to get a skeleton and produce a continuous skeleton.

This is the basic implementation given in section 8.6.2 of "Computer Vision", 5th Edition, by E.R. Davies
On average, it takes 5 minutes to segment a layer. Not good.
'''
def thin_cloud_continuous(points):
    skeleton = points.copy()

    while (True):
        skeleton, thinned_n = strip_north_pts(skeleton)
        skeleton, thinned_s = strip_south_pts(skeleton)
        skeleton, thinned_e = strip_east_pts(skeleton)
        skeleton, thinned_w = strip_west_pts(skeleton)

        # If no thinning occurred in this last iteration, we are finished.
        if not (thinned_n or thinned_s or thinned_e or thinned_w):
            break

    return skeleton


def strip_north_pts(mask):
    thinned = False
    points_to_remove = []
    for point in mask:
        # get neighbors we care about: (x, y+1) and (x, y-1)
        x = point[0]
        y = point[1]
        sigma, chi = calculate_params(point, mask)
        # check if chi == 2 and sigma != 1:
        # (implied that center pixel is active if it's in the mask, so no need to check)
        if chi == 2 and sigma != 1:
            if tuple((x, y + 1)) not in mask:  # i.e. it's 0
                if tuple((x, y - 1)) in mask:  # i.e. it's 1
                    # remove the pixel from the mask
                    points_to_remove.append(point)
                    thinned = True
    print("Removing ", len(points_to_remove), " points.")
    for point in points_to_remove:
        mask.remove(point)
    return mask, thinned


def strip_south_pts(mask):
    thinned = False
    points_to_remove = []
    for point in mask:
        # get neighbors we care about: (x, y+1) and (x, y-1)
        x = point[0]
        y = point[1]
        sigma, chi = calculate_params(point, mask)
        # check if chi == 2 and sigma != 1:
        # (implied that center pixel is active if it's in the mask, so no need to check)
        if chi == 2 and sigma != 1:
            if tuple((x, y - 1)) not in mask:
                if tuple((x, y + 1)) in mask:
                    # remove the pixel from the mask
                    points_to_remove.append(point)
                    thinned = True
    print("Removing ", len(points_to_remove), " points.")
    for point in points_to_remove:
        mask.remove(point)
    return mask, thinned


def strip_east_pts(mask):
    thinned = False
    points_to_remove = []
    for point in mask:
        # get neighbors we care about: (x, y+1) and (x, y-1)
        x = point[0]
        y = point[1]
        sigma, chi = calculate_params(point, mask)
        # check if chi == 2 and sigma != 1:
        # (implied that center pixel is active if it's in the mask, so no need to check)
        if chi == 2 and sigma != 1:
            if tuple((x + 1, y)) not in mask:  # '0'
                if tuple((x - 1, y)) in mask:  # '1'
                    # remove the pixel from the mask
                    points_to_remove.append(point)
                    thinned = True
    print("Removing ", len(points_to_remove), " points.")
    for point in points_to_remove:
        mask.remove(point)
    return mask, thinned


def strip_west_pts(mask):
    thinned = False
    points_to_remove = []
    for point in mask:
        # get neighbors we care about: (x, y+1) and (x, y-1)
        x = point[0]
        y = point[1]
        sigma, chi = calculate_params(point, mask)
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


def calculate_params(point, mask):
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
    sigma = calculate_sigma(point, mask)
    return sigma, chi


def calculate_sigma(point, mask):
    row = [0, 1, -1]
    col = [0, 1, -1]
    pt_x = point[0]
    pt_y = point[1]
    ctr = 0
    for x in row:
        for y in col:
            if x != 0 or y != 0:  # ignore (0,0)
                if tuple((pt_x + x, pt_y + y)) in mask:
                    ctr += 1
    return ctr


parser = argparse.ArgumentParser(description="Thin pointsets to a skeleton.")
parser.add_argument("pointset_path", type=str, help="Full path to the directory containing the mask files.")
args = parser.parse_args()

thin_pts(args.pointset_path)
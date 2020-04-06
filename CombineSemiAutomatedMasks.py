import glob
import os

import cv2
import numpy as np
import ujson

'''
path to mask directories
'''
def combine_semi_automated_masks(path):
    page_data = {}

    for page in range(0, 16):
        file = glob.glob(os.path.join(path, "synthetic_" + str(page), "synthetic_" + str(page) + ".txt*"))#glob.glob(path + "synthetic_" + str(page) + "/*.txt" )#os.path.join(path, "synthetic_" + str(page)) + "*.txt")
        print("Page " + str(page))
        f = open(file[0])
        data = ujson.load(f)
        page_data[page] = data

    complete_file = open(os.path.join(path, "complete/complete_masks.txt"), 'w')
    complete_file.write(ujson.dumps(page_data))
    make_instance_volume(os.path.join(path, "complete/"), "/Volumes/Research/1. Research/Experiments/M910TrainingSubvolumes/grayscale/300/synthetic_x83_y1347_s0")
    return page_data

def make_instance_volume(path_to_complete_dir, path_to_raw_imgs):
    imgs = glob.glob(os.path.join(path_to_raw_imgs, "*.tif")) #Just load the image to get the desired shape... We're going to make a black image with the same shape.
    im = cv2.imread(imgs[0], 0)

    f = open(os.path.join(path_to_complete_dir, "complete_masks.txt"))
    mask_pts = ujson.load(f)

    pg_colors = {} #Each page will have its own unique grey value (chosen randomly)

    for slice, img in enumerate(imgs): #For now, we're assuming we start from slice 0 because that is all I need.
        img = np.zeros((im.shape[1], im.shape[0]), dtype="uint8")  # black image

        for page in mask_pts:
            if page in pg_colors:
                pg_color = pg_colors[page]
            else:
                pg_color = np.random.choice(range(256), size=1)[0]
                while pg_color in pg_colors:  # Not sure this works, check it out. Meant to try to avoid assigning same color to multiple pages. (Probably won't happen, but would be a problem if it did.)
                    pg_color = np.random.choice(range(256), size=1)
                pg_colors[page] = pg_color
                #pg_color = pg_colors[page]

            for pt in mask_pts[page][str(slice)]:
                y = pt[1]
                x = pt[0]
                img[y][x] = pg_color
        cv2.imwrite(os.path.join(path_to_complete_dir, str(slice).zfill(4)+".tif"), img)

combine_semi_automated_masks("/Volumes/Research/1. Research/Experiments/ExtrapolateMask/Synthetic/")
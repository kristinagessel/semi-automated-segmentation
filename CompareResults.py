'''
Contains functions that allow us to compare across the different forms of segmentation we have.
- VC manual
- Flood-Fill Network
- Semi-Automated Segmentation
- Manually-Annotated Ground Truth

'''
import argparse
import glob
import os
import numpy as np

import cv2

'''
Process the hand-made ground truth mask -- literally just count each unique live pixel and assign it to the object label it belongs to.
Input: absolute path to the mask image
Output: A dictionary containing a count of each identified instances's pixel locations.
'''
def process_mask(image):
    img_shape = image.shape
    instance_dict = {} #use to keep track of what pixels belong to each individual page

    #Check all pixels in the image
    for y in range(0, img_shape[0]): #y
        for x in range(0, img_shape[1]): #x
            intensity = image[y][x]
            if intensity != 0: #black means background. For anything that isn't the background, add it to the dictionary.
                if str(intensity) not in instance_dict:
                    instance_dict[str(intensity)] = [tuple((x, y))]
                else:
                    instance_dict[str(intensity)].append(tuple((x, y)))

    return instance_dict


#TODO: can calculate the IoU of the flood-fill from VC points only for a single image -- slice 0 of the ground truth and mask.
'''
Intersection Over Union (IoU) (Entire Instance Mask?)
Calculating IoU of masks.
Between:
    Manually-Annotated Ground Truth and Flood-Filling Network
    Manually-Annotated Ground Truth and Semi-Automated Segmentation (Need to generate this segmentation)
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)
    Flood-Filling Network and Manually-Annotated Ground Truth (Over time, try all the inference results I have and compare them. Hopefully the IoU increases with more training.)
Input: OpenCV images of the ground truth and the segmentation
'''
def intersection_over_union(ground_truth, segmentation):
    iou_scores = []
    max_intersection = []
    max_union = []
    for gt_list_elem in ground_truth:
        for seg_list_elem in segmentation:
            intersect = set(ground_truth[gt_list_elem]).intersection(set(segmentation[seg_list_elem]))
            #Trying to find the instance with the most overlap to judge from since these pages don't have clean labels that will agree with the ground truth:
            if len(intersect) > len(max_intersection):
                max_intersection = intersect
                max_union = set(ground_truth[gt_list_elem]).union(set(segmentation[seg_list_elem]))
                #print("IoU is: " + str(len(max_intersection)) + " / " + str(len(max_union)) + " = " + str(len(max_intersection)/len(max_union)))
                iou_scores.append(len(max_intersection)/len(max_union))
    #compute an average IoU score for this segmentation using the iou_scores list:
    average_iou = np.average(iou_scores)
    return average_iou

'''
TODO: Intersection Over Union (IoU) (Single Page Mask?)
Between:
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)
'''



'''
Calculate precision and recall metrics
Between:
    Manually-Annotated Ground Truth and Flood-Filling Network
    Manually-Annotated Ground Truth and Semi-Automated Segmentation (Need to generate this segmentation)
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Not possible because I can't load synthetic subvolume in VC)
Input: OpenCV images of the ground truth and the segmentation
'''
def precision_and_recall(ground_truth, segmentation):
    #To determine which segmentation belongs to which ground truth, at least some of the pixels between both have to be the same.
    #Perhaps do an intersection between them and see which intersection has the greatest overlap. Then, go with that for checking true positives.
    #find true positives (agree across ground truth and segmentation that it is page)
    max_intersection = 0
    false_negatives = []
    false_positives = []
    precision_list = []
    recall_list = []
    for gt_list_elem in ground_truth:
        for seg_list_elem in segmentation:
            intersect = set(ground_truth[gt_list_elem]).intersection(set(segmentation[seg_list_elem]))
            # Trying to find the instance with the most overlap to judge from since these pages don't have clean labels that will agree with the ground truth:
            if intersect > max_intersection:
                max_intersection = intersect

                # False negatives would be the set difference of ground truth - segmentation
                false_negatives = set(ground_truth[gt_list_elem]) - set(segmentation[seg_list_elem])

                # False positives would be the set difference of segmentation - ground truth
                false_positives = set(segmentation[seg_list_elem]) - set(ground_truth[gt_list_elem])
            print("Computed intersection")
        true_positives = max_intersection
        precision = true_positives / (true_positives + len(false_positives))
        precision_list.append(precision)
        recall = true_positives / (true_positives + len(false_negatives))
        recall_list.append(recall)
    avg_precision = np.average(precision_list)
    avg_recall = np.average(recall_list)
    return avg_precision, avg_recall #TODO: return either an average precision/recall number or individual.

'''
Pixel Accuracy
Between:
    Manually-Annotated Ground Truth and Flood-Filling Network
    Manually-Annotated Ground Truth and Semi-Automated Segmentation (Need to generate this segmentation)
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)
Input: OpenCV images of the ground truth and the segmentation and the total number of pixels in either image.
'''
def pixel_accuracy(ground_truth, segmentation, pixel_num):
    #To determine which segmentation belongs to which ground truth, at least some of the pixels between both have to be the same.
    #Perhaps do an intersection between them and see which intersection has the greatest overlap. Then, go with that for checking true positives.
    #find true positives (agree across ground truth and segmentation that it is page)
    max_intersection = set()
    true_negatives = 0 #Initialize to make PyCharm happy
    false_negatives = []
    false_positives = []
    accuracy_list = []
    for gt_list_elem in ground_truth:
        for seg_list_elem in segmentation:
            intersect = set(ground_truth[gt_list_elem]).intersection(set(segmentation[seg_list_elem]))
            # Trying to find the instance with the most overlap to judge from since these pages don't have clean labels that will agree with the ground truth:
            if len(intersect) > len(max_intersection):
                max_intersection = intersect

                # False negatives would be the set difference of ground truth - segmentation
                false_negatives = set(ground_truth[gt_list_elem]) - set(segmentation[seg_list_elem])

                # False positives would be the set difference of segmentation - ground truth
                false_positives = set(segmentation[seg_list_elem]) - set(ground_truth[gt_list_elem])

                #True negatives is the intersection of the inverse of the masks? Or the entire size of the image minus the sum of all of the 3 above?
                true_negatives = pixel_num - len(max_intersection) - len(false_negatives) - len(false_positives)

            #print("Computed intersection")
        true_positives = max_intersection
        if (len(true_positives) + true_negatives + len(false_positives) + len(false_negatives)) > 0:
            accuracy = (len(true_positives) + true_negatives) / (len(true_positives) + true_negatives + len(false_positives) + len(false_negatives))
        else:
            accuracy = 0
        accuracy_list.append(accuracy)
    avg_accuracy = np.average(accuracy_list)
    return avg_accuracy



def pixel_wise_difference(ground_truth_mask, segmentation_mask):
    difference_list = []
    for i, gt_img in enumerate(sorted(glob.glob(os.path.join(args.ground_truth_path, "*")))):
        for j, seg_img in enumerate(sorted(glob.glob(os.path.join(args.segmentation_path, "*")))):
            if i == j:  # We only care about comparing the same two images.
                difference = len(ground_truth_mask) - len(segmentation_mask)
                difference_list.append(difference)
    return np.average(difference_list) #return the pixel-wise difference (+ if ground truth mask has more, - if segmentation mask has more)

'''
TODO: Point Cloud Comparison (Find algorithm)
Between:
    Flood-Filling Network and Semi-Automated Segmentation (Can't do a manually-annotated ground truth, unless I use the existing synthetic one and run Semi-Automated on it.)

'''

'''
Execute:
'''
def compute_iou_single_image(gt, seg):
    gt_pts = process_mask(gt)
    seg_pts = process_mask(seg)
    result = intersection_over_union(gt_pts, seg_pts)
    print("IoU: " + str(result))

'''Input: arguments given to the script.'''
def analyze_3d_volumes(args):
    complete_iou = []
    complete_pixel_accuracy = []
    pixel_difference = []

    # We want the masks, not the skeletons (for semi-automated segmentation)
    for i, gt_img in enumerate(sorted(glob.glob(os.path.join(args.ground_truth_path, "*")))):
        for j, seg_img in enumerate(sorted(glob.glob(os.path.join(args.segmentation_path, "*")))):
            if i == j:  # We only care about comparing the same two images.
                gt_im = cv2.imread(gt_img, 0)
                ground_truth_dict = process_mask(gt_im)

                seg_im = cv2.imread(seg_img, 0)
                seg_dict = process_mask(seg_im)

                # pixel_difference.append(pixel_wise_difference(ground_truth_dict, seg_dict))

                iou_result = intersection_over_union(ground_truth_dict, seg_dict)
                print(str(i) + " Intersection over union: " + str(iou_result))
                if not np.math.isnan(iou_result):
                    complete_iou.append(iou_result)
                else:
                    complete_iou.append(0.)

    # print("Average pixel difference: ", np.average(pixel_difference)) #Not helpful.
    print("Average IoU: ", np.average(complete_iou))



parser = argparse.ArgumentParser(description="Perform semi-automated segmentation with flood-filling and skeletonization")
parser.add_argument("ground_truth_path", type=str, help="Path to the ground truth image stack") #The only real ground truth hand-annotated image stack I have is synthetic. I can run semi-automated seg on it maybe, FFN has been.
parser.add_argument("segmentation_path", type=str, help="Path to the segmentation image stack")
args = parser.parse_args()

gt_img = cv2.imread(args.ground_truth_path, 0)
seg_img = cv2.imread(args.segmentation_path, 0)
compute_iou_single_image(gt_img, seg_img)


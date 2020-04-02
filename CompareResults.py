'''
Contains functions that allow us to compare across the different forms of segmentation we have.
- VC manual
- Flood-Fill Network
- Semi-Automated Segmentation
- Manually-Annotated Ground Truth

'''
import cv2

'''
Process the hand-made ground truth mask -- literally just count each unique live pixel and assign it to the object label it belongs to.
'''
def process_hand_annotated_mask(mask_image_path):
    gt_mask = cv2.imread(mask_image_path, 0)  # open the image of this slice (as greyscale)
    img_shape = gt_mask.shape
    instance_dict = {} #use to keep track of what pixels belong to each individual page

    #Check all pixels in the image
    for y in range(0, img_shape[0]): #y
        for x in range(0, img_shape[1]): #x
            intensity = gt_mask[y][x]
            if intensity != 0: #black means background. For anything that isn't the background, add it to the dictionary.
                if str(intensity) not in instance_dict:
                    instance_dict[str(intensity)] = tuple((x, y))

    return instance_dict


'''
TODO: Mean Square Error?
Between:
    Manually-Annotated Ground Truth and Flood-Filling Network
    Manually-Annotated Ground Truth and Semi-Automated Segmentation (Need to generate this segmentation)
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)
'''


'''
TODO: Intersection Over Union (IoU) (Entire Instance Mask?)
Calculating IoU of masks.
Between:
    Manually-Annotated Ground Truth and Flood-Filling Network
    Manually-Annotated Ground Truth and Semi-Automated Segmentation (Need to generate this segmentation)
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)
    

'''

'''
TODO: Intersection Over Union (IoU) (Single Page Mask?)
Between:
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)


'''

'''
TODO: Pixel Accuracy
(Just do a difference between the complete mask and the ground truth.)
Between:
    Manually-Annotated Ground Truth and Flood-Filling Network
    Manually-Annotated Ground Truth and Semi-Automated Segmentation (Need to generate this segmentation)
    Manually-Annotated Ground Truth and Semi-Automated Segmentation and Bounded Flood-Fill on VC Points (Need to generate semi-automated for these)
'''
def pixel_wise_difference(ground_truth_mask, segmentation_mask):
    return len(ground_truth_mask) - len(segmentation_mask) #return the pixel-wise difference (+ if ground truth mask has more, - if segmentation mask has more)

'''
TODO: Point Cloud Comparison (Find algorithm)
Between:
    Flood-Filling Network and Semi-Automated Segmentation (Can't do a manually-annotated ground truth, unless I use the existing synthetic one and run Semi-Automated on it.)

'''
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def iou_validation(box1, box2):

    # Get the Width and Height of each bounding box
    width_b1 = box1[2]
    height_b1 = box1[3]
    width_b2 = box2[2]
    height_b2 = box2[3]
    
    # Find the vertical and the Horizontal edges of the union of the two bounding boxes
    # IOU = Area of Intersection/ Area of Union
    min_x = min(box1[0] - width_b1/2.0, box2[0] - width_b2/2.0)
    max_x = max(box1[0] + width_b1/2.0, box2[0] + width_b2/2.0)
    min_y = min(box1[1] - height_b1 / 2.0, box2[1] - height_b2 / 2.0)
    max_y = max(box1[1] + height_b1 / 2.0, box2[1] + height_b2 / 2.0)

    # Calculate the width and height of the union of the two bounding boxes
    union_width = max_x - min_x
    union_height = max_y - min_y
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_b1 + width_b2 - union_width
    intersection_height = height_b1 + height_b2 - union_height
   
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height

    # Calculate the area of the each bounding box
    area_box1 = width_b1 * height_b1
    area_box2 = width_b2 * height_b2
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area / union_area
    
    return iou


def nms_validation(boxes, iou_thresh):
    
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes
    
    # Create a PyTorch Tensor to keep track of the detection confidence of each predicted bounding box
    detect_conf = torch.zeros(len(boxes))
    
    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        detect_conf[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence.
    _, sort_id = torch.sort(detect_conf, descending=True)
    
    # Create an empty list to store the best bounding boxes after Non-Maximal Suppression (NMS) is performed
    best_box = []

    # Perform NMS
    for i in range(len(boxes)):
        
        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sort_id[i]]
        
        # Check that the detection confidence is not zero
        if box_i[4] > 0:
            
            # Save the bounding box 
            best_box.append(box_i)
            # Go through the rest of the bounding boxes in the list and
            # calculate their IOU with respect to the previous selected box_i.
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sort_id[j]]
                
                # If the IOU of box_i and box_j is higher than the given IOU
                # threshold set box_j's detection confidence to zero.
                if iou_validation(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
                    
    return best_box


def detect_objects(model, img, iou_thresh, nms_thresh):

    # Start the time. This is done to calculate how long the detection takes.
    start = time.time()

    # Set the Darknet model to evaluation mode.
    model.eval()
    
    # Convert the image from a NumPy ndarray to a PyTorch Tensor with the correct shape.
    # The image is transposed, then converted to a FloatTensor of dtype float32.
    # It is then Normalized to values between 0 and 1 by dividing with 255.0
    # finally unsqueezed to have the correct shape of (1 x 3 x width x height)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    
    # Feed the image to the Darknet neural network model with the corresponding NMS threshold.
    # NMS is used to remove all bounding boxes that have a very low probability of detection.
    # All predicted bounding boxes with a value less than the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)
    
    # Create a new list with all the bounding boxes that are returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    
    # Now we perform NMS on the bounding boxes returned by the neural network.
    # Here, we keep the best bounding boxes and eliminate all the bounding boxes
    # whose IOU value is higher than the given IOU threshold
    boxes = nms_validation(boxes, iou_thresh)
    # Stop the time. 
    finish = time.time()

    # Total time taken to detect all the objects
    time_taken = round(finish-start)

    # Print the time it took to detect objects
    print('\n\nIt took {}'.format(time_taken), 'seconds to detect the objects in the image.\n')
    
    # Print the number of objects detected
    print('Total Number of Objects Detected:', len(boxes), '\n')
    
    return boxes, time_taken


def load_coco_names(names):
    
    # Create an empty list to hold the object classes
    class_names = []

    # Open the file containing the COCO object classes in read-only mode
    # The coco.names file contains only one object class per line.
    # Read the file line by line and save all the lines in a list.
    with open(names, 'r') as fp:
        lines = fp.readlines()

    # Get the object class names
    # Take the name in each line any remove any whitespaces
    # Append the object class name into class_names
    for name in lines:
        line = name.rstrip()
        class_names.append(line)
    return class_names


def label_objects(boxes, class_names):
    result = {}
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_id = box[6]
            w = class_names[cls_id]
            if w in result.keys():
                result[w] = result[w] + 1
            else:
                result[w] = 1
    return result


def plot_object_boxes(img, boxes, class_names):
    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]

    # Create a figure and plot the image
    _, t = plt.subplots(1, 1)
    t.imshow(img)

    # Plot the bounding boxes and their labels on top of the image
    for i in range(len(boxes)):

        # Get each bounding box
        box = boxes[i]

        # Get the (x,y) pixel coordinates of the upper-left and upper-right corners of the bounding box
        x1 = int(np.around((box[0] - box[2] / 2.0) * width))
        y1 = int(np.around((box[1] - box[3] / 2.0) * height))
        x2 = int(np.around((box[0] + box[2] / 2.0) * width))
        y2 = int(np.around((box[1] + box[3] / 2.0) * height))

        # Set the default rgb value to red
        rgb = (1, 0, 0)

        # Use the same color to plot the bounding boxes of the same object class
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            len_class = len(class_names)
            offset = cls_id * 123457 % len_class
            red = set_color(2, offset, len_class) / 255
            green = set_color(1, offset, len_class) / 255
            blue = set_color(0, offset, len_class) / 255
            rgb = (red, green, blue)

        # Calculate the width and height of the bounding box relative to
        # the size of the image.
        width_x = x2 - x1
        width_y = y1 - y2

        # Position the bounding boxes. (x1, y2) is the pixel coordinate of the upper-left corner of the bounding box
        rect = patches.Rectangle((x1, y2), width_x, width_y, linewidth=0.8, edgecolor=rgb, facecolor='none')

        # Draw the bounding box on top of the image
        t.add_patch(rect)

        # Create a string with the object class name and the corresponding object class Confidence
        conf_tx = class_names[cls_id] + ': {:.1f}'.format(cls_conf)

        # Define x and y offsets for the labels
        x_offset = (img.shape[1] * 0.200) / 100
        y_offset = (img.shape[0] * 1.140) / 100

        # Draw the labels on top of the image
        t.text(x1 + x_offset, y1 - y_offset, conf_tx, fontsize=6,
               color='k', bbox=dict(facecolor=rgb, edgecolor=rgb, alpha=0.8))

    plt.axis('off')
    plt.savefig("Object_Detection/static/test.jpeg", dpi=360, bbox_inches='tight')


def set_color(c, offset, len_class):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    ratio = float(offset) / len_class * 5
    p = int(np.floor(ratio))
    q = int(np.ceil(ratio))
    ratio = ratio - p
    r = (1 - ratio) * colors[p][c] + ratio * colors[q][c]
    return int(r * 255)

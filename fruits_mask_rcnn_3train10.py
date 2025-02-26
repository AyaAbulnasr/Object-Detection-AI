


# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from skimage import color
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#tf.config.run_functions_eagerly(True)

#import re
tf.compat.v1.enable_eager_execution()



# class that defines and loads the kangaroo dataset
class FruitsDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define classes
        self.add_class("dataset", 1, "apple")
        self.add_class("dataset", 2, "banana")
        self.add_class("dataset", 3, "orange")
        
        # define data locations
        images_dir = dataset_dir + '/images_sample/'
        annotations_dir = dataset_dir + '/annotation/'
       
             
		# find all images
        for filename in listdir(images_dir):
            print(filename)
			# extract image id
            image_id = filename[:-4]
			#print('IMAGE ID: ',image_id)
			
			# skip all images after 115 if we are building the train set
            if is_train and int(image_id) >= 8:
                continue
			# skip all images before 115 if we are building the test/val set
            if not is_train and int(image_id) < 8:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2,3])


	# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
		# load and parse the file
        tree = ElementTree.parse(filename)
		# get the root of the document
        root = tree.getroot()
		# extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text   #Add label name to the box list
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
		# extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

	# load the masks for an image
    def load_mask(self, image_id):
		# get details of image
        info = self.image_info[image_id]
		# define box file location
        path = info['annotation']
        #return info, path
        
        
		# load XML
        boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            
            
            # box[4] will have the name of the class 
            if (box[4] == 'apple'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('apple'))
            elif(box[4] == 'banana'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('banana')) 
            elif(box[4] == 'orange'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('orange'))
          
        return masks, asarray(class_ids, dtype='int32')
        

	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# train set
dataset_dir='dataset'

train_set = FruitsDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = FruitsDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


import random
num=random.randint(0, len(train_set.image_ids))
# define image id
image_id = num
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)



# define a configuration for the model
class FruitsConfig(Config):
    # Define the name of the configuration
    NAME = "fruits_cfg"
    BATCH_SIZE = 1
    # Number of classes (background + 3 fruits)
    NUM_CLASSES = 1 + 3
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 2
    # Pre NMS limit
    PRE_NMS_LIMIT = 6000
    # Minimum confidence for detection
    DETECTION_MIN_CONFIDENCE = 0.3
    GPU_COUNT = 1
    
    

# prepare config
config = FruitsConfig()
config.display()

import os
# Define a valid log directory
LOG_DIR = os.path.join(os.getcwd(), "logs2/profile")
LOG_DIR
# Ensure the directory exists
os.makedirs(LOG_DIR, exist_ok=True)

import tensorflow as tf

# Start the profiling before training
#log_dir = "logs2/profile"
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, profile_batch=0)

# Define your model and training configuration


# Set the log directory in the model config

###############
print("Training set path:", train_set)
print("Test set path:", test_set)


# define the model
model = MaskRCNN(mode='training', model_dir="logs2/profile", config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


# train weights (output layers or 'heads')
model.train(
    train_set,
    test_set,
    learning_rate=config.LEARNING_RATE,
    epochs=3,
    layers='heads'
)

tf.profiler.experimental.stop()
#########################################

#INFERENCE

###################################################

from matplotlib.patches import Rectangle


class PredictionConfig(Config):
    # Define the name of the configuration
    NAME = "fruits_cfg"
    # Number of classes (background + 3 fruits)
    NUM_CLASSES = 1 + 3
    # Simplify GPU config
    GPU_COUNT = 1
    Batch Size = 1
    IMAGES_PER_GPU = 1  
    PRE_NMS_LIMIT = 6000  
    DETECTION_MIN_CONFIDENCE = 0.3  


# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights(r'C:\Users\X1\Downloads\Fruits\logs\fruits_cfg20250201T0055\mask_rcnn_fruits_cfg_0003.h5', by_name=True)



#Test on a few images
import skimage
from matplotlib import pyplot
from matplotlib.patches import Rectangle

#Test on a single image

#fruit_img = skimage.io.imread("datasets/renamed_to_numbers/images/184.jpg") #Try 028, 120, 222, 171

#Download a new image for testing...
#https://c2.peakpx.com/wallpaper/603/971/645/fruit-fruit-bowl-fruits-apple-wallpaper-preview.jpg
fruit_img = skimage.io.imread("pictestgroup.jpg")
detected = model.detect([fruit_img])[0]


# Display the image
pyplot.imshow(fruit_img)
ax = pyplot.gca()

# Class names (assuming these are the classes for detection)
class_names = ['apple', 'banana', 'orange']

# Loop through the detected bounding boxes
for i, box in enumerate(detected['rois']):
    detected_class_id = detected['class_ids'][i]  # Get the class id for the box
    score = detected['scores'][i]  # Get the score for the box
    
    # Map the class id to the class name
    class_name = class_names[detected_class_id - 1]  # Adjust for 0-based indexing
    
    y1, x1, y2, x2 = box
    width, height = x2 - x1, y2 - y1
    
    # Draw annotations and rectangles for detected objects
    ax.annotate(f'{class_name}: {score:.2f}', (x1, y1), color='black', weight='bold', fontsize=10, ha='center', va='center')
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    ax.add_patch(rect)

# Show the image with annotations
pyplot.show()

# Check if all expected fields are present in the 'detected' object
print(f"Detected ROIs: {detected['rois']}")
print(f"Detected Scores: {detected['scores']}")
print(f"Detected Class IDs: {detected['class_ids']}")

# Run detection
results = model.detect([image], verbose=0)



##Evaluation metrics
print(f"Predicted Masks: {r['masks']}")

import numpy as np
from mrcnn.utils import compute_ap
from mrcnn.model import mold_image
from skimage import io

# Function to calculate IoU for a single pair of boxes
# Function to extract bounding boxes from masks
def extract_bboxes(masks):
    """
    Extract bounding boxes from a set of masks.
    masks: A numpy array of shape (height, width, num_instances) representing the masks
    Returns: A list of bounding boxes in the format [y1, x1, y2, x2]
    """
    bboxes = []
    for i in range(masks.shape[-1]):  # Iterate over each mask
        mask = masks[:, :, i]
        # Find the bounding box of the mask
        y_coords, x_coords = np.where(mask)  # Get the coordinates of the non-zero pixels
        y1, x1 = np.min(y_coords), np.min(x_coords)
        y2, x2 = np.max(y_coords), np.max(x_coords)
        bboxes.append([y1, x1, y2, x2])
    return np.array(bboxes)

# Inside your evaluate_model function
# Function to calculate Average Precision
from mrcnn.utils import compute_ap


import numpy as np
from mrcnn.utils import compute_ap

def compute_iou(gt_bbox, pred_bbox):
    """
    Compute the Intersection over Union (IoU) for two bounding boxes
    Args:
        gt_bbox: Ground truth bounding box [y1, x1, y2, x2]
        pred_bbox: Predicted bounding box [y1, x1, y2, x2]
    Returns:
        IoU: Intersection over Union score
    """
    y1_gt, x1_gt, y2_gt, x2_gt = gt_bbox
    y1_pred, x1_pred, y2_pred, x2_pred = pred_bbox

    # Compute the coordinates of the intersection rectangle
    y1_inter = max(y1_gt, y1_pred)
    x1_inter = max(x1_gt, x1_pred)
    y2_inter = min(y2_gt, y2_pred)
    x2_inter = min(x2_gt, x2_pred)

    # Compute the area of the intersection rectangle
    intersection_area = max(0, y2_inter - y1_inter) * max(0, x2_inter - x1_inter)

    # Compute the area of both bounding boxes
    gt_area = (y2_gt - y1_gt) * (x2_gt - x1_gt)
    pred_area = (y2_pred - y1_pred) * (x2_pred - x1_pred)

    # Compute the area of the union
    union_area = gt_area + pred_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def evaluate_model(dataset, model, cfg):
    APs = []
    IoUs = []
    
    for image_id in dataset.image_ids:
        # Load image, ground truth bounding boxes, and masks
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        
        # Extract ground truth bounding boxes from the masks
        gt_bbox = extract_bboxes(mask)
        
        # Run detection
        results = model.detect([image], verbose=0)
        
        # Check if results are empty
        if results:
            r = results[0]  # Access the first result from the list
            
            # Extract predicted bounding boxes and masks
            pred_bbox = r["rois"]
            pred_masks = r["masks"]
            pred_class_ids = r["class_ids"]
            pred_scores = r["scores"]
            
            # Compute Average Precision (AP) for each class
            AP, _, _, _ = compute_ap(mask, class_ids, pred_bbox, pred_class_ids, pred_scores, pred_masks)
            APs.append(AP)
            
            # Compute IoU for each ground truth box
            for i in range(len(gt_bbox)):
                iou_scores = [compute_iou(gt_bbox[i], pred_box) for pred_box in pred_bbox]
                if iou_scores:
                    IoUs.append(max(iou_scores))  # Take the highest IoU score for a ground truth box
        else:
            print("No detection results found.")
    
    # Calculate mean Average Precision and mean IoU
    mean_AP = np.mean(APs) if APs else 0
    mean_IoU = np.mean(IoUs) if IoUs else 0
    
    print(f"Mean AP: {mean_AP:.4f}")
    print(f"Mean IoU: {mean_IoU:.4f}")
    
    return mean_AP, mean_IoU


# Evaluate the model
mean_AP, mean_IoU = evaluate_model(test_set, model, cfg)

print("Evaluation Metrics:")
print(f"Mean Average Precision (mAP): {mean_AP:.4f}")
print(f"Mean Intersection over Union (IoU): {mean_IoU:.4f}")






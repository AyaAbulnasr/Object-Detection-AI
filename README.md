# Object-Detection-AI
Part of Object detection using Mask RCNN model. 2 codes inserted, simple task for fruits detection and the other incomplete insects in paddy field detection

# README - Fruits Detection using Mask R-CNN
## Project Overview
This project implements an object detection model using Mask R-CNN to detect and classify fruits (apples, bananas, and oranges) in images. The model is trained on a custom dataset and evaluation metrics using Mean Average Precision (mAP) and Intersection over Union (IoU) metrics.
The Implementation of Mask R-CNN I used for this project: Matterport Mask R-CNN

### Installation Requirements
### Dependencies
Libraries were used:

!pip install tensorflow numpy matplotlib scikit-image mrcnn

Additional dependencies:
•	Python 3.7.3
•	numpy==1.20.3
•	scipy==1.4.1
•	Pillow==8.4.0
•	cython==0.29.24
•	matplotlib
•	protobuf==3.20.0
•	scikit-image==0.16.2
•	tensorflow==2.0.0
•	keras== 2.2.4 
•	opencv-python==4.5.4.60
•	h5py==2.10.0
•	imgaug==0.4.0
•	Matterport Mask R-CNN library
•	scikit-image
•	ElementTree (for XML parsing)

### Dataset Structure
The dataset should be organized as follows:

/dataset
  /images_sample
    - image_1.jpg
    - image_2.jpg
    ...
  /annotation
    - image_1.xml
    - image_2.xml
    ...
•	images_sample/ contains images of fruits.
•	annotation/ contains XML files with bounding box information.
### Training the Model
1.	Prepare the Dataset
o	The dataset is loaded and split into training and testing sets.
o	XML annotations are parsed to extract bounding boxes and class labels.
2.	Train the Model
o	Load the COCO pre-trained weights (mask_rcnn_coco.h5).
o	Train for 3 epochs using the heads layers.
To start training, run:
python train_model.py

### Inference and Testing
After training, use the trained model to detect objects in new images:
python inference.py --image_path path_to_test_image.jpg
This will output the image with detected objects, bounding boxes, and confidence scores.
### Evaluation Metrics
The model is evaluated using:
•	Mean Average Precision (mAP)
•	Intersection over Union (IoU)
Run evaluation with:
python evaluate.py
Example Evaluation Results:

 ![image](https://github.com/user-attachments/assets/7a78a209-4203-40e4-b3a3-64a216dab450)

Mean AP: 0.2317
Mean IoU: 0.7284

Future Improvements
•	Increase dataset size for better accuracy.
•	Experiment with hyperparameters.
•	Implement real-time object detection.
•	Deploy on edge devices.
License
https://github.com/ahmedfgad/Mask-RCNN-TF2/blob/master/LICENSE


# README - Insects Detection using Mask R-CNN
This code is part of insects 'object' detection to minimize the effort and losses happening in agriculture and food field. Model developed using Mask-RCNN techbnique. It also highlights challenges and opportunities in objecting it and model challenges using IP102 dataset which is my local machine capability to process the high quality images without GPU usage.
<img width="383" alt="image" src="https://github.com/user-attachments/assets/5d1e9153-8d43-4914-b091-eab7e4e1aee6" />

The project is incomplete yet.


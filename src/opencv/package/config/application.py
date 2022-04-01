##
# Copyright 2022, Ammar Ali Khan
# Licensed under MIT.
##
import cv2

# OpenCV configuration

# Caffe models configurations
CAFFE_MODELS = {
    'object_net': ['./src/opencv/dnn/prototxt/deploy_mobile_net_ssd.prototxt.txt',
                   './src/opencv/dnn/model/deploy_mobile_net_ssd.caffemodel',
                   (300, 300),
                   (127.5, 127.5, 127.5)]

}
# Default Caffe models confidence
CONFIDENCE = 0.2

# Labels of objects
CLASS_LABELS = { 
    0: 'background',
    1: 'aeroplane', 
    2: 'bicycle', 
    3: 'bird', 
    4: 'boat',
    5: 'bottle', 
    6: 'bus', 
    7: 'car', 
    8: 'cat', 
    9: 'chair',
    10: 'cow', 
    11: 'diningtable', 
    12: 'dog', 
    13: 'horse',
    14: 'motorbike', 
    15: 'person', 
    16: 'pottedplant',
    17: 'sheep', 
    18: 'sofa', 
    19: 'train', 
    20: 'tvmonitor' 
}

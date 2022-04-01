##
# Copyright 2022, Ammar Ali Khan
# Licensed under MIT.
##

import cv2
from src.opencv.package.config import application as config_opencv
from src.common.package.io.handler import Handler as io_handler


##
# Handler class
# This class is a wrapper for Open Source Computer Vision (OpenCV)
#
# @see: https://opencv.org/
##
class Handler:

    def __init__(self):
        print('[INFO] OpenCV - Initialising...')

        # Caffe models
        self.model_object = config_opencv.CAFFE_MODELS['object_net']

        self._object_net = cv2.dnn.readNetFromCaffe(io_handler.absolute_path(self.model_object[0]),
                                                    io_handler.absolute_path(self.model_object[1]))

        # Multi trackers
        self.multi_tracker = cv2.MultiTracker_create()

    ##
    # Method dnn_object_detector()
    # Method to return OpenCV dnn_object_detector
    #
    # @param frame - image frame
    # @param scale_factor - reduce image
    # @param swap_rb - swap channel
    #
    # @return Array of detection(s)
    ##
    def dnn_object_detector(self,   
                            frame,
                            scale_factor=1.0,
                            swap_rb=True):

        size = self.model_object[2]
        mean = self.model_object[3]

        # Convert frame to a blob
        blob = cv2.dnn.blobFromImage(image=cv2.resize(frame, size),
                                     scalefactor=scale_factor,
                                     size=size,
                                     mean=mean)

        # Pass the blob through the network and obtain the detections and predictions
        self._object_net.setInput(blob)
        return self._object_net.forward()

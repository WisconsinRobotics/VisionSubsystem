# -*- coding: utf-8 -*-
"""
@author: David Fei
"""
import tensorflow as tf
import cv2 as cv

# Make sure opencv is working
print(cv.__version__)

classes = ['tennis ball', 'no tennis ball']
num_classes = len(classes)
train_path = 'training_data'

# validation split
# This value can be adjusted if we want to use more training images for validation.
validation_size = 0.2

batch_size = 16

# TODO: Implement the dataset class to give it the functionality to read images from the training
# and test data folders.
# data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:06:01 2017

@author: David Fei
"""
import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed to display the images.
# %matplotlib inline

# Object detection module stuff
# To use this, we have to clone the tensorflow models repo and add the directory that contains 
# the object detection library to the path.
# For this to work in all cases, make sure to clone the models repo into the same directory as the
# VisionSubsystem repo.
sys.path.append('../../models/research/')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Specify the model to use
# FIXME: These variables should all be filled in once our model has been created.
MODEL_NAME = ''
MODEL_FILE = MODEL_NAME + '.tar.gz'
# Path to frozen detection graph - this is the actual model
PATH_TO_CKPT = MODEL_NAME + ''
# List of the strings that is used to add correct label for each box.
# We only need to classify tennis balls, so only one class is needed.
PATH_TO_LABELS = os.path.join('', '')
NUM_CLASSES = 1

# Load the frozen Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Map array indices to label names (e.g. when the CNN predicts 0 use the label "tennis ball")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, 
                                                            max_num_classes=NUM_CLASSES, 
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Image loading function
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

# Specify path to the images to use for the classifier
# FIXME: Fill in these variables when we have images to classify
PATH_TO_TEST_IMAGES_DIR = ''
NUM_IMGS = 0
TEST_IMAGE_PATHS = []
for i in range(1, NUM_IMGS):
    TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)))

# Image size specification
IMG_SIZE = (10, 10)

# Tensorflow test code
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
print(sess.run([node1, node2]))



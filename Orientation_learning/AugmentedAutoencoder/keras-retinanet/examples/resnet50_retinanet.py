#!/usr/bin/env python
# coding: utf-8

# Load necessary modules

import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import pathlib
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf




# ## Load RetinaNet model

# In[ ]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = '/home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/snapshots/resnet50_csv_13_inference.h5'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'tool'}


# ## Run detection on example

# In[ ]:

image_folder = '/home/user/Orientation_learning/AugmentedAutoencoder/test_images_tool/'
bbox_folder = '/home/user/Orientation_learning/AugmentedAutoencoder/test_images_tool/bboxes/'
if not os.path.exists(bbox_folder):
            os.makedirs(bbox_folder)
file_names = list(pathlib.Path(image_folder).rglob('*.jpg'))

for file_name in file_names:
    # load image
    image = read_image_bgr(image_folder +file_name.name)
    image = cv2.resize(image, (960,720))
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image = cv2.resize(image, (960,720))
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # TO DO: ASSERT THAT THERE IS ONLY ONE DETECTION
        # scores are sorted so we can break
        if score < 0.7:
            break
        print(score,label)
        bbox_file_name = bbox_folder+ file_name.name.strip('.jpg')+'.txt'
        with open(bbox_file_name, 'w') as f:
            f.write( str(box) +','+ str(score)+','+  str(label))
            f.write('\n')
        f.close()
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


  
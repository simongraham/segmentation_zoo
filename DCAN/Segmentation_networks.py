'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf
import numpy as np

import Segmentation_parameters
import Segmentation_layers as layers
FLAGS = tf.app.flags.FLAGS

#-----------------------------------------------------------------------------------------------------
def dcan(images, keep_prob):
  with tf.variable_scope('conv1') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64)
  with tf.variable_scope('conv2') as scope:
    conv2 = layers.conv2D(conv1, 64, 64)
    pool1 = layers.maxPool(conv2, 3, 2, name='pool1')

  with tf.variable_scope('conv3') as scope:
    conv3 = layers.conv2D(pool1, 64, 128)
  with tf.variable_scope('conv4') as scope:
    conv4 = layers.conv2D(conv3, 128, 128)
    pool2 = layers.maxPool(conv4, 3, 2, name='pool2')

  with tf.variable_scope('conv5') as scope:
    conv5 = layers.conv2D(pool2, 128, 256)
  with tf.variable_scope('conv6') as scope:
    conv6 = layers.conv2D(conv5, 256, 256)
  with tf.variable_scope('conv7') as scope:
    conv7 = layers.conv2D(conv6, 256, 256)
    pool3 = layers.maxPool(conv7, 3, 2, name='pool3')

  with tf.variable_scope('conv8') as scope:
    conv8 = layers.conv2D(pool3, 256, 512)
  with tf.variable_scope('conv9') as scope:
    conv9 = layers.conv2D(conv8, 512, 512)
  with tf.variable_scope('conv10') as scope:
    conv10 = layers.conv2D(conv9, 512, 512)
    pool4 = layers.maxPool(conv10, 3, 1, name='pool4')

  with tf.variable_scope('conv11') as scope:
    conv11 = layers.conv2D_dilated(pool4, 512, 512, rate=2)
  with tf.variable_scope('conv12') as scope:
    conv12 = layers.conv2D_dilated(conv11, 512, 512, rate=2)
  with tf.variable_scope('conv13') as scope:
    conv13 = layers.conv2D_dilated(conv12, 512, 512, rate=2)
    pool5 = layers.maxPool(conv13, 3, 1, name='pool5')

  with tf.variable_scope('fullyConnected') as scope:
    fc1 = layers.conv2D_dilated(pool5, 512, 1024, rate=4)
    fc2 = layers.conv2D_dilated(fc1, 1024, 1024, rate=4, name_ext='_2')
    fc2 = tf.nn.dropout(fc2, keep_prob)

  with tf.variable_scope('output') as scope:
    output1 = layers.conv2D(fc2, 1024, 128, kernel_size=(1,1))
    output2 = layers.conv2D(conv13, 512, 128, kernel_size=(1,1), name_ext='_2')
    output3 = layers.conv2D(conv10, 512, 128, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('upSample_gland') as scope:
    upSample_gland1 = layers.upSample2D(output1, 128, 
      128, upscale_factor=8)
    upSample_gland1 = layers.conv2D(upSample_gland1, 128, 128)
    upSample_gland1 = layers.conv2D(upSample_gland1, 128, FLAGS.n_classes, name_ext='_2', activation=False)
    upSample_gland2 = layers.upSample2D(output2, 128, 
      128, upscale_factor=8, name_ext='_2')
    upSample_gland2 = layers.conv2D(upSample_gland2, 128, 128, name_ext='_3')
    upSample_gland2 = layers.conv2D(upSample_gland2, 128, FLAGS.n_classes, activation=False, name_ext='_4')
    upSample_gland3 = layers.upSample2D(output3, 128, 
      128, upscale_factor=8, name_ext='_3')
    upSample_gland3 = layers.conv2D(upSample_gland3, 128, 128, name_ext='_5')
    upSample_gland3 = layers.conv2D(upSample_gland3, 128, FLAGS.n_classes, activation=False, name_ext='_6')

  with tf.variable_scope('upSample_contour') as scope:
    upSample_contour1 = layers.upSample2D(output1, 128, 
      128, upscale_factor=8)
    upSample_contour1 = layers.conv2D(upSample_contour1, 128, 128)
    upSample_contour1 = layers.conv2D(upSample_contour1, 128, FLAGS.n_classes, name_ext='_2', activation=False)
    upSample_contour2 = layers.upSample2D(output2, 128, 
      128, upscale_factor=8, name_ext='_2')
    upSample_contour2 = layers.conv2D(upSample_contour2, 128, 128, name_ext='_3')
    upSample_contour2 = layers.conv2D(upSample_contour2, 128, FLAGS.n_classes, activation=False, name_ext='_4')
    upSample_contour3 = layers.upSample2D(output3, 128, 
      128, upscale_factor=8, name_ext='_3')
    upSample_contour3 = layers.conv2D(upSample_contour3, 128, 128, name_ext='_5')
    upSample_contour3 = layers.conv2D(upSample_contour3, 128, FLAGS.n_classes, activation=False, name_ext='_6')

  with tf.variable_scope('softmax_gland') as scope:
    softmax_gland1 = layers.softMax(upSample_gland1)
    softmax_gland2 = layers.softMax(upSample_gland2)
    softmax_gland3 = layers.softMax(upSample_gland3)

  with tf.variable_scope('softmax_contour') as scope:
    softmax_contour1 = layers.softMax(upSample_contour1)
    softmax_contour2 = layers.softMax(upSample_contour2)
    softmax_contour3 = layers.softMax(upSample_contour3)

  with tf.variable_scope('softmax_fusion_gland') as scope:
    softmax_fusion_gland = tf.add(upSample_gland1, upSample_gland2)
    softmax_fusion_gland = tf.add(softmax_fusion_gland, upSample_gland3)
    softmax_fusion_gland = layers.softMax(softmax_fusion_gland)

  with tf.variable_scope('softmax_fusion_contour') as scope:
    softmax_fusion_contour = tf.add(upSample_contour1, upSample_contour2)
    softmax_fusion_contour = tf.add(softmax_fusion_contour, upSample_contour3)
    softmax_fusion_contour = layers.softMax(softmax_fusion_contour)

  return softmax_gland1, softmax_gland2, softmax_gland3, softmax_fusion_gland, softmax_contour1, softmax_contour2, softmax_contour3, softmax_fusion_contour

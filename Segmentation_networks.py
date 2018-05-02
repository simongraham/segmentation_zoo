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
def unet(images, keep_prob):
  '''
  Implementation of U-Net
  '''
  with tf.variable_scope('conv1') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64, padding='VALID')
  with tf.variable_scope('conv2') as scope:
    conv2 = layers.conv2D(conv1, 64, 64, padding='VALID')
    pool1 = layers.maxPool(conv2, 2, 2, name='pool1')

  with tf.variable_scope('conv3') as scope:
    conv3 = layers.conv2D(pool1, 64, 128, padding='VALID')
  with tf.variable_scope('conv4') as scope:
    conv4 = layers.conv2D(conv3, 128, 128, padding='VALID')
    pool2 = layers.maxPool(conv4, 2, 2, name='pool2')

  with tf.variable_scope('conv5') as scope:
    conv5 = layers.conv2D(pool2, 128, 256, padding='VALID')
  with tf.variable_scope('conv6') as scope:
    conv6 = layers.conv2D(conv5, 256, 256, padding='VALID')
    pool3 = layers.maxPool(conv6, 2, 2, name='pool3')

  with tf.variable_scope('conv7') as scope:
    conv7 = layers.conv2D(pool3, 256, 512, padding='VALID')
  with tf.variable_scope('conv8') as scope:
    conv8 = layers.conv2D(conv7, 512, 512, padding='VALID')
    drop1 = tf.nn.dropout(conv8, keep_prob)
    pool4 = layers.maxPool(drop1, 2, 2, name='pool4')

  with tf.variable_scope('conv9') as scope:
    conv9 = layers.conv2D(pool4, 512, 1024, padding='VALID')
  with tf.variable_scope('conv10') as scope:
    conv10 = layers.conv2D(conv9, 1024, 1024, padding='VALID')
    drop2 = tf.nn.dropout(conv10, keep_prob)

  with tf.variable_scope('upsample1') as scope:
    upsample1 = layers.upSample2D_concat(drop2 , drop1, 1024, 512, upscale_factor=2)
  with tf.variable_scope('conv11') as scope:
    conv11 = layers.conv2D(upsample1, 1024, 512, padding='VALID')
  with tf.variable_scope('conv12') as scope:
    conv12 = layers.conv2D(conv11, 512, 512, padding='VALID')

  with tf.variable_scope('upsample2') as scope:
    upsample2 = layers.upSample2D_concat(conv12 , conv6, 512, 256, upscale_factor=2)  
  with tf.variable_scope('conv13') as scope:
    conv13 = layers.conv2D(upsample2, 512, 256, padding='VALID')
  with tf.variable_scope('conv14') as scope:
    conv14 = layers.conv2D(conv13, 256, 256, padding='VALID')

  with tf.variable_scope('upsample3') as scope:
    upsample3 = layers.upSample2D_concat(conv14 , conv4, 256, 128, upscale_factor=2)
  with tf.variable_scope('conv15') as scope:
    conv15 = layers.conv2D(upsample3, 256, 128, padding='VALID')
  with tf.variable_scope('conv16') as scope:
    conv16 = layers.conv2D(conv15, 128, 128, padding='VALID')

  with tf.variable_scope('upsample4') as scope:
    upsample4 = layers.upSample2D_concat(conv16 , conv2, 128, 64, upscale_factor=2)
  with tf.variable_scope('conv17') as scope:
    conv17 = layers.conv2D(upsample4, 128, 64, padding='VALID')
  with tf.variable_scope('conv18') as scope:
    conv18 = layers.conv2D(conv17, 64, 64, padding='VALID')

  with tf.variable_scope('fc') as scope:
    conv19 = layers.conv2D(conv18, 64, FLAGS.n_classes, kernel_size=(1, 1), padding='VALID', activation=False)

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(conv19)

  return softmax
  
#----------------------------------------------------------------------------------------------------- 
def fcn8(images, keep_prob):
  '''
  Implementation of FCN-8
  '''
  with tf.variable_scope('conv1') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64)
  with tf.variable_scope('conv2') as scope:
    conv2 = layers.conv2D(conv1, 64, 64)
    pool1 = layers.maxPool(conv2, 2, 2, name='pool1')

  with tf.variable_scope('conv3') as scope:
    conv3 = layers.conv2D(pool1, 64, 128)
  with tf.variable_scope('conv4') as scope:
    conv4 = layers.conv2D(conv3, 128, 128)
    pool2 = layers.maxPool(conv4, 2, 2, name='pool2')

  with tf.variable_scope('conv5') as scope:
    conv5 = layers.conv2D(pool2, 128, 256)
  with tf.variable_scope('conv6') as scope:
    conv6 = layers.conv2D(conv5, 256, 256)
  with tf.variable_scope('conv7') as scope:
    conv7 = layers.conv2D(conv6, 256, 256)
    pool3 = layers.maxPool(conv7, 2, 2, name='pool3')

  with tf.variable_scope('conv8') as scope:
    conv8 = layers.conv2D(pool3, 256, 512)
  with tf.variable_scope('conv9') as scope:
    conv9 = layers.conv2D(conv8, 512, 512)
  with tf.variable_scope('conv10') as scope:
    conv10 = layers.conv2D(conv9, 512, 512)
    pool4 = layers.maxPool(conv10, 2, 2, name='pool4')

  with tf.variable_scope('conv11') as scope:
    conv11 = layers.conv2D(pool4, 512, 512)
  with tf.variable_scope('conv12') as scope:
    conv12 = layers.conv2D(conv11, 512, 512)
  with tf.variable_scope('conv13') as scope:
    conv13 = layers.conv2D(conv12, 512, 512)
    pool5 = layers.maxPool(conv13, 2, 2, name='pool5')

  with tf.variable_scope('fullyConnected') as scope:
    fc1 = layers.conv2D(pool5, 512, 4096, kernel_size=(7,7))
    drop1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = layers.conv2D(drop1, 4096, 4096, kernel_size=(1,1), name_ext='_2')
    drop2 = tf.nn.dropout(fc2, keep_prob)
    score1 = layers.conv2D(drop2, 4096, FLAGS.n_classes, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('upsample1') as scope:
    score_pool4 = layers.conv2D(pool4, 512, FLAGS.n_classes, kernel_size=(1,1))
    upsample1 = layers.upSample2D(score1, FLAGS.n_classes, FLAGS.n_classes, upscale_factor=2, initializer='bilinear')
    fuse1 = tf.add(upsample1, score_pool4)

  with tf.variable_scope('upsample2') as scope:
    score_pool3 = layers.conv2D(pool3, 256, FLAGS.n_classes, kernel_size=(1,1))
    upsample2 = layers.upSample2D(fuse1, FLAGS.n_classes, FLAGS.n_classes, upscale_factor=2, initializer='bilinear')
    fuse2 = tf.add(upsample2, score_pool3)

  with tf.variable_scope('upsample3') as scope:
    upsample3 = layers.upSample2D(fuse2, FLAGS.n_classes, FLAGS.n_classes, upscale_factor=8, initializer='bilinear')

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(upsample3)

  return softmax

#-----------------------------------------------------------------------------------------------------
def fcn16(images, keep_prob):
  '''
  Implementation of FCN-16
  '''
  with tf.variable_scope('conv1') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64)
  with tf.variable_scope('conv2') as scope:
    conv2 = layers.conv2D(conv1, 64, 64)
    pool1 = layers.maxPool(conv2, 2, 2, name='pool1')

  with tf.variable_scope('conv3') as scope:
    conv3 = layers.conv2D(pool1, 64, 128)
  with tf.variable_scope('conv4') as scope:
    conv4 = layers.conv2D(conv3, 128, 128)
    pool2 = layers.maxPool(conv4, 2, 2, name='pool2')

  with tf.variable_scope('conv5') as scope:
    conv5 = layers.conv2D(pool2, 128, 256)
  with tf.variable_scope('conv6') as scope:
    conv6 = layers.conv2D(conv5, 256, 256)
  with tf.variable_scope('conv7') as scope:
    conv7 = layers.conv2D(conv6, 256, 256)
    pool3 = layers.maxPool(conv7, 2, 2, name='pool3')

  with tf.variable_scope('conv8') as scope:
    conv8 = layers.conv2D(pool3, 256, 512)
  with tf.variable_scope('conv9') as scope:
    conv9 = layers.conv2D(conv8, 512, 512)
  with tf.variable_scope('conv10') as scope:
    conv10 = layers.conv2D(conv9, 512, 512)
    pool4 = layers.maxPool(conv10, 2, 2, name='pool4')

  with tf.variable_scope('conv11') as scope:
    conv11 = layers.conv2D(pool4, 512, 512)
  with tf.variable_scope('conv12') as scope:
    conv12 = layers.conv2D(conv11, 512, 512)
  with tf.variable_scope('conv13') as scope:
    conv13 = layers.conv2D(conv12, 512, 512)
    pool5 = layers.maxPool(conv13, 2, 2, name='pool5')

  with tf.variable_scope('fullyConnected') as scope:
    fc1 = layers.conv2D(pool5, 512, 4096, (7,7))
    drop1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = layers.conv2D(drop1,4096, 4096, (1,1), name_ext='_2')
    drop2 = tf.nn.dropout(fc2, keep_prob)
    score1 = layers.conv2D(drop2, 4096, FLAGS.n_classes, (1,1), name_ext='_3')

  with tf.variable_scope('upsample1') as scope:
    score_pool4 = layers.conv2D(pool4, 512, FLAGS.n_classes, (1,1))
    upsample1 = layers.upSample2D(score1, FLAGS.n_classes, FLAGS.n_classes, upscale_factor=2, initializer='bilinear')
    fuse1 = tf.add(upsample1, score_pool4)

  with tf.variable_scope('upsample2') as scope:
    upsample2 = layers.upSample2D(fuse1, FLAGS.n_classes, FLAGS.n_classes, upscale_factor=16, initializer='bilinear')

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(upsample2)

  return softmax

#-----------------------------------------------------------------------------------------------------
def fcn32(images, keep_prob):
  '''
  Implementation of FCN-32
  '''
  with tf.variable_scope('conv1') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64)
  with tf.variable_scope('conv2') as scope:
    conv2 = layers.conv2D(conv1, 64, 64)
    pool1 = layers.maxPool(conv2, 2, 2, name='pool1')

  with tf.variable_scope('conv3') as scope:
    conv3 = layers.conv2D(pool1, 64, 128)
  with tf.variable_scope('conv4') as scope:
    conv4 = layers.conv2D(conv3, 128, 128)
    pool2 = layers.maxPool(conv4, 2, 2, name='pool2')

  with tf.variable_scope('conv5') as scope:
    conv5 = layers.conv2D(pool2, 128, 256)
  with tf.variable_scope('conv6') as scope:
    conv6 = layers.conv2D(conv5, 256, 256)
  with tf.variable_scope('conv7') as scope:
    conv7 = layers.conv2D(conv6, 256, 256)
    pool3 = layers.maxPool(conv7, 2, 2, name='pool3')

  with tf.variable_scope('conv8') as scope:
    conv8 = layers.conv2D(pool3, 256, 512)
  with tf.variable_scope('conv9') as scope:
    conv9 = layers.conv2D(conv8, 512, 512)
  with tf.variable_scope('conv10') as scope:
    conv10 = layers.conv2D(conv9, 512, 512)
    pool4 = layers.maxPool(conv10, 2, 2, name='pool4')

  with tf.variable_scope('conv11') as scope:
    conv11 = layers.conv2D(pool4, 512, 512)
  with tf.variable_scope('conv12') as scope:
    conv12 = layers.conv2D(conv11, 512, 512)
  with tf.variable_scope('conv13') as scope:
    conv13 = layers.conv2D(conv12, 512, 512)
    pool5 = layers.maxPool(conv13, 2, 2, name='pool5')

  with tf.variable_scope('fullyConnected') as scope:
    fc1 = layers.conv2D(pool5, 512, 1024, (7,7))
    drop1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = layers.conv2D(drop1, 1024, 1024, (1,1), name_ext='_2')
    drop2 = tf.nn.dropout(fc2, keep_prob)
    score1 = layers.conv2D(drop2, 1024, FLAGS.n_classes, (1,1), name_ext='_3')

  with tf.variable_scope('upsample1') as scope:
    upsample1 = layers.upSample2D(score1, FLAGS.n_classes, FLAGS.n_classes, upscale_factor=32, initializer='bilinear')

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(upsample1)

  return softmax

#-----------------------------------------------------------------------------------------------------
def segnet(images, keep_prob, is_training):
  '''
  Implementation of SegNet
  '''
  #encode
  with tf.variable_scope('conv1') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64, activation=False)
    conv1_bn = layers.batchNorm(conv1, 'batch_norm1', is_training)
    conv1_relu = layers._activation(conv1_bn)
  with tf.variable_scope('conv2') as scope:
    conv2 = layers.conv2D(conv1_relu, 64, 64, activation=False)
    conv2_bn = layers.batchNorm(conv2, 'batch_norm2', is_training)
    conv2_relu = layers._activation(conv2_bn)
  with tf.variable_scope('pool1') as scope:
    pool1, pool1_indicies = layers.maxPool(conv2_relu, 2, 2, name='pool1', index=True)

  with tf.variable_scope('conv3') as scope:
    conv3 = layers.conv2D(pool1, 64, 128, activation=False)
    conv3_bn = layers.batchNorm(conv3, 'batch_norm3', is_training)
    conv3_relu = layers._activation(conv3_bn)
  with tf.variable_scope('conv4') as scope:
    conv4 = layers.conv2D(conv3_relu, 128, 128, activation=False)
    conv4_bn = layers.batchNorm(conv4, 'batch_norm4', is_training)
    conv4_relu = layers._activation(conv4_bn)
  with tf.variable_scope('pool2') as scope:
    pool2, pool2_indicies = layers.maxPool(conv4_relu, 2, 2, name='pool2', index=True)

  with tf.variable_scope('conv5') as scope:
    conv5 = layers.conv2D(pool2, 128, 256, activation=False)
    conv5_bn = layers.batchNorm(conv5, 'batch_norm5', is_training)
    conv5_relu = layers._activation(conv5_bn)
  with tf.variable_scope('conv6') as scope:
    conv6 = layers.conv2D(conv5_relu, 256, 256, activation=False)
    conv6_bn = layers.batchNorm(conv6, 'batch_norm6', is_training)
    conv6_relu = layers._activation(conv6_bn)
  with tf.variable_scope('conv7') as scope:
    conv7 = layers.conv2D(conv6_relu, 256, 256, activation=False)
    conv7_bn = layers.batchNorm(conv7, 'batch_norm7', is_training)
    conv7_relu = layers._activation(conv7_bn)
  with tf.variable_scope('pool3') as scope:
    pool3, pool3_indicies = layers.maxPool(conv7_relu, 2, 2, name='pool3', index=True)

  with tf.variable_scope('conv8') as scope:
    conv8 = layers.conv2D(pool3, 256, 512, activation=False)
    conv8_bn = layers.batchNorm(conv8, 'batch_norm8', is_training)
    conv8_relu = layers._activation(conv8_bn)
  with tf.variable_scope('conv9') as scope:
    conv9 = layers.conv2D(conv8_relu, 512, 512, activation=False)
    conv9_bn = layers.batchNorm(conv9, 'batch_norm9', is_training)
    conv9_relu = layers._activation(conv9_bn)
  with tf.variable_scope('conv10') as scope:
    conv10 = layers.conv2D(conv9_relu, 512, 512, activation=False)
    conv10_bn = layers.batchNorm(conv10, 'batch_norm10', is_training)
    conv10_relu = layers._activation(conv10_bn)
  with tf.variable_scope('pool4') as scope:
    pool4, pool4_indicies = layers.maxPool(conv10_relu, 2, 2, name='pool4', index=True)

  with tf.variable_scope('conv11') as scope:
    conv11 = layers.conv2D(pool4, 512, 512, activation=False)
    conv11_bn = layers.batchNorm(conv11, 'batch_norm11', is_training)
    conv11_relu = layers._activation(conv11_bn)
  with tf.variable_scope('conv12') as scope:
    conv12 = layers.conv2D(conv11_relu, 512, 512, activation=False)
    conv12_bn = layers.batchNorm(conv12, 'batch_norm12', is_training)
    conv12_relu = layers._activation(conv12_bn)
  with tf.variable_scope('conv13') as scope:
    conv13 = layers.conv2D(conv12_relu, 512, 512, activation=False)
    conv13_bn = layers.batchNorm(conv13, 'batch_norm13', is_training)
    conv13_relu = layers._activation(conv13_bn)
  with tf.variable_scope('pool5') as scope:
    pool5, pool5_indicies = layers.maxPool(conv13_relu, 2, 2, name='pool5', index=True)

  # decode 
  with tf.variable_scope('unpool1') as scope:
    unpool1 = layers.unpool_with_argmax(pool5, ind=pool5_indicies, name='unpool_1')
  with tf.variable_scope('conv14') as scope:
    conv14 = layers.conv2D(unpool1, 512, 512, activation=False)
    conv14_bn = layers.batchNorm(conv14, 'batch_norm14', is_training)
    conv14_relu = layers._activation(conv14_bn)
  with tf.variable_scope('conv15') as scope:
    conv15 = layers.conv2D(conv14_relu, 512, 512, activation=False)
    conv15_bn = layers.batchNorm(conv15, 'batch_norm15', is_training)
    conv15_relu = layers._activation(conv15_bn)
  with tf.variable_scope('conv16') as scope:
    conv16 = layers.conv2D(conv15_relu, 512, 512, activation=False)
    conv16_bn = layers.batchNorm(conv16, 'batch_norm16', is_training)
    conv16_relu = layers._activation(conv16_bn)

  with tf.variable_scope('unpool2') as scope:
    unpool2 = layers.unpool_with_argmax(conv16_relu, ind=pool4_indicies, name='unpool_2')
  with tf.variable_scope('conv17') as scope:
    conv17 = layers.conv2D(unpool2, 512, 512, activation=False)
    conv17_bn = layers.batchNorm(conv17, 'batch_norm17', is_training)
    conv17_relu = layers._activation(conv17_bn)
  with tf.variable_scope('conv18') as scope:
    conv18 = layers.conv2D(conv17_relu, 512, 512, activation=False)
    conv18_bn = layers.batchNorm(conv18, 'batch_norm18', is_training)
    conv18_relu = layers._activation(conv18_bn)
  with tf.variable_scope('conv19') as scope:
    conv19 = layers.conv2D(conv18_relu, 512, 256, activation=False)
    conv19_bn = layers.batchNorm(conv19, 'batch_norm19', is_training)
    conv19_relu = layers._activation(conv19_bn)

  with tf.variable_scope('unpool3') as scope:
    unpool3 = layers.unpool_with_argmax(conv19_relu, ind=pool3_indicies, name='unpool_3')
  with tf.variable_scope('conv20') as scope:
    conv20 = layers.conv2D(unpool3, 256, 256, activation=False)
    conv20_bn = layers.batchNorm(conv20, 'batch_norm20', is_training)
    conv20_relu = layers._activation(conv20_bn)
  with tf.variable_scope('conv21') as scope:
    conv21 = layers.conv2D(conv20_relu, 256, 256, activation=False)
    conv21_bn = layers.batchNorm(conv21, 'batch_norm21', is_training)
    conv21_relu = layers._activation(conv21_bn)
  with tf.variable_scope('conv22') as scope:
    conv22 = layers.conv2D(conv21_relu, 256, 128, activation=False)
    conv22_bn = layers.batchNorm(conv22, 'batch_norm22', is_training)
    conv22_relu = layers._activation(conv22_bn)

  with tf.variable_scope('unpool4') as scope:
    unpool4 = layers.unpool_with_argmax(conv22_relu, ind=pool2_indicies, name='unpool_4')
  with tf.variable_scope('conv23') as scope:
    conv23 = layers.conv2D(unpool4, 128, 128, activation=False)
    conv23_bn = layers.batchNorm(conv23, 'batch_norm23', is_training)
    conv23_relu = layers._activation(conv23_bn)
  with tf.variable_scope('conv24') as scope:
    conv24 = layers.conv2D(conv23_relu, 128, 64, activation=False)
    conv24_bn = layers.batchNorm(conv24, 'batch_norm24', is_training)
    conv24_relu = layers._activation(conv24_bn)

  with tf.variable_scope('unpool5') as scope:
    unpool5 = layers.unpool_with_argmax(conv24_relu, ind=pool1_indicies, name='unpool_5')
  with tf.variable_scope('conv25') as scope:
    conv25 = layers.conv2D(unpool5, 64, 64, activation=False)
    conv25_bn = layers.batchNorm(conv25, 'batch_norm25', is_training)
    conv25_relu = layers._activation(conv25_bn)

  with tf.variable_scope('output') as scope:
    conv26 = layers.conv2D(conv25_relu, 64, FLAGS.n_classes, activation=False)
    conv26_bn = layers.batchNorm(conv26, 'batch_norm26', is_training)
    conv26_relu = layers._activation(conv26_bn)

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(conv26_relu)

  return softmax

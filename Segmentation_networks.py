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
def fcn8_2(images, keep_prob):
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

  with tf.variable_scope('new') as scope:
    new= layers.conv2D(upsample3, FLAGS.n_classes, FLAGS.n_classes, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(new)

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

#-----------------------------------------------------------------------------------------------------
def deeplab_v2(images, keep_prob):
  '''
  Implementation of Deeplab v2
  '''
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

  with tf.variable_scope('ASPP_4') as scope:
    fc1_1 = layers.conv2D_dilated(pool5, 512, 1024, rate=6)
    drop1_1 = tf.nn.dropout(fc1_1, keep_prob)
    fc2_1 = layers.conv2D(drop1_1, 1024, 1024, kernel_size=(1,1), name_ext='_2')
    drop2_1 = tf.nn.dropout(fc2_1, keep_prob)
    fc3_1 = layers.conv2D(drop2_1, 1024, FLAGS.n_classes, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('ASPP_8') as scope:
    fc1_2 = layers.conv2D_dilated(pool5, 512, 1024, rate=12)
    drop1_2 = tf.nn.dropout(fc1_2, keep_prob)
    fc2_2 = layers.conv2D(drop1_2, 1024, 1024, kernel_size=(1,1), name_ext='_2')
    drop2_2 = tf.nn.dropout(fc2_2, keep_prob)
    fc3_2 = layers.conv2D(drop2_2, 1024, FLAGS.n_classes, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('ASPP_12') as scope:
    fc1_3 = layers.conv2D_dilated(pool5, 512, 1024, rate=18)
    drop1_3 = tf.nn.dropout(fc1_3, keep_prob)
    fc2_3 = layers.conv2D(drop1_3, 1024, 1024, kernel_size=(1,1), name_ext='_2')
    drop2_3 = tf.nn.dropout(fc2_3, keep_prob)
    fc3_3 = layers.conv2D(drop2_3, 1024, FLAGS.n_classes, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('ASPP_16') as scope:
    fc1_4 = layers.conv2D_dilated(pool5, 512, 1024, rate=24)
    drop1_4 = tf.nn.dropout(fc1_4, keep_prob)
    fc2_4 = layers.conv2D(drop1_4, 1024, 1024, kernel_size=(1,1), name_ext='_2')
    drop2_4 = tf.nn.dropout(fc2_4, keep_prob)
    fc3_4 = layers.conv2D(drop2_4, 1024, FLAGS.n_classes, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('ASPP_sum') as scope:
    sum1 = tf.add(fc3_1, fc3_2)
    sum2 = tf.add(sum1, fc3_3)
    sum3 = tf.add(sum2, fc3_4)

  with tf.variable_scope('upSample') as scope:
    upsample = tf.image.resize_images(sum3, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])

  with tf.variable_scope('softmax') as scope:
    softmax = layers.softMax(upsample)

  return softmax

#-----------------------------------------------------------------------------------------------------
def proposed_single(images, keep_prob, is_training):
  with tf.variable_scope('input') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64, activation=True)
    conv2 = layers.conv2D(conv1, 64, 64, name_ext='_2', activation=True)
    pool1 = layers.maxPool(conv2, 2, 2, 'pool1')

  with tf.variable_scope('resize_1') as scope:
    resize_1 = tf.image.resize_bicubic(images, [int((FLAGS.train_image_target_size)/2),int((FLAGS.train_image_target_size)/2)])

  with tf.variable_scope('res1') as scope:
    res1 = layers.resUnit_x1(pool1, 64, 128, is_training, resize = resize_1)
  with tf.variable_scope('res2') as scope:
    res2 = layers.resUnit_x1(res1,  128, 128, is_training)
    pool2 = layers.maxPool(res2, 2, 2, 'pool2')

  with tf.variable_scope('resize_2') as scope:
    resize_2 = tf.image.resize_bicubic(images, [int((FLAGS.train_image_target_size)/4),int((FLAGS.train_image_target_size)/4)])

  with tf.variable_scope('res3') as scope:
    res3 = layers.resUnit_x1(pool2, 128, 256, is_training, resize = resize_2)
  with tf.variable_scope('res4') as scope:
    res4 = layers.resUnit_x1(res3, 256, 256, is_training)
    pool3 = layers.maxPool(res4, 2, 2, 'pool3')
 
  with tf.variable_scope('resize_2') as scope:
    resize_3 = tf.image.resize_bicubic(images, [int((FLAGS.train_image_target_size)/8),int((FLAGS.train_image_target_size)/8)])

  with tf.variable_scope('res5') as scope:
    res5 = layers.resUnit_x2(pool3, 256, 512, is_training, resize = resize_3)
  with tf.variable_scope('res6') as scope:
    res6 = layers.resUnit_x2(res5, 512, 512, is_training)
    res6 = layers.maxPool(res6, 2, 1 , 'pool_res6')

  with tf.variable_scope('res7') as scope:
    res7 = layers.resUnit_x2(res6, 512, 512, is_training, dilation_rate=2)
  with tf.variable_scope('res8') as scope:
    res8 = layers.resUnit_x2(res7, 512, 512, is_training, dilation_rate=2)
    res8 = layers.maxPool(res8, 2, 1, 'pool_res8')

  with tf.variable_scope('res9') as scope:
    res9 = layers.resUnit_x2(res8, 512, 1024, is_training, increase_dim=True, dilation_rate=4)
    res9 = tf.nn.dropout(res9, keep_prob)
  with tf.variable_scope('res10') as scope:
    res10 = layers.resUnit_x2(res9, 1024, 1024, is_training, dilation_rate=4)

  with tf.variable_scope('aspp') as scope:
    one_one = layers.conv2D(res10, 1024, 1024, kernel_size=(1,1), name_ext='_2')
    one_one = layers.conv2D(one_one, 1024, 256, kernel_size=(1,1), name_ext='_3')
    dilated_6 = layers.conv2D_dilated(res10, 1024, 1024, rate=6, name_ext = '_4')
    dilated_6 = layers.conv2D(dilated_6, 1024, 256, kernel_size=(1,1), name_ext='_5')
    dilated_12 = layers.conv2D_dilated(res10, 1024, 1024, rate=12, name_ext = '_6')
    dilated_12 = layers.conv2D(dilated_12, 1024, 256, kernel_size=(1,1), name_ext='_7')
    dilated_18 = layers.conv2D_dilated(res10, 1024, 1024, rate=18, name_ext = '_8')
    dilated_18 = layers.conv2D(dilated_18, 1024, 256, kernel_size=(1,1), name_ext='_9')

  with tf.variable_scope('Global_pooling') as scope:
    av_pool1 = layers.avgPool(res10, 60, stride=60, name='_1', padding='VALID')
    av_pool1 = layers.conv2D(av_pool1, 1024, 128, kernel_size=(1,1), name_ext='_1')
    av_pool1 = tf.image.resize_images(av_pool1, [int(FLAGS.train_image_target_size/4), int(FLAGS.train_image_target_size/4)])
    av_pool2 = layers.avgPool(res10, 30, stride=30, name='_2', padding='VALID')
    av_pool2 = layers.conv2D(av_pool2, 1024, 64, kernel_size=(1,1), name_ext='_2')
    av_pool2 = tf.image.resize_images(av_pool2, [int(FLAGS.train_image_target_size/2), int(FLAGS.train_image_target_size/2)])
    av_pool3 = layers.avgPool(res10, 15, stride=15, name='_3', padding='VALID')
    av_pool3 = layers.conv2D(av_pool3, 1024, 32, kernel_size=(1,1), name_ext='_3')
    av_pool3 = tf.image.resize_images(av_pool3, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
   
  with tf.variable_scope('Concat') as scope:
    concat = tf.concat([one_one, dilated_6, dilated_12, dilated_18], axis=3)
    conv3 = layers.conv2D(concat, 1024, 512, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('Upsample1') as scope:
    upsample1 = tf.image.resize_images(conv3, [int(FLAGS.train_image_target_size/4), int(FLAGS.train_image_target_size/4)])
    upsample1_concat = tf.concat([av_pool1, res4, upsample1], axis=3)
    upsample1_conv = layers.conv2D(upsample1_concat, 896, 256, kernel_size=(3,3), name_ext='_3')
    upsample1_conv2 = layers.conv2D(upsample1_conv, 256, 256, kernel_size=(3,3), name_ext='_4')

  with tf.variable_scope('Upsample2') as scope:
    upsample2 = tf.image.resize_images(upsample1_conv2, [int(FLAGS.train_image_target_size/2), int(FLAGS.train_image_target_size/2)])
    upsample2_concat = tf.concat([av_pool2, res2, upsample2], axis=3)
    upsample2_conv = layers.conv2D(upsample2_concat, 448, 128, kernel_size=(3,3), name_ext='_3')
    upsample2_conv2 = layers.conv2D(upsample2_conv, 128, 128, kernel_size=(3,3), name_ext='_4')

  with tf.variable_scope('Upsample3') as scope:
    upsample3 = tf.image.resize_images(upsample2_conv2, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    upsample3_concat = tf.concat([av_pool3, conv2, upsample3], axis=3)
    upsample3_conv = layers.conv2D(upsample3_concat, 224, 64, kernel_size=(3,3), name_ext='_3')
    upsample3_conv2 = layers.conv2D(upsample3_conv, 64, 64, kernel_size=(3,3), name_ext='_4')
    drop = tf.nn.dropout(upsample3_conv2, keep_prob)
    output = layers.conv2D(drop, 64, FLAGS.n_classes, kernel_size=(1,1), name_ext='_5')
    softmax = layers.softMax(output)

  with tf.variable_scope('Auxiliary') as scope:
    aux_output1 = tf.image.resize_images(res4, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    aux_output1 = tf.nn.dropout(aux_output1, keep_prob)
    aux_output1 = layers.conv2D(aux_output1, 64, FLAGS.n_classes, kernel_size=(1,1), name_ext='_2', activation=False)
    aux_output1 = layers.softMax(aux_output1)

  return softmax, aux_output1
  #-----------------------------------------------------------------------------------------------------
def proposed_contour(images, keep_prob, is_training):
  with tf.variable_scope('input') as scope:
    conv1 = layers.conv2D(images, FLAGS.n_channels, 64, activation=True)
    conv2 = layers.conv2D(conv1, 64, 64, name_ext='_2', activation=True)
    pool1 = layers.maxPool(conv2, 2, 2, 'pool1')

  with tf.variable_scope('resize_1') as scope:
    resize_1 = tf.image.resize_bicubic(images, [int((FLAGS.train_image_target_size)/2),int((FLAGS.train_image_target_size)/2)])

  with tf.variable_scope('res1') as scope:
    res1 = layers.resUnit_x1(pool1, 64, 128, is_training, resize = resize_1)
  with tf.variable_scope('res2') as scope:
    res2 = layers.resUnit_x1(res1,  128, 128, is_training)
    pool2 = layers.maxPool(res2, 2, 2, 'pool2')

  with tf.variable_scope('resize_2') as scope:
    resize_2 = tf.image.resize_bicubic(images, [int((FLAGS.train_image_target_size)/4),int((FLAGS.train_image_target_size)/4)])

  with tf.variable_scope('res3') as scope:
    res3 = layers.resUnit_x1(pool2, 128, 256, is_training, resize = resize_2)
  with tf.variable_scope('res4') as scope:
    res4 = layers.resUnit_x1(res3, 256, 256, is_training)
    pool3 = layers.maxPool(res4, 2, 2, 'pool3')
 
  with tf.variable_scope('resize_2') as scope:
    resize_3 = tf.image.resize_bicubic(images, [int((FLAGS.train_image_target_size)/8),int((FLAGS.train_image_target_size)/8)])

  with tf.variable_scope('res5') as scope:
    res5 = layers.resUnit_x2(pool3, 256, 512, is_training, resize = resize_3)
  with tf.variable_scope('res6') as scope:
    res6 = layers.resUnit_x2(res5, 512, 512, is_training)
    res6 = layers.maxPool(res6, 2, 1 , 'pool_res6')

  with tf.variable_scope('res7') as scope:
    res7 = layers.resUnit_x2(res6, 512, 512, is_training, dilation_rate=2)
  with tf.variable_scope('res8') as scope:
    res8 = layers.resUnit_x2(res7, 512, 512, is_training, dilation_rate=2)
    res8 = layers.maxPool(res8, 2, 1, 'pool_res8')

  with tf.variable_scope('res9') as scope:
    res9 = layers.resUnit_x2(res8, 512, 1024, is_training, increase_dim=True, dilation_rate=4)
    res9 = tf.nn.dropout(res9, keep_prob)
  with tf.variable_scope('res10') as scope:
    res10 = layers.resUnit_x2(res9, 1024, 1024, is_training, dilation_rate=4)

  with tf.variable_scope('aspp') as scope:
    one_one = layers.conv2D(res10, 1024, 1024, kernel_size=(1,1), name_ext='_2')
    one_one = layers.conv2D(one_one, 1024, 256, kernel_size=(1,1), name_ext='_3')
    dilated_6 = layers.conv2D_dilated(res10, 1024, 1024, rate=6, name_ext = '_4')
    dilated_6 = layers.conv2D(dilated_6, 1024, 256, kernel_size=(1,1), name_ext='_5')
    dilated_12 = layers.conv2D_dilated(res10, 1024, 1024, rate=12, name_ext = '_6')
    dilated_12 = layers.conv2D(dilated_12, 1024, 256, kernel_size=(1,1), name_ext='_7')
    dilated_18 = layers.conv2D_dilated(res10, 1024, 1024, rate=18, name_ext = '_8')
    dilated_18 = layers.conv2D(dilated_18, 1024, 256, kernel_size=(1,1), name_ext='_9')

  with tf.variable_scope('Global_pooling') as scope:
    av_pool1 = layers.avgPool(res10, 60, stride=60, name='_1', padding='VALID')
    av_pool1 = layers.conv2D(av_pool1, 1024, 128, kernel_size=(1,1), name_ext='_1')
    av_pool1 = tf.image.resize_images(av_pool1, [int(FLAGS.train_image_target_size/4), int(FLAGS.train_image_target_size/4)])
    av_pool2 = layers.avgPool(res10, 30, stride=30, name='_2', padding='VALID')
    av_pool2 = layers.conv2D(av_pool2, 1024, 64, kernel_size=(1,1), name_ext='_2')
    av_pool2 = tf.image.resize_images(av_pool2, [int(FLAGS.train_image_target_size/2), int(FLAGS.train_image_target_size/2)])
    av_pool3 = layers.avgPool(res10, 15, stride=15, name='_3', padding='VALID')
    av_pool3 = layers.conv2D(av_pool3, 1024, 32, kernel_size=(1,1), name_ext='_3')
    av_pool3 = tf.image.resize_images(av_pool3, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
   
  with tf.variable_scope('Concat') as scope:
    concat = tf.concat([one_one, dilated_6, dilated_12, dilated_18], axis=3)
    conv3 = layers.conv2D(concat, 1024, 512, kernel_size=(1,1), name_ext='_3')

  with tf.variable_scope('Upsample1') as scope:
    upsample1 = tf.image.resize_images(conv3, [int(FLAGS.train_image_target_size/4), int(FLAGS.train_image_target_size/4)])
    upsample1_concat = tf.concat([av_pool1, res4, upsample1], axis=3)
    upsample1_conv = layers.conv2D(upsample1_concat, 896, 256, kernel_size=(3,3), name_ext='_3')
    upsample1_conv2 = layers.conv2D(upsample1_conv, 256, 256, kernel_size=(3,3), name_ext='_4')

  with tf.variable_scope('Upsample2') as scope:
    upsample2 = tf.image.resize_images(upsample1_conv2, [int(FLAGS.train_image_target_size/2), int(FLAGS.train_image_target_size/2)])
    upsample2_concat = tf.concat([av_pool2, res2, upsample2], axis=3)
    upsample2_conv = layers.conv2D(upsample2_concat, 448, 128, kernel_size=(3,3), name_ext='_3')
    upsample2_conv2 = layers.conv2D(upsample2_conv, 128, 128, kernel_size=(3,3), name_ext='_4')

  with tf.variable_scope('Upsample3') as scope:
    upsample3 = tf.image.resize_images(upsample2_conv2, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    upsample3_concat = tf.concat([av_pool3, conv2, upsample3], axis=3)
    upsample3_conv = layers.conv2D(upsample3_concat, 224, 64, kernel_size=(3,3), name_ext='_3')
    upsample3_conv2 = layers.conv2D(upsample3_conv, 64, 64, kernel_size=(3,3), name_ext='_4')
    drop = tf.nn.dropout(upsample3_conv2, keep_prob)
    output_gland = layers.conv2D(drop, 64, FLAGS.n_classes, kernel_size=(1,1), name_ext='_5', activation=False)
    softmax_gland = layers.softMax(output_gland)
    output_contour = layers.conv2D(drop, 64, FLAGS.n_classes, kernel_size=(1,1), name_ext='_6', activation=False)
    softmax_contour = layers.softMax(output_contour)

  with tf.variable_scope('fusion') as scope:
    concat = tf.concat([softmax_gland, softmax_contour], axis=3)
    conv1_fusion = layers.conv2D(concat, 4, 64, activation=True)
    conv2_fusion = layers.conv2D_dilated(conv1_fusion, 64, 128, rate=2, name_ext='_2', activation=True)
    conv3_fusion = layers.conv2D_dilated(conv2_fusion, 128, 256, rate=2, name_ext='_2', activation=True)
    conv4_fusion = layers.conv2D_dilated(conv3_fusion, 256, 256, rate=4, name_ext='_3', activation=True)
    output_fusion = layers.conv2D(conv4_fusion, 256, FLAGS.n_classes, kernel_size=(1,1), name_ext='_4', activation=False)
    softmax_fusion = layers.softMax(output_fusion)

  with tf.variable_scope('Auxiliary') as scope:
    aux_output1 = tf.image.resize_images(res4, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    aux_output1 = layers.conv2D(aux_output1, 256, 64, kernel_size=(3,3), name_ext='_1')
    aux_output1 = tf.nn.dropout(aux_output1, keep_prob)
    aux_output1 = layers.conv2D(aux_output1, 64, FLAGS.n_classes, kernel_size=(1,1), name_ext='_2', activation=False)
    aux_output1 = layers.softMax(aux_output1)
    aux_output2 = tf.image.resize_images(res8, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    aux_output2 = layers.conv2D(aux_output2, 512, 64, kernel_size=(3,3), name_ext='_3')
    aux_output2 = tf.nn.dropout(aux_output2, keep_prob)
    aux_output2 = layers.conv2D(aux_output2, 64, FLAGS.n_classes, kernel_size=(1,1), name_ext='_4', activation=False)
    aux_output2 = layers.softMax(aux_output2)

  return softmax_contour, softmax_gland, softmax_fusion, aux_output1, aux_output2

#-----------------------------------------------------------------------------------------------------
def psp50(images, keep_prob, is_training):
  '''
  Implementation of pyramid scene parsing network
  '''
  with tf.variable_scope('input') as scope:
    initial_depth = 64
    conv1 = layers.conv2D(images, FLAGS.n_channels, initial_depth, kernel_size=(7,7), stride=2, activation=False)
    conv1_bn = layers.batchNorm(conv1, 'conv1_bn', is_training)
    conv1_relu = layers._activation(conv1_bn)
    pool1 = layers.maxPool(conv1_relu, 3, 1, name='pool1')

  with tf.variable_scope('residual_block1_1') as scope:
    res1_1 = layers.resUnitBottleneck_orig(pool1, 64, 64, 256, is_training, stride=1, connection=False)
  with tf.variable_scope('residual_block1_2') as scope:
    res1_2 = layers.resUnitBottleneck_orig(res1_1, 256, 64, 256, is_training)
  with tf.variable_scope('residual_block1_3') as scope:
    res1_3 = layers.resUnitBottleneck_orig(res1_2, 256, 64, 256, is_training)

  with tf.variable_scope('residual_block2_1') as scope:
    res2_1 = layers.resUnitBottleneck_orig(res1_3, 256, 128, 512, is_training, connection=False)
  with tf.variable_scope('residual_block2_2') as scope:
    res2_2 = layers.resUnitBottleneck_orig(res2_1, 512, 128, 512, is_training)
  with tf.variable_scope('residual_block2_3') as scope:
    res2_3 = layers.resUnitBottleneck_orig(res2_2, 512, 128, 512, is_training)
  with tf.variable_scope('residual_block2_4') as scope:
    res2_4 = layers.resUnitBottleneck_orig(res2_3, 512, 128, 512, is_training)

  with tf.variable_scope('residual_block3_1') as scope:
    res3_1 = layers.resUnitBottleneck_orig(res2_4, 512, 256, 1024, is_training, connection=False, dilated=True)
  with tf.variable_scope('residual_block3_2') as scope:
    res3_2 = layers.resUnitBottleneck_orig(res3_1, 1024, 256, 1024, is_training, dilated=True)
  with tf.variable_scope('residual_block3_3') as scope:
    res3_3 = layers.resUnitBottleneck_orig(res3_2, 1024, 256, 1024, is_training, dilated=True)
  with tf.variable_scope('residual_block3_4') as scope:
    res3_4 = layers.resUnitBottleneck_orig(res3_3, 1024, 256, 1024, is_training, dilated=True)
  with tf.variable_scope('residual_block3_5') as scope:
    res3_5 = layers.resUnitBottleneck_orig(res3_4, 1024, 256, 1024, is_training, dilated=True)
  with tf.variable_scope('residual_block3_6') as scope:
    res3_6 = layers.resUnitBottleneck_orig(res3_5, 1024, 256, 1024, is_training, dilated=True)

  with tf.variable_scope('residual_block4_1') as scope:
    res4_1 = layers.resUnitBottleneck_orig(res3_6, 1024, 256, 1024, is_training, stride=1, connection=False, dilated=True, dilation_rate=4)
  with tf.variable_scope('residual_block4_2') as scope:
    res4_2 = layers.resUnitBottleneck_orig(res4_1, 1024, 256, 1024, is_training, dilated=True, dilation_rate=4)
  with tf.variable_scope('residual_block4_3') as scope:
    res4_3 = layers.resUnitBottleneck_orig(res4_2, 1024, 256, 1024, is_training, dilated=True, dilation_rate=4)
    upsample1 = tf.image.resize_images(res4_3, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])

  # Average Pyramid pooling
  with tf.variable_scope('PyramidPooling') as scope:
    av_pool1 = layers.avgPool(res4_3, 56, stride=56, name='_1', padding='VALID')
    av_pool1 = layers.conv2D(av_pool1, 1024, 256, kernel_size=(1,1), name_ext='_1', activation=False)
    av_pool1= layers.batchNorm(av_pool1, 'av_pool1_bn', is_training)
    av_pool1 = layers._activation(av_pool1)
    av_pool1 = tf.nn.dropout(av_pool1, keep_prob)
    av_pool1 = tf.image.resize_images(av_pool1, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    av_pool2 = layers.avgPool(res4_3, 28, stride=28, name='_2', padding='VALID')
    av_pool2 = layers.conv2D(av_pool2, 1024, 256, kernel_size=(1,1), name_ext='_2', activation=False)
    av_pool2 = layers.batchNorm(av_pool2, 'av_pool2_bn', is_training)
    av_pool2 = layers._activation(av_pool2)
    av_pool2 = tf.nn.dropout(av_pool2, keep_prob)
    av_pool2 = tf.image.resize_images(av_pool2, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    av_pool3 = layers.avgPool(res4_3, 14, stride=14, name='_3', padding='VALID')
    av_pool3 = layers.conv2D(av_pool3, 1024, 256, kernel_size=(1,1), name_ext='_3', activation=False)
    av_pool3= layers.batchNorm(av_pool3, 'av_pool3_bn', is_training)
    av_pool3 = layers._activation(av_pool3)
    av_pool3 = tf.nn.dropout(av_pool3, keep_prob)
    av_pool3 = tf.image.resize_images(av_pool3, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])
    av_pool4 = layers.avgPool(res4_3, 7, stride=7, name='_4', padding='VALID')
    av_pool4 = layers.conv2D(av_pool4, 1024, 256, kernel_size=(1,1), name_ext='_4', activation=False)
    av_pool4 = layers.batchNorm(av_pool4, 'av_pool4_bn', is_training)
    av_pool4 = layers._activation(av_pool4)
    av_pool4 = tf.nn.dropout(av_pool4, keep_prob)
    av_pool4 = tf.image.resize_images(av_pool4, [FLAGS.train_image_target_size, FLAGS.train_image_target_size])

  with tf.variable_scope('concat') as scope:
    concat = tf.concat([upsample1, av_pool1, av_pool2, av_pool3, av_pool4], axis=3)

  with tf.variable_scope('Classification') as scope:
    conv_output = layers.conv2D(concat, 2048, FLAGS.n_classes, kernel_size=(1,1), activation=False)
    conv_output_bn = layers.batchNorm(conv_output, 'bn_1', is_training)
    conv_output_relu = layers._activation(conv_output_bn)
    softmax = layers.softMax(conv_output_relu)

  return softmax

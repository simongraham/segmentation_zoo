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
FLAGS = tf.app.flags.FLAGS

#-----------------------------------------------------------------------------------------------------
##################
#VARIABLE INIT   #
##################

def weightVariable(name, shape, stddev):
  """
  Weight initialization function.
  Args:
     name: name scope of weight variable
     shape: shape of the kernel
     stddev: standard deviation used for initialization
  Return:
     w: weight variable
  """
  w = tf.get_variable(name, shape,
    initializer = tf.contrib.layers.xavier_initializer_conv2d())
  tf.summary.histogram("weights", w)
  return w

#-----------------------------------------------------------------------------------------------------
def bilinearFilter(filter_shape, upscale_factor, name_ext):
  """
  Weight initialization function initialization with a bilinear filter
  Args:
     filter_shape: shape of the kernel
     upscale_factor: rate at which to upsample
     name_ext: extension of name scope
  Return:
     bilinear_weights: weight variable with bilinear initialization
  """
  kernel_size = filter_shape[1]
  ### Centre location of the filter for which value is calculated
  if kernel_size % 2 == 1:
    centre_location = upscale_factor - 1
  else:
    centre_location = upscale_factor - 0.5
 
  bilinear = np.zeros([filter_shape[0], filter_shape[1]])
  for x in range(filter_shape[0]):
    for y in range(filter_shape[1]):
      ##Interpolation Calculation
      value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
      bilinear[x, y] = value
  weights = np.zeros(filter_shape)
  for i in range(filter_shape[2]):
    for j in range(filter_shape[3]):
      weights[:, :, i, j] = bilinear
  init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
 
  bilinear_weights = tf.get_variable(name="decon_bilinear_filter" + name_ext, initializer=init,
    shape=weights.shape)
    
  tf.summary.histogram("weights", bilinear_weights)  
  return bilinear_weights

#-----------------------------------------------------------------------------------------------------
def biasVariable(name, shape, constant):
  """
  Bias initialization function
  Args:
     name: name scope of bias variable
     shape: shape of bias variable
     constant: constant used for bias initialization
  Return:
     b: bias variable
  """
  b = tf.get_variable(name, shape, 
    initializer = tf.constant_initializer(constant))
  tf.summary.histogram("biases", b)
  return b

#-----------------------------------------------------------------------------------------------------
########################
#ACTIVATION FUNCTIONS  #
########################

def _activation(inTensor):
  """
  Non-linearity layer
  Args:
     inTensor: Tensor to have non-linearity applied
  Return:
     Tensor with non-linearity applied
  """
  if FLAGS.activation == 'relu':
    return tf.nn.relu(inTensor)
  elif FLAGS.activation == 'elu':
    return tf.nn.elu(inTensor)
  elif FLAGS.activation == 'softmax':
    return tf.nn.softmax(inTensor)

#-----------------------------------------------------------------------------------------------------
def softMax(inTensor):
  """
  Softmax layer
  Args:
     inTensor: Input tensor for softmax
  Return:
     softmax: input tensor with softmax
  """
  max_ = tf.reduce_max(inTensor, reduction_indices=[3], keep_dims=True)
  numerator = tf.exp(inTensor - max_)
  denominator = tf.reduce_sum(numerator,reduction_indices=[3],keep_dims=True)
  softmax = tf.div(numerator,denominator)
  return softmax

#-----------------------------------------------------------------------------------------------------
########################
#POOLING LAYERS        #
########################

def maxPool(inTensor, kernel_size, stride, name, padding='SAME', index=False):
  """
  Max-pooling layer. Downsamples values within a kernel to the 
  maximum value within the corresponding kernel
  Args:
     inTensor: Input to the max-pooling layer
     kernel_size: Size of kernel where max-pooling is applied
     stride: Determines the downsample factor
     name: Name scope for operation
     padding: Same or valid padding
     index: Boolean- whether to return pooling indicies
  Return:
     pool: Tensor with max-pooling
     argmax: Indicies of maximal values computed in each kernel (use with segnet)
  """
  strides = [1,stride,stride,1]
  with tf.name_scope('max_pool'):
    ksize = [1, kernel_size, kernel_size, 1]

  if index == False:
    return tf.nn.max_pool(inTensor, ksize, strides,
                      padding=padding, name=name)
  else:
    pool, argmax = tf.nn.max_pool_with_argmax(inTensor, ksize, strides,
                      padding=padding, name=name)
    return pool, argmax

#-----------------------------------------------------------------------------------------------------
def avgPool(inTensor, kernel_size, stride, name, padding='SAME'):
  """
  Average-pooling layer. Downsamples values within a kernel to the 
  average value within the corresponding kernel
  Args:
     inTensor: Input to the max-pooling layer
     kernel_size: Size of kernel where max-pooling is applied
     stride: Determines the downsample factor
     name: Name scope for operation
     padding: Same or valid padding
  Return:
     pool: Tensor with average-pooling
  """
  strides = [1,stride,stride,1]
  with tf.name_scope('max_pool'):
    ksize = [1, kernel_size, kernel_size, 1]

  return tf.nn.max_pool(inTensor, ksize, strides,
                     padding=padding, name=name)

#-----------------------------------------------------------------------------------------------------
def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):
  """
  Unpooling layer after max_pool_with_argmax.
  Args:
     pool: max pooled output tensor
     ind: argmax indices
     ksize:ksize is the same as for the pool
  Return:
     unpool: unpooling tensor
  """
  with tf.variable_scope(name):
    input_shape = pool.get_shape().as_list()
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    flat_input_size = np.prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
    ret = tf.reshape(ret, output_shape)
    return ret

#-----------------------------------------------------------------------------------------------------
########################
#POOLING LAYERS        #
########################

def batchNorm(inTensor, scope, is_training):
  """
  Batch normalization layer. Explain and give references
  Args:
     inTensor: Input to the batch normalization layer
     scope: Name scope for operation
     is_training: Boolean that determines whether in train/test phase.
     This determines whether to use the mini-batch mean or the population mean,
  Return:
     inTensor with batch normalization
  """
  return tf.layers.batch_normalization(inputs=inTensor, axis=-1, momentum=0.9,
   epsilon=0.001, center=True, scale=True, training=is_training, name = scope)

#-----------------------------------------------------------------------------------------------------
########################
#CONVOLUTIONAL LAYERS  #
########################

def conv2D(images, in_channels, out_channels,  stride=1, kernel_size=(3, 3), 
  name_ext='_1', padding='SAME', activation=True, bias= True, activation_before=False):
  '''
  describe function
  '''
  strides = [1,stride,stride,1]
  stddev = tf.sqrt(2/(kernel_size[0]*kernel_size[1]*in_channels))
  kernel = weightVariable(name = 'weights' + name_ext, 
    shape = [kernel_size[0], kernel_size[1], in_channels, out_channels],
    stddev = stddev)
  biases = biasVariable(name ='biases' + name_ext, 
                           shape = [out_channels], 
                           constant = 0.1)
  
  if activation_before == 'False':
    conv = tf.nn.conv2d(images, kernel, strides, padding=padding)
    bias = tf.nn.bias_add(conv, biases)
    if activation:
      output = _activation(bias)
    else:
      output = bias
  else:
    if activation:
      images = _activation(images)
    conv = tf.nn.conv2d(images, kernel, strides, padding=padding)
  if bias: 
    bias = tf.nn.bias_add(conv, biases)
    output = bias

  else:
    output = conv

  return output

#-----------------------------------------------------------------------------------------------------
def conv2D_dilated(images, in_channels, out_channels, rate, kernel_size=(3, 3),
  name_ext='_1', padding='SAME', activation=True, bias=True, activation_before=False):
  '''
  describe function
  '''
  stddev = tf.sqrt(2/(kernel_size[0]*kernel_size[1]*in_channels))
  kernel = weightVariable(name = 'weights' + name_ext, 
    shape = [kernel_size[0], kernel_size[1], in_channels, out_channels],
    stddev = stddev)
  biases = biasVariable(name ='biases' + name_ext, 
                           shape = [out_channels], 
                           constant = 0.1)
  
  if activation_before == 'False':
    conv = tf.nn.atrous_conv2d(images, kernel, rate, padding=padding)
    bias = tf.nn.bias_add(conv, biases)
    if activation:
      output = _activation(bias)
    else:
      output = bias
  else:
    if activation:
      images = _activation(images)
    conv = tf.nn.atrous_conv2d(images, kernel, rate, padding=padding)
    if bias:
      bias = tf.nn.bias_add(conv, biases)
    output = bias

  return output

#-----------------------------------------------------------------------------------------------------
def conv2DTranspose(inTensor, in_channels, out_channels, output_shape, stride, kernel_size, name_ext, padding, initializer):
  '''
  describe function
  '''
  kshape = [kernel_size, kernel_size, out_channels, in_channels]
  
  if initializer == 'bilinear':
    deconv_filter = bilinearFilter(kshape, stride, name_ext) 
  else:
    stddev = tf.sqrt( 2 / (kshape[0] * kshape[1] * kshape[2]))
    deconv_filter = weightVariable(name = 'deconv_weights' + name_ext, 
                              shape = kshape, stddev = stddev)
            
  deconv_bias = biasVariable(name = 'deconv_bias' + name_ext, 
                            shape = kshape[2],
                            constant = 0.1)
  return tf.nn.bias_add(tf.nn.conv2d_transpose(inTensor, deconv_filter, output_shape, strides = [1,stride,stride,1] , padding =padding), deconv_bias)

#-----------------------------------------------------------------------------------------------------
########################
#UPSAMPLING LAYERS  #
########################

def upSample2D_concat(inTensor, symTensor, in_channels, out_channels, upscale_factor, 
  name_ext='_1', padding='SAME', initializer='xavier'):
  '''
  describe function
  '''
  if FLAGS.model_name == 'unet':
    kernel_size = 2
  else:
    kernel_size = 2*upscale_factor - upscale_factor%2
  stride = upscale_factor
  inTensor_shape = inTensor.get_shape().as_list()
  with tf.variable_scope('deconv') as scope:
    out_shape = [FLAGS.train_batch_size, stride*inTensor_shape[1], stride*inTensor_shape[1], out_channels]
    deconv = conv2DTranspose(inTensor, in_channels, out_channels, out_shape, stride, kernel_size, name_ext, padding, initializer)

  with tf.variable_scope('concat') as scope:
      symTensor_shape = symTensor.get_shape().as_list()
      diff = (symTensor_shape[1]-(stride*inTensor_shape[1]))/2
      crop1 = int((symTensor_shape[1]-(stride*inTensor_shape[1]))/2)
      if diff.is_integer():
        crop2 = symTensor_shape[1]-crop1
      else:
        crop2 = (symTensor_shape[1]-crop1)-1
      cropped = symTensor[:, crop1 : crop2 , crop1 : crop2, :]
      concatinated = tf.concat([cropped, deconv],3)

  return concatinated

#-----------------------------------------------------------------------------------------------------
def upSample2D(inTensor, in_channels, out_channels, upscale_factor, name_ext='_1', 
  padding='SAME', initializer='xavier'):
  '''
  describe function
  '''
  kernel_size = (2*upscale_factor) - (upscale_factor%2)
  stride = upscale_factor
  inTensor_shape = inTensor.get_shape().as_list()
  with tf.variable_scope('deconv') as scope:
    out_shape = [FLAGS.train_batch_size, stride*inTensor_shape[1], stride*inTensor_shape[1], out_channels]
    deconv = conv2DTranspose(inTensor, in_channels, out_channels, out_shape, stride, kernel_size, name_ext, padding, initializer)

  return deconv

#-----------------------------------------------------------------------------------------------------
########################
#RESIDUAL UNITS        #
########################

#-----------------------------------------------------------------------------------------------------
def resUnit(inTensor, in_channels, out_channels, is_training, increase_dim = False, resize=None, kernel_size=(3,3), stride=1, padding='SAME', dilation_rate=None):
  '''
  describe function
  '''

  if dilation_rate is None:
    conv1 = conv2D(inTensor, in_channels, out_channels, name_ext='_1', activation=True)
    conv2 = conv2D(conv1, out_channels, out_channels, name_ext='_3', activation=False)
  else:
    conv1 = conv2D_dilated(inTensor, in_channels, out_channels, dilation_rate, name_ext='_1', activation=True)
    conv2 = conv2D_dilated(conv1, out_channels, out_channels, dilation_rate, name_ext='_3', activation=False)

  if resize is not None:
    conv_resize = conv2D(resize, 3, in_channels, name_ext='_4', activation=True)
    concat = tf.concat([inTensor, conv_resize], axis=3)
    inTensor = conv2D(concat, out_channels, out_channels, name_ext='_5', activation=True)

  if increase_dim:
    inTensor = conv2D(inTensor, in_channels, out_channels, stride=1, kernel_size=(1,1), name_ext='_6', activation=True)

  output = tf.add(inTensor, conv2)
  output = _activation(output)

  return output

#-----------------------------------------------------------------------------------------------------
##################
#LOSS FUNCTIONS  #
##################

def weighted_cross_entropy(softmax, labels, weights):
  with tf.name_scope('loss'):
    epsilon=1e-6
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss = -tf.reduce_sum(labels * tf.log(truncated_softmax), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss = tf.multiply(weights,cross_entropy_log_loss)
    avg_cross_entropy_log_loss = tf.reduce_sum(cross_entropy_log_loss,reduction_indices=[0,1,2,3])
    tf.summary.scalar("xent_loss", avg_cross_entropy_log_loss)

  return avg_cross_entropy_log_loss

#-----------------------------------------------------------------------------------------------------
def cross_entropy(softmax, labels):
  with tf.name_scope('loss'):
    epsilon=1e-6
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss = -tf.reduce_sum(labels * tf.log(truncated_softmax), reduction_indices=[3],keep_dims=True)
    avg_cross_entropy_log_loss = tf.reduce_sum(cross_entropy_log_loss,reduction_indices=[0,1,2,3])
    tf.summary.scalar("xent_loss", avg_cross_entropy_log_loss)

  return avg_cross_entropy_log_loss

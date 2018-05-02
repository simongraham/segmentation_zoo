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
def conv2D_sep(images, in_channels, out_channels, stride=1, kernel_size=(3, 3), bias=True,
  name_ext='_1', padding='SAME', rate=None):
  '''
  describe function
  '''
  strides = [1,stride,stride,1]
  stddev = tf.sqrt(2/(kernel_size[0]*kernel_size[1]*in_channels))
  depthwise_filter = weightVariable(name = 'weights' + name_ext + '_1', 
    shape = [kernel_size[0], kernel_size[1], in_channels, 1],
    stddev = stddev)
  pointwise_filter = weightVariable(name = 'weights' + name_ext+ '_2', 
    shape = [1, 1, in_channels*1, out_channels],
    stddev = stddev)
  biases = biasVariable(name ='biases' + name_ext, 
                           shape = [out_channels], 
                           constant = 0.1)
  
  conv = tf.nn.separable_conv2d(images, depthwise_filter, pointwise_filter, strides, padding, rate)
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

def resUnit(inTensor, n_channels, is_training, kernel_size=(3,3), stride=1, padding='SAME', dilated=False, dilation_rate=2):
  '''
  describe function
  '''
  if dilated:
    conv1 = batchNorm(inTensor, 'bn_1', is_training)
    conv1_relu = _activation(conv1)
    conv1_out = conv2D_dilated(conv1_relu, n_channels, n_channels, dilation_rate, activation=False, bias=False)

    conv2 = batchNorm(conv1_out, 'bn_2', is_training)
    conv2_relu = _activation(conv2)
    conv2_out = conv2D_dilated(conv2_relu, n_channels, n_channels, dilation_rate, name_ext = '_2', activation=False, bias=False)
    sum_ = tf.add(inTensor, conv2_out)
  else:
    conv1 = batchNorm(inTensor, 'bn_1', is_training)
    conv1_relu = _activation(conv1)
    conv1_out = conv2D(conv1_relu, n_channels, n_channels, activation=False, bias=False)

    conv2 = batchNorm(conv1_out, 'bn_2', is_training)
    conv2_relu = _activation(conv2)
    conv2_out = conv2D(conv2_relu, n_channels, n_channels, name_ext = '_2', activation=False, bias=False)
    sum_ = tf.add(inTensor, conv2_out)

  return sum_

#-----------------------------------------------------------------------------------------------------
def resUnit2(inTensor, n_channels, is_training, kernel_size=(3,3), stride=1, padding='SAME', dilated=False, dilation_rate=2):
  '''
  describe function
  '''
  if dilated:

    conv1 = conv2D_dilated(inTensor, n_channels, n_channels, dilation_rate, activation=False)
    conv1_relu = _activation(conv1)

    conv2 = conv2D_dilated(conv1_relu, n_channels, n_channels, dilation_rate, name_ext = '_2', activation=False)
    sum_ = tf.add(inTensor, conv2)
    output = _activation(sum_)

  else:
    conv1 = conv2D(inTensor, n_channels, n_channels,  activation=False)
    conv1_relu = _activation(conv1)

    conv2 = conv2D(conv1_relu, n_channels, n_channels, name_ext = '_2', activation=False)
    sum_ = tf.add(inTensor, conv2)
    output = _activation(sum_)

  return sum_

#-----------------------------------------------------------------------------------------------------
def resUnitBottleneck_orig(inTensor, in_channels, n_channels1, n_channels2, is_training, stride=2, kernel_size=(3,3), padding='SAME', connection=True, dilated=False, dilation_rate=2):
  '''
  describe function
  '''
  if connection:
    if dilated:
      conv1 = conv2D(inTensor, in_channels, n_channels1, kernel_size=(1,1), name_ext='_1', activation=False)
      conv1 = batchNorm(conv1, 'bn_1', is_training)
      conv1_relu = _activation(conv1)

      conv2 = conv2D_dilated(conv1_relu, n_channels1, n_channels1, dilation_rate, name_ext='_2', activation=False)
      conv2 = batchNorm(conv2, 'bn_2', is_training)
      conv2_relu = _activation(conv2)

      conv3 = conv2D(conv2_relu, n_channels1, n_channels2, kernel_size=(1,1), name_ext='_3', activation=False)
      conv3 = batchNorm(conv3, 'bn_3', is_training)

      sum_ = tf.add(inTensor, conv3)
      output = _activation(sum_)

    else:
      conv1 = conv2D(inTensor, in_channels, n_channels1, kernel_size=(1,1), name_ext='_1', activation=False)
      conv1 = batchNorm(conv1, 'bn_1', is_training)
      conv1_relu = _activation(conv1)

      conv2 = conv2D(conv1_relu, n_channels1, n_channels1, name_ext='_2', activation=False)
      conv2 = batchNorm(conv2, 'bn_2', is_training)
      conv2_relu = _activation(conv2)

      conv3 = conv2D(conv2_relu, n_channels1, n_channels2, name_ext='_3', kernel_size=(1,1), activation=False)
      conv3 = batchNorm(conv3, 'bn_3', is_training)

      sum_ = tf.add(inTensor, conv3)
      output = _activation(sum_)
  else:
    if dilated:
      conv1 = conv2D(inTensor, in_channels, n_channels1, stride, kernel_size=(1,1), name_ext='_1', activation=False)
      conv1 = batchNorm(conv1, 'bn_1', is_training)
      conv1_relu = _activation(conv1)

      conv2 = conv2D_dilated(conv1_relu, n_channels1, n_channels1, dilation_rate, name_ext='_2', activation=False)
      conv2 = batchNorm(conv2, 'bn_2', is_training)
      conv2_relu = _activation(conv2)

      conv3 = conv2D(conv2_relu, n_channels1, n_channels2, kernel_size=(1,1), name_ext='_3', activation=False)
      conv3 = batchNorm(conv3, 'bn_3', is_training)
      output = _activation(conv3)

    else:
      conv1 = conv2D(inTensor, in_channels, n_channels1, stride ,kernel_size=(1,1), name_ext='_1', activation=False)
      conv1 = batchNorm(conv1, 'bn_1', is_training)
      conv1_relu = _activation(conv1)

      conv2 = conv2D(conv1_relu, n_channels1, n_channels1, name_ext='_2', activation=False)
      conv2 = batchNorm(conv2, 'bn_2', is_training)
      conv2_relu = _activation(conv2)

      conv3 = conv2D(conv2_relu, n_channels1, n_channels2, name_ext='_3', kernel_size=(1,1), activation=False)
      conv3 = batchNorm(conv3, 'bn_3', is_training)
      output = _activation(conv3)

  return output

#-----------------------------------------------------------------------------------------------------
def splitResUnit(resizeTensor, inTensor, n_channels, is_training, kernel_size=(3,3), stride=1, padding='SAME', dilated=False):
  '''
  describe function
  '''
  conv1_bn = batchNorm(inTensor, 'conv1_bn', is_training)
  conv1_relu = _activation(conv1_bn)
  conv1 = conv2D(conv1_relu, n_channels, n_channels, name_ext='_1', activation=False)

  conv_resize_bn = batchNorm(resizeTensor, 'resize_bn', is_training)
  conv_resize_relu = _activation(conv_resize_bn)
  conv_resize = conv2D(conv_resize_relu, 3, n_channels, name_ext = '_2', activation=False)

  conv_concat = tf.concat([conv1, conv_resize], axis=3)
  conv_concat_bn = batchNorm(conv_concat, 'concat_bn', is_training)
  conv_concat_relu = _activation(conv_concat_bn)
  conv_concat = conv2D(conv_concat_relu, 2*n_channels, n_channels, kernel_size=(1,1), name_ext = '_3', activation=False)

  conv2_bn = batchNorm(conv_concat, 'con2_bn', is_training)
  conv2_relu = _activation(conv2_bn)
  conv2 = conv2D(conv2_relu, n_channels, n_channels, name_ext='_4', activation=False)
  
  output = tf.add(inTensor, conv2)

  return output

#-----------------------------------------------------------------------------------------------------
def splitResUnit2(resizeTensor, inTensor, n_channels, is_training, kernel_size=(3,3), stride=1, padding='SAME', dilated=False):
  '''
  describe function
  '''

  conv1 = conv2D(inTensor, n_channels, n_channels, name_ext='_1', activation=False)
  conv1_relu = _activation(conv1)

  conv_resize = conv2D(resizeTensor, 3, n_channels, name_ext = '_2', activation=False)
  conv_resize_relu = _activation(conv_resize)

  conv_concat = tf.concat([conv1_relu, conv_resize_relu], axis=3)
  conv_concat = conv2D(conv_concat, 2*n_channels, n_channels, kernel_size=(1,1), name_ext = '_3', activation=False)

  conv2 = conv2D(conv_concat, n_channels, n_channels, name_ext='_4', activation=False)
  sum2 = tf.add(inTensor, conv2)
  output = _activation(sum2)

  return output

#-----------------------------------------------------------------------------------------------------
def resUnit_x1(inTensor, in_channels, out_channels, is_training, resize=None, kernel_size=(3,3), stride=1, padding='SAME', dilated=False):
  '''
  describe function
  '''
  conv1 = conv2D(inTensor, in_channels, out_channels, name_ext='_1', activation=True)
  conv2 = conv2D(conv1, out_channels, out_channels, name_ext='_2',activation=False)

  if resize is not None:
    conv_resize = conv2D(resize, 3, in_channels, name_ext='_3', activation=True)
    concat = tf.concat([inTensor, conv_resize], axis=3)
    inTensor = conv2D(concat, out_channels, out_channels, name_ext='_4', activation=True)
  
  output = tf.add(inTensor, conv2)
  output = _activation(output)

  return output

#-----------------------------------------------------------------------------------------------------
def resUnit_x2(inTensor, in_channels, out_channels, is_training, increase_dim = False, resize=None, kernel_size=(3,3), stride=1, padding='SAME', dilation_rate=None):
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
def res_aspp(inTensor, n_channels, is_training, dilation1, dilation2, dilation3, dropout=False, kernel_size=(3,3), padding='SAME'):
  '''
  describe function
  '''

  conv1 = conv2D_dilated(inTensor, n_channels, n_channels, rate=dilation1, name_ext = '_1', activation=False)
  conv1 = _activation(conv1)
  if dropout:
    conv1 = tf.nn.dropout(conv1,0.5)
  conv1 = conv2D(conv1, n_channels, n_channels, kernel_size = (1,1), name_ext = '_3', activation=False)
  conv1 = _activation(conv1)

  conv2 = conv2D_dilated(inTensor, n_channels, n_channels, rate=dilation2, name_ext = '_4', activation=False)
  conv2 = _activation(conv2)
  if dropout:
    conv2 = tf.nn.dropout(conv2, 0.5)
  conv2 = conv2D(conv2, n_channels, n_channels, kernel_size = (1,1), name_ext = '_6', activation=False)
  conv2 = _activation(conv2)

  conv3 = conv2D_dilated(inTensor, n_channels, n_channels, rate=dilation3, name_ext = '_7', activation=False)
  conv3 = _activation(conv3)
  if dropout:
    conv3 = tf.nn.dropout(conv3, 0.5)
  conv3 = conv2D(conv3, n_channels, n_channels, kernel_size = (1,1), name_ext = '_9', activation=False)
  conv3 = _activation(conv3)

  sum_ = tf.add(conv1, conv2)
  sum_ = tf.add(sum_, conv3)

  output = tf.add(sum_, inTensor)

  return output

#-----------------------------------------------------------------------------------------------------
def res_aspp_bn(inTensor, n_channels, is_training, dilation1, dilation2, dilation3, dropout=False, kernel_size=(3,3), padding='SAME'):
  '''
  describe function
  '''

  conv1 = conv2D_dilated(inTensor, n_channels, n_channels, rate=dilation1, name_ext = '_1', activation=False)
  conv1 = batchNorm(conv1, 'conv1_bn', is_training)
  conv1 = _activation(conv1)
  if dropout:
    conv1 = tf.nn.dropout(conv1,0.5)
  conv1 = conv2D(conv1, n_channels, n_channels, kernel_size = (1,1), name_ext = '_3', activation=False)
  conv1 = batchNorm(conv1, 'conv1_bn2', is_training)
  conv1 = _activation(conv1)

  conv2 = conv2D_dilated(inTensor, n_channels, n_channels, rate=dilation2, name_ext = '_4', activation=False)
  conv2 = batchNorm(conv2, 'conv2_bn', is_training)
  conv2 = _activation(conv2)
  if dropout:
    conv2 = tf.nn.dropout(conv2, 0.5)
  conv2 = conv2D(conv2, n_channels, n_channels, kernel_size = (1,1), name_ext = '_6', activation=False)
  conv2 = batchNorm(conv2, 'conv2_bn2', is_training)
  conv2 = _activation(conv2)

  conv3 = conv2D_dilated(inTensor, n_channels, n_channels, rate=dilation3, name_ext = '_7', activation=False)
  conv3 = batchNorm(conv3, 'conv3_bn', is_training)
  conv3 = _activation(conv3)
  if dropout:
    conv3 = tf.nn.dropout(conv3, 0.5)
  conv3 = conv2D(conv3, n_channels, n_channels, kernel_size = (1,1), name_ext = '_9', activation=False)
  conv3 = batchNorm(conv3, 'conv3_bn2', is_training)
  conv3 = _activation(conv3)

  sum_ = tf.add(conv1, conv2)
  sum_ = tf.add(sum_, conv3)

  output = tf.add(sum_, inTensor)

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

#-----------------------------------------------------------------------------------------------------
def loss_dcan(softmax_gland1, softmax_gland2, softmax_gland3, softmax_glandf, 
  softmax_contour1, softmax_contour2, softmax_contour3, softmax_contourf,
  labels_gland, labels_contour, discount_weight):
  
  with tf.name_scope('loss'):
    epsilon=1e-6
    
    w = discount_weight
    w = tf.cast(w, tf.float32)

    truncated_softmax_gland1 = tf.clip_by_value(softmax_gland1, epsilon, 1.0 - epsilon)
    truncated_softmax_gland2 = tf.clip_by_value(softmax_gland2, epsilon, 1.0 - epsilon)
    truncated_softmax_gland3 = tf.clip_by_value(softmax_gland3, epsilon, 1.0 - epsilon)
    truncated_softmax_glandf = tf.clip_by_value(softmax_glandf, epsilon, 1.0 - epsilon)
    truncated_softmax_contour1 = tf.clip_by_value(softmax_contour1, epsilon, 1.0 - epsilon)
    truncated_softmax_contour2 = tf.clip_by_value(softmax_contour2, epsilon, 1.0 - epsilon)
    truncated_softmax_contour3 = tf.clip_by_value(softmax_contour3, epsilon, 1.0 - epsilon)
    truncated_softmax_contourf = tf.clip_by_value(softmax_contourf, epsilon, 1.0 - epsilon)

    cross_entropy_log_loss_g1 = -tf.reduce_sum(labels_gland * tf.log(truncated_softmax_gland1), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_g2 = -tf.reduce_sum(labels_gland * tf.log(truncated_softmax_gland2), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_g3 = -tf.reduce_sum(labels_gland * tf.log(truncated_softmax_gland3), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_gf = -tf.reduce_sum(labels_gland * tf.log(truncated_softmax_glandf), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_c1 = -tf.reduce_sum(labels_contour * tf.log(truncated_softmax_contour1), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_c2 = -tf.reduce_sum(labels_contour * tf.log(truncated_softmax_contour2), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_c3 = -tf.reduce_sum(labels_contour * tf.log(truncated_softmax_contour3), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss_cf = -tf.reduce_sum(labels_contour * tf.log(truncated_softmax_contourf), reduction_indices=[3],keep_dims=True)

    avg_cross_entropy_log_loss_g1 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss_g1,reduction_indices=[0,1,2,3]))
    avg_cross_entropy_log_loss_g2 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss_g2,reduction_indices=[0,1,2,3]))
    avg_cross_entropy_log_loss_g3 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss_g3,reduction_indices=[0,1,2,3]))
    avg_cross_entropy_log_loss_gf = tf.reduce_sum(cross_entropy_log_loss_gf,reduction_indices=[0,1,2,3])
    avg_cross_entropy_log_loss_c1 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss_c1,reduction_indices=[0,1,2,3]))
    avg_cross_entropy_log_loss_c2 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss_c2,reduction_indices=[0,1,2,3]))
    avg_cross_entropy_log_loss_c3 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss_c3,reduction_indices=[0,1,2,3]))
    avg_cross_entropy_log_loss_cf = tf.reduce_sum(cross_entropy_log_loss_cf,reduction_indices=[0,1,2,3])

    overall_loss =  avg_cross_entropy_log_loss_g1 +  avg_cross_entropy_log_loss_g2 +  avg_cross_entropy_log_loss_g3 +  avg_cross_entropy_log_loss_gf + \
     avg_cross_entropy_log_loss_c1 +  avg_cross_entropy_log_loss_c2 +  avg_cross_entropy_log_loss_c3 +  avg_cross_entropy_log_loss_cf

    #tf.summary.scalar("xent_loss", avg_cross_entropy_log_loss)
  return overall_loss

#-----------------------------------------------------------------------------------------------------
def weighted_cross_entropy_aux(softmax, aux_softmax1, labels, weights, discount_weight):
  with tf.name_scope('loss'):
    epsilon=1e-6

    w = discount_weight
    w = tf.cast(w, tf.float32)

    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss = -tf.reduce_sum(labels * tf.log(truncated_softmax), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss = tf.multiply(weights,cross_entropy_log_loss)
    avg_cross_entropy_log_loss = tf.reduce_sum(cross_entropy_log_loss,reduction_indices=[0,1,2,3])

    truncated_softmax2 = tf.clip_by_value(aux_softmax1, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss2 = -tf.reduce_sum(labels * tf.log(truncated_softmax2), reduction_indices=[3],keep_dims=True)
    cross_entropy_log_loss2 = tf.multiply(weights,cross_entropy_log_loss2)
    avg_cross_entropy_log_loss2 = tf.scalar_mul(w,tf.reduce_sum(cross_entropy_log_loss2,reduction_indices=[0,1,2,3]))

    overall_loss = avg_cross_entropy_log_loss + avg_cross_entropy_log_loss2


    tf.summary.scalar("xent_loss", overall_loss)

  return overall_loss
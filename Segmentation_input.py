'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from six.moves import xrange  
import tensorflow as tf
import numpy as np
import cv2
import Segmentation_parameters
import scipy.io as sio
import csv
from tensorflow.python.ops import control_flow_ops

FLAGS = tf.app.flags.FLAGS

source_size = FLAGS.train_image_source_size
target_size = FLAGS.train_image_target_size 
ground_truth_size = FLAGS.ground_truth_size

#####################################################################################
def get_data_list(images_path, labels_path, weights_path, image_ext):
  images_list = glob.glob(os.path.join(images_path, '*' + image_ext))
  tmp_images_list = []
  tmp_labels_list = []
  tmp_weights_list = []

  for image_name in images_list:
    basename = os.path.basename(image_name)
    basename = basename.split('.')[0]
    label_name = os.path.join(labels_path, basename + image_ext)
    if weights_path is not None:
      weight_name = os.path.join(weights_path, basename + image_ext)

    if os.path.isfile(label_name):
      tmp_images_list.append(image_name)
      tmp_labels_list.append(label_name)
      if weights_path is not None:
        tmp_weights_list.append(weight_name)

  images_list = tmp_images_list
  labels_list = tmp_labels_list
  if weights_path is not None:
    weights_list = tmp_weights_list
  else:
    weights_list = []

  if not images_list:
    print("no data in %s or %s" % (images_path,labels_path))
    raise ValueError('terminate!')

  if not labels_list:
    print()
    raise ValueError('terminate!')

  return images_list, labels_list, weights_list

#####################################################################################
def calculate_mean_variance_image(list_images):
  
  image = cv2.imread(list_images[0])
  
  image = np.dstack((image[:,:,2],image[:,:,1],image[:,:,0]))

  if np.random.randint(2, size=1) == 1:
    image = np.flipud(image)
  if np.random.randint(2, size=1) == 1:
    image = np.fliplr(image)
  image = np.float32(image)

  mean_image = image
  variance_image = np.zeros(shape = image.shape, dtype = np.float32)

  for t, image_file in enumerate(list_images[1:]):
    image = cv2.imread(image_file)
    
    image = np.dstack((image[:,:,2],image[:,:,1],image[:,:,0]))

    if np.random.randint(2, size=1) == 1:
        image = np.flipud(image)
    if np.random.randint(2, size=1) == 1:
        image = np.fliplr(image)
    image = np.float32(image)

    mean_image = (np.float32(t + 1)*mean_image + image)/np.float32(t+2)
    
    variance_image = np.float32(t+1)/np.float32(t+2)*variance_image \
                      + np.float32(1)/np.float32(t+1)*((image - mean_image)**2)
  
  return mean_image, variance_image
#####################################################################################
def export_result(save_filename,score, prediction):
  if not tf.gfile.Exists(FLAGS.result_dir):
    tf.gfile.MakeDirs(FLAGS.result_dir)

  sio.savemat(os.path.join(FLAGS.result_dir,save_filename+'.mat'), {'score':score, 'prediction':prediction})
#####################################################################################
def MergePatches_test(patches,stride,image_size,sizeInputPatch,sizeOutputPatch):
  patches = np.float32(patches)

  rowRange = range(0, image_size[0]-sizeInputPatch[0], stride[0])
  colRange = range(0, image_size[1]-sizeInputPatch[1], stride[1])

  centreIndexRow = int((sizeInputPatch[0] - sizeOutputPatch[0])/2)
  centreIndexCol = int((sizeInputPatch[1] - sizeOutputPatch[1])/2)

  image = np.zeros([image_size[0],image_size[1],patches.shape[3]], dtype = np.float32)
  count = np.zeros([image_size[0],image_size[1],patches.shape[3]], dtype = np.float32)

  for index1, row in enumerate(rowRange):
    for index2, col in enumerate(colRange):       
      image[row + centreIndexRow : row + centreIndexRow + sizeOutputPatch[0],
              col + centreIndexCol : col + centreIndexCol + sizeOutputPatch[1],
              :] += patches[(index1*len(colRange))+index2,:,:,:]
      count[row + centreIndexRow : row + centreIndexRow + sizeOutputPatch[0],
              col + centreIndexCol : col + centreIndexCol + sizeOutputPatch[1],
              :] += 1.0
              
  mask = count > 0.0
  image[mask] = np.divide(image[mask],count[mask])

  # crop image
  image = image[centreIndexRow + 2*stride[0]:image.shape[0]-(centreIndexRow + 2*stride[0]),
                centreIndexCol + 2*stride[1]:image.shape[1]-(centreIndexCol + 2*stride[1]),
                :]
  return image

#####################################################################################
def read_data(filename_queue):

  image_content = tf.read_file(filename_queue[0])
  label_content = tf.read_file(filename_queue[1])
  if len(filename_queue)==3:
    weight_content = tf.read_file(filename_queue[2])
  
  image = tf.image.decode_image(image_content, channels=3)
  label = tf.image.decode_image(label_content, channels=3)
  if len(filename_queue)==3:
    weight = tf.image.decode_image(weight_content, channels=1)

  image.set_shape([FLAGS.train_image_source_size,FLAGS.train_image_source_size,3])
  label.set_shape([FLAGS.train_image_source_size,FLAGS.train_image_source_size,3])
  if len(filename_queue)==3:
    weight.set_shape([FLAGS.train_image_source_size,FLAGS.train_image_source_size,1])
  else:
    weight = None

  return image, label, weight

#####################################################################################
def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

#####################################################################################
def distort_color(image, color_ordering=0, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  image = image/255

  with tf.name_scope(scope, 'distort_color', [image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=0.1)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    elif color_ordering == 1:
      image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
      image = tf.image.random_brightness(image, max_delta=0.1)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.04)
    elif color_ordering == 2:
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_brightness(image, max_delta=0.1)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    elif color_ordering == 3:
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_brightness(image, max_delta=0.1)
    else:
      raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = image*255

    return image
  
#####################################################################################
def process_image_and_label(image, label, weight, source_size, target_size, 
  ground_truth_size, mean_image, variance_image):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.
  """
  if image.dtype != tf.float32:
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    if weight is not None:
      weight = tf.cast(weight,tf.float32)

  # Randomly crop the image
  if source_size >= target_size:
    slice_index = int((target_size-ground_truth_size)/2)
    if weight is not None:
      concat = tf.concat([image, label, weight], axis=2)
      crop = tf.random_crop(concat, [target_size,target_size,7])
      image = tf.slice(crop,[0,0,0],[target_size,target_size,3])
      label = tf.slice(crop,[slice_index,slice_index,3],[ground_truth_size,ground_truth_size,2]) 
      weight = tf.slice(crop,[slice_index,slice_index,6],[ground_truth_size,ground_truth_size,1])
    else:
      concat = tf.concat([image, label], axis=2)
      crop = tf.random_crop(concat, [target_size,target_size,6])
      image = tf.slice(crop,[0,0,0],[target_size,target_size,3])
      label = tf.slice(crop,[slice_index,slice_index,3],[ground_truth_size,ground_truth_size,2]) 
  else:
    # Resize the image to the specified target height and target width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [target_size, target_size], align_corners=False)
    image = tf.squeeze(image, [0])
    label = tf.expand_dims(label, 0)
    label = tf.image.resize_bilinear(label, [ground_truth_size, ground_truth_size], align_corners=False)
    label = tf.squeeze(label, [0])
    if weight is not None:
      weight = tf.expand_dims(weight, 0)
      weight = tf.image.resize_bilinear(weight, [ground_truth_size, ground_truth_size], align_corners=False)
      weight = tf.squeeze(weight, [0])
  tf.summary.image('cropped_image', tf.expand_dims(image, 0))

  if FLAGS.random_rotation:
    # Randomly rotate the image from 90,180,270,360 degrees.
    num_rotation = tf.random_uniform([1],minval=0,maxval=5,dtype=tf.int32)
    image = tf.image.rot90(image, k=num_rotation[0])
    image.set_shape([target_size, target_size, 3])
    label = tf.image.rot90(label, k=num_rotation[0])
    label.set_shape([ground_truth_size, ground_truth_size, 2])
    if weight is not None:
      weight = tf.image.rot90(weight, k=num_rotation[0])
      weight.set_shape([ground_truth_size, ground_truth_size, 1])
    tf.summary.image('rotated_image', tf.expand_dims(image, 0))

  if FLAGS.random_flipping:
    # Randomly flip the image horizontally and vertically.
    rand1 = np.random.randint(0,2)
    rand2 = np.random.randint(0,2)
    if rand1 == 1:
      image = tf.image.flip_left_right(image)
      label = tf.image.flip_left_right(label)
      if weight is not None:
        weight = tf.image.flip_left_right(weight)
    if rand2 == 1:
      image = tf.image.flip_up_down(image)
      label = tf.image.flip_up_down(label)
      if weight is not None:
        weight = tf.image.flip_up_down(weight)
    tf.summary.image('flipped_image', tf.expand_dims(image, 0))

  # Randomly flip the image horizontally and vertically.
  	random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
  	image= control_flow_ops.cond(pred=tf.equal(random_var, 0),
  		fn1=lambda: tf.image.flip_left_right(image),
  		fn2=lambda: image)
  	label = control_flow_ops.cond(pred=tf.equal(random_var, 0),
  		fn1=lambda: tf.image.flip_left_right(label),
  		fn2=lambda: label)
  	if weight is not None:
  		weight = control_flow_ops.cond(pred=tf.equal(random_var, 0),
  			fn1=lambda: tf.image.flip_left_right(weight),
  			fn2=lambda: weight)

  	random_var2 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
  	image= control_flow_ops.cond(pred=tf.equal(random_var2, 0),
  		fn1=lambda: tf.image.flip_up_down(image),
  		fn2=lambda: image)
  	label = control_flow_ops.cond(pred=tf.equal(random_var2, 0),
  		fn1=lambda: tf.image.flip_up_down(label),
  		fn2=lambda: label)
  	if weight is not None:
  		weight = control_flow_ops.cond(pred=tf.equal(random_var2, 0),
  			fn1=lambda: tf.image.flip_up_down(weight),
  			fn2=lambda: weight)

  	image.set_shape([target_size, target_size, 3])
  	label.set_shape([ground_truth_size, ground_truth_size, 2])
  	weight.set_shape([ground_truth_size, ground_truth_size, 1])
      
    tf.summary.image('flipped_image', tf.expand_dims(image, 0))

  if FLAGS.color_distortion:
    # Randomly distort the colors. There are 4 ways to do it.
    distorted_image = apply_with_random_selector(image, lambda x, ordering: distort_color(x, ordering), num_cases=4)
  else:
    distorted_image = image
  tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

  epsilon = 1e-6
  if FLAGS.zero_centre:
    distorted_image = distorted_image - mean_image
  if FLAGS.normalize:
    distorted_image = distorted_image/tf.sqrt(variance_image + epsilon)

  if weight is not None:
    return distorted_image, label, weight
  else:
    return distorted_image, label

#####################################################################################  
def process_image_test(patches,mean_image,variance_image):
  epsilon = 1e-6
  for ipatch in range(patches.shape[0]):
    image = patches[ipatch,:,:,:]
    if FLAGS.zero_centre:
      image = image - mean_image
    if FLAGS.normalize:
      image = image/np.sqrt(variance_image + epsilon)
    patches[ipatch,:,:,:] = image

  return patches
#####################################################################################
def generate_image_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  return images, labels

#####################################################################################
def generate_image_label_weight_batch(image, label, weight, min_queue_examples, batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, labels, weights= tf.train.shuffle_batch(
        [image, label, weight],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, labels, weights = tf.train.batch(
        [image, label, weight],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  return images, labels, weights

#####################################################################################
def generate_image_and_label_batch_test(image, min_queue_examples,batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  return images  
#####################################################################################
def inputs_train(mean_image,variance_image):
  image_ext = FLAGS.image_ext
  images_path = os.path.join(FLAGS.train_dir, 'Images')
  labels_path = os.path.join(FLAGS.train_dir, 'Labels')
  if FLAGS.loss_function  == 'weighted_cross_entropy' or FLAGS.loss_function == 'weighted_cross_entropy_aux':
    weights_path = os.path.join(FLAGS.train_dir, 'Weights')
  else:
    weights_path = None

  image_names, label_names, weight_names = get_data_list(images_path, labels_path, weights_path, image_ext)

  min_queue_examples = int(len(image_names) * FLAGS.min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
  # Create a queue that produces the filenames to read.

  if weights_path is not None:
    filename_queue = tf.train.slice_input_producer([image_names, label_names, weight_names],shuffle = True)
  else:
    filename_queue = tf.train.slice_input_producer([image_names, label_names],shuffle = True)

  # Read examples from files in the filename queue.
  image, label, weight= read_data(filename_queue)
  if weights_path is not None:
    image, label, weight = process_image_and_label(image, label, weight, source_size, 
      target_size, ground_truth_size, mean_image,variance_image)
    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_label_weight_batch(image, label, weight, min_queue_examples, 
      FLAGS.train_batch_size, shuffle=False)
  else:
    image, label = process_image_and_label(image, label, weight, source_size, 
      target_size, ground_truth_size, mean_image,variance_image)
    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_label_batch(image, label, min_queue_examples, 
      FLAGS.train_batch_size, shuffle=False)
#####################################################################################
def inputs_val(mean_image,variance_image):
  image_ext = FLAGS.image_ext
  images_path = os.path.join(FLAGS.val_dir, 'Images')
  labels_path = os.path.join(FLAGS.val_dir, 'Labels')
  if FLAGS.loss_function  == 'weighted_cross_entropy' or FLAGS.loss_function == 'weighted_cross_entropy_aux':
    weights_path = os.path.join(FLAGS.val_dir, 'Weights')
  else:
    weights_path = None

  image_names, label_names, weight_names = get_data_list(images_path, labels_path, weights_path, image_ext)

  min_queue_examples = int(len(image_names) * FLAGS.min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
  # Create a queue that produces the filenames to read.

  if weights_path is not None:
    filename_queue = tf.train.slice_input_producer([image_names, label_names, weight_names],shuffle = True)
  else:
    filename_queue = tf.train.slice_input_producer([image_names, label_names],shuffle = True)

  # Read examples from files in the filename queue.
  image, label, weight= read_data(filename_queue)
  if weights_path is not None:
    image, label, weight = process_image_and_label(image, label, weight, source_size, 
      target_size, ground_truth_size, mean_image,variance_image)
    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_label_weight_batch(image, label, weight, min_queue_examples, 
      FLAGS.train_batch_size, shuffle=False)
  else:
    image, label = process_image_and_label(image, label, weight, source_size, 
      target_size, ground_truth_size, mean_image,variance_image)
    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_label_batch(image, label, min_queue_examples, 
      FLAGS.train_batch_size, shuffle=False)
 
#####################################################################################
def inputs_test(patches, mean_image,variance_image):
  patches = process_image_test(patches,mean_image,variance_image)
  return patches
#####################################################################################
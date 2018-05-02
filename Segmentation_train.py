'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import glob
import numpy as np
import scipy.io as sio
from six.moves import xrange
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline
import pandas as pd

import Segmentation_input
import Segmentation_parameters
import Segmentation_networks
import Segmentation_layers as layers

FLAGS = tf.app.flags.FLAGS

 ##############################################################################
def configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.
  Args:
  num_samples_per_epoch: The number of samples in each epoch of training.
  global_step: The global_step tensor.
  Returns:
  A `Tensor` representing the learning rate.
  Raises:
  ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.train_batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)

##############################################################################
def configure_optimizer(learning_rate):
  """Configures the optimizer used for training.
  Args:
    learning_rate: A scalar or `Tensor` learning rate.
  Returns:
    An instance of an optimizer.
  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
    """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate, 
      rho=FLAGS.adadelta_rho,epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate,
      initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate,
      beta1=FLAGS.adam_beta1,beta2=FLAGS.adam_beta2,epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(learning_rate,learning_rate_power=FLAGS.ftrl_learning_rate_power,
      initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
      l1_regularization_strength=FLAGS.ftrl_l1,l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate,
      momentum=FLAGS.momentum,name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=FLAGS.rmsprop_decay,
      momentum=FLAGS.rmsprop_momentum,epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

###################################################################
def train():
  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  with tf.Graph().as_default():
    #with tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False, name = 'global_step')
    # Epoch counter
    curr_epoch = tf.Variable(0, trainable=False, name = 'curr_epoch')
    update_curr_epoch = tf.assign(curr_epoch, tf.add(curr_epoch, tf.constant(1)))
    # drop out
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    discount_weight = tf.Variable(1.0, trainable=False, name = 'curr_epoch')
    update_discount_weight = tf.assign(discount_weight, tf.div(discount_weight, 10))

    # network stats
    # Load network stats
    mat_contents = sio.loadmat(os.path.join(FLAGS.stats_dir,'network_stats.mat'))
    mean_img = np.float32(mat_contents['mean_image'])
    variance_img = np.float32(mat_contents['variance_image'])

    mean_image = tf.Variable(mean_img, trainable=False, name = 'mean_image')
    variance_image = tf.Variable(variance_img, trainable=False, name = 'variance_image')

    # Get images and labels.
    with tf.name_scope('inputs'):
      if FLAGS.loss_function == 'weighted_cross_entropy' or FLAGS.loss_function == 'weighted_cross_entropy_aux':
        images_train, labels_train, weights_train = Segmentation_input.inputs_train(mean_image,variance_image)
        images_val, labels_val, weights_val = Segmentation_input.inputs_val(mean_image,variance_image)
      else:
        images_train, labels_train = Segmentation_input.inputs_train(mean_image,variance_image)
        images_val, labels_val = Segmentation_input.inputs_val(mean_image,variance_image)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope("network") as scope:
      '''
      Passes the data through the deep learning model
      '''
      if FLAGS.model_name == 'unet':
        softmaxs_train = Segmentation_networks.unet(images_train,keep_prob)
        scope.reuse_variables()
        softmaxs_val = Segmentation_networks.unet(images_val,keep_prob)
      elif FLAGS.model_name == 'fcn8':
        softmaxs_train = Segmentation_networks.fcn8(images_train,keep_prob)
        scope.reuse_variables()
        softmaxs_val = Segmentation_networks.fcn8(images_val,keep_prob)
      elif FLAGS.model_name == 'fcn16':
        softmaxs_train = Segmentation_networks.fcn16(images_train,keep_prob)
        scope.reuse_variables()
        softmaxs_val = Segmentation_networks.fcn16(images_val,keep_prob)
      elif FLAGS.model_name == 'fcn32':
        softmaxs_train = Segmentation_networks.fcn32(images_train,keep_prob)
        scope.reuse_variables()
        softmaxs_val = Segmentation_networks.fcn32(images_val,keep_prob)
      elif FLAGS.model_name == 'segnet':
        softmaxs_train = Segmentation_networks.segnet(images_train,keep_prob, is_training)
        scope.reuse_variables()
        softmaxs_val = Segmentation_networks.segnet(images_val,keep_prob, is_training)
      else:
        raise ValueError('Network architecture not recognised')
    
    # Calculate loss.
    if FLAGS.loss_function == 'weighted_cross_entropy':
      loss_train = layers.weighted_cross_entropy(softmaxs_train, labels_train, weights_train)
      loss_val = layers.weighted_cross_entropy(softmaxs_val, labels_val, weights_val)
    elif FLAGS.loss_function == 'cross_entropy':
      loss_train = layers.cross_entropy(softmaxs_train, labels_train)
      loss_val = layers.cross_entropy(softmaxs_val, labels_val)
    else:
      raise ValueError('Loss function not recognised')

    # Accuracy for each class
    with tf.name_scope('training_predictions'):
      predict_train = tf.argmax(softmaxs_train,3)
      predict_train = tf.reshape(predict_train,[-1])
    with tf.name_scope('training_labels'):
      actual_train = tf.squeeze(tf.cast(labels_train-1,dtype = tf.int64))
      actual_train = tf.argmax(actual_train, 3)
      actual_train = tf.reshape(actual_train,[-1])
    with tf.name_scope('training_accuracy'):
      correct_prediction_train = tf.equal(predict_train, actual_train)
      accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))
    with tf.name_scope('validation_predictions'):
      predict_val = tf.argmax(softmaxs_val,3)
      predict_val = tf.reshape(predict_val,[-1])
    with tf.name_scope('validation_labels'):
      actual_val = tf.squeeze(tf.squeeze(tf.cast(labels_val-1,dtype = tf.int64)))
      actual_val = tf.argmax(actual_val, 3)
      actual_val = tf.reshape(actual_val,[-1])
    with tf.name_scope('validation_accuracy'):
      correct_prediction_val = tf.equal(predict_val,actual_val)
      accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    num_examples_per_epoch_for_train = len(glob.glob(os.path.join(FLAGS.train_dir, 'Images', '*' + FLAGS.image_ext)))
    learning_rate = configure_learning_rate(num_examples_per_epoch_for_train, global_step)
    optimizer = configure_optimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_train)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Create a saver.
    #saver = tf.train.Saver(tf.global_variables(),max_to_keep = 0)
    saver = tf.train.Saver(max_to_keep = 0)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=FLAGS.num_cores,
                                          intra_op_parallelism_threads=FLAGS.num_cores)) as sess:
    
      sess.run(init)
      # Create a summary file writer
      writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      # Start the queue runners.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        mat_contents = sio.loadmat(os.path.join(FLAGS.stats_dir,'variables.mat'))
        all_avgTrainLoss = mat_contents['all_avgTrainLoss']
        all_avgTrainLoss = all_avgTrainLoss[:,0:sess.run(curr_epoch)]
        all_avgValidationLoss = mat_contents['all_avgValidationLoss']
        all_avgValidationLoss = all_avgValidationLoss[:,0:sess.run(curr_epoch)]
        all_avgTrainAcc = mat_contents['all_avgTrainAcc']
        all_avgTrainAcc = all_avgTrainAcc[:,0:sess.run(curr_epoch)]
        all_avgValidationAcc = mat_contents['all_avgValidationAcc']
        all_avgValidationAcc = all_avgValidationAcc[:,0:sess.run(curr_epoch)]
      else:
        print('No checkpoint file found')
        all_avgTrainLoss = np.empty([1,1],dtype = np.float32 )
        all_avgValidationLoss = np.empty([1,1],dtype = np.float32 )
        all_avgTrainAcc = np.empty([1,1],dtype = np.float32 )
        all_avgValidationAcc = np.empty([1,1],dtype = np.float32 )

      num_examples_per_epoch_for_train = len(glob.glob(os.path.join(FLAGS.train_dir, 'Images', '*' + FLAGS.image_ext)))
      num_examples_per_epoch_for_val = len(glob.glob(os.path.join(FLAGS.val_dir, 'Images', '*' + FLAGS.image_ext)))
      nTrainBatches = int((num_examples_per_epoch_for_train/FLAGS.train_batch_size)+1)
      nValBatches = int((num_examples_per_epoch_for_val/FLAGS.train_batch_size)+1)


      for epoch in xrange(sess.run(curr_epoch), FLAGS.num_epochs + 1):
        avgTrainLoss = 0.0
        avgValLoss = 0.0
        avgTrainAcc  = 0.0
        avgValAcc =  0.0

        # Training loop
        for step in xrange(nTrainBatches):
          start_time = time.time()
          _, __, loss_value_train, acc_value_train, predict_value_train, actual_value_train= \
                  sess.run([train_op, extra_update_ops, loss_train, accuracy_train, predict_train, actual_train],
                  feed_dict = {keep_prob:FLAGS.keep_prob, is_training:True}) 

          duration = time.time() - start_time
          assert not np.isnan(loss_value_train), 'Model diverged with loss = NaN'
          avgTrainLoss += loss_value_train
          avgTrainAcc += acc_value_train
          format_str = ('%s: epoch %d, step %d/ %d, Training Loss = %.2f, Training Accuracy = %.2f (%.2f sec/step)')
          print (format_str % (datetime.now(), epoch, step+1, nTrainBatches, loss_value_train, acc_value_train, float(duration)))

          predict_value_train = np.reshape(predict_value_train,-1)
          actual_value_train = np.reshape(actual_value_train,-1)
          predict_value_train = pd.Series(predict_value_train, name='Predicted')
          actual_value_train = pd.Series(actual_value_train, name='Actual')
          print(pd.crosstab(predict_value_train, actual_value_train,margins=True))

        # Validation loop
        for step in xrange(nValBatches):
          start_time = time.time()
          loss_value_val, acc_value_val, predict_value_val, actual_value_val = \
                        sess.run([loss_val,accuracy_val, predict_val, actual_val],
                          feed_dict = {keep_prob:1.0, is_training:False})

          duration = time.time() - start_time
          assert not np.isnan(loss_value_val), 'Model diverged with loss = NaN'
          avgValLoss += loss_value_val
          avgValAcc += acc_value_val
          format_str = ('%s: epoch %d, step %d/ %d, Validation Loss = %.2f,  Validation Accuracy = %.2f (%.2f sec/step)')
          print (format_str % (datetime.now(), epoch, step+1, nValBatches, loss_value_val, acc_value_val, float(duration)))

          predict_value_val = np.reshape(predict_value_val,-1)
          actual_value_val = np.reshape(actual_value_val,-1)
          predict_value_val = pd.Series(predict_value_val, name='Predicted')
          actual_value_val = pd.Series(actual_value_val, name='Actual')
          print(pd.crosstab(predict_value_val, actual_value_val,margins=True))

        #Average loss on training and validation
        avgTrainLoss_per_epoch = avgTrainLoss/nTrainBatches
        avgTrainAcc_per_epoch = avgTrainAcc/nTrainBatches
        if epoch == 0:
          all_avgTrainLoss[epoch] = avgTrainLoss_per_epoch
          all_avgTrainAcc[epoch] = avgTrainAcc_per_epoch
        else:
          all_avgTrainLoss = np.append(all_avgTrainLoss,avgTrainLoss_per_epoch)
          all_avgTrainAcc = np.append(all_avgTrainAcc,avgTrainAcc_per_epoch)

        avgValidationLoss_per_epoch = avgValLoss/nValBatches
        avgValidationAcc_per_epoch = avgValAcc/nValBatches
        if epoch == 0:
          all_avgValidationLoss[epoch] = avgValidationLoss_per_epoch
          all_avgValidationAcc[epoch] = avgValidationAcc_per_epoch
        else:
          all_avgValidationLoss = np.append(all_avgValidationLoss,avgValidationLoss_per_epoch)
          all_avgValidationAcc = np.append(all_avgValidationAcc,avgValidationAcc_per_epoch)

        #Save the model after each epoch.
        sess.run(update_curr_epoch)
        if FLAGS.discount_weight is True:
          if (epoch+1) % 5 == 0:
            sess.run(update_discount_weight)
        
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)

        sio.savemat(os.path.join(FLAGS.stats_dir,'variables.mat'),
          {'all_avgTrainLoss':all_avgTrainLoss, 'all_avgTrainAcc':all_avgTrainAcc,
           'all_avgValidationLoss':all_avgValidationLoss,'all_avgValidationAcc':all_avgValidationAcc})

        ########################################################################################
        plt.figure(2)
        plt.ion()
        ax = plt.gca()
        plt.cla()
        plt.subplot(121)
        train_line, = plt.plot(np.log(all_avgTrainLoss),'b', label= 'train_line')
        val_line, = plt.plot(np.log(all_avgValidationLoss),'--r', label = 'val_line')
        plt.ylabel('average loss (log)')
        plt.xlabel('epoch')
        plt.legend([train_line,val_line],['Training', 'Validation'],loc = 0)
        plt.draw()

        plt.subplot(122)
        train_line, = plt.plot(all_avgTrainAcc,'b', label= 'train_line')
        val_line, = plt.plot(all_avgValidationAcc,'--r', label = 'val_line')
        plt.ylabel('average accuracy')
        plt.xlabel('epoch')
        plt.legend([train_line,val_line],['Training', 'Validation'],loc = 0)
        plt.draw()

        with PdfPages(os.path.join(FLAGS.stats_dir,'performance.pdf')) as pdf:
          pdf.savefig()

      coord.request_stop()
      coord.join(threads)
      plt.close()
###################################################################
def find_optim_net():
  print('Find an optimal network from validation accuracy')
  mat_contents = sio.loadmat(os.path.join(FLAGS.stats_dir, 'variables.mat'))
  all_avgValidationAcc = mat_contents['all_avgValidationAcc']

  x = np.arange(all_avgValidationAcc.shape[1])
  y = np.reshape(np.log(all_avgValidationAcc), -1)
  f = UnivariateSpline(x, y, s=0.1)
  ysmooth = f(x)

  plt.ion()
  plt.cla()
  plt.plot(x, y, 'o', x, ysmooth, '--')
  plt.ylabel('average validation accuracy (log)')
  plt.xlabel('epoch')
  plt.draw()
  with PdfPages(os.path.join(FLAGS.stats_dir, 'optim_net.pdf')) as pdf:
    pdf.savefig()
  plt.close()

  optim_epoch = np.argmax(ysmooth)
  print('The optimal epoch (based 0) is %d' % optim_epoch)
  checkpointlist = glob.glob(os.path.join(FLAGS.checkpoint_dir, 'model*meta'))
  temp = []
  for filepath in checkpointlist:
    basename = os.path.basename(filepath)
    temp.append(int(float(basename.split('-')[-1].split('.')[0])))
  temp = np.sort(temp)
  optim_model_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-' + str(temp[optim_epoch]))
  return optim_model_path
  ###################################################################
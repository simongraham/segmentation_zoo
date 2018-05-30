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
    discount_weight = tf.Variable(1.0, trainable=False, name = 'curr_epoch')
    update_discount_weight = tf.assign(discount_weight, tf.div(discount_weight, 10))

    # drop out
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # network stats
    # Load network stats
    mat_contents = sio.loadmat(os.path.join(FLAGS.stats_dir,'network_stats.mat'))
    mean_img = np.float32(mat_contents['mean_image'])
    variance_img = np.float32(mat_contents['variance_image'])

    mean_image = tf.Variable(mean_img, trainable=False, name = 'mean_image')
    variance_image = tf.Variable(variance_img, trainable=False, name = 'variance_image')

    # Get images and labels.
    with tf.name_scope('inputs'):
      images_train, labels_train, contours_train = Segmentation_input.inputs_train(mean_image,variance_image)
      images_val, labels_val, contours_val = Segmentation_input.inputs_val(mean_image,variance_image)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope("network") as scope:
      '''
      Passes the data through the deep learning model
      '''
      (softmax_train_gland1, softmax_train_gland2, softmax_train_gland3, softmax_train_fusion_gland, softmax_train_contour1, 
        softmax_train_contour2, softmax_train_contour3, softmax_train_fusion_contour) = Segmentation_networks.dcan(images_train,keep_prob)
      scope.reuse_variables()
      (softmax_val_gland1, softmax_val_gland2, softmax_val_gland3, softmax_val_fusion_gland, softmax_val_contour1, 
        softmax_val_contour2, softmax_val_contour3, softmax_val_fusion_contour) = Segmentation_networks.dcan(images_val,keep_prob)
    
    # Calculate loss.
    loss_train = layers.loss_dcan(softmax_train_gland1, softmax_train_gland2, softmax_train_gland3, softmax_train_fusion_gland, softmax_train_contour1, 
      softmax_train_contour2, softmax_train_contour3, softmax_train_fusion_contour, labels_train, contours_train , discount_weight)
    loss_val = layers.loss_dcan(softmax_val_gland1, softmax_val_gland2, softmax_val_gland3, softmax_val_fusion_gland, softmax_val_contour1, 
      softmax_val_contour2, softmax_val_contour3, softmax_val_fusion_contour, labels_val, contours_val , discount_weight)

    # Accuracy for each class- gland
    with tf.name_scope('training_predictions_gland'):
      predict_train_g = tf.argmax(softmax_train_fusion_gland,3)
      predict_train_g = tf.reshape(predict_train_g,[-1])
    with tf.name_scope('training_labels_gland'):
      actual_train_g = tf.squeeze(tf.cast(labels_train-1,dtype = tf.int64))
      actual_train_g = tf.argmax(actual_train_g, 3)
      actual_train_g = tf.reshape(actual_train_g,[-1])
    with tf.name_scope('training_accuracy_gland'):
      correct_prediction_train_g = tf.equal(predict_train_g, actual_train_g)
      accuracy_train_g = tf.reduce_mean(tf.cast(correct_prediction_train_g, tf.float32))
    with tf.name_scope('validation_predictions_gland'):
      predict_val_g = tf.argmax(softmax_val_fusion_gland,3)
      predict_val_g = tf.reshape(predict_val_g,[-1])
    with tf.name_scope('validation_labels_gland'):
      actual_val_g = tf.squeeze(tf.squeeze(tf.cast(labels_val-1,dtype = tf.int64)))
      actual_val_g = tf.argmax(actual_val_g, 3)
      actual_val_g = tf.reshape(actual_val_g,[-1])
    with tf.name_scope('validation_accuracy_gland'):
      correct_prediction_val_g = tf.equal(predict_val_g,actual_val_g)
      accuracy_val_g = tf.reduce_mean(tf.cast(correct_prediction_val_g, tf.float32))

     # Accuracy for each class - contour
    with tf.name_scope('training_predictions_contour'):
      predict_train_c = tf.argmax(softmax_train_fusion_contour,3)
      predict_train_c = tf.reshape(predict_train_c,[-1])
    with tf.name_scope('training_labels_contour'):
      actual_train_c = tf.squeeze(tf.cast(contours_train-1,dtype = tf.int64))
      actual_train_c = tf.argmax(actual_train_c, 3)
      actual_train_c = tf.reshape(actual_train_c,[-1])
    with tf.name_scope('training_accuracy_contour'):
      correct_prediction_train_c = tf.equal(predict_train_c, actual_train_c)
      accuracy_train_c = tf.reduce_mean(tf.cast(correct_prediction_train_c, tf.float32))
    with tf.name_scope('validation_predictions_contour'):
      predict_val_c = tf.argmax(softmax_val_fusion_contour,3)
      predict_val_c = tf.reshape(predict_val_c,[-1])
    with tf.name_scope('validation_labels_contour'):
      actual_val_c = tf.squeeze(tf.squeeze(tf.cast(contours_val-1,dtype = tf.int64)))
      actual_val_c = tf.argmax(actual_val_c, 3)
      actual_val_c = tf.reshape(actual_val_c,[-1])
    with tf.name_scope('validation_accuracy_contour'):
      correct_prediction_val_c = tf.equal(predict_val_c,actual_val_c)
      accuracy_val_c = tf.reduce_mean(tf.cast(correct_prediction_val_c, tf.float32))

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    num_examples_per_epoch_for_train = len(glob.glob(os.path.join(FLAGS.train_dir, 'Images', '*' + FLAGS.image_ext)))
    learning_rate = configure_learning_rate(num_examples_per_epoch_for_train, global_step)
    optimizer = configure_optimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_train)
    if FLAGS.l2_reg == True:
      vars = tf.trainable_variables()
      avg_cross_entropy_log_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])
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
        all_avgTrainAcc_g = mat_contents['all_avgTrainAcc_g']
        all_avgTrainAcc_g = all_avgTrainAcc_g[:,0:sess.run(curr_epoch)]
        all_avgValidationAcc_g = mat_contents['all_avgValidationAcc_g']
        all_avgValidationAcc_g = all_avgValidationAcc_g[:,0:sess.run(curr_epoch)]
        all_avgTrainAcc_c = mat_contents['all_avgTrainAcc_c']
        all_avgTrainAcc_c = all_avgTrainAcc_c[:,0:sess.run(curr_epoch)]
        all_avgValidationAcc_c = mat_contents['all_avgValidationAcc_c']
        all_avgValidationAcc_c = all_avgValidationAcc_c[:,0:sess.run(curr_epoch)]
      else:
        print('No checkpoint file found')
        all_avgTrainLoss = np.empty([1,1],dtype = np.float32 )
        all_avgValidationLoss = np.empty([1,1],dtype = np.float32 )
        all_avgTrainAcc_g = np.empty([1,1],dtype = np.float32 )
        all_avgValidationAcc_g = np.empty([1,1],dtype = np.float32 )
        all_avgTrainAcc_c = np.empty([1,1],dtype = np.float32 )
        all_avgValidationAcc_c = np.empty([1,1],dtype = np.float32 )

      num_examples_per_epoch_for_train = len(glob.glob(os.path.join(FLAGS.train_dir, 'Images', '*' + FLAGS.image_ext)))
      num_examples_per_epoch_for_val = len(glob.glob(os.path.join(FLAGS.val_dir, 'Images', '*' + FLAGS.image_ext)))
      nTrainBatches = int((num_examples_per_epoch_for_train/FLAGS.train_batch_size)+1)
      nValBatches = int((num_examples_per_epoch_for_val/FLAGS.train_batch_size)+1)


      for epoch in xrange(sess.run(curr_epoch), FLAGS.num_epochs + 1):
        avgTrainLoss = 0.0
        avgValLoss = 0.0
        avgTrainAcc_g  = 0.0
        avgValAcc_g =  0.0
        avgTrainAcc_c  = 0.0
        avgValAcc_c =  0.0

        # Training loop
        for step in xrange(nTrainBatches):
          start_time = time.time()
          _, loss_value_train, acc_value_train_g, predict_value_train_g, actual_value_train_g, acc_value_train_c, predict_value_train_c, actual_value_train_c, it, lt, ct= \
                  sess.run([train_op, loss_train, accuracy_train_g, predict_train_g, actual_train_g, accuracy_train_c, predict_train_c, actual_train_c, images_train, labels_train, contours_train],
                  feed_dict = {keep_prob:FLAGS.keep_prob, is_training:True})

          duration = time.time() - start_time
          assert not np.isnan(loss_value_train), 'Model diverged with loss = NaN'
          avgTrainLoss += loss_value_train
          avgTrainAcc_g += acc_value_train_g
          avgTrainAcc_c += acc_value_train_c
          format_str = ('%s: epoch %d, step %d/ %d, Training Loss = %.2f, Training Accuracy Gland = %.2f, Training Accuracy Contour = %.2f (%.2f sec/step)')
          print (format_str % (datetime.now(), epoch, step+1, nTrainBatches, loss_value_train, acc_value_train_g, acc_value_train_c, float(duration)))

          predict_value_train_g = np.reshape(predict_value_train_g,-1)
          actual_value_train_g = np.reshape(actual_value_train_g,-1)
          predict_value_train_g = pd.Series(predict_value_train_g, name='Predicted')
          actual_value_train_g = pd.Series(actual_value_train_g, name='Actual')
          print(pd.crosstab(predict_value_train_g, actual_value_train_g,margins=True))

        # Validation loop
        for step in xrange(nValBatches):
          start_time = time.time()
          loss_value_val, acc_value_val_g, predict_value_val_g, actual_value_val_g, acc_value_val_c, predict_value_val_c, actual_value_val_c = \
                        sess.run([loss_val,accuracy_val_g, predict_val_g, actual_val_g,accuracy_val_c, predict_val_c, actual_val_c],
                          feed_dict = {keep_prob:1.0, is_training:False})

          duration = time.time() - start_time
          assert not np.isnan(loss_value_val), 'Model diverged with loss = NaN'
          avgValLoss += loss_value_val
          avgValAcc_g += acc_value_val_g
          avgValAcc_c += acc_value_val_c
          format_str = ('%s: epoch %d, step %d/ %d, Validation Loss = %.2f,  Validation Accuracy Gland = %.2f,  Validation Accuracy Contour = %.2f (%.2f sec/step)')
          print (format_str % (datetime.now(), epoch, step+1, nValBatches, loss_value_val, acc_value_val_g, acc_value_val_c, float(duration)))

          predict_value_val_c = np.reshape(predict_value_val_c,-1)
          actual_value_val_c = np.reshape(actual_value_val_c,-1)
          predict_value_val_c = pd.Series(predict_value_val_c, name='Predicted')
          actual_value_val_c = pd.Series(actual_value_val_c, name='Actual')
          print(pd.crosstab(predict_value_val_c, actual_value_val_c,margins=True))

        #Average loss on training and validation
        avgTrainLoss_per_epoch = avgTrainLoss/nTrainBatches
        avgTrainAcc_per_epoch_g = avgTrainAcc_g/nTrainBatches
        avgTrainAcc_per_epoch_c = avgTrainAcc_c/nTrainBatches
        if epoch == 0:
          all_avgTrainLoss[epoch] = avgTrainLoss_per_epoch
          all_avgTrainAcc_g[epoch] = avgTrainAcc_per_epoch_g
          all_avgTrainAcc_c[epoch] = avgTrainAcc_per_epoch_c
        else:
          all_avgTrainLoss = np.append(all_avgTrainLoss,avgTrainLoss_per_epoch)
          all_avgTrainAcc_g = np.append(all_avgTrainAcc_g,avgTrainAcc_per_epoch_g)
          all_avgTrainAcc_c = np.append(all_avgTrainAcc_c,avgTrainAcc_per_epoch_c)

        avgValidationLoss_per_epoch = avgValLoss/nValBatches
        avgValidationAcc_per_epoch_g = avgValAcc_g/nValBatches
        avgValidationAcc_per_epoch_c = avgValAcc_c/nValBatches
        if epoch == 0:
          all_avgValidationLoss[epoch] = avgValidationLoss_per_epoch
          all_avgValidationAcc_g[epoch] = avgValidationAcc_per_epoch_g
          all_avgValidationAcc_c[epoch] = avgValidationAcc_per_epoch_c
        else:
          all_avgValidationLoss = np.append(all_avgValidationLoss,avgValidationLoss_per_epoch)
          all_avgValidationAcc_g = np.append(all_avgValidationAcc_g,avgValidationAcc_per_epoch_g)
          all_avgValidationAcc_c = np.append(all_avgValidationAcc_c,avgValidationAcc_per_epoch_c)

        #Save the model after each epoch.
        sess.run(update_curr_epoch)
        if (epoch+1) % 5 == 0:
          sess.run(update_discount_weight)

        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)

        sio.savemat(os.path.join(FLAGS.stats_dir,'variables.mat'),
          {'all_avgTrainLoss':all_avgTrainLoss, 'all_avgTrainAcc_g':all_avgTrainAcc_g, 'all_avgTrainAcc_c':all_avgTrainAcc_c,
           'all_avgValidationLoss':all_avgValidationLoss,'all_avgValidationAcc_g':all_avgValidationAcc_g,'all_avgValidationAcc_c':all_avgValidationAcc_c})

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
        train_line, = plt.plot(all_avgTrainAcc_g,'b', label= 'train_gland')
        train_line2, = plt.plot(all_avgTrainAcc_c,'--b', label= 'train_contour')
        val_line, = plt.plot(all_avgValidationAcc_g,'r', label = 'val_gland')
        val_line2, = plt.plot(all_avgValidationAcc_c,'--r', label = 'val_contour')
        plt.ylabel('average accuracy')
        plt.xlabel('epoch')
        plt.legend([train_line, train_line2, val_line, val_line2],['Training_gland', 'Training_contour', 'Validation_gland', 'Validation_contour'],loc = 0)
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
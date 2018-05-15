'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

data_root = # Path to the data directory
model_name = 'segnet'
# choose from unet, fcn8, fcn16, fcn32, segnet
loss_function = 'cross_entropy'
# choose from cross_entropy, weighted_cross_entropy
activation = 'relu'
# choose from relu, elu, softmax
exp_id = '1'

######################
# Training data      #
######################

tf.app.flags.DEFINE_string('train_dir', data_root + 'train','Directory where training patches are saved.')
tf.app.flags.DEFINE_string('val_dir', data_root + 'valid', 'Directory where validation patches are saved.')
tf.app.flags.DEFINE_integer('train_image_source_size', 200, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_target_size', 120, 'Train image size')
tf.app.flags.DEFINE_integer('ground_truth_size', 120, 'Ground truth size (will be different for unet)')
tf.app.flags.DEFINE_integer('n_channels', 3, 'Number of channels in input image')
tf.app.flags.DEFINE_integer('n_classes', 2,'number of classes at output')
tf.app.flags.DEFINE_boolean('random_rotation', True, 'Random rotation of 90,180, or 270')
tf.app.flags.DEFINE_boolean('random_flipping', True, 'Random horizontal and vertical flipping')
tf.app.flags.DEFINE_boolean('color_distortion', True, 'Random color distorsion')
tf.app.flags.DEFINE_boolean('zero_centre', True, 'Centre the dataset about the mean')
tf.app.flags.DEFINE_boolean('normalize', True, 'divide dataset by the variance')
tf.app.flags.DEFINE_string('image_ext', '.png', 'file extension')
tf.app.flags.DEFINE_string('test_image_ext', '.tif', 'file extension')

######################
# Network training   #
######################

tf.app.flags.DEFINE_string('model_name', model_name, 'The name of the architecture to train.')
tf.app.flags.DEFINE_string('loss_function', loss_function, 'Loss function to use.')
tf.app.flags.DEFINE_string('activation', activation, 'Activation function to use.')
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join('checkpoints/checkpoints'+'_'+model_name+'_'+exp_id), 'Directory where checkpoint files are saved.')
tf.app.flags.DEFINE_string('log_dir', os.path.join('logs/logs'+'_'+model_name+'_'+exp_id), 'Directory where log files are saved.')
tf.app.flags.DEFINE_string('stats_dir', os.path.join('stats/stats'+'_'+model_name+'_'+exp_id), 'Directory where stats files are saved.')
tf.app.flags.DEFINE_integer('num_epochs', 60, 'Number of epochs to run.')
tf.app.flags.DEFINE_integer('train_batch_size',5 , 'Number of samples in a training batch.') 
tf.app.flags.DEFINE_integer('eval_batch_size', 5, 'Number of samples in an evaluation batch.') 
tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.4, 'fraction of examples pre-read in queue')
tf.app.flags.DEFINE_integer('num_cores',8, 'number of cpu cores')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float('keep_prob', 0.5, 'Fraction of dropped units, when using Dropout.')
tf.app.flags.DEFINE_bool('l2_reg', False, 'Apply l2 regularization on the weights.')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('optimizer', 'adam','The name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('adadelta_rho', 0.95,'The decay rate for adadelta.')
tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1,'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float('adam_beta1', 0.9,'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999,'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,'The learning rate power.')
tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1,'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float('momentum', 0.9,'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string('learning_rate_decay_type','fixed','Specifies how the learning rate is decayed. One of "fixed", "exponential", or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_bool('sync_replicas', False,'Whether or not to synchronize the replicas during training.')
tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1,'The Number of gradients to collect before updating params.')
tf.app.flags.DEFINE_float('moving_average_decay', None,'The decay to use for the moving average.''If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_bool('discount_weight', True, 'Whether to apply discount weights to auxiliary classifier.')

######################
# Network testing    #
######################

tf.app.flags.DEFINE_string('test_dir', data_root + 'test', 'Directory where test images are saved')
tf.app.flags.DEFINE_string('result_dir', os.path.join('Results/Results'+'_'+model_name+'_'+exp_id), 'Directory where test output images are saved')
tf.app.flags.DEFINE_string('target_image_path', os.path.join('target.png'), 'Path for target image for stain normalization')

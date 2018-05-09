'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

import Segmentation_train
import test_classification
import Segmentation_input
import mean_images
import tensorflow as tf
import glob
import os
import sys
import scipy.io as sio
from PIL import Image
import numpy as np 
import scipy.misc
import cv2 as cv


FLAGS = tf.app.flags.FLAGS

##############################################################################
def main(argv):
    # Training 
  if argv[0] == 'train':

    if not os.path.exists(FLAGS.stats_dir):
      os.mkdir(FLAGS.stats_dir)
  
    if not os.path.exists(os.path.join(FLAGS.stats_dir,'mean_calc')):
      os.mkdir(os.path.join(FLAGS.stats_dir,'mean_calc'))


    '''
    Below we calculate the mean and variance of all training and validation images. This
    is so that we can use mean subtraction and normalization. If the target size
    is different to the source size, we must first crop the source images so that we can 
    subsequently use mean subratction and normalization.

    Images created for the calculation will be written to /stats/mean_calc 

    If the images are cropped, the cropped image size is displayed in the folder name
    '''

    if FLAGS.train_image_source_size != FLAGS.train_image_target_size:
      network_stats_file_path = os.path.join(FLAGS.stats_dir,'network_stats.mat')
      if not os.path.isfile(network_stats_file_path):
        if not os.path.exists(os.path.join(FLAGS.stats_dir,'mean_calc','Images_')+str(FLAGS.train_image_target_size)):
          os.mkdir(os.path.join(FLAGS.stats_dir,'mean_calc','Images_')+str(FLAGS.train_image_target_size))
          print('Cropping and writing images for mean and variance calculation...')
          mean_images.crop_and_write()
        list_images = glob.glob(os.path.join(FLAGS.stats_dir,'mean_calc','Images_')+str(FLAGS.train_image_target_size)+'/*'+FLAGS.image_ext)
        print('Calculating mean and variance...')
        mean_image, variance_image = Segmentation_input.calculate_mean_variance_image(list_images)
        sio.savemat(network_stats_file_path,
          {'mean_image': mean_image, 'variance_image': variance_image})

    else:
      network_stats_file_path = os.path.join(FLAGS.stats_dir,'network_stats.mat')
      if not os.path.isfile(network_stats_file_path):
        if not os.path.exists(os.path.join(FLAGS.stats_dir,'mean_calc','Images_')):
          os.mkdir(os.path.join(FLAGS.stats_dir,'mean_calc','Images_'))
          print('Writing images for mean and variance calculation...')
          mean_images.write()
        list_images = glob.glob(os.path.join(FLAGS.stats_dir,'mean_calc','Images_', '*') + FLAGS.image_ext)
        print('Calculating mean and variance...')
        mean_image, variance_image = Segmentation_input.calculate_mean_variance_image(list_images)
        sio.savemat(network_stats_file_path,
          {'mean_image': mean_image, 'variance_image': variance_image})
        
    if not os.path.exists(FLAGS.checkpoint_dir):
      os.mkdir(FLAGS.checkpoint_dir)
        
    if not os.path.exists(FLAGS.log_dir):
      os.mkdir(FLAGS.log_dir)

    Segmentation_train.train()

    # Testing
  elif argv[0] == 'test':
    #model_path = Segmentation_train.find_optim_net()

    model_path = FLAGS.checkpoint_dir + '/model.ckpt-172198'
    path_id = os.path.basename(model_path)
    path_id = path_id.split('-')[1]

    print(model_path)

    if not os.path.exists(os.path.join('Results')):
      os.mkdir(os.path.join('Results'))

    result_dir = FLAGS.result_dir
    images_path = FLAGS.test_dir
      
    images_list = test_classification.get_image_list(images_path)
    filename_list = images_list
    print('Processing test images ...')
    test_classification.testing(model_path, path_id, filename_list)

if __name__ == '__main__':
    main(sys.argv[1:])

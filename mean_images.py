import numpy as np 
import tensorflow as tf
import cv2
import glob
import os 

FLAGS = tf.app.flags.FLAGS

"""
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
"""

######################################################################################
def crop_and_write():
	source_size = FLAGS.train_image_source_size
	target_size = FLAGS.train_image_target_size

	diff = (source_size-target_size)/2
	crop1 = int((source_size-target_size)/2)
	if float(diff).is_integer():
		crop2 = source_size-crop1
	else:
		crop2 = (source_size-crop1)-1

	train_ims = glob.glob(FLAGS.train_dir + '/Images/*')
	num_ims = len(train_ims)

	for i in range(num_ims):
		im = train_ims[i]
		basename = os.path.basename(im)
		basename = basename.split('.')[0]
		im = cv2.imread(im)
		im = im[crop1:crop2,crop1:crop2,:]
		cv2.imwrite(os.path.join(FLAGS.stats_dir,'mean_calc','Images_')+str(target_size)+'/'+basename+'.png', im)

def write():
	source_size = FLAGS.train_image_source_size
	target_size = FLAGS.train_image_target_size

	diff = (source_size-target_size)/2
	crop1 = int((source_size-target_size)/2)
	if float(diff).is_integer():
		crop2 = source_size-crop1
	else:
		crop2 = (source_size-crop1)-1

	train_ims = glob.glob(FLAGS.train_dir + '/Images/*')
	num_ims = len(train_ims)

	for i in range(num_ims):
		im = train_ims[i]
		basename = os.path.basename(im)
		basename = basename.split('.')[0]
		im = cv2.imread(im)
		im = im[crop1:crop2,crop1:crop2,:]
		cv2.imwrite(os.path.join(FLAGS.stats_dir,'mean_calc','Images/')+basename+'.png', im)
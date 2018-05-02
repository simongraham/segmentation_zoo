'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

import numpy as np
import os
import sys
import glob
import time
import cv2 as cv
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import tensorflow as tf
import scipy.misc
import math

import stain_norm
import Segmentation_networks
import Segmentation_input
import mirror

FLAGS = tf.app.flags.FLAGS

[patch_w,patch_h] = [FLAGS.train_image_target_size,FLAGS.train_image_target_size]
[output_w, output_h] = [FLAGS.ground_truth_size,FLAGS.ground_truth_size]
[stride_w, stride_h] = [117,117]
padding = int((patch_w - output_w)/2)

#-------------------------------------------------------------------------------------------------------
def get_image_list(images_path):
	images_list = glob.glob(images_path + '/*')
	tmp_images_list = []

	for image_name in images_list:
		basename = os.path.basename(image_name)
		basename = basename.split('.')[0]

		tmp_images_list.append(image_name)

	images_list = tmp_images_list

	if not images_list:
		print("no data in %s or %s" % (images_path))
		raise ValueError('terminate!')

	return images_list

#----------------------------------------------------------------------------------------------------
def extract_patches(filename, patch_size , stride):
	
	image = cv.imread(filename) 
	original_shape = image.shape
	image = mirror.mirror(image,padding,output_w)
	# switch bgr to rgb
	image = np.dstack((image[:,:,2],image[:,:,1],image[:,:,0]))

	target = FLAGS.target_image_path
	target = cv.imread(target)
	b, g, r = cv.split(target)  # get b,g,r
	target = cv.merge([r, g, b])

	# pad array
	# centreIndexRow = int((size_input_patch[0] - size_input_patch[0])/2)
	# centreIndexCol = int((size_input_patch[1] - size_input_patch[1])/2)
	centreIndexRow = 0
	centreIndexCol = 0
	padrow = centreIndexRow + 2*stride[0]
	padcol = centreIndexCol + 2*stride[1]

	#image = np.lib.pad(image, ((padrow, padrow), (padcol,padcol), (0,0)), 'symmetric')

	rowRange = range(0, (image.shape[0]-patch_h)+1, stride[0])
	colRange = range(0, (image.shape[1]-patch_w)+1, stride[1])
	nPatches = len(rowRange)*len(colRange)

	patches = np.empty([nPatches,patch_h,patch_w,image.shape[2]], 
	        dtype = image.dtype)

	for index1, row in enumerate(rowRange):
		for index2, col in enumerate(colRange): 

			patch1 = image[row:row+FLAGS.train_image_target_size,col:col+FLAGS.train_image_target_size,:]
			patch1 = stain_norm.norm_rn(patch1, target)
			patches[(index1*len(colRange))+index2,:,:] = patch1

	return patches, image.shape, original_shape

#-----------------------------------------------------------------------------------------------------
def generate_patches_prob_map(pred, patch_size, patches, batch_index, filename):
	# initialize batch of patches  
	batch  =  np.zeros([pred.shape[0], pred.shape[1], pred.shape[2], 1])

	for index in range(patches.shape[0]):
		prob =  pred[index]
		roi_prob = prob[:,:,0]
		non_roi_prob = prob[:,:,1]
		prob_map = np.zeros([prob.shape[1], prob.shape[0], 1])
		pat =  patches[index, :,:,:]
		
		if np.mean(pat) > 240:
			roi_prob = np.zeros([output_h,output_w,1])

		roi_prob = np.reshape(roi_prob, [output_h,output_w,1])
		prob_map =  prob_map + roi_prob

		batch[index, :,:,:] =  batch[index, :,:,:] +  prob_map

	return batch
#-----------------------------------------------------------------------------------------------------
def MergePatches_test(patches, patch_size ,stride, image_size, original_shape):

	output_shape_x = int(np.ceil(float(original_shape[1])/float(output_w)) * output_w)
	output_shape_y = int(np.ceil(float(original_shape[0])/float(output_w)) * output_w)

	patches = np.float32(patches)
	patch_size = [patch_w,patch_h] 
	# rowRange = range(0, (image_size[0]-output_h)+1, stride[1])
	# colRange = range(0, (image_size[1]-output_w)+1, stride[0])
	rowRange = range(0, (output_shape_y-output_h)+1, stride[1])
	colRange = range(0, (output_shape_x-output_w)+1, stride[0])
	centreIndexRow = 0 #int((sizeInputPatch[0] - sizeOutputPatch[0])/2)
	centreIndexCol = 0 #int((sizeInputPatch[1] - sizeOutputPatch[1])/2)

	image = np.zeros([image_size[0]-(padding*2),image_size[1]-(padding*2),patches.shape[3]], dtype = np.float32)
	count = np.zeros([image_size[0]-(padding*2),image_size[1]-(padding*2),patches.shape[3]], dtype = np.float32)

	for index1, row in enumerate(rowRange):
		for index2, col in enumerate(colRange):

			image[row: row + patches.shape[1],col : col + patches.shape[2],:] += patches[(index1*len(colRange))+index2,:,:,:]
			count[row : row + patches.shape[1],col: col + patches.shape[2],:] += 1.0

	image = image/count

	divide_x = float(original_shape[1]) / float(output_w)
	divide_y = float(original_shape[0]) / float(output_h)
	extra_padding_x =int((math.ceil(divide_x)* output_w) - original_shape[1])
	extra_padding_y =int((math.ceil(divide_y)* output_h) - original_shape[0])

	# crop image
	image = image[:image.shape[0]-extra_padding_y,:image.shape[1]-extra_padding_x,:]

	return image
#-----------------------------------------------------------------------------------------------------
def batch_processing(filename, sess, logits_test, images_test, keep_prob, is_training, mean_image, variance_image):
	# Read image and extract patches
	patch_size = [patch_w, patch_h]
	output_size = [output_w, output_h]
	stride = [stride_w, stride_h]
	patches, image_size, original_shape =  extract_patches(filename,  patch_size , stride)

	# Construct batch indexes
	batch_index = list(range(0,patches.shape[0],FLAGS.eval_batch_size))
	if patches.shape[0] not in batch_index:
		batch_index.append(patches.shape[0])

	# initialize all patches, 1 for one channel output	
	all_patches =  np.zeros([patches.shape[0], output_size[0], output_size[1], 1], dtype =  np.float32)

	for ipatch in range(len(batch_index)-1):
		start_time = time.time()
		start_idx = batch_index[ipatch]
		end_idx = batch_index[ipatch+1]
		temp = Segmentation_input.inputs_test(patches[start_idx:end_idx,:,:,:], mean_image, variance_image)

		if temp.shape[0] < FLAGS.eval_batch_size:
			rep = np.tile(temp[-1,:,:,:], [FLAGS.eval_batch_size - temp.shape[0], 1 ,1, 1])
			temp = np.vstack([temp,rep])

		pred = sess.run(logits_test, feed_dict={images_test:temp,keep_prob:1.0, is_training:False})
	
		batch_prob_maps =  generate_patches_prob_map(pred, patch_size, patches[start_idx:end_idx,:,:,:], ipatch, filename) # for each batch
		all_patches[start_idx:end_idx,:,:,:] = batch_prob_maps[range(end_idx-start_idx),:,:,:]

		duration = time.time() - start_time
		print('Processing step %d/%d (%.2f sec/step)'%(ipatch+1,len(batch_index)-1,duration))


	results =  MergePatches_test(all_patches, output_size, stride, image_size, original_shape)	
	return results

#-----------------------------------------------------------------------------------------------------
def testing(model_path, path_id, filename_list):
	mat_contents = sio.loadmat(os.path.join(FLAGS.stats_dir, 'network_stats.mat'))
	mean_image = np.float32(mat_contents['mean_image'])
	variance_image = np.float32(mat_contents['variance_image'])

	if not os.path.exists(FLAGS.result_dir + '_' + path_id):    #+str(FLAGS.train_image_target_size)):
		os.mkdir(FLAGS.result_dir + '_' + path_id)

	with tf.Graph().as_default():
		#with tf.device('/cpu:0'):
		keep_prob = tf.placeholder(tf.float32)
		is_training = tf.placeholder(tf.bool, [], name='is_training')
		# Place holder for patches
		shape  = np.hstack((FLAGS.eval_batch_size,FLAGS.train_image_target_size,FLAGS.train_image_target_size,FLAGS.n_channels))
		images_test = tf.placeholder(tf.float32, shape = shape)

		# Network
		with tf.variable_scope("network") as scope:
			if FLAGS.model_name == 'unet':
				logits_test = Segmentation_networks.unet(images_test,keep_prob)
			elif FLAGS.model_name == 'fcn8':
				logits_test = Segmentation_networks.fcn8(images_test,keep_prob)
			elif FLAGS.model_name == 'fcn16':
				logits_test = Segmentation_networks.fcn16(images_test,keep_prob)
			elif FLAGS.model_name == 'fcn32':
				logits_test = Segmentation_networks.fcn32(images_test,keep_prob)
			elif FLAGS.model_name == 'fast_fcn':
				logits_test = Segmentation_networks.fast_fcn(images_test,keep_prob)
			elif FLAGS.model_name == 'segnet':
				logits_test = Segmentation_networks.segnet(images_test, keep_prob, is_training)
			elif FLAGS.model_name == 'deeplab_v2':
				logits_test = Segmentation_networks.deeplab_v2(images_test,keep_prob)
			elif FLAGS.model_name == 'proposed':
				logits_test, _, __ = Segmentation_networks.proposed(images_test, keep_prob, is_training)
			elif FLAGS.model_name == 'proposed_sep':
				logits_test, _ = Segmentation_networks.proposed_sep(images_test, keep_prob, is_training)
			else:
				raise ValueError('Network architecture not recognised')

		# Saver and initialization
		saver =  tf.train.Saver()
		init = tf.global_variables_initializer()

		with tf.Session() as sess:
		# Initialise and load variables
			sess.run(init)
			saver.restore(sess, model_path)

			for iImage, file in enumerate(filename_list):
				print(file)
				basename = os.path.basename(file)
				basename = basename.split('.')[0]

				if not os.path.exists(os.path.join(FLAGS.result_dir, basename + '.mat')):
					print('Processing image %d/%d' % (iImage + 1, len(filename_list)))
					result = batch_processing(file,sess,logits_test,images_test,keep_prob, is_training, mean_image, variance_image)
					result =  np.squeeze(result)/ (np.amax(np.squeeze(result)))
					plt.imsave(FLAGS.result_dir + '_' + path_id + '/' + basename+'.png', result, cmap=cm.jet)
					sio.savemat(FLAGS.result_dir + '_' + path_id + '/' + basename+'.mat', {'result':result})


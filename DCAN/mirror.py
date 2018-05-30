import numpy as np 
import random
import matplotlib.cm as cm
import cv2
import math

'''
Author: Simon Graham
Tissue Image Analytics Lab
Department of Computer Science, 
University of Warwick, UK.
'''

######################################################################################
def mirror(image,padding, output_size):

	shape_x = image.shape[1]
	shape_y = image.shape[0]
	divide_x = float(shape_x) / float(output_size)
	extra_padding_x =int((math.ceil(divide_x)*output_size) - shape_x)
	divide_y = float(shape_y) / float(output_size)
	extra_padding_y =int((math.ceil(divide_y)*output_size) - shape_y)
	padding2 = int(padding + extra_padding_x)
	padding3 = int(padding + extra_padding_y)
	padding = int(padding)
	im1 = image[:padding,:,:]
	im2 = image[shape_y-padding3:,:,:]
	im3 = image[:,:padding,:]
	im4 = image[:,shape_x-padding2:,:]
	im5 = image[:padding,:padding,:]
	im6 = image[shape_y-padding3:,shape_x-padding2:,:]
	im7 = image[:padding,shape_x-padding2:,:]
	im8 = image[shape_y-padding3:,:padding,:]

	im1 = cv2.flip(im1,0)
	im2 = cv2.flip(im2,0)
	im3 = cv2.flip(im3,1)
	im4 = cv2.flip(im4,1)
	im5 = cv2.flip(im5,0)
	im5 = cv2.flip(im5,1)
	im6 = cv2.flip(im6,0)
	im6 = cv2.flip(im6,1)
	im7 = cv2.flip(im7,0)
	im7 = cv2.flip(im7,1)
	im8 = cv2.flip(im8,0)
	im8 = cv2.flip(im8,1)
	output = np.zeros([shape_y+(padding*2)+extra_padding_y,shape_x+(padding*2)+extra_padding_x,3])
	shape2_x = output.shape[1]
	shape2_y = output.shape[0]
	output[padding:shape2_y-padding3,padding:shape2_x-padding2,:] = image
	output[:padding,padding:shape2_x-padding2,:] = im1
	output[shape2_y-padding3:,padding:shape2_x-padding2,:] = im2
	output[padding:shape2_y-padding3,shape2_x-padding2:,:] = im4
	output[padding:shape2_y-padding3,:padding,:] = im3
	output[:padding,:padding,:] = im5
	output[shape2_y-padding3:,shape2_x-padding2:,:] = im6
	output[:padding,shape2_x-padding2:,:] = im7
	output[shape2_y-padding3:,:padding,:] = im8

	return output

######################################################################################
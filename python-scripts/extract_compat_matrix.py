# -*- coding: utf-8 -*-
"""
Small script to extract the compatibility function (matrix) of a trained model
"""

caffe_root = '../caffe/'
import sys, getopt
sys.path.insert(0, caffe_root + 'python')

#from PIL import Image as PILImage
import caffe
import matplotlib.pyplot as plt
import numpy as np

MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
PRETRAIN_FILE = 'TVG_CRFRNN_COCO_VOC.caffemodel'


# The output parameters of the # fully displayed
# if not this one, because too many parameters, the middle with an ellipsis"......" In the form of
np.set_printoptions(threshold='nan')

# deploy file
#MODEL_FILE ='caffe_deploy.prototxt'
# pre training Caffe model well
#PRETRAIN_FILE ='caffe_iter_10000.caffemodel'

# save the parameter file
params_txt ='params.txt'
pf = open (params_txt,'w')

# let Caffe read network parameters in test mode
net = caffe.Net (MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

# traversal of each layer
for param_name in net.params.keys ():

    if param_name == 'inference1':
	# weight parameter
	weight = net.params[param_name][0].data
	# offset parameter
	bias = net.params[param_name][1].data

	#The # layer corresponds to the name "top" in the prototxt file
	pf.write (param_name)
	pf.write ('\n')

        # write weight parameters
	pf.write (param_name +'_weight:\n')
	# weight parameter is a multidimensional array, in order to facilitate the output, to separate the array
	weight.shape = (-1,1)

	for w in weight:
	    pf.write ('%f,' % w)

	# write offset parameter
	pf.write ('\n'+ param_name +'_bias:\n')
	# bias is a multidimensional array, in order to facilitate the output, to separate the array
	bias.shape = (-1,1)
	for b in bias:
	    pf.write ('%f,' % b)

	pf.write ('\n\n')

pf.close

# -*- coding: utf-8 -*-
"""
Small script to extract the compatibility function (matrix) of a trained model
to CSV file
"""

caffe_root = '../caffe/'
import sys, getopt
sys.path.insert(0, caffe_root + 'python')

#from PIL import Image as PILImage
import caffe
import matplotlib.pyplot as plt
import numpy as np

MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
#PRETRAIN_FILE = 'TVG_CRFRNN_COCO_VOC.caffemodel'
PRETRAIN_FILE = 'snapshots/crfrnn_iter_28000.caffemodel'

# The output parameters of the # fully displayed
# if not this one, because too many parameters, the middle with an ellipsis"......" In the form of
np.set_printoptions(threshold='nan')


# save the parameter file
params_txt ='compat.csv'
pf = open (params_txt,'w')

# let Caffe read network parameters in test mode
net = caffe.Net (MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

# traversal of each layer
for param_name in net.params.keys ():

    if param_name == 'inference1-ft':

	#write the column header
	pf.write('background,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor')
	pf.write ('\n')

	#get the blob[2], where the compatibility matrix is stored
	idx_data = net.params[param_name][2].data
	#print '{0}'.format(idx_data)
	for item in idx_data:
	    for second in item:
		for third in second:
		    first=True
		    for fourth in third:
			if not first:
			    pf.write(',%f' % fourth)
			else:
			    pf.write('%f' % fourth)
			    first=False
		    
		    pf.write('\n')
	    
#	pf.write ('\n\n')

pf.close

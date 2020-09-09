#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
from mss import mss

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

import time




def args_str2bool(arg_value):
	"""

	:param arg_value:
	:return:
	"""
	if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
		return True

	elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
	"""

	:param input_arr:
	:return:
	"""
	min_val = np.min(input_arr)
	max_val = np.max(input_arr)

	output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

	return output_arr




weights_path = r"D:\AI and ML\Github\AI-Asseto-Corsa\src\lanenet\model\tusimple_lanenet.ckpt"

input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

net = lanenet.LaneNet(phase='test')
binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

postprocessor = lanenet_postprocess.LaneNetPostProcessor()

# Set sess configuration
sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
sess_config.gpu_options.allocator_type = 'BFC'

sess = tf.Session(config=sess_config)

# define moving average version of the learned variables for eval
with tf.variable_scope(name_or_scope='moving_avg'):
	variable_averages = tf.train.ExponentialMovingAverage(
		CFG.SOLVER.MOVING_AVE_DECAY)
	variables_to_restore = variable_averages.variables_to_restore()

# define saver
saver = tf.train.Saver(variables_to_restore)


height, width = 800, 1200

with sess.as_default():
	saver.restore(sess=sess, save_path=weights_path)

	bounding_box = {'top': int(1440/2) - int(height/2), 'left': int(2560/2) - int(width/2), 'width': width, 'height': height}

	sct = mss()



	while True:
		frame = np.array(sct.grab(bounding_box))
		image = cv2.resize(frame[:,:,:3], (512, 256), interpolation=cv2.INTER_LINEAR)
		image = image / 127.5 - 1.0
		#frame = frame.resize(1, 256, 512, 3)
		#cv2.imshow("Frame", frame[:,:,:3])

		# t_start = time.time()
		# loop_times = 500
		# for i in range(loop_times):
		# 	binary_seg_image, instance_seg_image = sess.run(
		# 		[binary_seg_ret, instance_seg_ret],
		# 		feed_dict={input_tensor: [image]}
		# 	)
		# t_cost = time.time() - t_start
		# t_cost /= loop_times
		# LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

		binary_seg_image, instance_seg_image = sess.run(
				[binary_seg_ret, instance_seg_ret],
				feed_dict={input_tensor: [image]}
			)

		postprocess_result = postprocessor.postprocess(
			binary_seg_result=binary_seg_image[0],
			instance_seg_result=instance_seg_image[0],
			source_image=frame
		)
		mask_image = postprocess_result['mask_image']


		# my_preds = my_preds.flatten()
		# my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])
		# my_preds = my_preds.reshape((height, width))*128/255.0
		
		#my_preds = binary_seg_image[0] * 255
		#my_preds = np.dstack([my_preds, my_preds, my_preds])
		# print (np.unique(my_preds))

		# for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
		# 	instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
		# embedding_image = np.array(instance_seg_image[0], np.uint8)


		cv2.imwrite("predictions/Pred"+str(time.time())+".jpeg", mask_image[:, :, (2, 1, 0)] )#binary_seg_image[0] * 255) ## embedding_image)#
		cv2.imshow("Pred", mask_image[:, :, (2, 1, 0)] )# binary_seg_image[0] * 150.0/255) #embedding_image)#, binary_seg_image[0] * 150.0/255)
		# cv2.waitKey(1)

		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			cv2.destroyAllWindows()
			break

		

	# for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
	#     instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
	# embedding_image = np.array(instance_seg_image[0], np.uint8)

	# plt.figure('mask_image')
	# plt.imshow(mask_image[:, :, (2, 1, 0)])
	# plt.figure('src_image')
	# plt.imshow(image_vis[:, :, (2, 1, 0)])
	# plt.figure('instance_image')
	# plt.imshow(embedding_image[:, :, (2, 1, 0)])
	# plt.figure('binary_image')
	# plt.imshow(binary_seg_image[0] * 255, cmap='gray')
	# plt.show()

sess.close()



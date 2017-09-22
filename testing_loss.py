#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import tensorflow as tf

import os
import sys
import glob
import argparse

import math
import numpy as np

from nets import inception_utils
from nets import inception_v4
from preprocessing import inception_preprocessing

import tensorflow.contrib.slim as slim


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Assess the testing loss after some training the neural network for some steps')
	parser.add_argument('model_path', help='Path to .meta file containing the graph')
	parser.add_argument('pos_path', help='Path to positive validation samples')
	parser.add_argument('neg_path', help='Path to negative validation samples')
	args = parser.parse_args()
	
	testing_loss = 0
	testing_loss2 = 0
	testing_loss3 = 0
	positive_occurence = len(os.listdir(args.pos_path)) / (len(os.listdir(args.pos_path)) + len(os.listdir(args.neg_path)))
	negative_occurence = len(os.listdir(args.neg_path)) / (len(os.listdir(args.pos_path)) + len(os.listdir(args.neg_path)))
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7) #do not set gpu memory usage to 100% otherwise the program crashes
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		arg_scope = inception_utils.inception_arg_scope()
		im_size = 299

		inputs = tf.placeholder(tf.float32, (None, im_size, im_size, 3))
		#inputs_processed = inception_preprocessing.preprocess_image(tf.squeeze(inputs), im_size, im_size,3)
		#inputs_processed = tf.expand_dims(inputs_processed, 0)
		
		with slim.arg_scope(arg_scope):
			logits, end_points = inception_v4.inception_v4(inputs)
			
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint('../backup_meta/'))
		
		
		lab= np.array(range(0, 1001)).reshape(1, 1001)
		loss_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=lab)
		probabilities_tensor = tf.nn.softmax(logits=logits)
		
		
		tf.get_default_graph().finalize() # checks that we do not add nodes to the graph after this line				
		print("processing positive images")
		image_list = os.listdir(args.pos_path)
		i = 0
		i_init=0
		while i < len(image_list):
			feed_images = []
			i_init = i
			while(i < i_init+32):
				image = cv2.imread(args.pos_path + image_list[i])
				image = cv2.resize(image, (im_size, im_size))
				feed_images.append(image)
				i = i+1
				if i == len(image_list):
					break
			
			logit_values = sess.run(logits, feed_dict={inputs:np.array(feed_images)})
			for j in range(32):			
				if len(logit_values) < 32:
					break
				#get the normalized probabilities by applying the softmax operation (should be done by the network itself ...)
				fed_logit = np.array(logit_values[j]).reshape(1, 1001)
				probabilities = sess.run(probabilities_tensor, feed_dict={logits:fed_logit})
				#computes the cross_entropy in two different way (tensorflow way, and with the binary cross entropy formula : - 1 /N * sum(y*ln(a) + (1-y)*ln(1-a))
				# with N : the number of samples in the test set, y : the class label, a : the probability given in the targeted class returned by the network
				loss = sess.run(loss_tensor, feed_dict={logits:fed_logit})
				testing_loss += loss[0]
				testing_loss2 += math.log(probabilities[0][1]) #log stands for natural logarithm, log10 is base 10 logarithm
				testing_loss3 += math.log2(probabilities[0][1])
				
			print("images processed : ", i)
		
		
		print("processing negative images")
		image_list = os.listdir(args.neg_path)
		i=0
		i_init=0
		while i < len(image_list):
			feed_images = []
			i_init = i
			while(i < i_init+32):
				image = cv2.imread(args.neg_path + image_list[i])
				image = cv2.resize(image, (im_size, im_size))
				feed_images.append(image)
				i = i + 1 
				if i == len(image_list):
					break
				
			logit_values = sess.run(logits, feed_dict={inputs:np.array(feed_images)})
			for j in range(32):		
				if len(logit_values) < 32:
					break
				fed_logit = np.array(logit_values[j]).reshape(1, 1001)
				probabilities = sess.run(probabilities_tensor, feed_dict={logits:fed_logit})
				loss = sess.run(loss_tensor, feed_dict={logits:fed_logit})
				testing_loss += loss[0]
				testing_loss2 += math.log(1-probabilities[0][0])
				
			print("images processed : ", i)	

		testing_loss =  (-1 / float((len(os.listdir(args.pos_path)) + len(os.listdir(args.neg_path))))) * testing_loss2
		testing_loss2 = (- 1 / float((len(os.listdir(args.pos_path)) + len(os.listdir(args.neg_path))))) * testing_loss2
		testing_loss3 = (- 1 / float((len(os.listdir(args.pos_path)) + len(os.listdir(args.neg_path))))) * testing_loss3
		
		print("testing_loss = ", testing_loss)
		print("testing_loss2 = ", testing_loss2)
		print("testing_loss3 = ", testing_loss3)
			
		
			
		
		
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import os, copy, random
import numpy as np

class feed_data(object):
	def __init__(self, dataset, pos_balance=0.5, shuffle=True, batch_size=32, samples=-1, im_size=224):
		self.dataset = dataset
		self.pos_balance = pos_balance
		self.samples = samples
		self.im_size = im_size
		self.data = self.build_image_catalogue()
		if shuffle:
			random.shuffle(self.data)
		self.batch_index = 0
		self.batch_size = batch_size
		self.length = len(self.data)

	def build_image_catalogue(self):
		# builds a set that points to all the files in this dataset
		if self.dataset == "train":
			folders = [sign+"A_d800mm_R"+str(index) for index in range(1,4+1) for sign in ["pos/", "neg/"]]
			folders += [sign+"B_No_d800mm_R"+str(index) for index in range(1,4+1) for sign in ["pos/", "neg/"]]
		elif self.dataset == "test":
			folders = [sign+"A_d800mm_R"+str(index) for index in range(5,8+1) for sign in ["pos/", "neg/"]]
			folders += [sign+"B_No_d800mm_R"+str(index) for index in range(5,7+1) for sign in ["pos/", "neg/"]]
		else:
			print("Dataset name error. Choose between 'train' or 'test'")

		# first get all images paths and labels, and shuffle them
		images_data = []
		pos, neg = 0, 0
		for folder in folders:
			for image in os.listdir(folder):
				if image.endswith(".png"):
					if "neg" in folder:
						neg += 1
						label = [1,0]
					else:
						pos += 1
						label = [0,1]
					# add if image is positive or negative balance not reached
					if label == [0,1] or pos < (neg+pos)*self.pos_balance:
						images_data.append([os.path.join(folder, image), label])
		random.shuffle(images_data)

		# update samples parameter
		if self.samples < 0:
			self.samples = len(images_data)

		# now keep only samples
		images_data = images_data[:self.samples]

		return images_data

	def read_images(self, images_data):
		batch = []
		for image_path, label in images_data:
			image = Image.open(image_path)
			image = image.resize((self.im_size, self.im_size), Image.ANTIALIAS)
			batch.append([np.array(image)/255., label])
		return batch

	def next_batch(self):
		to_process = []
		X, y = [], []
		start = self.batch_index
		if self.batch_index + self.batch_size > self.length:
			# send what was left
			to_process = copy.deepcopy(self.data[start:])
			# reshuffle
			random.shuffle(self.data)
			# update index
			self.batch_index = self.batch_size - (self.length - start)
			# add missing to complete batch size
			to_process += copy.deepcopy(self.data[:self.batch_index])
			
		else:
			end = self.batch_index + self.batch_size
			self.batch_index = end
			to_process = copy.deepcopy(self.data[start:end])

		processed = self.read_images(to_process)

		for pair in processed:
			X.append(pair[0])
			y.append(pair[1])

		X = np.asarray(X)
		y = np.asarray(y)

		return (X, y)

	def generate(self):
		while True:
			yield self.next_batch()

if __name__ == "__main__":
	feed = feed_data('train', samples=2)
	print(feed.next_batch()[0].shape)
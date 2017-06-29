#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import glob
import numpy as np
import os

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100

#HOG descriptor size for 56x56 window = 1296
x = tf.placeholder(tf.float32, shape=[None, 1296], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')


#read_samples : reads the dataset and shapes the descriptors and labels correctly to be fed to the NN
#@param pos_path : path to positive samples
#@param neg_path : path to negative samples
def read_samples(pos_path, neg_path):
    images = []
    labels = []
    test = []
    test_labels = [] 

    images_pos = []
    images_neg = []
                                                                                                                                      
    sub_dir = os.listdir(pos_path)
    #keeps 2 directories for test set
    nb = len(sub_dir)-3
    print "Positives"
    #creates train and test subsets 
    for sd in sub_dir:
        print sd
        files = glob.glob(pos_path + sd + "/*.png")
        for f in files:
            if nb<0:
                test_img = cv2.imread(f, cv2.IMREAD_COLOR)
                test_img = cv2.resize(test_img, (56,56))
                hog = cv2.HOGDescriptor((56,56), (16,16), (8,8), (8,8), 9)
                descriptor = hog.compute(test_img)
                descriptor = np.array(descriptor).reshape(1, 1296)
                test.append(descriptor)
                test_labels.append(np.array([1, 0]).reshape(1, 2))                
            else:
                image = cv2.imread(f, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (56,56))
                hog = cv2.HOGDescriptor((56,56), (16,16), (8,8), (8,8), 9)
                descriptor = hog.compute(image)
                descriptor = np.array(descriptor).reshape(1, 1296)
                images_pos.append(descriptor)
                labels.append(np.array([1, 0]).reshape(1, 2))
        nb = nb-1

    sub_dir = os.listdir(neg_path)
    nb = len(sub_dir)-3
    neg_images = []
    print "\nNegatives"
    for sd in sub_dir:
        print sd
        files = glob.glob(neg_path + sd + "/*.png")
        for f in files:
            if nb<0:
                test_img = cv2.imread(f, cv2.IMREAD_COLOR)
                test_img = cv2.resize(test_img, (56,56))
                hog = cv2.HOGDescriptor((56,56), (16,16), (8,8), (8,8), 9)
                descriptor = hog.compute(test_img)
                descriptor = np.array(descriptor).reshape(1, 1296) 
                test.append(descriptor)
                test_labels.append(np.array([0, 1]).reshape(1, 2))              
            else:
                image = cv2.imread(f, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (56,56))
                hog = cv2.HOGDescriptor((56,56), (16,16), (8,8), (8,8), 9)
                descriptor = hog.compute(image)
                descriptor = np.array(descriptor).reshape(1, 1296)
                images_neg.append(descriptor)
                labels.append(np.array([0, 1]).reshape(1, 2))
        nb = nb-1


    return images_pos, images_neg, labels, test, test_labels

images = []
labels = []
test = []
test_labels = []
images_pos = [] 
images_neg = [] 


images_pos, images_neg, labels, test, test_labels = read_samples("/home/mathieu/STAGE/underground_dataset/pos/train/", "/home/mathieu/STAGE/underground_dataset/neg/train/")      
print "images size = ", len(images)


#reduce size of training samples (decrease training time, should also decrease performances)
reduc_pos = images_pos[0:3000]
reduc_neg = images_neg[-3000:]

images = []
labels = []
#mix positive and negative samples in order not to learn all positives first 
for idx, im in enumerate(reduc_pos):
    images.append(im)
    labels.append(np.array([1, 0]).reshape(1, 2))
    images.append(reduc_neg[idx])
    labels.append(np.array([0, 1]).reshape(1, 2))

print labels


print "labels size = ", len(labels)
print "train subset size = ", len(images)
print "test subset size = ", len(test)

#creates neural network structure 
hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([1296, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')

#neural_network_model : defines operations used by each layer
#@param data : single data or list of data to be processed
#return : score for each class
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weight']), output_layer['bias'], name="output_op")

    return output

#train_neural_network : train the neural network until the error is satisfactory
#@param x : placeholder defined earlier
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y), name="cost" )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

	epoch_loss = 1000000000
        epoch = 0
        while epoch_loss > 100.0:
            epoch_loss = 0
            i = 0
            while i < len(images):
                start = i
                end = i + batch_size
                print "batch ", start, ":", end
                batch_x = np.array(images[start:end])
                batch_y = np.array(labels[start:end])
                
                #should feed batch directly instead of single tensor but requires a powerful computer
                for idx, tensor in enumerate(batch_x):                        
                    _, c = sess.run([optimizer, cost], feed_dict={x:tensor, y: batch_y[idx]})
                    epoch_loss += c
                i+=batch_size
                
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            epoch += 1

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        true_positives = 0.
        true_negatives = 0.
        false_positives = 0.
        false_negatives = 0.
        positive_samples = 0
        negative_samples = 0
        for idx, tensor in enumerate(test):
            res = accuracy.eval({x:tensor, y:test_labels[idx]})
            if test_labels[idx][0][0]==1 and res==1.0:
                true_positives += 1.0
                positive_samples += 1
            else:
                if test_labels[idx][0][0]==1 and res==0.0:
                    false_negatives += 1.0
                    positive_samples += 1
                else:
                    if test_labels[idx][0][1]==1 and res==0.0:
                        false_positives += 1.0
                        negative_samples += 1
                    else:
                        true_negatives += 1.0
                        negative_samples += 1

        print "positive samples = ", positive_samples
        print "negative samples = ", negative_samples
        print "true_positives = ", true_positives
        print "true_negatives = ", true_negatives
        print "false_positives = ", false_positives
        print "false_negatives = ", false_negatives
        precision = true_positives / float(true_positives + false_positives)
        recall = true_positives / float(true_positives + false_negatives)        

        print "precision = " + str(precision)
        print "recall = " + str(recall)

        #saves the graph in the current directory
        saver = tf.train.Saver()
        saver.save(sess, "/home/mathieu/STAGE/underground_dataset/results/models/NN_HOG_6000/graph")

train_neural_network(x)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import tensorflow as tf

import os
import sys
import glob
import argparse

import numpy as np

#TF-slim
#from nets import inception_utils
#from nets import inception_v4
#from preprocessing import inception_preprocessing
#import tensorflow.contrib.slim as slim

#Roberto's work + scikit-learn + keras
import data_feed
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import load_model
from tqdm import tqdm

import csv


viz = False			# flag to display debug / visualization images
FRAME_DIFF = True	# flag to use frame difference background subtraction
MOG = False 			# flag to use Mixture of Gaussian background modelling
DET_RES = True		# flag to output raw detection results
NMS_RES = True 		# flag to output raw detection results after NMS
TRACK_RES = True	# flag to output raw tracking results


#function used to sort list of detections by confidence
def getKey(dt):
    return dt.confidence

#detection class
class dt :
    def __init__(self, bb, conf):
        self.bbox = bb
        self.confidence = conf
		
class kcf:
	def __init__(self, bb, frame, nf):
		self.locations = []
		self.numFrame = []
		self.noMeasureCount = 0
		self.bbox = bb
		self.numFrame.append(nf)
		self.locations.append(bb)
		self.tracker = cv2.TrackerKCF_create()
		self.tracker.init(frame, bb)

        
        
class BgsubTrack:
    #constructor
    def __init__(self):
        self.hog = cv2.HOGDescriptor((56,56), (16,16), (8,8), (8,8), 9)
        self.bgsub = cv2.createBackgroundSubtractorMOG2()

    #slidingWindow : applies a sliding window over a given image
    #@param image : input image on which we have to extract descriptors
    #@param multiscale : defines if we have to apply multiple downscale to the image
    #@param scaleFactor : downscale factor in percent
    #return : list of descriptors and list of locations in the image
    def slidingWindow(self, image, multiscale=False, scaleFactor=0.0):
        if image is None:
            print("One of the image dimension is 0")
            return [], []
        if image.shape[1] %self.hog.blockStride[0] != 0 or image.shape[0] %self.hog.blockStride[1] != 0:
            print("Image size must be a multiple of block stride")
            return [], []

        if image.shape[1] < self.hog.winSize[1] or image.shape[0] < self.hog.winSize[0]:
            print("Image is smaller than detection window")
            return [], []

        descriptors = []
        descriptor = []
        roi = []
        nbDownscale = 0
        while self.hog.winSize[0] < image.shape[0] and self.hog.winSize[1] < image.shape[1]:
            if multiscale==True and scaleFactor != 0.0:
                image = cv2.resize(image, (int(image.shape[1] - nbDownscale*scaleFactor*image.shape[1]), int(image.shape[0] - nbDownscale*scaleFactor*image.shape[0])))
            
            for i in range(0, image.shape[0], self.hog.blockStride[0]):
                for j in range(0, image.shape[1], self.hog.blockStride[1]):
                    # breaks when the sliding windows starts to be out of bounds
                    if i > image.shape[0] - self.hog.winSize[1] or j > image.shape[1] - self.hog.winSize[0]:
                        break;
                    
                    #if viz:
                    #    clone = np.copy(image)
                    #    cv2.rectangle(clone, (j,i), (j+self.hog.winSize[0], i+self.hog.winSize[1]), (0, 0, 255))
                    #    cv2.imshow("sliding window", clone)
                    #    cv2.waitKey(30)

                    #extracts the window and computes HOG descriptors
                    window = image[i:i+self.hog.winSize[1], j:j+self.hog.winSize[0]];
                    roi.append((j, i, self.hog.winSize[0], self.hog.winSize[1]))
                    dsc = []
                    dsc = self.hog.compute(window)
                    descriptors.append(dsc)
            if multiscale==False:
                break
            nbDownscale += 1

        return descriptors, roi


    #resize : resizes images so we can run the sliding window without OOB errors
    #@param regions : input / ouput list of images samples
    #return : correctly shaped regions
    def resize(self, regions):
        for idx, region in enumerate(regions):
            if region.shape[0] ==0 or region.shape[1] == 0:
                regions[idx]=None
                continue
            if region.shape[0] %self.hog.blockStride[1] != 0 or region.shape[1] % self.hog.blockStride[0] != 0:
                region=cv2.resize(region, (region.shape[1] + self.hog.blockStride[0] - region.shape[1] % self.hog.blockStride[0], 
                                    region.shape[0] + self.hog.blockStride[1] - region.shape[0] % self.hog.blockStride[1]))

            if region.shape[0] < self.hog.winSize[1]:
                region=cv2.resize(region, (region.shape[1], self.hog.winSize[1]))
            if region.shape[1] < self.hog.winSize[0]:   
                region = cv2.resize(region, (self.hog.winSize[0], region.shape[0]))
				
            regions[idx] = region
        return regions

    #roiSelection : selects regions on dimension criteria to reduce the research of pedestrians in these regions
    #@param rectangles : regions that were extracted by subtracting the background
    #@param frame : input / output original image
    #return : selected regions and their corresponding locations in the image + image with regions drawn (used for visualization)
    def roiSelection(self, rectangles, frame):
        regions = []
        boundingLocations = []
        original_frame = np.copy(frame)
        for bRect in rectangles:
            if bRect[2] * bRect[3] > 750 and bRect[2] * bRect[3] < 6000:
                if bRect[0] < 30 or bRect[1] < 30 or bRect[0] > frame.shape[0] - bRect[2] - 60 or bRect[1] > frame.shape[1] - bRect[3] -60: 
                    continue
                regions.append(np.copy(original_frame[bRect[1]-30:bRect[1]+ bRect[3]+60, bRect[0]-30:bRect[0]+bRect[2]+60]))
                boundingLocations.append((bRect[0]-30, bRect[1]-30, bRect[2]+60, bRect[3]+60))
                cv2.rectangle(frame, (bRect[0]-30, bRect[1]-30), (bRect[0]+bRect[2]+60, bRect[1]+bRect[3]+60), (0,0,255), 1)
                    
        return regions, boundingLocations, frame

    #nms : applies Non Maxima Suppression on given detections
    #@param posDetections : input / output list of dt object corresponding to windows that were classified as positives 
    #return : list of detections with highest confidence (posDetections size is reduced)
    def nms(self, posDetections):
        for i in range(len(posDetections)-1, -1, -1):
            for det2 in posDetections :
                if posDetections[i]==det2:
                    break

                x_overlap = max(0, min(posDetections[i].bbox[0]+posDetections[i].bbox[2], det2.bbox[0]+det2.bbox[2]) - max(posDetections[i].bbox[0], det2.bbox[0]))
                y_overlap = max(0, min(posDetections[i].bbox[1]+posDetections[i].bbox[3], det2.bbox[1]+det2.bbox[3]) - max(posDetections[i].bbox[1], det2.bbox[1]))
                overlap_area = x_overlap*y_overlap
                total_area = (posDetections[i].bbox[2]*posDetections[i].bbox[3]) + (det2.bbox[2]*det2.bbox[3]) - overlap_area
                if overlap_area/float(total_area) > 0.1:
                    del posDetections[i]
                    break

        return posDetections


    #addNewTrackers : adds trackers that does not overlap already tracked regions
    #@param posDetections : input list of dt object corresponding to windows that were classified as positives
    #@param trackerList : input / output list of colorHistTracker object
    #@param frame : input original image used to instantiate new tracker
    #@pram nframe : number of the current frame
    #return : list of trackers containing old and new trackers
    def addNewTrackers(self, posDetections, trackerList, frame, nFrame):
        for det in posDetections:
            if len(trackerList)==0:
                trackerList.append(kcf((det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]), frame, nFrame))        
            else:
                flag_nms = False
                for tracker in trackerList:
                    x_overlap = max(0, min(det.bbox[0]+det.bbox[2], tracker.bbox[0]+tracker.bbox[2]) - max(det.bbox[0], tracker.bbox[0]))
                    y_overlap = max(0, min(det.bbox[1]+det.bbox[3], tracker.bbox[1]+tracker.bbox[2]) - max(det.bbox[1], tracker.bbox[1]))
                    overlap_area = x_overlap*y_overlap
                    total_area = (det.bbox[2]*det.bbox[3])+(tracker.bbox[2]*tracker.bbox[3])-overlap_area
                    if overlap_area / float(total_area) > 0.1:
                        flag_nms = True
                        break
                if not flag_nms:
                    trackerList.append(kcf((det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]), frame, nFrame))
            
        return trackerList

    #updateTrackers : tries to define new locations of all existing trackers
    #@param posDetections : input list of dt object corresponding to windows that were classified as positives
    #@param trackerList : input / output list of colorHistTracker object
    #@param frame : input original image used to calculate histogram distance
    #@param nframe : number of the current frame
    #@param significantTrackers : input / output list of trackers that lasted more than 20 frames
	#@param model : input pre-loaded model used to do the predictions
    #return : updated list of trackers, updated list of trackers that produced relevant tracks
    def updateTrackers(self, posDetections, trackerList, frame, nframe, significantTrackers, model):
        for i in range(len(trackerList)-1, -1, -1):
            flag = False
            for det in posDetections:
                x_overlap = max(0, min(trackerList[i].bbox[0] + trackerList[i].bbox[2], det.bbox[0]+det.bbox[2]) - max(trackerList[i].bbox[0], det.bbox[0]))
                y_overlap = max(0, min(trackerList[i].bbox[1] + trackerList[i].bbox[3], det.bbox[1]+det.bbox[3]) - max(trackerList[i].bbox[1], det.bbox[1]))
                overlap_area = x_overlap*y_overlap
                total_area = trackerList[i].bbox[2] * trackerList[i].bbox[3] + det.bbox[2]*det.bbox[3] - overlap_area

                if overlap_area / float(total_area) > 0.5:
                    #updates the tracker from measurement
                    if distance==0 or (len(trackerList[i].distances)>=1 and trackerList[i].distances[-1]==0) or distance <= 1.5*mean_distance or len(trackerList[i].distances)==0:
                        trackerList[i].bbox = det.bbox
                        trackerList[i].locations.append(det.bbox)
                        trackerList[i].numFrame.append(nframe)
                        trackerList[i].noMeasureCount = 0
                        flag = True
                        break
            #if we could not find a measurement that matches the current tracker
            if not flag:
                if len(trackerList[i].locations) >= 2:
                    #estimates the new location of the pedestrian (we assume that the speed is linear and the direction remains the same between 2 frames)
                    dx = trackerList[i].locations[-1][0] - trackerList[i].locations[-2][0]
                    dy = trackerList[i].locations[-1][1] - trackerList[i].locations[-2][1]
                    predict_x = trackerList[i].locations[-1][0] + dx
                    predict_y = trackerList[i].locations[-1][1] + dy
                    roi_x = predict_x-20
                    roi_y = predict_y-20
                    width = 96
                    height = 96
                    
                    #gets a larger window to avoid relying too much on our estimation
                    if roi_x < 0:
                        roi_x = 0
                    if roi_x + width > frame.shape[0]:
                        width = 96 - (roi_x+96-frame.shape[0])
                    if roi_y < 0:
                        roi_y = 0
                    if roi_y + height > frame.shape[1]:
                        height = 96 - (roi_y+96 - frame.shape[1])

                    #should contain only one region but resize function is designed to iterate over a list
                    region = []
                    region.append(np.copy(frame[roi_y:roi_y+height, roi_x:roi_x+width]))
					
                    region = self.resize(region)
                    descriptors, roi = self.slidingWindow(region[0])
                    redetections = []
					
                    for r in roi:
                        feed_im = np.copy(frame[roi_y+r[1]:roi_y+r[1]+r[3], roi_x+r[0]:roi_x+r[0]+r[2]])
                        feed_im = cv2.resize(feed_im, (im_size, im_size))
                        feed_im = np.array([feed_im])/255.
                        prediction = model.predict(feed_im)
				
                        if np.argmax(prediction) == 1:
                            accurateRect = []
                            accurateRect.append(roi_x+r[0])
                            accurateRect.append(roi_y+r[1])
                            accurateRect.append(r[2])
                            accurateRect.append(r[3])
                            redetections.append(dt(accurateRect, prediction[0][1]))
							
                    sorted(redetections, key=getKey, reverse=True)
                    flagNoRedetect = False
                    if len(redetections) > 0:
                        if (redetections[0].bbox[0])+(redetections[0].bbox[2]) > frame.shape[0]:
                            redetections[0].bbox[2]=frame.shape[0]-redetections[0].bbox[0]
                        if (redetections[0].bbox[1])+(redetections[0].bbox[3])>frame.shape[1]:
                            redetections[0].bbox[3] = frame.shape[1]-redetections[0].bbox[1]

                        #updates from redetection
                        if distance==0 or trackerList[i].distances[-1]==0 or distance <= 1.5*mean_distance:
                            trackerList[i].bbox = redetections[0].bbox
                            trackerList[i].locations.append(redetections[0].bbox)
                            trackerList[i].numFrame.append(nframe)
                            trackerList[i].noMeasureCount = 0
                            flag = True
                    
                    #if we could not redetect the head then the target is lost for this frame
                    else:
                        trackerList[i].noMeasureCount += 1
                        flagNoRedetect = True
                        if trackerList[i].noMeasureCount == 5:
                            #if the track length was higher than 20 frames we consider this track as relevant to be a pedestrian
                            if len(trackerList[i].locations) > 20:
                                significantTrackers.append(trackerList[i])
                            del trackerList[i]


                    if not flag and not flagNoRedetect:
                        trackerList[i].noMeasureCount += 1
                        if trackerList[i].noMeasureCount == 5:
                            if len(trackerList[i].locations) > 20:
                                significantTrackers.append(trackerList[i])
                            del trackerList[i]
                #still increases the no measure count because new tracked regions that does not generate atleast 2 consecutive measures are
                #most likely false detections
                else:
                    trackerList[i].noMeasureCount += 1
                    if trackerList[i].noMeasureCount == 5:
                        if len(trackerList[i].locations) > 20:
                            significantTrackers.append(trackerList[i])
                        del trackerList[i]

        return trackerList, significantTrackers



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect pedestrians on surveillance videos')
    parser.add_argument('model_path', help='Path to the model hdf5 file')
    parser.add_argument('video_path', help='Path to the video to be processed')
    parser.add_argument('detection_path', help='Path to the csv file in which raw detection results will be written')
    parser.add_argument('NMS_path', help='Path to the csv file in which raw after NMS results will be written')
    parser.add_argument('tracking_path', help='Path to the csv file in which raw tracking detection results will be written')
    args = parser.parse_args()
   
 
    csv_file = csv_nms_file = csv_tracking_file = ""
    writer = writer_nms = writer_tracking = ""
	
    if DET_RES:
        csv_file = open(args.detection_path, "w")
        writer = csv.writer(csv_file)
	
    if NMS_RES:
        csv_nms_file = open(args.NMS_path, "w")
        writer_nms = csv.writer(csv_nms_file)
	
    if TRACK_RES:
        csv_tracking_file = open(args.tracking_path, "w")
        writer_tracking = csv.writer(csv_tracking_file)
	
    bgsubTrack = BgsubTrack()
    cap = cv2.VideoCapture(args.video_path)
	
	#load inception_V3 model
    model = load_model(args.model_path)
    if model is None:
        print("Could not load the model")
        exit(-1)

    trackerList = []
    significantTrackers = []
    im_size = 224

    currentFrame = np.ndarray(shape=(0,0))
    previousFrame = np.ndarray(shape=(0,0))
    numFrame = 0
    while(True):
        ret, frame = cap.read()
        if ret==False:
            break

        numFrame += 1
        print("frame n°{}".format(numFrame))
        true_detections = np.copy(frame)
        after_nms = np.copy(frame)
        trackingResults = np.copy(frame)
        currentFrame = np.copy(frame)

		
        diff_mask = np.ndarray(shape=currentFrame.shape, dtype=currentFrame.dtype)
        contours = []
		
        if FRAME_DIFF:
            if currentFrame.shape != (0,0) and previousFrame.shape != (0,0):
                diff_mask = cv2.absdiff(previousFrame, currentFrame)
                diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
                ret, diff_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY)
                im2, contours, hierarchy = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		
        if MOG:
            fgmask = bgsubTrack.bgsub.apply(frame)
            ret, thresholded = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
            im2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rectangles = []
        for c in contours:
            rectangles.append(cv2.boundingRect(c))
		
        regions, boundingLocations, frame_roi = bgsubTrack.roiSelection(rectangles, frame)
        regions = bgsubTrack.resize(regions)
		
        posDetections = []
		#extract HOG descriptors from each region
        for idxr, region in enumerate(regions):
            descriptors = []
            roi = []
            descriptors, roi = bgsubTrack.slidingWindow(region)
			#descriptors, roi = bgsubTrack.slidingWindow(frame)
			#classify extracted descriptors
            for idxd, reg in enumerate(roi):
				#desc = desc.reshape(1, desc.shape[0])                   
                feed_im = np.copy(currentFrame[ reg[1]:reg[3] + reg[1],  reg[0]: reg[0] + reg[2]])
                feed_im = cv2.resize(feed_im, (im_size, im_size))
					
                feed_im = np.array([feed_im])/255.
                prediction = model.predict(feed_im)
				
                if np.argmax(prediction)==1:
					#relocate accurately the window
                    accurateRect = []
                    accurateRect.append(roi[idxd][0])
                    accurateRect.append(roi[idxd][1])
                    accurateRect.append(roi[idxd][2])
                    accurateRect.append(roi[idxd][3])
                    posDetections.append(dt(accurateRect, prediction[0][1]))
                    if viz:
                        if accurateRect[0] + accurateRect[2] > frame.shape[0]:
                            accurateRect[2] = frame.shape[0] - accurateRect[0]
                        if accurateRect[1] + accurateRect[3] > frame.shape[1]:
                            accurateRect[3] = frame.shape[1] - accurateRect[1]

                        cv2.rectangle(true_detections, (accurateRect[0], accurateRect[1]), (accurateRect[0]+accurateRect[2], accurateRect[1]+accurateRect[3]), (0, 0, 255));
					#Ouput detections results (CSV format)
                    if DET_RES:
                        writer.writerow([numFrame, "1", accurateRect[0], accurateRect[1], accurateRect[2], accurateRect[3], prediction[0][1]])
                else:
                    accurateRect = []
                    accurateRect.append(roi[idxd][0])
                    accurateRect.append(roi[idxd][1])
                    accurateRect.append(roi[idxd][2])
                    accurateRect.append(roi[idxd][3])
                    if DET_RES:
                        writer.writerow([numFrame, "0", accurateRect[0], accurateRect[1], accurateRect[2], accurateRect[3], prediction[0][0]])

		#if viz:
		#    cv2.imshow("true_detections", true_detections)

        posDetections = sorted(posDetections, key=getKey, reverse=True) 
        posDetections = bgsubTrack.nms(posDetections)
		
        if NMS_RES:
            for d in posDetections :
                writer_nms.writerow([numFrame, "1", d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence])
				
        for t in trackerList:
            t.tracker.update(frame, bbox)
            if bbox[2] !=0 and bbox[3] !=0:
                t.bbox = bbox
		
        trackerList, significantTrackers = bgsubTrack.updateTrackers(posDetections, trackerList, frame, numFrame, significantTrackers, model)
        trackerList = bgsubTrack.addNewTrackers(posDetections, trackerList, frame, numFrame)

        previousFrame = np.copy(currentFrame)
		
        if viz:
			#cv2.imshow("foreground", fgmask)
			#cv2.imshow("thresholded", thresholded)
			#cv2.imshow("opened", opened)
            cv2.imshow("frame diff", diff_mask)
            cv2.imshow("rectangles", frame_roi)
            for detection in posDetections : 
                cv2.rectangle(after_nms, (detection.bbox[0], detection.bbox[1]), (detection.bbox[0]+detection.bbox[2], detection.bbox[1]+detection.bbox[3]), (0, 0, 255))
            cv2.imshow("after_nms", after_nms)
			
            for tracker in trackerList:
                cv2.rectangle(trackingResults, (tracker.bbox[0], tracker.bbox[1]), (tracker.bbox[0]+tracker.bbox[2], tracker.bbox[1]+tracker.bbox[3]), (0, 0, 255))
            cv2.imshow("trackers", trackingResults)
			
            cv2.waitKey(30)
		
    if TRACK_RES:
	    for idx, t in enumerate(significantTrackers):
	        for idx2, nf in enumerate(t.numFrame):
		        writer_tracking.writerow(["head"+str(idx), nf, t.locations[idx2][0]+(t.locations[idx2][2]/2), t.locations[idx2][1] + (t.locations[idx2][3] / 2)])
			
    csv_file.close()
    csv_nms_file.close()
    csv_tracking_file.close()

        
    

    
    

    
    

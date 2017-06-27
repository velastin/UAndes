#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import tensorflow as tf

import os
import sys
import glob

import numpy as np

viz = True

#function used to sort list of detections by confidence
def getKey(dt):
    return dt.confidence

#detection class
class dt :
    def __init__(self, bb, conf):
        self.bbox = bb
        self.confidence = conf

#tracker class
class colorHistTracker:
    def __init__(self, location, f, nf):
        self.distances = []
        self.locations = []
        self.numFrame = []
        self.hist = []
        self.bbox = location
        self.locations.append(location)
        self.numFrame.append(nf)
        self.noMeasureCount = 0

        self.hist.append(cv2.calcHist([f],[0],None,[256],[0,256]))
        self.hist.append(cv2.calcHist([f],[1],None,[256],[0,256]))
        self.hist.append(cv2.calcHist([f],[2],None,[256],[0,256]))
        
        
class BgsubTrack:
    #constructor
    def __init__(self):
        self.hog = cv2.HOGDescriptor((56,56), (16,16), (8,8), (8,8), 9)
        self.bgsub = cv2.createBackgroundSubtractorMOG2()

    #getHistDistance : Calculates the distance between 2 histogram
    #@param roi : part of an image on which we have to calculate the corresponding histogram
    #@param histogram : reference histogram used to do the comparison
    #return : batacharrya distance between the 2 histograms, and the calculated histogram
    def getHistDistance(self, roi, histogram): 
        hist = []
        hist.append(cv2.calcHist([roi],[0],None,[256],[0,256]))
        hist.append(cv2.calcHist([roi],[1],None,[256],[0,256]))
        hist.append(cv2.calcHist([roi],[2],None,[256],[0,256]))
        
        distance = 0
        distance += cv2.compareHist(hist[0], histogram[0], cv2.HISTCMP_HELLINGER)
        distance += cv2.compareHist(hist[1], histogram[1], cv2.HISTCMP_HELLINGER)
        distance += cv2.compareHist(hist[2], histogram[2], cv2.HISTCMP_HELLINGER)
        distance = distance /3.0
        
        return distance, hist

    #slidingWindow : applies a sliding window over a given image
    #@param image : input image on which we have to extract descriptors
    #@param multiscale : defines if we have to apply multiple downscale to the image
    #@param scaleFactor : downscale factor in percent
    #return : list of descriptors and list of locations in the image
    def slidingWindow(self, image, multiscale=False, scaleFactor=0.0):
        if image.shape[1] %self.hog.blockStride[0] != 0 or image.shape[0] %self.hog.blockStride[1] != 0:
            print "Image size must be a multiple of block stride"
            exit()

        if image.shape[1] < self.hog.winSize[1] or image.shape[0] < self.hog.winSize[0]:
            print "Image is smaller than detection window"
            exit()

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
                        #clone = np.copy(image);
                        #cv2.rectangle(clone, (j,i), (j+self.hog.winSize[0], i+self.hog.winSize[1]), (0, 0, 255));
                        #cv2.imshow("sliding window", clone);
                        #cv2.waitKey(30);

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
    #return : selected regions and their corresponding locations in the image, image with regions drawn (used for visualization)
    def roiSelection(self, rectangles, frame):
        regions = []
        boundingLocations = []
        for bRect in rectangles:
            if bRect[2] * bRect[3] > 750 and bRect[2] * bRect[3] < 6000:
                if bRect[0] < 40 or bRect[1] < 40 or bRect[0] > frame.shape[0] - bRect[2] - 40 or bRect[1] > frame.shape[1] - bRect[1] -40: 
                    continue
                regions.append(np.copy(frame[bRect[1]-40:bRect[1]+ bRect[3], bRect[0]-40:bRect[0]+bRect[2]]))
                boundingLocations.append((bRect[0]-40, bRect[1]-40, bRect[2]+40, bRect[3]+40))
                cv2.rectangle(frame, (bRect[0]-40, bRect[1]-40), (bRect[0]+bRect[2], bRect[1]+bRect[3]), (0,0,255), 1)
                    
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
                trackerList.append(colorHistTracker(det.bbox, frame[det.bbox[1]:det.bbox[1]+det.bbox[3], det.bbox[0]:det.bbox[2]] , nFrame))        
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
                    trackerList.append(colorHistTracker(det.bbox, frame[det.bbox[1]:det.bbox[1]+det.bbox[3], det.bbox[0]:det.bbox[2]] , nFrame))
            
        return trackerList

    #updateTrackers : tries to define new locations of all existing trackers
    #@param posDetections : input list of dt object corresponding to windows that were classified as positives
    #@param trackerList : input / output list of colorHistTracker object
    #@param frame : input original image used to calculate histogram distance
    #@param nframe : number of the current frame
    #@param significantTrackers : input / output list of trackers that lasted more than 20 frames
    def updateTrackers(self, posDetections, trackerList, frame, nframe, significantTrackers):
        for i in range(len(trackerList)-1, -1, -1):
            flag = False
            for det in posDetections:
                x_overlap = max(0, min(trackerList[i].bbox[0] + trackerList[i].bbox[2], det.bbox[0]+det.bbox[2]) - max(trackerList[i].bbox[0], det.bbox[0]))
                y_overlap = max(0, min(trackerList[i].bbox[1] + trackerList[i].bbox[3], det.bbox[1]+det.bbox[3]) - max(trackerList[i].bbox[1], det.bbox[1]))
                overlap_area = x_overlap*y_overlap
                total_area = trackerList[i].bbox[2] * trackerList[i].bbox[3] + det.bbox[2]*det.bbox[3] - overlap_area

                if overlap_area / float(total_area) > 0.5:
                    distance, hist = self.getHistDistance(np.copy(frame[det.bbox[1]:det.bbox[1]+det.bbox[3], det.bbox[0]:det.bbox[0]+det.bbox[2]]), trackerList[i].hist)

                    mean_distance = 0
                    for dist in trackerList[i].distances:
                        mean_distance += dist
                    if len(trackerList[i].distances) > 0:
                        mean_distance = mean_distance / len(trackerList[i].distances)

                    #updates the tracker from measurement
                    if distance==0 or (len(trackerList[i].distances)>=1 and trackerList[i].distances[-1]==0) or distance <= 1.5*mean_distance or len(trackerList[i].distances)==0:
                        trackerList[i].distances.append(distance)
                        trackerList[i].hist = hist
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
                    dy = trackerList[i].locations[-1][1] - trackerList[i].locations[-2][0]
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

                    with tf.Session() as sess:
                        saver = tf.train.import_meta_graph('graph.meta')
                        saver.restore(sess, tf.train.latest_checkpoint('./'))
                        graph = tf.get_default_graph()
                        x = graph.get_tensor_by_name("x:0")
                        y = graph.get_tensor_by_name("y:0")
                        op_to_restore = graph.get_tensor_by_name("output_op:0")
            
                        #tries to redetect the lost head
                        for idxd, desc in enumerate(descriptors):
                            desc = desc.reshape(1, desc.shape[0])
                            feed_dict = {x:desc}
                            regression = sess.run(op_to_restore, feed_dict)
                            if regression[0][0] > regression[0][1]:
                                accurateRect = (roi_x+roi[idxd][0], roi_y+roi[idxd][1], roi[idxd][2], roi[idxd][3])
                                redetections.append(dt(accurateRect, regression[0][0]))
                    sorted(redetections, key=getKey, reverse=True)
                    flagNoRedetect = False
                    if len(redetections) > 0:
                        if (redetections[0].bbox[0])+(redetections[0].bbox[2]) > frame.shape[0]:
                            redetections[0].bbox[2]=frame.shape[0]-redetections[0].bbox[0]
                        if (redetections[0].bbox[1])+(redetections[0].bbox[3])>frame.shape[1]:
                            redetections[0].bbox[3] = frame.shape[1]-redetections[0].bbox[1]
                        
                        distance, hist = self.getHistDistance(frame[redetections[0].bbox[1]:redetections[0].bbox[3], redetections[0].bbox[0]:redetections[0].bbox[2]], trackerList[i].hist)

                        mean_distance = 0
                        for d in trackerList[i].distances:
                            mean_distance+=d
                        if len(trackerList[i].distances)>0:
                            mean_distance = mean_distance / len(trackerList[i].distances)

                        #updates from redetection
                        if distance==0 or trackerList[i].distances[-1]==0 or distance <= 1.5*mean_distance:
                            trackerList[i].distances.append(distance)
                            trackerList[i].hist = hist
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
    bgsubTrack = BgsubTrack()
    cap = cv2.VideoCapture("/home/mathieu/STAGE/underground_dataset/pos/test/A_d800mm_R6.mpg")
    #svm = cv2.ml.SVM_load("/home/mathieu/STAGE/underground_dataset/results/models/openCV_model.xml")

    trackerList = []
    significantTrackers = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('graph.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        op_to_restore = graph.get_tensor_by_name("output_op:0")
        print "loaded graph successfully"
        numFrame = -1
        while(True):
            ret, frame = cap.read()
            if ret==False:
                break

            numFrame += 1
            true_detections = np.copy(frame)
            after_nms = np.copy(frame)
            trackingResults = np.copy(frame)

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

                #classify extracted descriptors
                for idxd, desc in enumerate(descriptors):
                    desc = desc.reshape(1, desc.shape[0])
                    feed_dict = {x:desc}
                    regression = sess.run(op_to_restore, feed_dict)
                    if regression [0][0] > regression[0][1]:
                        #relocate accurately the window
                        accurateRect = (boundingLocations[idxr][0]+roi[idxd][0], boundingLocations[idxr][1]+roi[idxd][1], roi[idxd][2], roi[idxd][3])
                        posDetections.append(dt(accurateRect, regression[0][0]))
                        if viz:
                            if accurateRect[0] + accurateRect[2] > frame.shape[0]:
                                accurateRect[2] = frame.shape[0] - accurateRect[0]
                            if accurateRect[1] + accurateRect[3] > frame.shape[1]:
                                accurateRect[3] = frame.shape[1] - accurateRect[1]

                            cv2.rectangle(true_detections, (accurateRect[0], accurateRect[1]), (accurateRect[0]+accurateRect[2], accurateRect[1]+accurateRect[3]), (0, 0, 255));

            if viz:
                cv2.imshow("true_detections", true_detections)

            sorted(posDetections, key=getKey, reverse=True)
            posDetections = bgsubTrack.nms(posDetections)
            
            trackerList, significantTrackers = bgsubTrack.updateTrackers(posDetections, trackerList, frame, numFrame, significantTrackers)
            trackerList = bgsubTrack.addNewTrackers(posDetections, trackerList, frame, numFrame)

            
            if viz:
                cv2.imshow("foreground", fgmask)
                cv2.imshow("thresholded", thresholded)
                #cv2.imshow("opened", opened)
                cv2.imshow("rectangles", frame_roi)
                for detection in posDetections : 
                    cv2.rectangle(after_nms, (detection.bbox[0], detection.bbox[1]), (detection.bbox[0]+detection.bbox[2], detection.bbox[1]+detection.bbox[3]), (0, 0, 255))
                cv2.imshow("after_nms", after_nms)
                
                for tracker in trackerList:
                    cv2.rectangle(trackingResults, (tracker.bbox[0], tracker.bbox[1]), (tracker.bbox[0]+tracker.bbox[2], tracker.bbox[1]+tracker.bbox[3]), (0, 0, 255))
                cv2.imshow("trackers", trackingResults)
                
                cv2.waitKey(30)

        
    

    
    

    
    

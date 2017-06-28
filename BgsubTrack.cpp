//OpenCV
#include <opencv2/core/core.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/flann.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/bgsegm.hpp"
#include "opencv2/photo.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

//C
#include <stdio.h>
#include <sys/time.h>

//C++
#include <iostream>
#include <sstream>
#include <cmath>

#include "BgsubTrack.hpp"

//#define VIZ 1

using namespace std;
using namespace cv;
using namespace ml;

BgsubTrack::BgsubTrack(const HOGDescriptor & hogD, const Ptr<BackgroundSubtractor> & gaussMix)
{
    this->hog = HOGDescriptor(hogD.winSize, hogD.blockSize, hogD.blockStride, hogD.cellSize, hogD.nbins);
    this->bgsub = gaussMix;
}

void BgsubTrack::addNewTrackers(const vector<dt> & posDetections, vector<colorHistTracker> * trackerList, const Mat & frame, const int & nframe)
{
    for(int i=0; i < posDetections.size(); i--)
    {
        // get the area of the detection
        int x1_tl = posDetections[i].boundingBox.x;
        int y1_tl = posDetections[i].boundingBox.y;
        int x1_br = posDetections[i].boundingBox.x + posDetections[i].boundingBox.width;
        int y1_br = posDetections[i].boundingBox.y + posDetections[i].boundingBox.height;
        int area_1 = posDetections[i].boundingBox.width * posDetections[i].boundingBox.height;

        if(trackerList->size() == 0)
            trackerList->push_back(colorHistTracker(posDetections[0].boundingBox, frame, nframe));
        else
        {
            bool flag_nms = false;
            for(int j=0; j < trackerList->size(); j++)
            {
                // gets the area of the already tracked region
                int x2_tl = (*trackerList)[j].bbox.x;
                int y2_tl = (*trackerList)[j].bbox.y;
                int x2_br = (*trackerList)[j].bbox.x + (*trackerList)[j].bbox.width;
                int y2_br = (*trackerList)[j].bbox.y + (*trackerList)[j].bbox.height;
                int area_2 = (*trackerList)[j].bbox.width * (*trackerList)[j].bbox.height;

                // Calculates overlapping area and applies Non Maxima Suppression
                int x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl));
                int y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl));
                int overlap_area = x_overlap * y_overlap;
                int total_area = area_1 + area_2 - overlap_area;
                if(overlap_area / (float) total_area > 0.1) // BOSS : 0.5 / subway : 0.1 - 0.2
                {
                    // the tracker should have already been updated in updateTrackers so we do nothing
                    flag_nms = true;
                    break;
                }
            }
            // if no overlapping with other trackers has been found at this step then we assume that the detection is a new person
            if(! flag_nms)
            {
                // instanciates a new tracker
                trackerList->push_back(colorHistTracker(posDetections[i].boundingBox, frame, nframe));
            }
        }
    }
}


void BgsubTrack::nms(vector<dt> * posDetections)
{
    //iterates from starting from the end of the vector since we will delete some elements
    for(int i=posDetections->size()-1; i >= 0; i--)
    {
        //gets the lowest confidence detection's top left and bottom right corners coordinates
        int x1_tl = (*posDetections)[i].boundingBox.x;
        int y1_tl = (*posDetections)[i].boundingBox.y;
        int x1_br = (*posDetections)[i].boundingBox.x + (*posDetections)[i].boundingBox.width;
        int y1_br = (*posDetections)[i].boundingBox.y + (*posDetections)[i].boundingBox.height;
        int area_1 = (*posDetections)[i].boundingBox.width * (*posDetections)[i].boundingBox.height;
        //compares the location with the highest confidence detections (first elements in the vector)
        for(int j=0; j < posDetections->size(); j++)
        {
            if(i == j)
                break;

            if((*posDetections)[i].confidence > -0.015)
            {
                posDetections->erase(posDetections->begin() +i);
                break;
            }
            //gets the highest confidence detection's top left and bottom right corners coordinates
            int x2_tl = (*posDetections)[j].boundingBox.x;
            int y2_tl = (*posDetections)[j].boundingBox.y;
            int x2_br = (*posDetections)[j].boundingBox.x + (*posDetections)[j].boundingBox.width;
            int y2_br = (*posDetections)[j].boundingBox.y + (*posDetections)[j].boundingBox.height;
            int area_2 = (*posDetections)[j].boundingBox.width * (*posDetections)[j].boundingBox.height;
            int x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl));
            int y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl));
            int overlap_area = x_overlap * y_overlap;
            int total_area = area_1 + area_2 - overlap_area;
            // if there's more than 10% overlapping we erase detections starting from the end of the vector (lowest confidences)
            if(overlap_area / (float) total_area > 0.1)
            {
                posDetections->erase(posDetections->begin() + i);
                break;
            }
        }
    }
}

void BgsubTrack::roiSelection(const vector<RotatedRect> & rectangles, vector<Mat> * regions, vector<Rect2d> * boundingLocations, const Mat & frame)
{
    for(int i=0; i < (int)rectangles.size(); i++)
    {
        Rect bRect = rectangles[i].boundingRect();
        if((bRect.size().width * bRect.size().height) > 750 && (bRect.size().width * bRect.size().height) < 6000)//(bRect.size().width * bRect.size().height > 3000))////(bRect.size().width * bRect.size().height > 3000))//(bRect.size().width * bRect.size().height) > 750 && (bRect.size().width * bRect.size().height) < 6000)// > 3000 BOSS dataset
        {
            // skips bounding boxes that are out of the image bounds
            if(bRect.x < 40 || bRect.y < 40 || bRect.x > frame.cols - bRect.width -40|| bRect.y > frame.rows - bRect.height -40)
                continue;

            // extracts a larger region of interest from the original image since blob detector is not really realiable for head top detection
            regions->push_back(frame(Rect(bRect.x-40, bRect.y-40, bRect.width+40, bRect.height+40)));
            boundingLocations->push_back(Rect2d(bRect.x-40, bRect.y-40, bRect.width+40, bRect.height+40));
        }
    }
}

void BgsubTrack::resize(vector<Mat> * regions)
{
    for(int i=0; i < (int)regions->size(); i++)
    {
        // checks if the region needs to be resized so the sliding window can run over it without size problem
        if((*regions)[i].rows %this->hog.blockStride.height !=0 || (*regions)[i].cols %this->hog.blockStride.width != 0)
            cv::resize((*regions)[i], (*regions)[i], Size((*regions)[i].cols + this->hog.blockStride.width - (*regions)[i].cols%this->hog.blockStride.width,
                                                (*regions)[i].rows + this->hog.blockStride.height - (*regions)[i].rows%this->hog.blockStride.height));

        //resizes too small regions
        if((*regions)[i].rows < this->hog.winSize.height )
            cv::resize((*regions)[i], (*regions)[i], Size((*regions)[i].cols, this->hog.winSize.height));
        if((*regions)[i].cols < this->hog.winSize.width)
            cv::resize((*regions)[i], (*regions)[i], Size(this->hog.winSize.width, (*regions)[i].rows));
    }
}


vector<vector<float> > BgsubTrack::slidingWindow(Mat image, vector<Rect2d> * roi, const bool & multiscale, const float & scaleFactor)
{
    if(image.rows %this->hog.blockStride.height != 0 || image.cols %this->hog.blockStride.width != 0)
    {
        cout << "Image size must be a multiple of block stride" << endl;
        exit(-1);
    }

    if(image.rows < this->hog.winSize.height || image.cols < this->hog.winSize.width)
    {
        cout << "Image is smaller than detection window " << endl;
        exit(-1);
    }

    vector<vector<float> > descriptors;
    vector<float> dsc;
    int nbDownscale = 0;

    while(this->hog.winSize.width < image.cols && this->hog.winSize.height < image.rows)
    {
        if(multiscale && scaleFactor != 0.0)
            cv::resize(image, image, Size(image.cols - nbDownscale*scaleFactor*image.cols, image.rows - nbDownscale*scaleFactor*image.rows));

        //makes the sliding window slide
        for(int i=0; i <= image.rows; i=i+this->hog.blockStride.height)
        {
            for(int j=0; j <= image.cols; j=j+this->hog.blockStride.width)
            {
                // breaks when the sliding windows starts to be out of bounds
                if(i > image.rows - this->hog.winSize.height || j > image.cols - this->hog.winSize.width)
                    break;

#ifdef VIZ
                /*Mat clone = image.clone();
                rectangle(clone, Point(j,i), Point(j+this->hog.winSize.width, i+this->hog.winSize.height), Scalar(0, 0, 255));
                imshow("sliding window", clone);
                waitKey(30);*/
#endif

                //extracts the window and computes HOG descriptors
                Mat window = image(Rect(j, i, this->hog.winSize.width, this->hog.winSize.height));
                roi->push_back(Rect2d(j, i, this->hog.winSize.width, this->hog.winSize.height));
                this->hog.compute(window, dsc);
                descriptors.push_back(dsc);
            }
        }
        //We do not want to iterate more if we apply a single scale detection
        if(!multiscale)
            break;
        //Otherwise we pass to the next scale
        nbDownscale++;
    }
    return descriptors;

}

klmFilter BgsubTrack::initFilter(const Rect & boundingBox, const int & nbFrame)
{
    //KalmanFilter kf(8, 6, 0, CV_32F);

    // state\meas  x    y   xk-1    yk-1    vx  vy  ax  ay  w   h
    // x   1                        dt      @               // @ = 0.5*ax(k-1)*dt²
    // y        1                       dt      @
    // xk-1         1
    // yk-1                 1
    // vx                           1
    // vy                                1
    // ax                                   1
    // ay                                       1
    // w                                            1
    // h                                                1
    KalmanFilter kf(10, 10, 0, CV_32F);
    // model the system's dynamics : x(k) = transitionMatrix * x(k-1)
    kf.transitionMatrix.at<float>(0, 0) = 1; // x(k) = x(k-1) *
    kf.transitionMatrix.at<float>(1, 1) = 1;
    kf.transitionMatrix.at<float>(2, 2) = 1;
    kf.transitionMatrix.at<float>(3, 3) = 1;
    kf.transitionMatrix.at<float>(4, 4) = 1;
    kf.transitionMatrix.at<float>(5, 5) = 1;
    kf.transitionMatrix.at<float>(6, 6) = 1;
    kf.transitionMatrix.at<float>(7, 7) = 1;
    kf.transitionMatrix.at<float>(8, 8) = 1;
    kf.transitionMatrix.at<float>(9, 9) = 1;
    kf.transitionMatrix.at<float>(0, 4) = 1/25.; // * x(k) = x(k-1) + dt*vx(k-1) **
    kf.transitionMatrix.at<float>(1, 5) = 1/25.;
    kf.transitionMatrix.at<float>(0, 6) = 0.5*pow(1/25., 2); // ** x(k) = x(k-1) + dt*vx(k-1) + 0.5*dt²*ax(k-1)
    kf.transitionMatrix.at<float>(1, 7) = 0.5*pow(1/25., 2);

    kf.measurementMatrix = Mat::zeros(10, 10, CV_32F);
    setIdentity(kf.measurementMatrix);

    kf.processNoiseCov.at<float>(0,0) = 1.5f; // Error x
    kf.processNoiseCov.at<float>(1,1) = 2.0f; // Error y
    kf.processNoiseCov.at<float>(2,2) = 1e-2; // Error xk-1
    kf.processNoiseCov.at<float>(3,3) = 1e-2; // Error yk-1
    kf.processNoiseCov.at<float>(4,4) = 1.0f; // Error vx
    kf.processNoiseCov.at<float>(5,5) = 1.0f; // Error vy
    kf.processNoiseCov.at<float>(6,6) = 1.0f; // Error ax
    kf.processNoiseCov.at<float>(7,7) = 1.0f; // Error ay
    kf.processNoiseCov.at<float>(8,8) = 1e-2; // Error w
    kf.processNoiseCov.at<float>(9,9) = 1e-2; // Error h

    setIdentity(kf.measurementNoiseCov, Scalar(1e-1));
    setIdentity(kf.errorCovPre);

    //When a new person is detected for the first time the measure = current state
    klmFilter klmf(kf, boundingBox);
    klmf.meas.at<float>(0) = boundingBox.x + boundingBox.width/2;
    klmf.meas.at<float>(1) = boundingBox.y + boundingBox.height/2;
    klmf.meas.at<float>(8) = (float) boundingBox.width;
    klmf.meas.at<float>(9) = (float) boundingBox.height;

    //Sets the current state with the measure
    klmf.state.at<float>(0) = klmf.meas.at<float>(0);
    klmf.state.at<float>(1) = klmf.meas.at<float>(1);
    klmf.state.at<float>(8) = klmf.meas.at<float>(8);
    klmf.state.at<float>(9) = klmf.meas.at<float>(9);

    klmf.kf.statePost = klmf.state;

    klmf.found = true;
    klmf.numFrame = nbFrame;

    return klmf;
}

void BgsubTrack::updateTrackers(const vector<dt> & posDetections, vector<colorHistTracker> * trackerList, const Mat & frame, const int & nframe,
                                const Ptr<ml::SVM> & svm, vector<colorHistTracker> * significantTrackers)
{
    for(int i=trackerList->size() -1; i >= 0; i--)
    {
        //cout << "TRACKER" << to_string(i) << endl;
        //cout << "noMeasure count = " << (*trackerList)[i].noMeasureCount << endl;

        // get the area of already tracked region
        int x1_tl = (*trackerList)[i].bbox.x;
        int y1_tl = (*trackerList)[i].bbox.y;
        int x1_br = (*trackerList)[i].bbox.x + (*trackerList)[i].bbox.width;
        int y1_br = (*trackerList)[i].bbox.y + (*trackerList)[i].bbox.height;
        int area_1 = (*trackerList)[i].bbox.width * (*trackerList)[i].bbox.height;

        bool flag = false;
        //for each of the new detections found by head-top detector
        for(int j=0; j < posDetections.size(); j++)
        {
            // gets the area of the detected region
            int x2_tl = posDetections[j].boundingBox.x;
            int y2_tl = posDetections[j].boundingBox.y;
            int x2_br = posDetections[j].boundingBox.x + posDetections[j].boundingBox.width;
            int y2_br = posDetections[j].boundingBox.y + posDetections[j].boundingBox.height;
            int area_2 = posDetections[j].boundingBox.width * posDetections[j].boundingBox.height;

            // Calculates overlapping area
            int x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl));
            int y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl));
            int overlap_area = x_overlap * y_overlap;
            int total_area = area_1 + area_2 - overlap_area;
            // Note : increase the jaccard coefficient threshold increases the accuracy of each tracker but can cause the deletion of "young" good tracks
            if(overlap_area / (float) total_area > 0.5)
            {
                //cout << "overlapping found" << endl;
                //imshow("measure", frame(posDetections[j].boundingBox));

                Mat b_hist, g_hist, r_hist;
                //calculates the distance between tracker and detection color histograms
                double distance = this->getHistDistance(frame(posDetections[j].boundingBox).clone(), &b_hist, &g_hist, &r_hist, (*trackerList)[i].histogram);

                // gets the mean distance between color histogram for the tracked person i
                double mean_distance=0;
                for(int k=0; k < (*trackerList)[i].distances.size(); k++)
                    mean_distance += (*trackerList)[i].distances[k];

                if((*trackerList)[i].distances.size() >=1)
                    mean_distance = mean_distance / (*trackerList)[i].distances.size();

                if(distance == 0 || ((*trackerList)[i].distances.size() >= 1 && (*trackerList)[i].distances[(*trackerList)[i].distances.size()-1] == 0 ) ||
                   distance <= 1.5*mean_distance || ((*trackerList)[i].distances.size() == 0))
                {
                    (*trackerList)[i].distances.push_back(distance);
                    (*trackerList)[i].histogram.clear();
                    (*trackerList)[i].histogram.push_back(b_hist);
                    (*trackerList)[i].histogram.push_back(g_hist);
                    (*trackerList)[i].histogram.push_back(r_hist);
                    (*trackerList)[i].bbox = posDetections[j].boundingBox;
                    (*trackerList)[i].locations.push_back(posDetections[j].boundingBox);
                    (*trackerList)[i].numFrame.push_back(nframe);
                    (*trackerList)[i].noMeasureCount = 0;
                    flag = true;                    
                    break;
                }
            }
        }
        // if no overlapping with detections has been found this can be due to missing detection or new person
        if(! flag)
        {
            // if we tracked atleast 2 locations of the same person then we can estimate where the person will be in the next frame
            if((*trackerList)[i].locations.size() >= 2)
            {
                int sizeLocations = (*trackerList)[i].locations.size();
                // calculates traveled distance between last and penultimate measures
                int dx = (*trackerList)[i].locations[sizeLocations-1].x - (*trackerList)[i].locations[sizeLocations-2].x;
                int dy = (*trackerList)[i].locations[sizeLocations-1].y - (*trackerList)[i].locations[sizeLocations-2].y;

                // doing the hypothesis that speed is constant between 2 frames and the direction remains the same as before, we can predict new x and y
                int predict_x = (*trackerList)[i].locations[sizeLocations-1].x + dx;
                int predict_y = (*trackerList)[i].locations[sizeLocations-1].y + dy;

                //try to extract a larger roi (-20p TL corner, +20 BR corner)
                int roi_x= predict_x-20;
                int roi_y = predict_y-20;
                int width = 96, height = 96;

                //avoid out of bounds errors
                if(roi_x < 0)
                    roi_x = 0;
                if(roi_x + width > frame.cols)
                    width = 96 - (roi_x+96 - frame.cols);
                if(roi_y < 0)
                    roi_y = 0;
                if(roi_y + height > frame.rows)
                    height = 96 - (roi_y+96 - frame.rows);

                // resizes the predicted region so the sliding window can run without problem
                vector<Mat> region;
                region.push_back(frame(Rect(roi_x, roi_y, width, height)).clone());
                this->resize(& region);

                vector<Rect2d> detectedRoi;
                vector<vector<float> > descriptors;
                // calculates HOG descriptor for each detection window
                descriptors = this->slidingWindow(region[0], & detectedRoi);

                vector<dt> redetections;
                for(int k=0; k < descriptors.size(); k++)
                {
                    //try to redetect the head in the predicted window
                    int prediction = svm->predict(descriptors[k]);
                    Mat conf;
                    svm->predict(descriptors[k], conf, StatModel::RAW_OUTPUT);
                    if(prediction != 0)
                    {
                        Rect accurateRect(roi_x + detectedRoi[k].x, roi_y + detectedRoi[k].y,
                                          detectedRoi[k].width, detectedRoi[k].height);

                        redetections.push_back(dt(accurateRect, conf.at<float>(0,0)));
                    }
                }
                sort(redetections.begin(), redetections.end(), less_than_confidence());
                Mat b_hist, g_hist, r_hist;

                bool flagNoRedetect = false;
                if(redetections.size() > 0)
                {
                    //we only use the first redetection since we are expecting only 1 head in the predicted window, so we take the one that
                    //produces the highest classifier confidence
                    if(redetections[0].boundingBox.x + redetections[0].boundingBox.width > frame.cols)
                        redetections[0].boundingBox.width = frame.cols - redetections[0].boundingBox.x ;
                    if(redetections[0].boundingBox.y + redetections[0].boundingBox.height > frame.rows)
                        redetections[0].boundingBox.height = frame.rows - redetections[0].boundingBox.y;

                    double distance = getHistDistance(frame(redetections[0].boundingBox).clone(), &b_hist, &g_hist, &r_hist, (*trackerList)[i].histogram);
                    double mean_distance=0;
                    for(int m=0; m < (*trackerList)[i].distances.size(); m++)
                        mean_distance += (*trackerList)[i].distances[m];

                    if((*trackerList)[i].distances.size() >=1)
                        mean_distance = mean_distance / (*trackerList)[i].distances.size();

                    if(distance == 0 || (*trackerList)[i].distances[(*trackerList)[i].distances.size()-1] == 0 ||
                       distance <= 1.5*mean_distance)
                    {
                        (*trackerList)[i].distances.push_back(distance);
                        (*trackerList)[i].histogram.clear();
                        (*trackerList)[i].histogram.push_back(b_hist);
                        (*trackerList)[i].histogram.push_back(g_hist);
                        (*trackerList)[i].histogram.push_back(r_hist);
                        (*trackerList)[i].bbox = redetections[0].boundingBox;
                        (*trackerList)[i].locations.push_back(redetections[0].boundingBox);
                        (*trackerList)[i].numFrame.push_back(nframe);
                        (*trackerList)[i].noMeasureCount = 0;
                        flag = true;
                    }
                }
                else
                {
                    //if we cannot find the target for 10 frames then we assume it was a false detection and remove the tracker
                    (*trackerList)[i].noMeasureCount ++ ;
                    flagNoRedetect = true;
                    if((*trackerList)[i].noMeasureCount == 5)
                    {
                        // if the track length is greater than 20 frames then it had a high chance to be a person
                        if((*trackerList)[i].locations.size() > 20)
                            significantTrackers->push_back((*trackerList)[i]);
                        trackerList->erase(trackerList->begin() +i);
                        //cout << "erase tracker" << to_string(i) << endl;
                    }
                }

                if(!flag && !flagNoRedetect)
                {
                    (*trackerList)[i].noMeasureCount++;
                    if((*trackerList)[i].noMeasureCount == 5)
                    {
                        if((*trackerList)[i].locations.size() > 20)
                            significantTrackers->push_back((*trackerList)[i]);
                        trackerList->erase(trackerList->begin() +i);
                        //cout << "erase tracker" << to_string(i) << endl;
                    }
                }
            }
            else
            {
                //still increases the no measure count because new tracked regions that does not generate atleast 2 consecutive measures are
                //most likely false detections
                (*trackerList)[i].noMeasureCount++;
                if((*trackerList)[i].noMeasureCount == 5)
                {
                    if((*trackerList)[i].locations.size() > 20)
                        significantTrackers->push_back((*trackerList)[i]);
                    trackerList->erase(trackerList->begin() +i);
                    //cout << "erase tracker" << to_string(i) << endl;
                }
            }
        }
        //waitKey(-1);
    }
    //cout << endl;
}

double BgsubTrack::getHistDistance(const Mat & roi, Mat * b_hist, Mat * g_hist, Mat * r_hist, vector<Mat> histogram)
{
    vector<Mat> bgr_planes;
    split(roi, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float * histRange = {range};

    // calculate histogram for each channel
    cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), (*b_hist), 1, &histSize, &histRange, true, false);
    cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), (*g_hist), 1, &histSize, &histRange, true, false);
    cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), (*r_hist), 1, &histSize, &histRange, true, false);

    double distance = 0;
    distance += compareHist((*b_hist), histogram[0], HISTCMP_HELLINGER);
    distance += compareHist((*g_hist), histogram[1], HISTCMP_HELLINGER);
    distance += compareHist((*r_hist), histogram[2], HISTCMP_HELLINGER);
    distance = distance / 3.;

    return distance;
}


int main( int argc, char** argv )
{
    // background subtraction
    Mat frame, fgmask;
    Ptr<BackgroundSubtractor> gaussMix = createBackgroundSubtractorMOG2();

    //SVM + HOG
    HOGDescriptor hogD(Size(56,56), Size(16,16), Size(8,8), Size(8,8), 9);
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm = ml::SVM::load(argv[2]);

    BgsubTrack bst(hogD, gaussMix);

    // Trackers
    vector<colorHistTracker> trackerList;
    vector<colorHistTracker> significantTrackers;

    // Video
    VideoCapture capture(argv[1]);
    if(!capture.isOpened())
    {
        cout << "Cannot open video file" << endl;
        exit(-1);
    }

    int nbFrame = 0;
    Mat currentFrame, previousFrame;
    // iterates until the last frame of the video
    while(1)
    {
        // reads next frame
        if(!capture.read(frame))
        {
            //cout << "Cannot read next frame " << endl;
            break;
        }

        currentFrame = frame.clone();

        //Applies background subtraction for the current frame and update weights of each pixel
        //bst.bgsub->apply(frame, fgmask); // BOSS dataset : add learning rate 0.001
        Mat diff_mask;
        if(currentFrame.size() != Size(0,0) && previousFrame.size() != Size(0,0))
        {
            absdiff(previousFrame, currentFrame, diff_mask);
            cvtColor(diff_mask, diff_mask, COLOR_BGR2GRAY);
            threshold(diff_mask, diff_mask, 30, 255, CV_THRESH_BINARY);
        }

        //Mat thresholded;
        vector<vector<Point> > contours;
        vector<cv::Vec4i> hierarchy;

        // binarizes image to extract contours
        //Mat opened;
        //threshold(fgmask, thresholded, 150, 255, CV_THRESH_BINARY); // BOSS dataset 90 / subway 150
        //Applies opening on the binarized image to remove small artefacts and to try to split regions that should not be linked (shadows etc)
        //morphologyEx(thresholded, opened, MORPH_OPEN, getStructuringElement(MORPH_CROSS, Size(5, 5))); // (9,9) BOSS dataset
        //findContours(opened, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        findContours(diff_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        Mat contoursImg = Mat::zeros(frame.size(), CV_8UC1);
        vector<RotatedRect> rectangles;
        //extract ROI from each contour
        for(int i=0; i < (int)contours.size(); i++)
        {
            rectangles.push_back(minAreaRect(contours[i]));
#ifdef VIZ
            for(int j=0; j < (int)contours[i].size(); j++)
                contoursImg.at<unsigned char>(contours[i][j].y, contours[i][j].x) = 255;
#endif
        }

        // keeps only ROI that are big enough to likely be human
        vector<Mat> regions; // small images that will be used to feed the classifier
        vector<Rect2d> boundingLocations; // need this to locate the region to track in the whole image
        bst.roiSelection(rectangles, &regions, &boundingLocations, frame);

#ifdef VIZ
        Mat bboxes;
        bboxes = frame.clone();
        for(int i=0; i < boundingLocations.size(); i++)
            rectangle(bboxes, boundingLocations[i], Scalar(0, 0, 255));
#endif

        //resize regions to avoid sliding window OOB problems
        if(regions.size() > 0)
            bst.resize(&regions);

        Mat true_detections = frame.clone();

        vector<dt> posDetections;
        // iterates over all selected regions in order to feed them to the classifier
        for(int i=0; i < (int)regions.size(); i++)
        {
            // gets all hog descriptors that it is possible to calculate in the given region of interest
            vector<vector<float> > descriptors;
            vector<Rect2d> * outputROI = new vector<Rect2d>();
            descriptors = bst.slidingWindow(regions[i], outputROI); // single scale detection
            for(int j = 0; j < (int)descriptors.size(); j++)
            {
                // asks the classifier whether it belongs to the positive class
                int prediction = svm->predict(descriptors[j]);
                Mat conf;
                svm->predict(descriptors[j], conf, StatModel::RAW_OUTPUT);

                if(prediction != 0) // != 0 subway dataset, ==1 BOSS dataset
                {
                    //relocates the head in the global image (frame)
                    Rect accurateRect(boundingLocations[i].x + (*outputROI)[j].x, boundingLocations[i].y + (*outputROI)[j].y,
                                      (*outputROI)[j].width, (*outputROI)[j].height);
                    posDetections.push_back(dt(accurateRect, conf.at<float>(0,0)));

                    //Ouput detections results (CSV format)
                    //cout << nbFrame << ", 1, " << accurateRect.x << ", " << accurateRect.y << ", " << accurateRect.width << ", " << accurateRect.height << endl;
#ifdef VIZ
                    if(accurateRect.x + accurateRect.width > frame.cols)
                        accurateRect.width = frame.cols - accurateRect.x;
                    if(accurateRect.y + accurateRect.height > frame.rows)
                        accurateRect.height = frame.rows - accurateRect.y;

                    rectangle(true_detections, accurateRect, Scalar(0, 0, 255));
#endif
                }
                /*else
                    cout << nbFrame << ", 0, " << (*outputROI)[j].x + boundingLocations[i].x << ", " << (*outputROI)[j].y + boundingLocations[i].y
                         << (*outputROI)[j].width << ", " << (*outputROI)[j].height << endl;*/
            }
        }

        //sorts detections, highest confidence first
        sort(posDetections.begin(), posDetections.end(), less_than_confidence());
        // Non Maxima suppression on the detections
        bst.nms(&posDetections);

#ifdef VIZ
        Mat highestconf = frame.clone();
        for(int i=0; i < posDetections.size(); i++)
            rectangle(highestconf, posDetections[i].boundingBox, Scalar(0, 0, 255));
        imshow("after nms", highestconf);
#endif

//==========================
// color histogram tracking |
//==========================
        //cout << "frame n°" << nbFrame << endl;
        bst.updateTrackers(posDetections, &trackerList, frame, nbFrame, svm, & significantTrackers);
        bst.addNewTrackers(posDetections, &trackerList, frame, nbFrame);

        Mat trackingResult = frame.clone();
        for(int i=0; i < trackerList.size(); i++)
        {
            rectangle(trackingResult, trackerList[i].bbox, Scalar(0, 0, 255));
        }

        previousFrame = currentFrame.clone();
#ifdef VIZ
        //imshow("original frame", frame);
        //imshow("opened", opened);
        //imshow("contours", contoursImg);
        //imshow("foreground", fgmask);
        //imshow("thresholded", thresholded);
        //imshow("bboxes", bboxes);
        //imshow("true_detections", true_detections);
        imshow("tracking res", trackingResult);
        //imshow("all detections", all_detections);
        waitKey(20);
#endif

        nbFrame++;
    }
    //cout << "significant trackers size = " << significantTrackers.size() << endl;

    // ouput tracking results (CSV format)
    for(int i=0; i < significantTrackers.size(); i++)
    {
        for(int j=0; j < significantTrackers[i].numFrame.size(); j++)
            cout << "head" << to_string(i) << ", " << significantTrackers[i].numFrame[j] << ", " <<
                    significantTrackers[i].locations[j].x + significantTrackers[i].locations[j].width / 2 << ", " <<
                    significantTrackers[i].locations[j].y + significantTrackers[i].locations[j].height / 2 << endl;
    }
}

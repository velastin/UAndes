/*!
 * \file BgsubTrack.hpp
 * \brief BgsubTrack class to apply background subtraction and select blobs
 * \version 1.0
 */

#ifndef BGSUB_TRACK_H
#define BGSUB_TRACK_H

//usage of timestamp : (t1 - t0) / 1000000.0L with t0 and t1 initialized with get_timestamp()
typedef unsigned long long timestamp_t;
static timestamp_t
get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

struct dt
{
    cv::Rect boundingBox;
    float confidence;

    dt(const cv::Rect & bb, const float & c):boundingBox(bb), confidence(c) {}
}typedef dt;


// struct used to sort detections by confidence (highest confidence first in the case of positive class gives negative confidence ...)
struct less_than_confidence
{
    inline bool operator() (const dt& struct1, const dt& struct2)
    {
        return (struct1.confidence < struct2.confidence);
    }
};

struct klmFilter
{
    cv::Rect bbox;
    cv::Mat state = cv::Mat::zeros(10, 1, CV_32F);
    cv::Mat meas = cv::Mat::zeros(10, 1, CV_32F);
    cv::KalmanFilter kf;
    bool found = false;
    int notFoundCount = 0;
    int numFrame = 0, numFrame2; // used to determine the time between the current measure and respectively that last one and the penultimate one
    int lastMeasureX=0, lastMeasureY = 0, lastMeasureX2 = 0, lastMeasureY2 = 0;

    klmFilter(const cv::KalmanFilter & k, const cv::Rect & bb):kf(k), bbox(bb){}
}typedef klmFilter;

struct colorHistTracker
{
    std::vector<cv::Rect> locations; // all the locations of the same person
    std::vector<double> distances; // all the distances between last histogram and current frame histogram
    std::vector<cv::Mat> histogram; // last BGR histogram
    cv::Rect bbox;
    std::vector<int> numFrame; // used to know in which frame was found the person (for calculating metrics)
    int noMeasureCount = 0, id;

    colorHistTracker(const cv::Rect & b, const cv::Mat & frame, const int & nf, const int & tracker_id):bbox(b), id(tracker_id)
    {
        locations.push_back(b);
        numFrame.push_back(nf);

        // split the image into its 3 channels
        std::vector<cv::Mat> bgr_planes;
        cv::split(frame(b).clone(), bgr_planes);

        int histSize = 256;
        float range[] = {0, 256};
        const float * histRange = {range};

        // calculate histogram for each channel
        cv::Mat b_hist, g_hist, r_hist;
        cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, true, false);
        cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, true, false);
        cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, true, false);

        histogram.push_back(b_hist);
        histogram.push_back(g_hist);
        histogram.push_back(r_hist);
    }
};

class BgsubTrack
{

public:

    cv::HOGDescriptor hog;
    cv::Ptr<cv::BackgroundSubtractor> bgsub;

    /**
     * @brief BgsubTrack : constructor
     * @param hogD : hog descriptor object used to train the classifier
     * @param gaussMix : pointer to a background subtractor object
     */
    BgsubTrack(const cv::HOGDescriptor & hogD, const cv::Ptr<cv::BackgroundSubtractor> & gaussMix);

    /**
     * @brief addNewTrackers : instanciate a new tracker for each detection that does not overlap and existing tracked region
     * @param posDetections : input new blob detections in the current frame
     * @param trackerList : input/output vector of trackers
     * @param frame : current frame
     * @param nframe : number of the current frame
     * @param id : current id available for new trackers
     */
    void addNewTrackers(const std::vector<dt> & posDetections, std::vector<colorHistTracker> * trackerList, const cv::Mat & frame, const int & nframe, int * id);

    /**
     * @brief nms : apply Non Maxima Suppression on the detections
     * @param posDetections : input / ouput list of positively classified samples, sorted by highest confidence first
     */
    void nms(std::vector<dt> * posDetections);

    /**
     * @brief roiSelection : select regions of interest base on size criterion
     * @param rectangles : input vector of rectangles extracted from blob detection
     * @param regions : output vector of selected small images
     * @param boundingLocations : output vector of rectangles. Each rectangle is equivalent to the region at same index in "regions" vector
     * @param frame : frame from which were extracted the rectangles
     */
    void roiSelection(const std::vector<cv::RotatedRect> & rectangles, std::vector<cv::Mat> * regions, std::vector<cv::Rect2d> * boundingLocations,
                      const cv::Mat & frame);

    /**
     * @brief resize : resize images so they are suitable for the following sliding window process
     * @param regions : input/ouput vector of small images
     */
    void resize(std::vector<cv::Mat> * regions);

    /**
     * @brief slidingWindow : apply the sliding window on the given image and return vector of descriptors
     * @param image : input image on which we want to extract descriptors
     * @param roi : output vector of windows (used to display results after prediction)
     * @param multiscale : input boolean to decide if we have to downscale image multiple times to detect (true when the scene contains same object with different scales)
     * @param scaleFactor : input downscale factor, used to resize image when multiscale is set to true;
     * @return : vector of descriptors
     */
    std::vector<std::vector<float> > slidingWindow(cv::Mat image, std::vector<cv::Rect2d> * roi, const bool & multiscale=false, const float & scaleFactor=0.0);

    /**
     * @brief initFilter : instanciate a Kalman filter with hard coded parameters
     * @param boundingBox : input measure of spatial location of the object to track
     * @param nbFrame : numero of the frame in which the object to tracker was detected
     * @return : intialized Kalman filter
     */
    klmFilter initFilter(const cv::Rect & boundingBox, const int & nbFrame);

    /**
     * @brief updateTrackers : try to update the new position of all instanciated trackers using new detections and jaccard coefficient
     * @param posDetections : input vector of detections
     * @param trackerList : input/output vector of trackers
     * @param frame : input current image
     * @param nframe : input number of the current frame
     * @param svm : input loaded svm model
     * @param significantTrackers : output vector that stores all tracks that lasted more than 20 frames
     */
    void updateTrackers(const std::vector<dt> & posDetections, std::vector<colorHistTracker> * trackerList, const cv::Mat & frame, const int & nframe,
                               const cv::Ptr<cv::ml::SVM> & svm, std::vector<colorHistTracker> * significantTrackers);


    /**
     * @brief getHistDistance : calculates the distance between a given image and a reference histogram
     * @param roi : input image to be compared
     * @param b_hist : output blue histogram of the roi
     * @param g_hist : output green histogram of the roi
     * @param r_hist : output red histogram of the roi
     * @param histogram : reference histogram
     * @return : distance using Hellinger histogram comparison
     */
    double getHistDistance(const cv::Mat & roi, cv::Mat * b_hist, cv::Mat * g_hist, cv::Mat * r_hist, std::vector<cv::Mat> histogram);


    /**
     * @brief replayTracks : display retained tracks
     * @param videoPath : path to the source video
     * @param significantTrackers : list of retained trackers
     */
    void replayTracks(const std::string &videoPath, const std::vector<colorHistTracker> & significantTrackers);

    /**
     * @brief fuseTrackers : try to recover from track fragmentation by re-identifying same target tracked twice
     * @param significantTrackers : input vector of all tracks that lasted more than 20 frames
     * @param listFrames : input vector of all frames, used to calculate histogram distances
     * @return : vector of fused tracks (output vector size <= input vector size)
     */
    std::vector<colorHistTracker> fuseTrackers(std::vector<colorHistTracker> significantTrackers, std::vector<cv::Mat> listFrames);
};


#endif // BGSUB_TRACK_H

/*!
 * \file TrainSVM.hpp
 * \brief TrainSVM class to read training sample and train an SVM model
 * \version 1.0
 */

// OpenCV 3.2
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <dirent.h> // only available for unix systems (translations exist for Windows)



#ifndef TRAIN_SVM_H
#define TRAIN_SVM_H

class TrainSVM
{

public :
    cv::Ptr<cv::ml::SVM> svm;
    cv::HOGDescriptor hog;

    /**
     * @brief TrainSVM : Constructor
     * @param C : C parameter of the linear classifier
     * @param winSize : window size for the hog descriptor
     * @param blockSize : size of a block in pixel
     * @param blockStride : step between two blocks
     * @param cellSize : size of cells in pixel
     * @param nBins : number of gradient orientations
     */
    TrainSVM(float C,const cv::Size & winSize, const cv::Size & blockSize, const cv::Size & blockStride, const cv::Size & cellSize, int nBins);

    /**
     * @brief readSamples : read positive and negative samples
     * @param posPath : path to positive samples
     * @param negPath : path to negative samples
     * @param posFileNames : output vector of positive file names
     * @param negFileNames : output vector of negative file names
     */
    void readSamples(const std::string & posPath, const std::string & negPath, std::vector<std::string> * posFileNames,
                     std::vector<std::string> * negFileNames);

    /**
     * @brief describe : apply description of files given
     * @param posFileNames : pointer to a string vector that contains names of positive samples
     * @param negFileNames : pointer to a string vector that contains names of negative samples
     * @param descriptors : output vector of descriptors
     * @param labels : output vector of labels
     * @param resize : boolean to decide wether we need to resize the samples (default false)
     * @param newSize : the new Size we have to use in case of resize=true
     */
    void describe(std::vector<std::string> * posFileNames, std::vector<std::string> * negFileNames, std::vector<std::vector<float> > *descriptors,
                  std::vector<int> *labels, bool rsz=false, const cv::Size & newSize=cv::Size(0,0));

    /**
     * @brief fit : train the classifier to find the best hyperplane and save it
     * @param descriptors : pointer to a vector of float descriptors
     * @param labels : pointer to a vector of class labels
     * @param outPath : path to save the trained model (must contain the file name)
     */
    void fit(std::vector<std::vector<float> > * descriptors, std::vector<int> * labels, const std::string & outPath);

};

#endif // TRAIN_SVM_H

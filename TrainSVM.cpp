#include "TrainSVM.hpp"

using namespace cv;
using namespace std;


TrainSVM::TrainSVM(float C, const Size & winSize, const Size & blockSize, const Size & blockStride, const Size & cellSize, int nBins)
{
    this->svm = ml::SVM::create();
    this->svm->setType(ml::SVM::C_SVC); // binary classifier
    this->svm->setKernel(ml::SVM::LINEAR); // Linear
    this->svm->setC(C);
    this->hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);
}

void TrainSVM::readSamples(const string & posPath, const string & negPath, vector<string> * posFileNames, vector<string> * negFileNames)
{
    // read all positive and negative samples
    DIR *dp;
    struct dirent *dirp;

    // try to open directory
    if((dp = opendir(posPath.c_str())) == NULL)
    {
        cout << "Cannot open directory : " << posPath << endl;
        exit(-1);
    }
    // reads files one by one and stores it in the output vector
    while((dirp = readdir(dp)) != NULL)
    {
        DIR * dp2;
        struct dirent *dirp2;

        // look for a subdirectory level under posPath
        if(string(dirp->d_name).compare(".") == 0 || string(dirp->d_name).compare("..") == 0) // skips "." and ".." directory that are listed by readdir
        {
            continue;
        }

        if((dp2 = opendir((posPath + dirp->d_name).c_str())) != NULL)
        {
            while((dirp2 = readdir(dp2)) != NULL)
            {
                if(string(dirp2->d_name).compare(".") != 0 && string(dirp2->d_name).compare("..") != 0)
                    posFileNames->push_back(posPath + string(dirp->d_name) + "/" + string(dirp2->d_name));
            }
        }
        else
        {
            posFileNames->push_back(posPath + string(dirp->d_name));
        }
    }
    closedir(dp);


    // same thing for the negative samples
    if((dp = opendir(negPath.c_str())) == NULL)
    {
        cout << "Cannot open directory : " << negPath << endl;
        exit(-1);
    }

    while((dirp = readdir(dp)) != NULL)
    {
        DIR * dp2;
        struct dirent * dirp2;

        if(string(dirp->d_name).compare(".") == 0 || string(dirp->d_name).compare("..") == 0)
            continue;

        if((dp2 = opendir((negPath + dirp->d_name).c_str())) != NULL)
        {
            while((dirp2 = readdir(dp2)) != NULL)
            {
                if(string(dirp2->d_name).compare(".") != 0 && string(dirp2->d_name).compare("..") != 0)
                    negFileNames->push_back(negPath + string(dirp->d_name) + "/" + string(dirp2->d_name));
            }
        }
        else
        {
            negFileNames->push_back(negPath + string(dirp->d_name));
        }
    }
    closedir(dp);

    cout << "Finished reading all training samples " << endl;
}


void TrainSVM::describe(vector<string> * posFileNames, vector<string> * negFileNames, vector<vector<float> > * descriptors, vector<int> * labels,
                        bool rsz, const Size & newSize)
{
    bool first=true;
    for(int i=0; i < posFileNames->size(); i++)
    {
        vector<float> dsc;
        Mat image = imread((*posFileNames)[i], cv::IMREAD_GRAYSCALE);
        if(image.size() == Size(0,0))
        {
            cout << "Could not read image file " << endl;
            exit(-1);
        }
        if(rsz)
        {
            if(newSize != Size(0,0))
                resize(image, image, newSize);
            else
                cout << "Cannot resize the image when the new size is not specified" << endl;
        }

        // computes HOG descriptor
        this->hog.compute(image, dsc);
        descriptors->push_back(dsc);
        labels->push_back(1);


        for(int j=0; j < dsc.size(); j++)
        {
            if(first)
            {
                cout << dsc[j];
                first = false;
            }
            else
                cout << ", " << dsc[j];
        }
    }
    //cout << "Processed all positive descriptors " << endl;

    for(int i=0; i < negFileNames->size(); i++)
    {
        vector<float> dsc;
        Mat image = imread((*negFileNames)[i], cv::IMREAD_GRAYSCALE);
        if(rsz)
        {
            if(newSize != Size(0,0))
                resize(image, image, newSize);
            else
                cout << "Cannot resize the image when the new size is not specified" << endl;
        }

        this->hog.compute(image, dsc);
        descriptors->push_back(dsc);
        labels->push_back(0);

        for(int j=0; j < dsc.size(); j++)
        {
            if(first)
            {
                cout << dsc[j];
                first = false;
            }
            else
                cout << ", " << dsc[j];
        }
    }

    //cout << "Processed all negative descriptors" << endl;
}



void TrainSVM::fit(vector<vector<float> > * descriptors, vector<int> * labels, const string & outPath)
{
    // converts vectors to matrices since "train" method needs matrices
    Mat trainingData = Mat::zeros(descriptors->size(), (*descriptors)[0].size(), CV_32FC1);
    Mat lb = Mat::zeros(labels->size(), 1, CV_32S);

    for(int i=0; i < descriptors-> size(); i++)
    {
        lb.at<float>(i, 0) = (*labels)[i];
        for(int j=0; j < (*descriptors)[i].size(); j++)
            trainingData.at<float>(i, j) = (*descriptors)[i][j];
    }

    cout << "Training classifier ..." << endl;
    this->svm->train(trainingData, ml::ROW_SAMPLE, lb);
    this->svm->save(outPath);

    cout << "Finished training classifier " << endl;
}


int main( int argc, char** argv )
{
    // Training
    TrainSVM train(0.002, Size(56,56), Size(16,16), Size(8,8), Size(8,8), 9);
    TrainSVM t(train);

    vector<string> * posFileNames = new vector<string>();
    vector<string> * negFileNames = new vector<string>();
    vector<vector<float> > * descriptors = new vector<vector<float> >();
    vector<int> * labels = new vector<int>();

    train.readSamples("/home/mathieu/STAGE/underground_dataset/pos/train/", "/home/mathieu/STAGE/underground_dataset/neg/test/", posFileNames, negFileNames);
    train.describe(posFileNames, negFileNames, descriptors, labels, true, Size(56,56));

    cout << "descriptors size = " << descriptors->size() << endl;
    cout << "labels size = " << labels->size() << endl;
    train.fit(descriptors, labels, "/tmp/test_model.xml");


    //VideoCapture cap("/home/mathieu/STAGE/Videos/Cell_phone_Spanish.Cam1.avi");
/*    VideoCapture cap("/home/mathieu/STAGE/underground_dataset/pos/test/A_d800mm_R6.mpg");
    if(!cap.isOpened())
    {
        cout << "Cannot read video file " << endl;
        exit(-1);
    }
*/
    // Test
/*    while(1)
    {
        Mat testImg;
        if(! cap.read(testImg))
        {
            cout << "Cannot read frame" << endl;
            exit(-1);
        }

        Ptr<ml::SVM> svm = ml::SVM::create();
        //svm = ml::SVM::load("/home/mathieu/STAGE/underground_dataset/results/models/openCV_model.xml");
        svm = ml::SVM::load("/home/mathieu/STAGE/SVM/LinearSergio/Cell_phone_Spanish.Cam1-SVM.xml");

        vector<vector<float> > descriptors;
        vector<Rect2d> * roi = new vector<Rect2d>();
        descriptors = train.slidingWindow(testImg, roi, true, 0.10);

        for(int i=0; i<descriptors.size(); i++)
        {
            int prediction = svm->predict(descriptors[i]);
            if(prediction > 0)
            {
                imshow("frame", testImg);
                Mat clone = testImg.clone();
                rectangle(clone, (*roi)[i], Scalar(0, 0, 255));
                imshow("true detection", clone);
                waitKey(30);
            }
        }
    }
*/

}

#include "TrainSVM.hpp"

using namespace cv;
using namespace std;

//#define HARD_TRAINING 1

TrainSVM::TrainSVM(const float & C, const Size & winSize, const Size & blockSize, const Size & blockStride, const Size & cellSize, int nBins)
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
        if(string(dirp->d_name).compare(".") == 0 || string(dirp->d_name).compare("..") == 0 || string(dirp->d_name).find(".mpg") != string::npos) // skips "." and ".." directory that are listed by readdir
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

        if(string(dirp->d_name).compare(".") == 0 || string(dirp->d_name).compare("..") == 0 || string(dirp->d_name).find(".mpg") != string::npos)
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

    cout << "Finished reading all samples " << endl;
}


void TrainSVM::describe(vector<string> * posFileNames, vector<string> * negFileNames, vector<vector<float> > * descriptors, vector<int> * labels,
                        bool rsz, const Size & newSize)
{
    for(int i=0; i < posFileNames->size(); i++)
    {
        vector<float> dsc;
        Mat image = imread((*posFileNames)[i], cv::IMREAD_GRAYSCALE);
        if(image.size() == Size(0,0))
        {
            cout << "Could not read image file : " << (*posFileNames)[i] << endl;
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
    }
    cout << "Processed all positive descriptors " << endl;

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
    }

    cout << "Processed all negative descriptors" << endl;
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

void TrainSVM::testModel(vector<string> * posTestFileNames, vector<string> * negTestFileNames, const Ptr<ml::SVM> & model, vector<int> * hardLabels,
                         vector<vector<float> > * hardDescriptors)
{
    cout << "testing model ..." << endl;
    vector<vector<float> > * descriptors = new vector<vector<float> >();
    vector<int> * labels = new vector<int>();
    this->describe(posTestFileNames, negTestFileNames, descriptors, labels, true, Size(56,56));

    int true_positives = 0, true_negatives = 0, false_positives=0, false_negatives=0;
    for(int i=0; i < descriptors->size(); i++)
    {
        int prediction = model->predict((*descriptors)[i]);
        if(prediction!=0 && (*labels)[i]==1)
            true_positives++;
        if(prediction!=0 && (*labels)[i]==0)
        {
            false_positives++;
            hardLabels->push_back(0);
            hardDescriptors->push_back((*descriptors)[i]);
        }
        if(prediction==0 && (*labels)[i]==0)
            true_negatives++;
        if(prediction==0 && (*labels)[i]== 1)
        {
            false_negatives++;
            hardLabels->push_back(1);
            hardDescriptors->push_back((*descriptors)[i]);
        }
    }

    cout << "precision : " << true_positives /(float)(true_positives +false_positives) << endl;
    cout << "recall : " << true_positives / (float)(true_positives + false_negatives) << endl;
}


int main( int argc, char** argv )
{
    if(argc < 6)
    {
        cout << "Need 5 arguments : " << endl;
        cout << "\t 1. Path to positive training samples " << endl;
        cout << "\t 2. Path to negative training samples " << endl;
        cout << "\t 3. Path to positive testing samples " << endl;
        cout << "\t 4. Path to negative testing samples " << endl;
        cout << "\t 5. Output path to save the trained model (must contain model's name) " << endl;
        exit(-1);
    }

    TrainSVM train(0.002, Size(56,56), Size(16,16), Size(8,8), Size(8,8), 9);
    TrainSVM t(train);

    vector<string> * posFileNames = new vector<string>();
    vector<string> * negFileNames = new vector<string>();
    vector<vector<float> > * descriptors = new vector<vector<float> >();
    vector<int> * labels = new vector<int>();

    //train model
    train.readSamples(argv[1], argv[2], posFileNames, negFileNames);
    train.describe(posFileNames, negFileNames, descriptors, labels, true, Size(56,56));

    cout << "descriptors size = " << descriptors->size() << endl;
    cout << "labels size = " << labels->size() << endl;
    train.fit(descriptors, labels, argv[5]);

    vector<string> * posTestFileNames = new vector<string>();
    vector<string> * negTestFileNames = new vector<string>();
    train.readSamples(argv[3], argv[4], posTestFileNames, negTestFileNames);

    // test model
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm = ml::SVM::load(argv[5]);
    vector<int> * hardLabels = new vector<int>();
    vector<vector<float> > * hardDescriptors = new vector<vector<float> >();
    train.testModel(posTestFileNames, negTestFileNames, svm, hardLabels, hardDescriptors);

#ifdef HARD_TRAINING
    cout << "hardDescriptors size = " << hardDescriptors->size() << endl;
    cout << "hardLabels size = " << hardLabels->size() << endl;

    // hard training. Need to retrain the entire model since openCV does not allow to fine-tune an existing model
    for(int i=0; i < hardDescriptors->size(); i++)
    {
        descriptors->push_back((*hardDescriptors)[i]);
        labels->push_back((*labels)[i]);
    }
    cout << "retrain descriptors = " << descriptors->size() << endl;
    cout << "retrain labels = " << labels->size() << endl;

    train.fit(descriptors, labels, argv[5]);

    cout << "test after hard training " << endl;
    svm = ml::SVM::load(argv[5]);
    train.testModel(posTestFileNames, negTestFileNames, svm, hardLabels, hardDescriptors);
#endif

}

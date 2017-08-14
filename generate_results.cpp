/*!
 * \file generate_results.cpp
 * \brief Standalone script to generate metrics for detection perfomance evaluation
 * \version 1.0
 */

// Standard
#include <iostream>
#include <fstream>

// Opencv 3.2
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"


using namespace std;
using namespace cv;

//#define TEST 1; // outputs a video where true positives, false positives and GT bounding boxes are displayed
//#define NMS_RES 1; // uncomment this line to generate results after Non Maxima Suppression instead of classic detection results

int main( int argc, char** argv )
{
    ifstream csv(argv[1], ios::in);
    ifstream gt(argv[2], ios::in);

    if(! (csv || gt))
    {
        cout << "Could not open one of the files given" << endl;
        exit(-1);
    }

    vector<int> frame_number;
    vector<int> classes;
    vector<Rect> detection_bboxes;
    string line;
    // reads detections created by the algorithm
    while(getline(csv, line))
    {
        string sub_str = line.substr(0, line.find(","));
        frame_number.push_back(stoi(sub_str));
        line.erase(0, line.find(",") + string(",").length());

        sub_str = line.substr(0, line.find(","));
        classes.push_back(stoi(sub_str));
        line.erase(0, line.find(",") + string(",").length());

        int x, y, width, height;
        sub_str = line.substr(0, line.find(","));
        x = stoi(sub_str);
        line.erase(0, line.find(",") + string(",").length());

        sub_str = line.substr(0, line.find(","));
        y = stoi(sub_str);
        line.erase(0, line.find(",") + string(",").length());

        sub_str = line.substr(0, line.find(","));
        width = stoi(sub_str);
        line.erase(0, line.find(",") + string(",").length());

        sub_str = line.substr(0, line.find(","));
        height = stoi(sub_str);

        detection_bboxes.push_back(Rect(x+(0.125*width), y+(0.125*height), width-(0.25*width), height-(0.25*height)));
    }

    cout << "detections bboxes size = " << detection_bboxes.size() << endl;

    // parses xgtf file by hand because c++ xml parsers seem not suitable for xgtf file, or the xgtf file is too complicated
    vector<int> frame_number_gt;
    vector<Rect> gt_bboxes;
    while(getline(gt, line))
    {
        if(line.find("data:obox") != string::npos)
        {
            string framespan1 = line.substr(line.find("framespan=")+11, line.find("height=")-line.find("framespan")-13);
            frame_number_gt.push_back(stoi(framespan1.substr(0, framespan1.find(":"))));

            string framespan2 = framespan1.substr(framespan1.find(":")+1, framespan1.find("height=")-3);

            string height = line.substr(line.find("height=")+ 8, line.find("rotation") - line.find("height") - 10);
            int h = stoi(height);

            string width = line.substr(line.find("width=") + 7, line.find("x=") - line.find("width=") - 9);
            int w = stoi(width);

            string x_coord = line.substr(line.find("x=") + 3, line.find("y=") - line.find("x=") - 5);
            int x = stoi(x_coord);

            string y_coord = line.substr(line.find("y=") + 3, line.find("/>") - 4);
            int y = stoi(y_coord);

            gt_bboxes.push_back(Rect(x, y, w, h));

            if(stoi(framespan1.substr(0, framespan1.find(":"))) != stoi(framespan2))
            {
                // in the case of the framespan is not made of a single frame we add the same bbox for each other frame of the framespan
                for(int i=stoi(framespan1.substr(0, framespan1.find(":")))+1; i <= stoi(framespan2); i++)
                {
                    frame_number_gt.push_back(i);
                    gt_bboxes.push_back(Rect(x, y, w, h));
                }
            }
        }
    }

    cout << "gt bboxes size = " << gt_bboxes.size() << endl;

    vector<int> usedElements;
    vector<int> detectedElements;
    int false_positives = 0, true_positives=0, false_negatives=0, true_negatives=0;
    vector<Rect> fn;
    vector<Rect> tp;
    vector<int> ffn, ftp;
    for(int i=0; i < detection_bboxes.size(); i++)
    {
        bool flag = false;
        for(int j=0; j < gt_bboxes.size(); j++)
        {
            // checks if we are looking at the same frame
            if(frame_number[i] == frame_number_gt[j])
            {
                int x_overlap = max(0, min(detection_bboxes[i].x + detection_bboxes[i].width, gt_bboxes[j].x + gt_bboxes[j].width) -
                                    max(detection_bboxes[i].x, gt_bboxes[j].x));
                int y_overlap = max(0, min(detection_bboxes[i].y + detection_bboxes[i].height, gt_bboxes[j].y + gt_bboxes[j].height) -
                                    max(detection_bboxes[i].y, gt_bboxes[j].y));
                int intersection_area = x_overlap * y_overlap;
                int union_area = ((detection_bboxes[i].width * detection_bboxes[i].height) + (gt_bboxes[j].width * gt_bboxes[j].height)) - intersection_area;
                double jaccardCoef = double(intersection_area) / double(union_area);

                if(jaccardCoef > 0.1)
                {
                    usedElements.push_back(j);
                    detectedElements.push_back(i);
                    if(classes[i]==1)
                    {
                        tp.push_back(detection_bboxes[i]);
                        ftp.push_back(frame_number[i]);
                        true_positives++;
                        flag = true;
                        break;
                    }
                    else
                    {
                        fn.push_back(detection_bboxes[i]);
                        ffn.push_back(frame_number[i]);
                        false_negatives++;
                        flag = true;
                        break;
                    }
                }
            }
        }
        if(!flag)
        {
            if(classes[i] == 0)
                true_negatives++;
            else
                false_positives++;
        }
    }

#ifdef NMS_RES
    int false_negatives_nms = 0;
    for(int i=0; i < gt_bboxes.size(); i++)
    {
        bool flag=false;
        for(int j=0; j < detection_bboxes.size(); j++)
        {
            int x_overlap = max(0, min(detection_bboxes[j].x + detection_bboxes[j].width, gt_bboxes[i].x + gt_bboxes[i].width) -
                                max(detection_bboxes[j].x, gt_bboxes[i].x));
            int y_overlap = max(0, min(detection_bboxes[j].y + detection_bboxes[j].height, gt_bboxes[i].y + gt_bboxes[i].height) -
                                max(detection_bboxes[j].y, gt_bboxes[i].y));
            int intersection_area = x_overlap * y_overlap;
            int union_area = ((detection_bboxes[j].width * detection_bboxes[j].height) + (gt_bboxes[i].width * gt_bboxes[i].height)) - intersection_area;
            double jaccardCoef = double(intersection_area) / double(union_area);
            if(jaccardCoef > 0.1)
            {
                flag=true;
                break;
            }
        }
        if(!flag)
            false_negatives_nms++;

    }
#endif

    int undetectedGT=0;
    for(int i=0; i < gt_bboxes.size(); i++)
    {
        if(find(usedElements.begin(), usedElements.end(), i) == usedElements.end())
            undetectedGT++;
    }
    cout << "true_positives = " << true_positives << endl;
    cout << "true_negatives = " << true_negatives << endl;
    cout << "false positives = " << false_positives << endl;
    cout << "false negatives = " << false_negatives << endl;
    cout << "number of classified samples = " << detection_bboxes.size() << endl;
    cout << "detected ratio = " << (gt_bboxes.size() - undetectedGT) / (double) gt_bboxes.size() << endl;
    cout << "precision = " << true_positives / double(true_positives + false_positives) << endl;
#ifndef NMS_RES
    cout << "recall = " << true_positives / double(true_positives + false_negatives) << endl;
#else
    cout << "recall after NMS= " << true_positives / double(true_positives + false_negatives_nms) << endl;
#endif
    cout << "accuracy = " << (true_negatives + true_positives) / double(detection_bboxes.size()) << endl << endl;

    csv.close();
    gt.close();


#ifdef TEST
    VideoWriter outputVideo;
    outputVideo.open("/home/mathieu/STAGE/underground_dataset/results/false_negatives.mpg",
                     VideoWriter::fourcc('M','P','E','G'), 25, Size(352,288), true);

    if(!outputVideo.isOpened())
    {
        cout << "Could not open output video file" << endl;
        exit(-1);
    }

    VideoCapture cap("/home/mathieu/STAGE/underground_dataset/videos/A_d800mm_R2.mpg");
    if(! cap.isOpened())
    {
        cout << "Could not open video file " << endl;
        exit(-1);
    }

    int nb_frame =-1;
    while(1)
    {
        Mat frame;
        if(!cap.read(frame))
            break;

        nb_frame++;

        for(int i=0; i < frame_number_gt.size(); i++)
        {
            if(frame_number_gt[i] == nb_frame)
            {
                rectangle(frame, gt_bboxes[i], Scalar(255, 0, 0));
            }
        }

        /*for(int i=0; i < frame_number.size(); i++)
        {
            if(frame_number[i] == nb_frame)
            {
                if(find(detectedElements.begin(), detectedElements.end(), i) != detectedElements.end() && classes[i] == 1)
                    rectangle(frame, detection_bboxes[i], Scalar(0, 255, 0));
                else
                    if(classes[i] == 1)
                        rectangle(frame, detection_bboxes[i], Scalar(0, 0, 255));
            }

        }*/

        for(int i=0; i < ffn.size(); i++)
        {
            if(ffn[i] == nb_frame)
            {
                rectangle(frame, fn[i], Scalar(0,0,255));
            }
        }
        outputVideo.write(frame);
    }
#endif
}

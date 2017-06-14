// Standard
#include <iostream>
#include <fstream>

// Opencv 3.2
#include <opencv2/core/utility.hpp>


using namespace std;
using namespace cv;
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

        detection_bboxes.push_back(Rect(x, y, width, height));
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

    int false_positives = 0, true_positives=0, false_negatives=0, true_negatives=0;
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

                if(jaccardCoef > 0.05)
                {
                    if(classes[i]==1)
                    {
                        true_positives++;
                        flag = true;
                        break;
                    }
                    else
                    {
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

    cout << "number of positive bboxes : " << true_positives + false_positives << endl;
    cout << "number of negative bboxes : " << true_negatives + false_negatives << endl;
    cout << "precision = " << true_positives / double(true_positives + false_positives) << endl;
    cout << "recall = " << true_positives / double(true_positives + false_negatives) << endl;
    cout << "accuracy = " << (true_negatives + true_positives) / double(detection_bboxes.size()) << endl;

    csv.close();
    gt.close();
}

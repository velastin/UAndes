// Standard
#include <iostream>
#include <fstream>
#include <string>

// Opencv 3.2
#include <opencv2/core/utility.hpp>


using namespace std;
using namespace cv;

//structure to store information about a track sequence
struct head
{
    string name;
    vector<int> numFrame;
    vector<Point> centroids;
    vector<Rect> bboxes;

    head(string n):name(n) {}
}typedef head;


int main( int argc, char** argv )
{
    ifstream csv(argv[1], ios::in);
    ifstream gt(argv[2], ios::in);

    if(! (csv || gt))
    {
        cout << "Could not open one of the files given" << endl;
        exit(-1);
    }

    string line;
    vector<head> heads;
    string lastName="";
    // reads detections created by the algorithm
    while(getline(csv, line))
    {
        string sub_str = line.substr(0, line.find(","));
        if(lastName.compare(sub_str) != 0)
        {
            heads.push_back(head(sub_str));
            lastName = sub_str;
        }
        line.erase(0, line.find(",") + string(",").length());

        sub_str = line.substr(0, line.find(","));
        heads[heads.size()-1].numFrame.push_back(stoi(sub_str));
        line.erase(0, line.find(",") + string(",").length());

        int x, y;
        sub_str = line.substr(0, line.find(","));
        x = stoi(sub_str);
        line.erase(0, line.find(",") + string(",").length());

        sub_str = line.substr(0, line.find(","));
        y = stoi(sub_str);
        heads[heads.size()-1].centroids.push_back(Point(x,y));

        // adds the bbox in order to calculate a jaccard coefficient
        heads[heads.size()-1].bboxes.push_back(Rect(x-28, y-28, 56, 56));
    }


    // parses xgtf file by hand because c++ xml parsers seem not suitable for xgtf file, or the xgtf file is too complicated
    vector<int> frame_number_gt;
    vector<Rect> gt_bboxes;
    vector<Point> gt_centroids;
    int gt_heads = 0;
    while(getline(gt, line))
    {
        if(line.find("\"Cabeza\"") != string::npos)
            gt_heads++;
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
            gt_centroids.push_back(Point(x+(w/2), y+(h/2)));

            if(stoi(framespan1.substr(0, framespan1.find(":"))) != stoi(framespan2))
            {
                // in the case of the framespan is not made of a single frame we add the same bbox for each other frame of the framespan
                for(int i=stoi(framespan1.substr(0, framespan1.find(":")))+1; i <= stoi(framespan2); i++)
                {
                    frame_number_gt.push_back(i);
                    gt_bboxes.push_back(Rect(x, y, w, h));
                    gt_centroids.push_back(Point(x+(w/2), y+(h/2)));
                }
            }
        }
    }

    int false_positives=0, trackedRegions = 0;
    vector<double> motp(heads.size());

    // iterates on tracked regions first in order to count the number of false positives
    for(int i=0; i < heads.size(); i++)
    {
        for(int j=0; j < heads[i].bboxes.size(); j++)
        {
            bool flag = false;
            trackedRegions++;
            int x1_tl = heads[i].bboxes[j].x;
            int y1_tl = heads[i].bboxes[j].y;
            int x1_br = heads[i].bboxes[j].x + heads[i].bboxes[j].width;
            int y1_br = heads[i].bboxes[j].y + heads[i].bboxes[j].height;
            int area_1 = heads[i].bboxes[j].width * heads[i].bboxes[j].height;

            double min_dist = 65656565;
            for(int k=0; k < gt_bboxes.size(); k++)
            {
                if(heads[i].numFrame[j] != frame_number_gt[k])
                    continue;

                int x2_tl = gt_bboxes[k].x;
                int y2_tl = gt_bboxes[k].y;
                int x2_br = gt_bboxes[k].x + gt_bboxes[k].width;
                int y2_br = gt_bboxes[k].y + gt_bboxes[k].height;
                int area_2 = gt_bboxes[k].width * gt_bboxes[k].height;

                // calculates jaccard coefficient between tracked area and gt area
                int x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl));
                int y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl));
                int overlap_area = x_overlap * y_overlap;
                int total_area = area_1 + area_2 - overlap_area;
                // if there is an overlapping then we can calculate the metrics
                if(overlap_area / (float) total_area > 0.3)
                {
                    // distance between centroids :
                    double distance = sqrt(pow(gt_centroids[k].x - heads[i].centroids[j].x, 2) + pow(gt_centroids[k].y - heads[i].centroids[j].y, 2));
                    if(min_dist > distance)
                        min_dist = distance;
                    flag = true;
                }
            }
            if(!flag)
                false_positives++;
            else
                motp[i] += min_dist;
        }
        motp[i] = motp[i] / heads[i].bboxes.size();
    }

    int misses = 0;
    //iterates on gt_bboxes first in order to count the missed regions
    for(int i=0; i < gt_bboxes.size(); i++)
    {
        bool flag = false;
        int x1_tl = gt_bboxes[i].x;
        int y1_tl = gt_bboxes[i].y;
        int x1_br = gt_bboxes[i].x + gt_bboxes[i].width;
        int y1_br = gt_bboxes[i].y + gt_bboxes[i].height;
        int area_1 = gt_bboxes[i].width * gt_bboxes[i].height;
        for(int j = 0; j < heads.size(); j++)
        {
            for(int k = 0; k < heads[j].bboxes.size(); k++)
            {
                if(heads[j].numFrame[k] != frame_number_gt[i])
                    continue;

                int x2_tl = heads[j].bboxes[k].x;
                int y2_tl = heads[j].bboxes[k].y;
                int x2_br = heads[j].bboxes[k].x + heads[j].bboxes[k].width;
                int y2_br = heads[j].bboxes[k].y + heads[j].bboxes[k].height;
                int area_2 = heads[j].bboxes[k].width * heads[j].bboxes[k].height;

                int x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl));
                int y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl));
                int overlap_area = x_overlap * y_overlap;
                int total_area = area_1 + area_2 - overlap_area;

                if(overlap_area / (float) total_area > 0.05)
                {
                    flag = true;
                    break;
                }
            }
            if(flag)
                break;
        }
        if(!flag)
            misses++;
    }

    //calculates the mean distance error
    double mean_motp = 0;
    for(int i=0; i < motp.size(); i++)
        mean_motp += motp[i];

    mean_motp = mean_motp / motp.size();

    cout << "number of heads detected : " << heads.size() << endl;
    cout << "number of GT heads : " << gt_heads << endl;
    cout << "number of tracked regions : " << trackedRegions << endl;
    cout << "number of gt regions : " << gt_bboxes.size() << endl;
    cout << "number of false positives : " << false_positives << endl;
    cout << "missrate : " << misses / (double) gt_bboxes.size() << endl;
    cout << "motp : " << mean_motp << endl;


}

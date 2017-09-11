/*!
 * \file generate_tracking_results.cpp
 * \brief Standalone script to generate metrics for tracking performance evaluation
 * \version 1.0
 */

// Standard
#include <iostream>
#include <fstream>
#include <string>

// Opencv 3.2
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#define TEST 1

using namespace std;
using namespace cv;

//==================================================
//  All the metrics used in this file are computed  |
//  according to the following paper :              |
//  "Quantitative evaluation of different aspects   |
//  of motion trackers under various challenges"    |
//==================================================


//structure to store information about a track sequence
struct head
{
    string name;
    vector<int> numFrame;
    vector<Point> centroids;
    vector<Rect> bboxes;

    head(string n):name(n) {}
}typedef head;


/**
 * @brief readResults : read CSV file of generated results
 * @param path : input path to the CSV file
 * @return : vector of "head" structures
 */
vector<head> readResults(const string & path)
{
    ifstream csv(path, ios::in);
    if(! (csv))
    {
        cout << "Could not open CSV file" << endl;
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

    return heads;
}

/**
 * @brief readGT : read the ground truth file (Viper xml format)
 * @param path : input path to the ground truth file
 * @return : vector of "head" structures
 */
vector<head> readGT(const string & path)
{
    ifstream gt(path, ios::in);

    if(! (gt))
    {
        cout << "Could not open GT file" << endl;
        exit(-1);
    }

    // parses xgtf file by hand because c++ xml parsers seem not suitable for xgtf file, or the xgtf file is too complicated
    vector<head> gt_heads;
    vector<int> frame_number_gt;
    vector<Rect> gt_bboxes;
    vector<Point> gt_centroids;
    int nb_gt_heads = 0;
    string line;
    while(getline(gt, line))
    {
        if(line.find("attribute name=\"Cabeza\"") != string::npos)
        {
            if(nb_gt_heads!=0)
            {
                gt_heads.push_back(head("gt_head"+to_string(nb_gt_heads-1)));
                gt_heads[gt_heads.size()-1].numFrame = frame_number_gt;
                gt_heads[gt_heads.size()-1].centroids = gt_centroids;
                gt_heads[gt_heads.size()-1].bboxes = gt_bboxes;
                frame_number_gt.clear();
                gt_centroids.clear();
                gt_bboxes.clear();
            }
            nb_gt_heads++;
        }
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
    //adds the last head
    gt_heads.push_back(head("gt_head"+to_string(nb_gt_heads-1)));
    gt_heads[gt_heads.size()-1].numFrame = frame_number_gt;
    gt_heads[gt_heads.size()-1].centroids = gt_centroids;
    gt_heads[gt_heads.size()-1].bboxes = gt_bboxes;

    return gt_heads;
}

/**
 * @brief computeOverlap : computes overlapping between regions of interest
 * @param r1 : input ROI 1
 * @param r2 : input ROI 2
 * @return : overlapping percentage between the 2 ROI
 */
double computeOverlap(const Rect & r1, const Rect & r2)
{
    int x_overlap = max(0, min(r1.x+r1.width, r2.x+r2.width) - max(r1.x, r2.x));
    int y_overlap = max(0, min(r1.y+r1.height, r2.y+r2.height) - max(r1.y, r2.y));
    int overlap_area = x_overlap * y_overlap;
    int total_area = (r1.width*r1.height) + (r2.width*r2.height) - overlap_area;

    double overlapping = overlap_area / (double)total_area;
    return overlapping;
}

/**
 * @brief computeTPFP : computes the number of True Positives (TP) and False Positives (FP)
 * @param heads : input vector of "head" structures coming from the CSV file, output of the algorithm to evaluate
 * @param gt_heads : input vector of ground truth "head" structures
 * @param fp : output number of false positives
 * @param pairs : output vector of pairs between GT and algorithm tracks
 * @return : number of true positives
 */
int computeTPFP(const vector<head> & heads, const vector<head> & gt_heads, int * fp, vector<pair<int, int> > * pairs)
{
    int true_positives=0;
    for(int i=0; i < heads.size(); i++)
    {
        bool flag = false;
        for(int j = 0; j < gt_heads.size(); j++)
        {
            int temporal_coherence = 0;
            double spatial_overlap = 0;
            for(int k=0; k < heads[i].numFrame.size(); k++)
            {
                for(int l=0; l < gt_heads[j].numFrame.size(); l++)
                {
                    if(heads[i].numFrame[k] == gt_heads[j].numFrame[l])
                    {
                        temporal_coherence++;
                        spatial_overlap += computeOverlap(heads[i].bboxes[k], gt_heads[j].bboxes[l]);
                    }
                }
            }
            if(temporal_coherence == 0)
                continue;

            spatial_overlap = spatial_overlap / (double) temporal_coherence;
            if(temporal_coherence > 0.15 * gt_heads[j].numFrame.size() && spatial_overlap > 0.15)
            {
                true_positives++;
                pairs->push_back(std::pair<int, int> (i, j));
                flag = true;
                break;
            }
        }
        if(! flag)
            (*fp) = (*fp)+1;
    }
    return true_positives;
}


/**
 * @brief computeFN : computes the number of False Negatives (FN)
 * @param heads : input vector of "head" structures coming from the CSV file, output of the algorithm to evaluate
 * @param gt_heads : input vector of ground truth "head" structures
 * @return : number of false negatives
 */
int computeFN(const vector<head> & heads, const vector<head> & gt_heads)
{
    int false_negatives=0;
    for(int i=0; i < gt_heads.size(); i++)
    {
        bool flag = false;
        for(int j = 0; j < heads.size(); j++)
        {
            int temporal_coherence = 0;
            double spatial_overlap = 0;
            for(int k=0; k < gt_heads[i].numFrame.size(); k++)
            {
                for(int l=0; l < heads[j].numFrame.size(); l++)
                {
                    if(heads[j].numFrame[l] == gt_heads[i].numFrame[k])
                    {
                        temporal_coherence++;
                        spatial_overlap += computeOverlap(heads[j].bboxes[l], gt_heads[i].bboxes[k]);
                    }
                }
            }
            if(temporal_coherence == 0)
                continue;


            spatial_overlap = spatial_overlap / (double) temporal_coherence;
            if(temporal_coherence > 0.15 * gt_heads[j].numFrame.size() && spatial_overlap > 0.15)
            {
                flag = true;
                break;
            }
        }
        if(!flag)
            false_negatives++;
    }
    return false_negatives;
}

/**
 * @brief computeCloseness : compute the closeness between pairs of algorithm and GT tracks and the corresponding standard deviation
 * @param heads : input vector of "head" structures coming from the CSV file, output of the algorithm to evaluate
 * @param gt_heads : input vector of ground truth "head" structures
 * @param pairs : input vector of pairs between GT and algorithm tracks
 * @param closenessDeviation : output closeness deviation from the mean closeness
 * @return : closeness value for all pairs of tracks
 */
double computeCloseness(const vector<head> & heads, const vector<head> & gt_heads, const vector<pair<int, int> > & pairs, double * closenessDeviation)
{
    vector<double> sum_closeness_vec(pairs.size());
    vector<vector<double> > deviation_vec(pairs.size());
    vector<double> mean_vec(pairs.size());

    for(int i=0; i < pairs.size(); i++)
    {
        int temporal_coherence = 0;
        for(int j=0; j < heads[pairs[i].first].numFrame.size(); j++)
        {
            for(int k=0; k < gt_heads[pairs[i].second].numFrame.size(); k++)
            {
                if(gt_heads[pairs[i].second].numFrame[k] == heads[pairs[i].first].numFrame[j])
                {
                    sum_closeness_vec[i] += computeOverlap(heads[pairs[i].first].bboxes[j], gt_heads[pairs[i].second].bboxes[k]);
                    deviation_vec[i].push_back(computeOverlap(heads[pairs[i].first].bboxes[j], gt_heads[pairs[i].second].bboxes[k]));
                    temporal_coherence++;
                    break;
                }
            }
        }
        // mean overlapping for this tracks pair
        mean_vec[i] = sum_closeness_vec[i] / (double)temporal_coherence;
    }

    // sums all overlapping for all track pairs
    double overall_overlapping_sum=0;
    for(int i=0; i < sum_closeness_vec.size(); i++)
        overall_overlapping_sum+= sum_closeness_vec[i];

    // sums products between all overlapping of one track pair and track mean overlapping. Do that for all track pairs
    double product=0;
    for(int i=0; i < deviation_vec.size(); i++)
        for(int j=0; j < deviation_vec[i].size(); j++)
            product+= deviation_vec[i][j]*mean_vec[i];

    double closeness = product / overall_overlapping_sum;


    // sums all square variance for each tracks pair
    vector<double> deviation_vec2(pairs.size());
    for(int i=0; i < deviation_vec.size(); i++)
        for(int j=0; j < deviation_vec[i].size(); j++)
            deviation_vec2[i] += pow(deviation_vec[i][j] - mean_vec[i], 2);

    //calculates weighted deviation for each tracks pair
    for(int i=0; i < deviation_vec2.size(); i++)
    {
        deviation_vec2[i] = deviation_vec2[i] / (deviation_vec[i].size()-1);
        deviation_vec2[i] = sqrt(deviation_vec2[i]);
    }

    // calculates closeness weighted standard deviation
    double sum_closeness_dev = 0;
    for(int i=0; i < deviation_vec.size(); i++)
    {
        for(int j=0; j < deviation_vec[i].size(); j++)
            sum_closeness_dev += deviation_vec[i][j] * deviation_vec2[i];
    }
    (*closenessDeviation) = sum_closeness_dev / overall_overlapping_sum;

    return closeness;
}

/**
 * @brief computeLatency : compute the latency for all tracks that matches a GT track
 * @param heads : input vector of "head" structures coming from the CSV file, output of the algorithm to evaluate
 * @param gt_heads : input vector of ground truth "head" structures
 * @param pairs : input vector of pairs between GT and algorithm tracks
 * @return : number of frame latency between beginning of GT track and earlieast beginning of algorithm track
 */
int computeLatency(const vector<head> & heads, const vector<head> & gt_heads, const vector<pair<int, int> > & pairs)
{
    int latency = 0;
    for(int i=0; i < pairs.size(); i++)
    {
        int min_latency = 65535;
        bool flag = false;
        // check if there is more than one track associated with a GT track (second element of the pair)
        for(int j=0; j < pairs.size(); j++)
        {
            if(i==j)
                continue;

            // if multiple tracks are associated with the same GT track we use the one that generates the minimal latency
            if(pairs[i].second == pairs[j].second)
            {
                int mini = min(heads[pairs[i].first].numFrame[0] - gt_heads[pairs[i].second].numFrame[0],
                           heads[pairs[j].first].numFrame[0] - gt_heads[pairs[j].second].numFrame[0]);
                if(mini < min_latency)
                    min_latency = mini;
                flag = true;
            }
        }
        if(flag)
            latency += min_latency;
        else
            latency += heads[pairs[i].first].numFrame[0] - gt_heads[pairs[i].second].numFrame[0];
    }

    // not sure that ponderate by number of ground truth tracks is meaningful
    latency = latency / gt_heads.size();
    return latency;
}

/**
 * @brief computeTDE : computes the Track Distance Error (TDE) and the corresponding standard deviation
 * @param heads : input vector of "head" structures coming from the CSV file, output of the algorithm to evaluate
 * @param gt_heads : input vector of ground truth "head" structures
 * @param pairs : input vector of pairs between GT and algorithm tracks
 * @param standardDeviation : output standard deviation of the track distance error
 * @return : track distance error value
 */
double computeTDE(const vector<head> & heads, const vector<head> & gt_heads, const vector<pair<int, int> > & pairs, double * standardDeviation)
{
    vector<double> mean_dist(pairs.size());
    vector<vector<double> > dist_vec(pairs.size());
    for(int i=0; i < pairs.size(); i++)
    {
        int temporal_coherence = 0;
        for(int j=0; j < heads[pairs[i].first].numFrame.size(); j++)
        {
            for(int k=0; k < gt_heads[pairs[i].second].numFrame.size(); k++)
            {
                if(heads[pairs[i].first].numFrame[j] == gt_heads[pairs[i].second].numFrame[k])
                {
                    // euclidean distance between two 2D points
                    mean_dist[i] += sqrt(pow(abs(gt_heads[pairs[i].second].centroids[k].x - heads[pairs[i].first].centroids[j].x), 2)
                                              + pow(abs(gt_heads[pairs[i].second].centroids[k].y - heads[pairs[i].first].centroids[j].y), 2));
                    dist_vec[i].push_back(sqrt(pow(abs(gt_heads[pairs[i].second].centroids[k].x - heads[pairs[i].first].centroids[j].x), 2)
                                          + pow(abs(gt_heads[pairs[i].second].centroids[k].y - heads[pairs[i].first].centroids[j].y), 2)));
                    temporal_coherence++;
                    break;
                }
            }
        }
        mean_dist[i] = mean_dist[i] / temporal_coherence;
    }

    // sums products between all distances of one track pair and track mean distance. Do that for all track pairs
    double product_sum = 0;
    for(int i=0; i < dist_vec.size(); i++)
        for(int j=0; j < dist_vec[i].size(); j++)
            product_sum += dist_vec[i][j] * mean_dist[i];

    double sum_distances = 0;
    for(int i=0; i < dist_vec.size(); i++)
        for(int j=0; j < dist_vec[i].size(); j++)
            sum_distances += dist_vec[i][j];

    double track_distance_error = product_sum / sum_distances;


    // computes the overall standard deviation of the track distance error
    vector<double> deviation(pairs.size());
    for(int i=0; i < dist_vec.size(); i++)
    {
        for(int j=0; j < dist_vec[i].size(); j++)
        {
            deviation[i] += pow(dist_vec[i][j] - mean_dist[i], 2);
        }
        deviation[i] = deviation[i] / (dist_vec[i].size() -1);
        deviation[i] = sqrt(deviation[i]);
    }

    double sum_product=0;
    for(int i=0; i < dist_vec.size(); i++)
        for(int j=0; j < dist_vec[i].size(); j++)
            sum_product += dist_vec[i][j] * deviation[i];

    (*standardDeviation) = sum_product / sum_distances;

    return track_distance_error;
}


/**
 * @brief computeFragmentation : compute the number of track fragmentations for all tracks
 * @param pairs : input vector of pairs between GT and algorithm tracks
 * @return : number of track fragmentations
 */
int computeFragmentation(const vector<pair<int, int> > & pairs)
{
    int track_fragmentation = 0;
    vector<int> gt_done;
    for(int i=0; i < pairs.size(); i++)
    {
        // looks for multiple association with the same ground truth track and count each additional association as track_fragmentation
        if(find(gt_done.begin(), gt_done.end(), pairs[i].second) != gt_done.end())
            track_fragmentation++;
        else
            gt_done.push_back(pairs[i].second);
    }
    return track_fragmentation;
}


/**
 * @brief computeIDC : computes the number of ID changes for all system tracks
 * @param pairs : input vector of pairs between GT and algorithm tracks
 * @return : number of ID changes
 */
int computeIDC(const vector<pair<int, int> > & pairs)
{
    int id_changes = 0;
    vector<pair<int, int> > done_pairs;
    vector<int> done_sys_track;
    for(int i=0; i < pairs.size(); i++)
    {
        //if the system track of the current pair has already been associated with a ground truth track
        if(find(done_sys_track.begin(), done_sys_track.end(), pairs[i].first) != done_sys_track.end())
        {
            // and the associated ground truth track is different from the previous one
            if(find(done_pairs.begin(), done_pairs.end(), pairs[i]) == done_pairs.end())
                id_changes++;
        }
        else
        {
            done_sys_track.push_back(pairs[i].first);
            done_pairs.push_back(pairs[i]);
        }
    }
    return id_changes;
}


/**
 * @brief computeCompleteness : computes the overall completeness of system tracks and the associated standard deviation
 * @param heads : input vector of "head" structures coming from the CSV file, output of the algorithm to evaluate
 * @param gt_heads : input vector of ground truth "head" structures
 * @param pairs : input vector of pairs between GT and algorithm tracks
 * @param standardDeviation : output standard deviation of the track completeness
 * @return : track completeness value
 */
double computeCompleteness(const vector<head> & heads, const vector<head> & gt_heads, const vector<pair<int, int> > & pairs, double * standardDeviation)
{
    double completeness = 0;
    vector<double> completeness_vec(pairs.size());
    for(int i=0; i < pairs.size(); i++)
    {
        double max_completeness = 0;
        bool flag = false;
        // check if there is more than one track associated with a GT track (second element of the pair)
        for(int j=0; j < pairs.size(); j++)
        {
            if(i==j)
                continue;

            // if multiple tracks are associated with the same GT track we use the one that generates the maximal completeness
            if(pairs[i].second == pairs[j].second)
            {
                double maxi = max(heads[pairs[i].first].numFrame.size() / (double)gt_heads[pairs[i].second].numFrame.size(),
                                  heads[pairs[j].first].numFrame.size() / (double)gt_heads[pairs[j].second].numFrame.size());
                if(maxi > max_completeness)
                    max_completeness = maxi;
                flag = true;
            }
        }
        if(flag)
        {
            completeness += max_completeness;
            completeness_vec[i] = max_completeness;
        }
        else
        {
            completeness += heads[pairs[i].first].numFrame.size() / (double)gt_heads[pairs[i].second].numFrame.size();
            completeness_vec[i] = heads[pairs[i].first].numFrame.size() / (double)gt_heads[pairs[i].second].numFrame.size();
        }
    }
    completeness = completeness / pairs.size();

    //calculates standard deviation of the track completeness;
    for(int i=0; i < completeness_vec.size(); i++)
        (*standardDeviation) += pow(completeness_vec[i] - completeness, 2);

    (*standardDeviation) = sqrt((*standardDeviation) / (double)(completeness_vec.size()-1));

    return completeness;
}


int main( int argc, char** argv )
{
    if(argc != 3)
    {
        cout << "Need 2 arguments : " << endl;
        cout << "\t 1. Path to csv file containing tracks" << endl;
        cout << "\t 2. Path to ground truth file " << endl;
        exit(-1);
    }

    vector<head> heads = readResults(argv[1]);
    vector<head> gt_heads = readGT(argv[2]);

    int false_positives = 0;
    vector<pair<int, int> > pairs;
    int true_positives = computeTPFP(heads, gt_heads, &false_positives, &pairs);
    int false_negatives = computeFN(heads, gt_heads);
    double closenessDeviation = 0;
    double closeness = computeCloseness(heads, gt_heads, pairs, &closenessDeviation);
    int latency = computeLatency(heads, gt_heads, pairs);
    double distanceDeviation = 0;
    double track_distance_error = computeTDE(heads, gt_heads, pairs, &distanceDeviation);
    int track_fragmentation = computeFragmentation(pairs);
    int id_changes = computeIDC(pairs);
    double completenessDeviation = 0;
    double completeness = computeCompleteness(heads, gt_heads, pairs, & completenessDeviation);

    cout << "GT pedestrians = " << gt_heads.size() << endl;
    cout << "true_positives = " << true_positives << endl;
    cout << "false_positives = " << false_positives << endl;
    cout << "false_negatives = " << false_negatives << endl;
    cout << "closeness = " << closeness << endl;
    cout << "closeness standard deviation = " << closenessDeviation << endl;
    cout << "latency = " << latency << endl;
    cout << "track distance error = " << track_distance_error << endl;
    cout << "distance error standard deviation = " << distanceDeviation << endl;
    cout << "track fragmentation = " << track_fragmentation << endl;
    cout << "id changes = " << id_changes << endl;
    cout << "completeness = " << completeness << endl;
    cout << "completeness standard deviation = " << completenessDeviation << endl;


    vector<pair<int, int> > detectedElements;
    vector<pair<int, int> > badlyDetectedElements;
    int dt_false_positives = 0, dt_true_positives=0;
    for(int i=0; i < heads.size(); i++)
    {
        for(int j=0; j < heads[i].bboxes.size(); j++)
        {
            bool flag = false;
            for(int k=0; k < gt_heads.size(); k++)
            {
                for(int l=0; l < gt_heads[k].bboxes.size(); l++)
                {
                    // checks if we are looking at the same frame
                    if(heads[i].numFrame[j] == gt_heads[k].numFrame[l])
                    {
                        int x_overlap = max(0, min(heads[i].bboxes[j].x + heads[i].bboxes[j].width, gt_heads[k].bboxes[l].x + gt_heads[k].bboxes[l].width) -
                                            max(heads[i].bboxes[j].x, gt_heads[k].bboxes[l].x));
                        int y_overlap = max(0, min(heads[i].bboxes[j].y + heads[i].bboxes[j].height, gt_heads[k].bboxes[l].y + gt_heads[k].bboxes[l].height) -
                                            max(heads[i].bboxes[j].y, gt_heads[k].bboxes[l].y));
                        int intersection_area = x_overlap * y_overlap;
                        int union_area = ((heads[i].bboxes[j].width * heads[i].bboxes[j].height) + (gt_heads[k].bboxes[l].width * gt_heads[k].bboxes[l].height)) - intersection_area;
                        double jaccardCoef = double(intersection_area) / double(union_area);

                        if(jaccardCoef > 0.1)
                        {
                            detectedElements.push_back(pair<int, int>(i,j));
                            dt_true_positives++;
                            flag=true;
                            break;
                        }
                    }
                }
                if(flag)
                    break;
            }
            if(!flag)
            {
                badlyDetectedElements.push_back(pair<int, int>(i,j));
                dt_false_positives++;
            }
        }
    }

    int dt_false_negatives=0;
    int nb_gt = 0;
    for(int i=0; i < gt_heads.size(); i++)
    {
        for(int j=0; j < gt_heads[i].bboxes.size(); j++)
        {
            nb_gt++;
            bool flag = false;
            for(int k=0; k < heads.size(); k++)
            {
                for(int l=0; l < heads[k].bboxes.size(); l++)
                {
                    // checks if we are looking at the same frame
                    if(gt_heads[i].numFrame[j] == heads[k].numFrame[l])
                    {
                        int x_overlap = max(0, min(gt_heads[i].bboxes[j].x + gt_heads[i].bboxes[j].width, heads[k].bboxes[l].x + heads[k].bboxes[l].width) -
                                            max(gt_heads[i].bboxes[j].x, heads[k].bboxes[l].x));
                        int y_overlap = max(0, min(gt_heads[i].bboxes[j].y + gt_heads[i].bboxes[j].height, heads[k].bboxes[l].y + heads[k].bboxes[l].height) -
                                            max(gt_heads[i].bboxes[j].y, heads[k].bboxes[l].y));
                        int intersection_area = x_overlap * y_overlap;
                        int union_area = ((gt_heads[i].bboxes[j].width * gt_heads[i].bboxes[j].height) + (heads[k].bboxes[l].width * heads[k].bboxes[l].height)) - intersection_area;
                        double jaccardCoef = double(intersection_area) / double(union_area);

                        if(jaccardCoef > 0.1)
                        {
                            flag=true;
                            break;
                        }
                    }
                }
                if(flag)
                    break;
            }
            if(!flag)
            {
                dt_false_negatives++;
            }
        }
    }
    cout << "detected ratio = " << (nb_gt - dt_false_negatives) / (double) nb_gt << endl;
    cout << "precision = " << dt_true_positives / (double)(dt_true_positives + dt_false_positives) << endl;
    cout << "recall = " << dt_true_positives / (double)(dt_true_positives + dt_false_negatives) << endl;

#ifdef TEST
    VideoWriter outputVideo;
    outputVideo.open("/home/mathieu/STAGE/underground_dataset/results/A_d800mm_R7_tracking.mpg",
                     VideoWriter::fourcc('M','P','E','G'), 25, Size(352,288), true);

    if(!outputVideo.isOpened())
    {
        cout << "Could not open output video file" << endl;
        exit(-1);
    }

    VideoCapture cap("/home/mathieu/STAGE/underground_dataset/pos/test/A_d800mm_R7.mpg");
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

        for(int i=0; i < gt_heads.size(); i++)
        {
            for(int j=0; j < gt_heads[i].numFrame.size(); j++)
            {
                if(gt_heads[i].numFrame[j] == nb_frame)
                {
                    rectangle(frame, gt_heads[i].bboxes[j], Scalar(255, 0, 0));
                }
            }
        }

        for(int i=0; i < heads.size(); i++)
        {
            for(int j=0; j < heads[i].numFrame.size(); j++)
            {
                if(heads[i].numFrame[j] == nb_frame)
                {
                    if(find(detectedElements.begin(), detectedElements.end(), pair<int, int>(i,j)) != detectedElements.end())
                        rectangle(frame, heads[i].bboxes[j], Scalar(0, 255, 0));
                    else
                        if(find(badlyDetectedElements.begin(), badlyDetectedElements.end(), pair<int, int>(i,j)) != detectedElements.end())
                            rectangle(frame, heads[i].bboxes[j], Scalar(0, 0, 255));
                }
            }
        }

        outputVideo.write(frame);
    }
#endif

}

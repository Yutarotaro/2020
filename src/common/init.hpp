#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define filepath1 "/Users/yutaro/research/2020/src"
#define filepath2 "/Users/yutaro/research/2020/src/pictures"

using namespace std;

namespace Init
{

class Feature
{
    cv::Ptr<cv::Feature2D> feature;

public:
    cv::Mat img;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point2f> featurePoint;


    Feature(std::string path)
    {
        img = cv::imread(path, 1);
        feature = cv::AKAZE::create();
        feature->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
    }

    Feature(cv::Mat src)
    {
        src.copyTo(img);
        feature = cv::AKAZE::create();
        feature->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
    }

    cv::Mat getHomography(Feature& obj)
    {
        //マッチング(knnマッチング)
        vector<vector<cv::DMatch>> knnmatch_points;
        cv::BFMatcher match(cv::NORM_HAMMING);
        match.knnMatch(this->descriptors, obj.descriptors, knnmatch_points, 2);

        //対応点を絞る
        const double match_par = 0.75;  //候補点を残す場合のしきい値originally 0.6
        vector<cv::DMatch> goodMatch;
        //KeyPoint -> Point2d
        vector<cv::Point2f> match_point1;
        vector<cv::Point2f> match_point2;

        for (size_t i = 0; i < knnmatch_points.size(); ++i) {
            double distance1 = knnmatch_points[i][0].distance;
            double distance2 = knnmatch_points[i][1].distance;


            //第二候補点から距離値が離れている点のみ抽出（いい点だけ残す）
            if (distance1 <= distance2 * match_par) {
                goodMatch.push_back(knnmatch_points[i][0]);
                match_point1.push_back(this->keypoints[knnmatch_points[i][0].queryIdx].pt);
                match_point2.push_back(obj.keypoints[knnmatch_points[i][0].trainIdx].pt);
            }
        }

        //ホモグラフィ行列推定
        cv::Mat masks;
        cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
        try {
            H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3);
        } catch (cv::Exception& e) {
            std::cerr << e.what() << std::endl;
            return H;
        }

        //RANSACで使われた対応点のみ抽出
        vector<cv::DMatch> inlierMatch;
        for (size_t i = 0; i < masks.rows; ++i) {
            uchar* inlier = masks.ptr<uchar>(i);
            if (inlier[0] == 1) {
                inlierMatch.push_back(goodMatch[i]);
            }
        }

        std::cout << "Number of match points" << inlierMatch.size() << std::endl;


        for (int i = 0; i < inlierMatch.size(); ++i) {
            int kp2_idx = inlierMatch[i].queryIdx;
            //this->featurePoint.push_back(cv::Point(this->keypoints[kp2_idx].pt.x, this->keypoints[kp2_idx].pt.y));
            this->featurePoint.push_back(this->keypoints[kp2_idx].pt);
        }

        for (int i = 0; i < inlierMatch.size(); ++i) {
            int kp2_idx = inlierMatch[i].trainIdx;
            obj.featurePoint.push_back(obj.keypoints[kp2_idx].pt);
        }

        //インライアの対応点のみ表示
        cv::Mat drawMatch_inlier;
        cv::drawMatches(this->img, this->keypoints, obj.img, obj.keypoints, inlierMatch, drawMatch_inlier);

        imshow("DrawMatch_inlier", drawMatch_inlier);
        //imwrite("./diffjust/" + meter_type_s + "/match/match_inlier_" + std::to_string(it) + ".png", drawMatch_inlier);

        return H;
    }
};

class Params
{
public:
    int thresh;          //2値化の閾値
    int total;           //枚数
    std::string picdir;  //ディレクトリ
    cv::Point tl;        //左上
    int l;               //矩形の大きさ
    double front_value;  //正面から読んだときの値
    double front_rad;    //正面から読み取った針の角度
    double k;            //1[deg]に対する変化率
};
int parseInit();
cv::Mat input_render(std::string s, int num);
cv::Mat input_images2(std::string s, std::string t);
cv::Mat input_images(std::string s);
}  // namespace Init

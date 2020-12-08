#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define filepath1 "/Users/yutaro/research/2020/src"
#define filepath2 "/Users/yutaro/research/2020/src/pictures"

namespace Init
{

class Feature
{
    cv::Ptr<cv::Feature2D> feature;

public:
    cv::Mat img;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;


    Feature(std::string path)
    {
        img = cv::imread(path, 1);
        feature = cv::AKAZE::create();
        feature->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
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

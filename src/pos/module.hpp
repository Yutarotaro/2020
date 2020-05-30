#ifndef MODULE_H_
#define MODULE_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

namespace Module
{

cv::Mat getHomography(cv::Mat Src1, cv::Mat Src2);

std::pair<cv::Point, int> circleDetect(cv::Mat img);

cv::Mat decomposeP(cv::Mat P);

struct pose {
    cv::Mat R;
    cv::Vec3f t;
};

pose decomposeH(cv::Mat H, cv::Mat A);

}  // namespace Module

#endif

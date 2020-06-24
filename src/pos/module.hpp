#ifndef MODULE_H_
#define MODULE_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

namespace Module
{

struct pose {
    cv::Mat position;
    cv::Mat orientation;
};

cv::Mat getHomography(cv::Mat Src1, cv::Mat Src2);

pose decomposeHomography(cv::Mat H, cv::Mat A);


}  // namespace Module

#endif

#ifndef D_H_
#define D_H_

#include "read.hpp"
#include <opencv2/opencv.hpp>

namespace Difference
{

std::pair<cv::Point, int> circleDetect(cv::Mat img);

void Lines(cv::Mat src, std::pair<cv::Point, int> circle, double& m);
Read::Data readMeter(cv::Mat src);

}  // namespace Difference

#endif

#ifndef R_H_
#define R_H_

#include <opencv2/opencv.hpp>

namespace Readability
{
double judge(cv::Mat src, int num, int flag);
std::pair<double, cv::Mat> edgeDetection(cv::Mat img);
std::pair<double, cv::Mat> pointerDetection(cv::Mat src, cv::Mat origin);
double read(cv::Mat src);
}  // namespace Readability

#endif

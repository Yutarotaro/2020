#ifndef R_H_
#define R_H_

#include <opencv2/opencv.hpp>

namespace Readability
{
int judge(cv::Mat src, int num, int flag);
cv::Mat edgeDetection(cv::Mat img);
double read(cv::Mat src);
}  // namespace Readability

#endif

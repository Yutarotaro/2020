#ifndef R_H_
#define R_H_

#include <opencv2/opencv.hpp>

namespace Readability
{
struct result {
    double value;
    int readability;
    cv::Mat img;
};
int judge(cv::Mat src, int num, int flag);
std::pair<double, cv::Mat> edgeDetection(cv::Mat img);
result pointerDetection(cv::Mat src, cv::Mat origin);

}  // namespace Readability

#endif

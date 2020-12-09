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
std::pair<double, cv::Mat> edgeDetection(cv::Mat img);
result pointerDetection(cv::Mat src, cv::Mat origin);

}  // namespace Readability

#endif

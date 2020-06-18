#ifndef D_H_
#define D_H_

#include <opencv2/opencv.hpp>

namespace Difference
{
class Data
{
public:
    double value;
    double percent;
};
std::pair<cv::Point, int> circleDetect(cv::Mat img);

void norm(std::pair<double, double>& x);
void Lines(cv::Mat src, std::pair<cv::Point, int> circle, std::pair<double, int>& m);
Data readMeter(cv::Mat src);

}  // namespace Difference

#endif

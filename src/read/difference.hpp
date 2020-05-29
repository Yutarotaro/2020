#ifndef D_H_
#define D_H_

#include "read.hpp"
#include <opencv2/opencv.hpp>

namespace Difference
{

std::vector<cv::Vec2f> Lines(cv::Mat src, std::pair<cv::Point, int> circle);
Read::Data readMeter(cv::Mat src);

}  // namespace Difference

#endif

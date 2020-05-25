#ifndef Read_H_
#define Read_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace Read
{

class Data
{
public:
    double value;
    double percent;
};

void lineDetect(cv::Mat src);

Data readMeter(cv::Mat src);
}  // namespace Read
#endif

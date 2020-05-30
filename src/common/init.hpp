#ifndef INIT_H_
#define INIT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define filepath1 "/Users/yutaro/research/2020/src"
#define filepath2 "/Users/yutaro/research/2020/src/pictures/A"

namespace Init
{
int parseA(cv::Mat& A);
cv::Mat input_images(std::string s);
void read_config();
}  // namespace Init


#endif

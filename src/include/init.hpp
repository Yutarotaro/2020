#ifndef INIT_H_
#define INIT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

//#define filepath1 "/Users/yutaro/research/2020/src/pictures/A"
//#define filepath2 "/pic"
//#define filepath3 ".jpg"

namespace Init
{
int input_images(int a, int b, cv::Mat& image);
void read_config();
}  // namespace Init


#endif

#ifndef INIT_H_
#define INIT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define filepath1 "/home/yutaro/research/src"
#define filepath2 "/home/yutaro/research/src/pictures"

namespace Init
{
int parseInit();
cv::Mat input_render(std::string s, int num);
cv::Mat input_images2(std::string s, std::string t);
cv::Mat input_images(std::string s);
}  // namespace Init


#endif

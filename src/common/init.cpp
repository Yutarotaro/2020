#include "init.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#define filepath1 "/Users/yutaro/research/2020/src/pictures/A"
#define filepath2 "/pic"
#define filepath3 ".jpg"

namespace Init
{
cv::Mat input_images(int a, int b)
{
    std::ostringstream ostr;
    ostr << filepath1 << std::to_string(a) << filepath2 << std::to_string(b) << filepath3;
    cv::Mat image = cv::imread(ostr.str(), 1);

    return image;
}

void read_config()
{
}
}  // namespace Init

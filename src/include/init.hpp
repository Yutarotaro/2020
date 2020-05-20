#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define filepath1 "/Users/yutaro/research/2020/src/pictures/A"
#define filepath2 "/pic"
#define filepath3 ".jpg"

namespace Init
{
int input_images(int a, int b, cv::Mat& image)
{
    try {
        image = cv::imread(filepath1 + std::to_string(a) + filepath2 + std::to_string(b) + filepath3, 1);
    } catch (cv::Exception& e) {
        std::cout << "wrong file format, please input the name of an IMAGE file" << std::endl;
        return -1;
    }

    return 0;
}

void read_config()
{
}
}  // namespace Init

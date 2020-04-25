#include "include/circle.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>


#define filepath "/Users/yutaro/research/2020/src/pictures/A3/pic1.jpg"

int main()
{

    cv::Mat img = cv::imread(filepath, 1);
    if (img.empty()) {
        std::cout << "failed to read pictures" << std::endl;
        return -1;
    }


    Circle::circleDetect(img);

    while (true) {
        const int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }


    cv::destroyAllWindows();

    return 0;
}

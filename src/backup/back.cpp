#include "include/init.hpp"
#include "include/module.hpp"
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{

    cv::Mat img;
    if (argc == 3) {
        int a = argv[1][0] - '0';
        int b = argv[2][0] - '0';
        Init::input_images(a, b, img);
        //Module::circleDetect(img);
        Module::ellipseDetect(img);
        //Module::showHSV(img);
    }
    //std::vector<cv::Mat> output = target(img);


    while (true) {
        const int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}

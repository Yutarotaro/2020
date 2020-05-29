#include "common/init.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pos/module.hpp"
#include "read/difference.hpp"
#include "read/read.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(void)
{
    //pose estimation
    cv::Mat Src1 = Init::input_images(1, 5);
    cv::Mat Src2 = Init::input_images(1, 3);

    //Src1, Src2間のHomographyを求める
    cv::Mat H = Module::getHomography(Src1, Src2);


    // read meter
    cv::Mat Src3 = Init::input_images(1, 5);

    Difference::Lines(Src3, Module::circleDetect(Src3));

    cv::waitKey();

    return 0;
}

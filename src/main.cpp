#include "common/init.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pos/module.hpp"
#include "read/difference.hpp"
#include "read/read.hpp"
//#include <Eigen/Dense>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(void)
{
    //parameter
    cv::Mat A;
    Init::parseA(A);

    return 0;


    //pose estimation
    cv::Mat Src1 = Init::input_images(1, 5);
    cv::Mat Src2 = Init::input_images(1, 3);

    cv::Mat H = Module::getHomography(Src1, Src2);
    std::cout << H << std::endl;


    //Difference::Lines(Src3, Module::circleDetect(Src3));
    Module::pose Homography = Module::decomposeH(H, A);

    cv::Vec3f R = cv::Vec3f(0., 0., 0.);
    cv::Vec3f t = cv::Vec3f(0., 0., 0.);


    //
    //
    //
    //
    // read meter
    cv::Mat Src3 = Init::input_images(1, 5);


    std::cout << "メータの位置: " << t + Homography.t << std::endl;
    //<< "メータの: " << R + Homography.R << std::endl;


    cv::waitKey();
    return 0;
}

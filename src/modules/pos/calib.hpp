#ifndef CALIB_H_
#define CALIB_H_

#include "Homography.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

namespace Calib
{

void calibration(cv::Mat& img, Module::pose& p, int flag);
}  // namespace Calib

#endif

#ifndef THRESH_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;

namespace Adaptive
{

void thresholdIntegral(cv::Mat& inputMat, cv::Mat& outputMat);
}  // namespace Adaptive
#endif

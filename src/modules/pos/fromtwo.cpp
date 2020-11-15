#include "module.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

using namespace cv;
using namespace std;


extern cv::Mat A;
extern cv::Mat R;
extern cv::Mat pos;  //World
extern cv::Mat t;    //Camera1

namespace Fromtwo
{

}  // namespace Fromtwo

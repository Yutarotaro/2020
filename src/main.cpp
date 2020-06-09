#include "common/init.hpp"
#include "opencv2/calib3d.hpp"
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


cv::Mat A;  //カメラ行列
cv::Mat R;  //基準姿勢
cv::Mat t;  //基準位置

//メータの位置をワールド座標の原点とする

int main(void)
{
    //parameter
    Init::parseInit();

    //TODO:homography分解によって複数の解が得られるので１つに絞りpose estimation


#if 0
    //Pose estimation
    cv::Mat Base = Init::input_images("Src1");
    cv::Mat Now = Init::input_images("now");


    auto [rot, tra] = Module::decomposeHomography(Module::getHomography(Base, Now), A);

    //    cv::Mat estimated_pose = rot * t + tra;

    //   std::cout << estimated_pose << std::endl;
#else

    // read meter
    cv::Mat Src3 = Init::input_images("target");
    Difference::readMeter(Src3);

    cv::waitKey();
#endif
    return 0;
}

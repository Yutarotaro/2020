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


cv::Mat A;    //カメラ行列
cv::Vec3f R;  //初期姿勢
cv::Vec3f t;  //初期位置

//メータの位置をワールド座標の原点とする

int main(void)
{
    //parameter
    Init::parseInit();


    while (false) {
        //Pose estimation
        cv::Mat Base = Init::input_images("Src1");
        cv::Mat Now = Init::input_images("now");


        cv::Mat H = Module::getHomography(Base, Now);
        std::cout << "Homography: \n"
                  << H << std::endl;
        //この時点でHomographyは正規化されてる(h33=1)

        Module::reconstructH(H, A);

        //std::cout << "メータの位置: " << ts_decomp << std::endl;
        //<< "メータの: " << R + Homography.R << std::endl;
    }

    // read meter
    cv::Mat Src3 = Init::input_images("Src3");
    cv::imshow("Src3", Src3);

    Difference::readMeter(Src3);

    //    Difference::Lines(Src3, Module::circleDetect(Src3));


    cv::waitKey();
    return 0;
}

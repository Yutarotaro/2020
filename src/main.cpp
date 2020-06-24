#include "common/init.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pos/calib.hpp"
#include "pos/fromtwo.hpp"
#include "pos/module.hpp"
#include "read/difference.hpp"
#include "read/template.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


cv::Mat A;  //カメラ行列
cv::Mat distCoeffs;
cv::Mat R;    //基準姿勢
cv::Mat pos;  //基準位置
cv::Mat t;    //cam1におけるpos

cv::Mat temp;
std::vector<cv::KeyPoint> keypoints1;
cv::Mat descriptors1;

//メータの位置をワールド座標の原点とする

int main(void)
{
    //parameter
    //A,R,pose,tの代入
    Init::parseInit();


#if 1
    //Pose estimation
    //cv::Mat Base_pointer = Init::input_images2("clock", "Base_pointer");
    // cv::Mat Base = Init::input_images("Base");
    // cv::Mat Base = Init::input_images2("meter2", "clota");
    cv::Mat Base = cv::imread("../pictures/meter2/pic0.JPG", 1);

    std::ofstream ofs("./resultZ=0.1115.csv");

    int N = 30;
    ofs << N << ',' << std::endl;

    //cv::Mat Now = Init::input_images("Now");
    //cv::Mat Now = Init::input_images2("clock", "clono");

    //tmpMat tmp = Init::input_images2("meter2", "meter_origin");
    /*
    for (int i = 22; i < 30; i++) {
        cv::Mat tmp = cv::imread("../pictures/meter2/pic" + std::to_string(i) + ".JPG", 1);


        cv::imshow("jjn", tmp);

        auto H = Module::getHomography(Base, tmp);
    }
    */

    cv::Mat Base2 = cv::imread("../pictures/clock/pic0.png", 1);
    for (int i = N / 5; i < N * 4 / 5 - 1; i++) {

        cv::Mat Now = Init::input_render("render", i);
        //cv::Mat Now = Init::input_images2("clock", "clono");

        //    cv::resize(Base, Base, cv::Size(), 0.25, 0.25);
        //   cv::resize(Now, Now, cv::Size(), 0.25, 0.25);

        //        Calib::calibration();

        //mask処理でchessboardを隠す

        auto H = Module::getHomography(Base2, Now);
        auto [position, ori] = Module::decomposeHomography(H, A);

        position += pos;


        //getHomographyの返り値のpositionをいじってるからワンチャンミスるかも

        ofs << i << ',' << position.at<double>(0, 0) << ',' << position.at<double>(0, 1) << ',' << position.at<double>(0, 2) << ',' << std::endl;


        if (false) {
            cv::Mat dst = cv::Mat::zeros(Base.rows + 100, Base.cols + 100, CV_8UC3);
            cv::warpPerspective(Now, dst, H.inv(), dst.size());


            cv::imshow("J", dst);
            cv::waitKey();
        }
    }


#else

    // read meter
    cv::Mat Src3 = Init::input_images("target");
    //cv::Mat Src3 = Init::input_images2("clock", "Base_pointer");
    //cv::resize(Src3, Src3, cv::Size(), 0.25, 0.25);
    Difference::readMeter(Src3);

    cv::waitKey();
#endif
    return 0;
}

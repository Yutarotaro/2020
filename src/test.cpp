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
cv::Mat t;    //camにおけるpos

//メータの位置をワールド座標の原点とする

int main(int argc, char** argv)
{
    //parameter
    Init::parseInit();

    /*
    cv::Mat img = cv::imread("../pictures/meter2/pic-1calib.png", 1);
    cv::Mat Now = cv::imread("../pictures/meter2/pic-1meter.png", 1);
    cv::Mat Base = cv::imread("../pictures/meter2/pic14.JPG", 1);
*/
    cv::Mat Base_calib = cv::imread("../pictures/sim/pic0.png", 1);
    cv::Mat Now_calib = cv::imread("../pictures/sim/pic1.png", 1);
    cv::Mat Base_clock = cv::imread("../pictures/sim/pic2.png", 1);
    cv::Mat Now_clock = cv::imread("../pictures/sim/pic3.png", 1);


    Module::pose p;
    Calib::calibration(Base_calib, p);  //pに基準とする画像のカメラ視点，位置が入る
    Module::pose q;
    Calib::calibration(Now_calib, q);  //qに基準とする画像のカメラ視点，位置が入る


    std::cout << "並進ベクトル" << std::endl
              << p.orientation.inv() * p.position - q.orientation.inv() * q.position << std::endl
              << std::endl
              << q.orientation * p.orientation << std::endl;


    //cv::resize(Base, Base, cv::Size(), 0.7, 0.7);


    auto H = Module::getHomography(Base_clock, Now_clock);

    Module::pose r = Module::decomposeHomography(H, A);

    std::cout << r.position << std::endl;
    std::cout << r.orientation << std::endl;

    return 0;

    /*
    cv::Mat dst = cv::Mat::zeros(Base.rows + 100, Base.cols + 100, CV_8UC3);
    cv::warpPerspective(Base, dst, H.inv(), dst.size());
    cv::imshow("f", dst);
    cv::waitKey();


    Template::readMeter(Base);

    return 0;
    */
}

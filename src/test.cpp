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
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

cv::Mat A;  //カメラ行列
cv::Mat distCoeffs;
cv::Mat R;    //基準姿勢
cv::Mat pos;  //基準位置
cv::Mat t;    //camにおけるpos

cv::Mat temp;

std::vector<cv::KeyPoint> keypoints1;
cv::Mat descriptors1;

int main(int argc, char** argv)
{
    //parameter
    Init::parseInit();

#if 1
    /*
    cv::Mat img = cv::imread("../pictures/meter2/pic-1calib.png", 1);
    cv::Mat Now = cv::imread("../pictures/meter2/pic-1meter.png", 1);
    cv::Mat Base = cv::imread("../pictures/meter2/pic14.JPG", 1);
*/
    /*
    cv::Mat Base_calib = cv::imread("../pictures/sim/pic0.png", 1);
    cv::Mat Now_calib = cv::imread("../pictures/sim/pic1.png", 1);
    cv::Mat Base_clock = cv::imread("../pictures/sim/pic2.png", 1);
    cv::Mat Now_clock = cv::imread("../pictures/sim/pic3.png", 1);
    */


    cv::Mat Base_calib = cv::imread("../pictures/meter_experiment/pic-1.JPG", 1);
    cv::Mat Base_clock_tmp = cv::imread("../pictures/meter_experiment/pic-1.JPG", 1);
    cv::Mat mask = cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
    cv::circle(mask, cv::Point(Base_clock_tmp.cols / 2 + 100, Base_clock_tmp.rows / 2 - 120), 365, cv::Scalar(255), -1, CV_AA);
    //    cv::circle(mask, cv::Point(1290, 2207), 370, cv::Scalar(255), -1, CV_AA);

    cv::Mat Base_clock;
    Base_clock_tmp.copyTo(Base_clock, mask);


    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();


    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);


    Module::pose p;
    Calib::calibration(Base_calib, p);  //pに基準とする画像のカメラ視点，位置が入る

    cv::Mat rvec;  //回転ベクトル

    int total = 127;


    std::ofstream ofs("./result.csv");

    for (int i = 0; i < total; i++) {
        if (i == 54)
            continue;
        std::cout << std::endl
                  << i + 1 << "回目" << std::endl;
        std::string path = "../pictures/meter_experiment/pic" + std::to_string(i) + ".JPG";


        cv::Mat Now_calib = cv::imread(path, 1);
        cv::Mat Now_clock = cv::imread(path, 1);


        Module::pose q;
        Calib::calibration(Now_calib, q);  //qに基準とする画像のカメラ視点，位置が入る

        cv::Rodrigues(q.orientation * p.orientation.inv(), rvec);

        cv::Mat calib_pos = p.orientation.inv() * p.position - q.orientation.inv() * q.position;

        std::cout
            << "chessboard並進ベクトル" << std::endl
            << calib_pos << std::endl
            << "回転ベクトル" << std::endl
            << std::endl
            << rvec << std::endl
            << std::endl;

        auto H = Module::getHomography(Base_clock, Now_clock);

        Module::pose r = Module::decomposeHomography(H, A);

        std::cout << "Homography分解で得られた並進ベクトル(world)" << std::endl
                  << r.position << std::endl;
        std::cout << "回転行列" << std::endl
                  << r.orientation << std::endl;

        cv::Mat R_calib = q.orientation * p.orientation.inv();
        cv::Mat R_estimated;
        cv::Rodrigues(r.orientation, R_estimated);


        double angle_error = cv::determinant(R_estimated * R_calib.inv());
        std::cout << std::fixed << std::setprecision(15) << angle_error << std::endl;

        std::cout << cv::norm(calib_pos) << ',' << cv::norm(calib_pos - r.position) << std::endl;

        if (cv::norm(calib_pos) > cv::norm(calib_pos - r.position)) {
            ofs << i << ',' << cv::norm(calib_pos) << ',' << cv::norm(calib_pos - r.position) << ',' << angle_error
                << ',' << std::endl;
        }
    }
    return 0;

/*
    cv::Mat dst = cv::Mat::zeros(Base.rows + 100, Base.cols + 100, CV_8UC3);
    cv::warpPerspective(Base, dst, H.inv(), dst.size());
    cv::imshow("f", dst);
    cv::waitKey();
    */
#else

    temp = cv::imread("/Users/yutaro/research/2020/src/pictures/meter2/pic-1.png", 1);


    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();


    feature->detectAndCompute(temp, cv::Mat(), keypoints1, descriptors1);


    cv::Mat meter = cv::imread("../pictures/meter_experiment/pic79.JPG", 1);
    cv::imshow("j", meter);
    cv::waitKey();

    Template::readMeter(meter);

    return 0;
#endif
}

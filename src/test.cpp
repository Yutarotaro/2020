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

std::vector<cv::KeyPoint> keypointsP;
cv::Mat descriptorsP;

std::vector<cv::Point> featurePoint;
cv::Mat H;
cv::Mat HP;

int main(int argc, char** argv)
{
    //parameter
    Init::parseInit();

#if 1

    cv::Mat Base_calib = cv::imread("../pictures/meter_experiment/pic-1.JPG", 1);
    cv::Mat Base_clock_tmp = cv::imread("../pictures/meter_experiment/pic-2.JPG", 1);
    std::cout << "pic-2" << std::endl;
    cv::Mat mask = cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
    cv::circle(mask, cv::Point(Base_clock_tmp.cols / 2 + 90, Base_clock_tmp.rows / 2 - 120), 360, cv::Scalar(255), -1, CV_AA);
    //    cv::circle(mask, cv::Point(1290, 2207), 370, cv::Scalar(255), -1, CV_AA);

    cv::Mat Base_clock;
    Base_clock_tmp.copyTo(Base_clock, mask);
    //cv::imwrite("mask.png", Base_clock);


    //基準画像の特徴点を事前に検出しておく
    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();
    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);


    Module::pose p;
    Calib::calibration(Base_calib, p);  //pに基準とする画像のカメラ視点，位置が入る

    cv::Mat rvec;  //回転ベクトル

    int st = 0;
    int to = 125;

    std::ofstream ofs("./result455forpresentation.csv");
    //    std::ofstream ofs("./resultAHAinv.csv");
    std::ofstream ofs2("./failure.csv");
    int failure = 0;  //i = 54をcountしている
    int ct = 0;

    for (int i = st; i <= to; i++) {
        std::cout << std::endl
                  << i + 1 << "回目" << std::endl;
        std::string path = "../pictures/meter_experiment/pic" + std::to_string(i) + ".JPG";

        cv::Mat Now_calib = cv::imread(path, 1);
        cv::Mat Now_clock = cv::imread(path, 1);

        Module::pose q;
        Calib::calibration(Now_calib, q);  //qに対象画像のカメラ視点，位置が入る

        cv::Mat R_calib = q.orientation * p.orientation.inv();
        cv::Rodrigues(R_calib, rvec);

        cv::Mat calib_pos = p.orientation.inv() * p.position - q.orientation.inv() * q.position;

        std::cout
            << "chessboard並進ベクトル" << std::endl
            << calib_pos << std::endl
            << "回転ベクトル" << std::endl
            << rvec << std::endl
            << std::endl;


        auto H = Module::getHomography(Base_clock, Now_clock);


        Module::pose r = Module::decomposeHomography(H, A);
        std::cout << "Homography分解で得られた並進ベクトル(world)" << std::endl
                  << r.position << std::endl
                  << std::endl
                  << "回転ベクトル" << std::endl
                  << r.orientation << std::endl
                  << std::endl;

        cv::Mat R_estimated;
        cv::Rodrigues(r.orientation, R_estimated);

        cv::Mat angle_error_mat = R_estimated * R_calib.inv();
        cv::Mat angle_error;
        cv::Rodrigues(angle_error_mat, angle_error);
        double error = cv::norm(angle_error);

        std::cout << "dist by calib(mm) = " << cv::norm(calib_pos) << std::endl
                  << "error dist(mm) = " << cv::norm(calib_pos - r.position) << std::endl
                  << cv::norm(r.orientation) * 180.0 / CV_PI << ' ' << cv::norm(rvec) * 180.0 / CV_PI << ' ' << error * 180.0 / CV_PI << std::endl;

        if (cv::norm(calib_pos) < 1000)
            ct++;
        if (cv::norm(calib_pos) < 1000 && cv::norm(calib_pos) > cv::norm(calib_pos - r.position)) {
            //            ofs << i << ',' << r.position << std::endl;
            ofs << i << ',' << cv::norm(calib_pos) << ',' << cv::norm(calib_pos - r.position) << ',' << cv::norm(calib_pos - r.position) / cv::norm(calib_pos) << ',' << cv::norm(r.orientation) * 180.0 / CV_PI << ',' << cv::norm(rvec) * 180.0 / CV_PI << ',' << error * 180.0 / CV_PI << ',' << std::endl;

        } else {
            failure++;
            ofs2 << i << ',' << failure << ',' << cv::norm(calib_pos) << ',' << std::endl;
        }
    }
    std::cout << "total number of images under 1000mm" << ct << std::endl;
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


    cv::Mat meter = cv::imread("../pictures/meter_experiment/pic66.JPG", 1);

    Template::readMeter(meter);

    return 0;
#endif
}

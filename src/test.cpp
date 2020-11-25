#include "common/init.hpp"
#include "modules/pos/Homography.hpp"
#include "modules/pos/calib.hpp"
#include "modules/read/difference.hpp"
#include "modules/read/template.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "params/pose_params.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

Camera_pose camera;

int type;

cv::Mat temp;
std::string meter_type_s = "dia_V";
int it;
int meter_type = 2;
std::vector<cv::KeyPoint> keypoints1;
cv::Mat descriptors1;

std::vector<cv::KeyPoint> keypointsP;
cv::Mat descriptorsP;

std::vector<cv::Point> featurePoint;
cv::Mat H;
cv::Mat HP;

Init::Params params[] = {{70, 126, "meter_experiment", cv::Point(1899, 979), 620, 25.2, 1.81970, 80. / CV_PI},
    {100, 92, "meter_experiment_V", cv::Point(1808, 1033), 680, -0.0101, 1.88299, 33.5 * 0.002 / CV_PI}};

int main(int argc, char** argv)
{
    //parameter
    Init::parseInit();


    // cv::Mat Base_calib = cv::imread("../pictures/meter_experiment/pic-1.JPG", 1);
    cv::Mat Base_calib = cv::imread("../pictures/dia_experiment_V/pic0.JPG", 1);
    //cv::Mat Base_clock_tmp = cv::imread("../pictures/meter_experiment/pic-2.JPG", 1);
    cv::Mat Base_clock = cv::imread("../pictures/meter_template/Base_clockdia_V.png", 1);


    //基準画像の特徴点を事前に検出しておく
    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();
    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);


    Module::pose p;
    Calib::calibration(Base_calib, p, 1);  //pに基準とする画像のカメラ視点，位置が入る
    //image上のchessboardからtemplateのカメラの位置姿勢を推定

    cv::Mat rvec;  //回転ベクトル

    int st = std::stoi(argv[1]);
    int to = st;

    std::ofstream ofs("./result_for_dia.csv");
    int ct = 0;


    for (int i = st; i <= to; i++) {
        std::cout << std::endl
                  << i << "-th image" << std::endl;
        std::string path = "../pictures/dia_experiment_V/pic" + std::to_string(i) + ".JPG";

        cv::Mat Now_calib = cv::imread(path, 1);
        cv::Mat Now_clock = cv::imread(path, 1);

        std::cout << "camera pose of test image" << std::endl;
        Module::pose q;
        Calib::calibration(Now_calib, q, 0);  //qに対象画像のカメラ視点，位置が入る

        cv::Mat R_calib = q.orientation * p.orientation.inv();
        cv::Rodrigues(R_calib, rvec);

        cv::Mat calib_pos = p.orientation.inv() * p.position - q.orientation.inv() * q.position;

        std::cout << "chessboard並進ベクトル" << std::endl
                  << calib_pos << std::endl
                  << "回転ベクトル" << std::endl
                  << rvec << std::endl
                  << std::endl;


        cv::Mat H = Module::getHomography(Base_clock, Now_clock);
        std::cout << "Homography: " << camera.A.inv() * H * camera.A << std::endl;
        cv::Mat warped = cv::Mat::zeros(Now_clock.rows, Now_clock.cols, CV_8UC3);
        cv::warpPerspective(Now_clock, warped, H.inv(), warped.size());

        cv::Mat H2 = Module::getHomography(Base_clock, warped);

        Module::pose r = Module::decomposeHomography(H * H2, camera.A);

        std::cout << std::endl
                  << "Homography分解で得られた並進ベクトル(world)" << std::endl
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
        }
    }
    std::cout << "total number of images under 1000mm" << ct << std::endl;
    return 0;
}

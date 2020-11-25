#include "common/init.hpp"
#include "modules/pos/calib.hpp"
#include "modules/pos/homography.hpp"
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


    //Base_clockの特徴点を保存
    Init::Feature Base(Base_clock);


    Module::pose p;
    Calib::calibration(Base_calib, p, 1);  //pに基準とする画像のカメラ視点，位置が入る
    //image上のchessboardからtemplateのカメラの位置姿勢を推定

    cv::Mat rvec;  //回転ベクトル

    int st = std::stoi(argv[1]);
    int to = st;

    std::string fileName = "./result_for_dia.csv";
    std::ofstream ofs(fileName, std::ios::app);

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

        //Homgraphy Estimation

#if 0

        cv::Mat roi = cv::imread("../pictures/dia_experiment_V/roi/pic" + std::to_string(i) + ".png", 1);
        double temp_size = 784.;
        double rate = std::max(temp_size / roi.cols, temp_size / roi.rows);

        cv::Mat resized_roi;
        cv::resize(roi, resized_roi, cv::Size(), rate, rate);


        //Base_clock -> resized_roi
        cv::Mat H_0 = Module::getHomography(Base_clock, resized_roi);

        //resized_roi -> roi
        Init::Feature f_resized_roi(resized_roi);
        cv::Mat H_1 = Module::getHomography(f_resized_roi.keypoints, f_resized_roi.descriptors, resized_roi, roi);

        //roi -> Now_clock
        Init::Feature f_roi(roi);
        cv::Mat H_2 = Module::getHomography(f_roi.keypoints, f_roi.descriptors, roi, Now_clock);
        //要改良　画像のサイズがあっていないと意味がない
        cv::Mat H_temp = H_0 * H_1 * H_2;


#endif
        //2 steps homography estimation
        cv::Mat H = Module::getHomography(Base.keypoints, Base.descriptors, Base_clock, Now_clock);
        cv::Mat warped = cv::Mat::zeros(Now_clock.rows, Now_clock.cols, CV_8UC3);
        cv::warpPerspective(Now_clock, warped, H.inv(), warped.size());

        cv::Mat H2 = Module::getHomography(Base.keypoints, Base.descriptors, Base_clock, warped);


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

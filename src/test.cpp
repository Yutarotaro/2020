#include "common/init.hpp"
#include "external/picojson.h"
#include "modules/pos/calib.hpp"
#include "modules/pos/homography.hpp"
#include "modules/read/difference.hpp"
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

Init::Params params;

cv::Point ROI_tl[1000];
int main(int argc, char** argv)
{
    //parameter
    Init::parseInit();

    // JSONデータの読み込み。
    std::ifstream ifs("save.json", std::ios::in);
    if (ifs.fail()) {
        std::cerr << "failed to read test.json" << std::endl;
        return 1;
    }
    const std::string json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    // JSONデータを解析する。
    picojson::value v;
    const std::string err = picojson::parse(v, json);
    if (err.empty() == false) {
        std::cerr << err << std::endl;
        return 2;
    }
    picojson::object& obj = v.get<picojson::object>();


    // cv::Mat Base_calib = cv::imread("../pictures/meter_experiment/pic-1.JPG", 1);
    cv::Mat Base_calib = cv::imread("../pictures/dia_experiment_V/pic0.JPG", 1);
    //cv::Mat Base_clock_tmp = cv::imread("../pictures/meter_experiment/pic-2.JPG", 1);
    cv::Mat Base_clock = cv::imread("../pictures/meter_template/Base_clockdia_V.png", 1);


    //Base_clockの特徴点を保存
    Init::Feature Base(Base_clock);
    //Init::Feature Base("../pictures/meter_template/Base_clockdia_V.png");


    Module::pose p;
    Calib::calibration(Base_calib, p, 1);  //pに基準とする画像のカメラ視点，位置が入る
    //image上のchessboardからtemplateのカメラの位置姿勢を推定

    cv::Mat rvec;  //回転ベクトル

    int st = std::stoi(argv[1]);
    int to = st;


    //読み取り結果を記録
    std::string fileName = "./diffjust/" + meter_type_s + "/data/pos/" + (argc >= 3 ? argv[2] : "") + ".csv";
    std::ofstream ofs(fileName, std::ios::app);

    int ct = 0;


    for (int i = st; i <= to; i++) {
        std::cout << i << "-th image" << std::endl;


        picojson::array& ary = obj[std::to_string(i)].get<picojson::array>();
        ROI_tl[i].x = std::stoi(ary[0].get<std::string>());
        ROI_tl[i].y = std::stoi(ary[1].get<std::string>());


        std::string path_calib = "../pictures/dia_experiment_V/pic" + std::to_string(i) + ".JPG";

        //mask image
        std::string path_clock = "../pictures/dia_experiment_V/mask/pic" + std::to_string(i) + ".png";

        cv::Mat Now_calib = cv::imread(path_calib, 1);
        // cv::Mat Now_clock = cv::imread(path_clock, 1);
        Init::Feature Now_clock(path_clock);

        Module::pose q;
        Calib::calibration(Now_calib, q, 0);  //qに対象画像のカメラ視点，位置が入る

        cv::Mat R_calib = q.orientation * p.orientation.inv();
        cv::Rodrigues(R_calib, rvec);


        //offset world coordinate
        cv::Mat offset = (cv::Mat_<double>(3, 1) << -337., -73.2, -410.);

        //横軸: メータからの距離(chess boardからではない)
        cv::Mat distance = -q.orientation.inv() * q.position + offset;

        std::cout << "from meter: " << distance << std::endl;


        cv::Mat calib_pos = p.orientation.inv() * p.position - q.orientation.inv() * q.position;
        //cv::Mat calib_pos = p.orientation.inv() * p.position - q.orientation.inv() * q.position;

        std::cout << "chessboard並進ベクトル" << std::endl
                  << calib_pos << std::endl
                  << "回転ベクトル" << std::endl
                  << rvec << std::endl
                  << std::endl;

        //Homgraphy Estimation


        Init::Feature ROI("../pictures/dia_experiment_V/roi/pic" + std::to_string(i) + ".png");
        cv::Mat roi = cv::imread("../pictures/dia_experiment_V/roi/pic" + std::to_string(i) + ".png", 1);
        double temp_size = 784.;
        double rate = std::max(temp_size / roi.cols, temp_size / roi.rows);

        cv::Mat resized_roi;
        cv::resize(roi, resized_roi, cv::Size(), rate, rate);
        Init::Feature R_ROI(resized_roi);

        Base.getHomography(R_ROI);
        for (auto j : R_ROI.featurePoint) {
            ROI.featurePoint.push_back(cv::Point(j.x / rate + ROI_tl[i].x, j.y / rate + ROI_tl[i].y));  //resized_roi内座標をNow_clock内座標に変換
        }

        cv::Mat masks;
        cv::Mat H4 = cv::findHomography(Base.featurePoint, ROI.featurePoint, masks, cv::RANSAC, 3);
        cv::Mat warped_2 = cv::Mat::zeros(Now_clock.img.rows, Now_clock.img.cols, CV_8UC3);
        cv::warpPerspective(Now_clock.img, warped_2, H4.inv(), warped_2.size());

        Init::Feature Warp(warped_2);
        cv::Mat H3 = Base.getHomography(Warp);


        Module::pose r = Module::decomposeHomography(H4 * H3, camera.A);

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

        std::cout << "distance to meter (mm) = " << cv::norm(distance) << std::endl
                  << "error dist(mm) = " << cv::norm(calib_pos - r.position) << std::endl
                  << cv::norm(r.orientation) * 180.0 / CV_PI << ' ' << cv::norm(rvec) * 180.0 / CV_PI << ' ' << error * 180.0 / CV_PI << std::endl;

        //if (cv::norm(calib_pos) > cv::norm(calib_pos - r.position)) {
        //            ofs << i << ',' << r.position << std::endl;
        ofs << i << ',' << cv::norm(distance) << ',' << cv::norm(calib_pos - r.position) << ',' << cv::norm(calib_pos - r.position) / cv::norm(calib_pos) << ',' << cv::norm(r.orientation) * 180.0 / CV_PI << ',' << cv::norm(rvec) * 180.0 / CV_PI << ',' << error * 180.0 / CV_PI << ',' << std::endl;
        //}
    }
    return 0;
}

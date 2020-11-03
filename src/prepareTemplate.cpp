#include "Eigen/Dense"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::string picdir = "meter_experiment";
cv::Mat temp;

int main(int argc, char** argv)
{


    std::cout << "type of analog meter: ThermoMeter -> T or Vacuum -> V" << std::endl;
    std::string meter_type;
    std::cin >> meter_type;

    cv::Mat Base_calib;
    cv::Mat Base_clock_tmp;

    if (meter_type == "T") {
        Base_calib = cv::imread("../pictures/" + picdir + "/pic-1.JPG", 1);
        Base_clock_tmp = cv::imread("../pictures/" + picdir + "/pic-3.JPG", 1);

    } else {
        std::cout << "not ready" << std::endl;
        return 0;
    }
    //文字盤以外にマスクをかける処理
    cv::Mat mask = cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
    cv::circle(mask, cv::Point(2214, 1294), 310, cv::Scalar(255), -1, CV_AA);

    cv::Mat Base_clock;  //メータ領域だけを残した基準画像
    Base_clock_tmp.copyTo(Base_clock, mask);

    cv::Rect roi_temp(cv::Point(1899, 979), cv::Size(620, 620));  //基準画像におけるメータ文字盤部分の位置
    temp = Base_clock(roi_temp);

    ///////////////////////////////

    cv::imwrite("../pictures/meter_template/Base_clock" + meter_type + ".png", Base_clock);
    cv::imwrite("../pictures/meter_template/temp" + meter_type + ".png", temp);
}

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat temp;
int meter_type;
std::string meter_type_s;
// type 0: normal, 1: pointer_considered
int type;
// record 0: no, 1:Yes
int record;

void message();

class Params {
public:
  cv::Point center;
  int radius;
  std::string picdir; //ディレクトリ
  std::string index;
  cv::Point tl; //左上
  int l;        //矩形の大きさ
};

// 0:T, 1:V
Params params[] = {
    {cv::Point(2214, 1294), 310, "meter_experiment", "-3", cv::Point(1899, 979),
     620},
    {cv::Point(2158, 1383), 340, "meter_experiment_V", "-1",
     cv::Point(1808, 1033), 680},
    {cv::Point(2160, 1708), 372, "dia_experiment_V", "-1",
     cv::Point(1762, 1315), 784},
    {cv::Point(2109, 1616), 243, "bthesis", "-2", cv::Point(1864, 1374), 486},
    {cv::Point(2263, 1800), 316, "bthesis", "-1", cv::Point(1947, 1482), 632}};

std::map<std::string, int> mp;

int main(int argc, char **argv) {

  message();

  cv::Mat Base_clock_tmp =
      cv::imread("../pictures/" + params[meter_type].picdir + "/pic" +
                     // params[meter_type].index + ".JPG",
                     "-3.JPG",
                 1);

  // 2158,1383
  //文字盤以外にマスクをかける処理
  cv::Mat mask =
      cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
  cv::circle(mask, params[meter_type].center, params[meter_type].radius,
             cv::Scalar(255), -1, CV_AA);

  cv::Mat Base_clock; //メータ領域だけを残した基準画像
  Base_clock_tmp.copyTo(Base_clock, mask);

  cv::Rect roi_temp(
      params[meter_type].tl,
      cv::Size(params[meter_type].l,
               params[meter_type].l)); //基準画像におけるメータ文字盤部分の位置
  temp = Base_clock(roi_temp);

  ///////////////////////////////

  cv::imwrite("../pictures/meter_template/temp.png");
  //  cv::imwrite("../pictures/meter_template/Base_clock" + meter_type_s +
  //  ".png",            Base_clock);
  // cv::imwrite("../pictures/meter_template/temp" + meter_type_s +
  // ".png", temp);
}

void message() {
  mp["T"] = 0;
  mp["V"] = 1;
  mp["dia_V"] = 2;
  mp["bthesis"] = 4;

  std::cout << "type of analog meter: ThermoMeter -> T or Vacuum -> V or dia_V "
               "-> dia_V"
            << std::endl;
  std::cin >> meter_type_s;

  meter_type = mp[meter_type_s];
}

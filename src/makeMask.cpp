#include "Eigen/Dense"
#include "common/init.hpp"
#include "pos/calib.hpp"
#include "pos/fromtwo.hpp"
#include "pos/module.hpp"
#include "read/difference.hpp"
#include "read/readability.hpp"
#include "read/template.hpp"
#include "sub/fit.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
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

//対応点
std::vector<cv::Point> featurePoint;
std::vector<cv::Point> featurePoint2;

//H:Template to Test Homography
cv::Mat H;
cv::Mat HP;

Eigen::Matrix<float, 6, 1> param;

//テスト画像のindex
int it;

int meter_type;
std::string meter_type_s;
//type 0: normal, 1: pointer_considered
int type;
//record 0: no, 1:Yes
int record;
int message(int argc, char** argv);

int opt_list[] = {5, 40, 45, 63, 89, 97, 104, 106};
int lis[] = {10, 11, 12, 17};
/*class Params
{
public:
    int thresh;          //2値化の閾値
    int total;           //枚数
    std::string picdir;  //ディレクトリ
    cv::Point tl;        //左上
    int l;               //矩形の大きさ
    double front_value;  //正面から読んだときの値
    double front_rad;    //正面から読み取った針の角度
    double k;            //1[deg]に対する変化率
};
*/

//一番マシなメータまでの距離
double z = 449.35;


//0:T, 1:V
Init::Params params[] = {{70, 126, "meter_experiment", cv::Point(1899, 979), 620, 25.2, 1.81970, 80. / CV_PI},
    {100, 92, "meter_experiment_V", cv::Point(1808, 1033), 680, -0.0101, 1.88299 /*CV_PI / 2. * 33 / 40.*/, 33. * 0.002 / CV_PI}};

std::map<std::string, int> mp;

int ite = 3;


int main(int argc, char** argv)
{
    //入力が正しいか確認

    if (message(argc, argv)) {
        std::cout << "unexpected inputs" << std::endl;
        return -1;
    }

    if (argc != 6) {
        std::cout << "argc" << std::endl;
    }


    cv::Mat Base_clock = cv::imread("../pictures/meter_template/Base_clock" + meter_type_s + ".png", 1);
    temp = cv::imread("../pictures/meter_template/temp" + meter_type_s + ".png", 1);


    //基準画像の特徴点を事前に検出しておく
    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();
    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);


    int it = std::stoi(argv[1]);


    std::cout << std::endl
              << "/////////////////////////////" << std::endl
              << "picture " << it << std::endl;
    std::string path = "../pictures/" + params[meter_type].picdir + "/pic" + std::to_string(it) + ".JPG";

    cv::Mat Now_clock_o = cv::imread(path, 1);  //for matching

    cv::Mat Now_clock;


    //////////for masking


    cv::Point topleft(std::stoi(argv[2]), std::stoi(argv[3]));
    cv::Point bottomright(std::stoi(argv[4]), std::stoi(argv[5]));

    cv::Mat mask_pa = cv::Mat::zeros(Now_clock_o.rows, Now_clock_o.cols, CV_8UC1);
    cv::rectangle(mask_pa, topleft, bottomright, cv::Scalar(255), -1, CV_AA);
    Now_clock_o.copyTo(Now_clock, mask_pa);

    cv::Rect roi(cv::Point(topleft), cv::Size(bottomright - topleft));
    cv::Mat img = Now_clock_o(roi);


    cv::Mat tmp = cv::imread("../pictures/meter_experiment_V/mask/pic" + std::to_string(it) + ".png", 1);

    try {
        cv::imshow("tmp", tmp);
    } catch (cv::Exception& e) {
        goto OK;
    }
    return 0;

OK:
    cv::imshow("mask", Now_clock);
    cv::imshow("masked", img);


    int key = cv::waitKey();

    if (key == 'q')
        return 0;

    cv::imwrite("../pictures/meter_experiment_V/mask/pic" + std::to_string(it) + ".png", Now_clock);
    cv::imwrite("../pictures/meter_experiment_V/roi/pic" + std::to_string(it) + ".png", img);


    //Homography: Template to Test
    //H = Module::getHomography(Base_clock, Now_clock);
    /////////////////////////////////////////////////////

    return 0;
}

int message(int argc, char** argv)
{
    mp["T"] = 0;
    mp["V"] = 1;


    //meter_type_s = argv[1];
    meter_type_s = "V";
    std::cout << "type of analog meter:" << (meter_type_s == "T" ? "ThermoMeter" : "Vacuum") << std::endl;
    meter_type = mp[meter_type_s];


    return 0;
}

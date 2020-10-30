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
#include <opencv2/ximgproc.hpp>
#include <string>
#include <vector>


std::string picdir = "meter_experiment";

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

//type 0: normal, 1: pointer_considered
int type;

int opt_list[] = {5, 40, 45, 63, 89, 97, 104, 106};


int main(int argc, char** argv)
{

    std::cout << "choose type of homography \n 0:scale-based, 1:pointer based" << std::endl;
    std::cin >> type;

    std::cout << "record in csv file?\n 0:No, 1: Yes" << std::endl;
    int record;
    std::cin >> record;

    //parameter
    Init::parseInit();

    cv::Mat Base_calib = cv::imread("../pictures/" + picdir + "/pic-1.JPG", 1);
    cv::Mat Base_clock_tmp = cv::imread("../pictures/" + picdir + "/pic-3.JPG", 1);

    //文字盤以外にマスクをかける処理
    cv::Mat mask = cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
    cv::circle(mask, cv::Point(2214, 1294), 310, cv::Scalar(255), -1, CV_AA);

    cv::Mat Base_clock;  //メータ領域だけを残した基準画像
    Base_clock_tmp.copyTo(Base_clock, mask);

    cv::Rect roi_temp(cv::Point(1899, 979), cv::Size(620, 620));  //基準画像におけるメータ文字盤部分の位置
    temp = Base_clock(roi_temp);
    ///////////////////////////////


    //基準画像の特徴点を事前に検出しておく
    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();
    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);

    int total = 126;


    int st = (argc == 3 ? std::stoi(argv[2]) : 61);  //start
    int to = 90;                                     //end


    //読み取り結果を記録
    std::ofstream ofs;
    if (record) {
        std::string t = (type ? "pointer" : "normal");
        ofs.open("./diffjust/reading/reading" + t + ".csv");
    }

    ///////
    double z = 449.35;
    pos.at<double>(0, 2) = z;
    t = R * pos;
    ///////


    //for (int itt = 0; itt < 77; ++itt) {
    for (int itt = 0; itt < total; ++itt) {
        //it = list[itt];
        it = itt;


        if (argc == 2) {
            it = std::stoi(argv[1]);
        }

        ////for more accurate Homography
        featurePoint2.clear();          //特徴点ベクトルの初期化
        featurePoint2.shrink_to_fit();  //メモリの開放


        std::cout << std::endl
                  << "picture " << it << std::endl;
        std::string path = "../pictures/meter_experiment/pic" + std::to_string(it) + ".JPG";

        cv::Mat Now_clock_o = cv::imread(path, 1);  //for matching


        cv::Mat Now_clock;
        //        Now_clock_o.copyTo(Now_clock, masko);
        Now_clock_o.copyTo(Now_clock);

        H = Module::getHomography(Base_clock, Now_clock);
        /////////////////////////////////////////////////////


        cv::Mat sub_dst = cv::Mat::zeros(Now_clock.rows, Now_clock.cols, CV_8UC3);

        try {
            cv::warpPerspective(Now_clock, sub_dst, (type ? Module::remakeHomography(H).inv() : H.inv()), sub_dst.size());
        } catch (cv::Exception& e) {
            if (record) {
                ofs << it << ',' << 0 << std::endl;
            }
            continue;
        }

        cv::Rect roi2(cv::Point(1899, 979), cv::Size(620, 620));  //基準画像におけるメータ文字盤部分の位置
        cv::Mat right = sub_dst(roi2);                            // 切り出し画像
        //right: テスト画像を正面視点へ変換し，切り取ったもの

        cv::imwrite("./diffjust/perspective/transformed" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", right);


        //remakeHomographyを使う前提
        //正面切り抜き画像tempをright中のスケールが消えるように変換
        //2段階homography
        std::vector<cv::KeyPoint> keypointsR;
        cv::Mat descriptorsR;

        cv::Ptr<cv::Feature2D> featureR;
        featureR = cv::AKAZE::create();
        featureR->detectAndCompute(right, cv::Mat(), keypointsR, descriptorsR);


        cv::Mat HR = Module::getHomography(keypointsR, descriptorsR, right, temp);
        //right -> tempのhomography

        cv::Mat temp_modified = cv::Mat::zeros(temp.rows, temp.cols, CV_8UC3);
        //cv::warpPerspective(right, right_modified, HR, right_modified.size());


        //right -> tempではなく，temp -> right して比較したい
        //tempのスケールを上手く変換し，right内の針以外の情報をなるべく消したい
        try {
            cv::warpPerspective(temp, temp_modified, HR.inv(), temp_modified.size());
        } catch (cv::Exception& e) {

            if (record) {
                ofs << it << ',' << 0 << std::endl;
            }
            continue;
        }

        cv::Mat diff;
        cv::absdiff(right, temp_modified, diff);

        //文字盤より外を黒く

        int d = 4;
        cv::Mat mask_for_dif = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
        cv::circle(mask_for_dif, cv::Point(310, 310), 310 - d, cv::Scalar(255), -1, 0);

        cv::Mat dif;  //メータ領域だけを残した基準画像
        diff.copyTo(dif, mask_for_dif);
        ///////////////////


        cv::Mat graydiff;
        //グレースケール化
        cv::cvtColor(dif, graydiff, CV_BGR2GRAY);

        //binarization
        //改善の余地あり
        cv::Mat bin;
        double thresh = 70;
        double maxval = 255;
        int thre_type = cv::THRESH_BINARY;
        cv::threshold(graydiff, bin, thresh, maxval, thre_type);


        cv::imshow("dif", bin);
        cv::imwrite("./diffjust/diff" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", bin);

        int iter = 0;
        cv::erode(bin, bin, cv::Mat(), cv::Point(-1, -1), iter);
        cv::dilate(bin, bin, cv::Mat(), cv::Point(-1, -1), iter);

        if (iter) {
            cv::imwrite("./diffjust/mor" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", bin);
        }
        cv::ximgproc::thinning(bin, bin, cv::ximgproc::WMF_EXP);

        cv::Rect roi_onlyPointer(cv::Point(220, 220), cv::Size(280, 280));
        cv::Mat pointerImage = bin(roi_onlyPointer);  // 切り出し画像

        //TODO:Hough Transform
        //PCAにしたい
        std::pair<double, cv::Mat> a;
        a.first = 0.;
        a = Readability::pointerDetection(pointerImage);

        std::cout << it << "-th read value = " << a.first << std::endl;
        cv::imwrite("./diffjust/reading/" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", a.second);

        if (record) {
            ofs << it << ',' << a.first << std::endl;
        }

        cv::waitKey(2);

        continue;
        //可読性判定part
        //Readability::judge(right, i);
        //judge:(target image, number of trial, read or not)
        //double value = Readability::judge(right_modified, it, 1);
        double value = Readability::judge(right, it, 1);
        if (record)
            ofs << ',' << (double)value << ',' << std::endl;
        std::cout << "value " << value << std::endl;

        //Readability::read(subImg);

        if (argc < 2) {
            cv::waitKey();
        }
        cv::Mat gray;


        continue;
    }

    return 0;
}

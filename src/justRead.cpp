#include "AdaptiveIntegralThresholding/thresh.hpp"
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


    //parameter
    Init::parseInit();


    cv::Mat Base_clock = cv::imread("../pictures/meter_template/Base_clock" + meter_type_s + ".png", 1);
    temp = cv::imread("../pictures/meter_template/temp" + meter_type_s + ".png", 1);

    //   cv::Mat hsv;
    //  cv::cvtColor(temp, hsv, cv::COLOR_BGR2HSV);
    //    cv::imshow("hsv", hsv);

    ///////////////////////////////


    //基準画像の特徴点を事前に検出しておく
    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();
    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);


    //読み取り結果を記録
    std::ofstream ofs;
    if (record) {
        std::string t = (type ? "pointer" : "normal");
        ofs.open("./diffjust/" + meter_type_s + "/reading/reading" + t + ".csv");
    }

    ///////
    double z = 449.35;
    pos.at<double>(0, 2) = z;
    t = R * pos;
    ///////

    int st = (argc > 4 ? std::stoi(argv[4]) : 0);
    int en = (argc > 5 ? std::stoi(argv[5]) : params[meter_type].total);


    //itt 回数
    //it  画像のindex
    for (int itt = st; itt < en; ++itt) {
        //    for (int itt = 0; itt < 3; ++itt) {
        it = itt;
        //    it = lis[itt];

        ////for more accurate Homography
        featurePoint2.clear();          //特徴点ベクトルの初期化
        featurePoint2.shrink_to_fit();  //メモリの開放


        std::cout << std::endl
                  << "/////////////////////////////" << std::endl
                  << "picture " << it << std::endl;
        std::string path = "../pictures/" + params[meter_type].picdir + "/pic" + std::to_string(it) + ".JPG";

        cv::Mat Now_clock_o = cv::imread(path, 1);  //for matching

        cv::Mat Now_clock;


        //////////for masking
        if (false) {
            /*            cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
            if (!fs.isOpened()) {
                std::cerr << "File can not be opened." << std::endl;
            }
            cv::Mat pa = cv::zeros(3, 1, int);

            fs["17"] >> pa;
            */


            cv::Point topleft[] = {cv::Point(1911, 1325), cv::Point(1940, 1197), cv::Point(2101, 1156), cv::Point(2234, 1257)};
            cv::Size bottomright[] = {cv::Point(2276, 1710), cv::Size(2220, 1597), cv::Size(2521, 1556), cv::Size(2394, 1857)};

            cv::Mat mask_pa = cv::Mat::zeros(Now_clock_o.rows, Now_clock_o.cols, CV_8UC1);
            cv::rectangle(mask_pa, topleft[itt], bottomright[itt], cv::Scalar(255), -1, CV_AA);
            Now_clock_o.copyTo(Now_clock, mask_pa);


            cv::imshow("masked", Now_clock);
            cv::waitKey();

        } else {
            Now_clock_o.copyTo(Now_clock);
        }

        cv::Mat hsv;
        cv::cvtColor(Now_clock, hsv, cv::COLOR_BGR2HSV);
        cv::imshow("hsv", hsv);

        //Homography: Template to Test
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


        cv::Rect roi2(params[meter_type].tl, cv::Size(params[meter_type].l, params[meter_type].l));  //基準画像におけるメータ文字盤部分の位置
        cv::Mat right = sub_dst(roi2);                                                               // 切り出し画像
        //right: テスト画像を正面視点へ変換し，切り取ったもの

        cv::imwrite("./diffjust/" + meter_type_s + "/perspective/transformed" + std::to_string(it) + (type ? "pointer" : "normal") + meter_type_s + ".png", right);


        //remakeHomographyを使う前提
        //正面切り抜き画像tempをright中のスケールが消えるように変換
        //2段階homography
        std::vector<cv::KeyPoint> keypointsR;
        cv::Mat descriptorsR;

        cv::Ptr<cv::Feature2D> featureR;
        featureR = cv::AKAZE::create();
        featureR->detectAndCompute(right, cv::Mat(), keypointsR, descriptorsR);


        cv::Mat HR = Module::getHomography(keypointsR, descriptorsR, right, temp);
        std::cout << HR << std::endl;
        //right -> tempのhomography

        //cv::Mat temp_modified = cv::Mat::zeros(temp.rows, temp.cols, CV_8UC3);
        cv::Mat right_modified = cv::Mat::zeros(temp.rows, temp.cols, CV_8UC3);
        //cv::warpPerspective(right, right_modified, HR, right_modified.size());


        //right -> tempではなく，temp -> right して比較したい
        //11/05 やっぱりright -> temp そうしないとtempが壊れる
        //tempのスケールを上手く変換し，right内の針以外の情報をなるべく消したい

        try {
            cv::warpPerspective(right, right_modified, HR, right_modified.size());
        } catch (cv::Exception& e) {
            if (record) {
                ofs << it << ',' << 0 << std::endl;
            }
            continue;
        }


        cv::Mat gray_right;
        cv::cvtColor(right_modified, gray_right, cv::COLOR_BGR2GRAY);
        cv::Mat bwr = cv::Mat::zeros(gray_right.size(), CV_8UC1);
        Adaptive::thresholdIntegral(gray_right, bwr);
        cv::erode(bwr, bwr, cv::Mat(), cv::Point(-1, -1), 1);


        cv::Mat gray_tempm;
        cv::cvtColor(temp, gray_tempm, cv::COLOR_BGR2GRAY);
        cv::Mat bwt = cv::Mat::zeros(gray_tempm.size(), CV_8UC1);
        Adaptive::thresholdIntegral(gray_tempm, bwt);
        cv::dilate(bwt, bwt, cv::Mat(), cv::Point(-1, -1), 1);

        cv::Mat diff;
        //cv::absdiff(right, temp_modified, diff);
        //        cv::absdiff(bwr, bwt, diff);
        //cv::bitwise_xor(bwr, bwt, diff);
        //diff = bwt - bwr;
        diff = bwr - bwt;


        cv::imshow("bwr", bwr);
        //        cv::imwrite("./diffjust/V/bwr.png", bwr);
        cv::imshow("bwr_origin", gray_right);
        cv::imshow("bwt", bwt);


        //文字盤より外を黒く

        //int d = 4;
        int d = 80;
        cv::Mat mask_for_dif = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
        cv::circle(mask_for_dif, cv::Point(params[meter_type].l / 2, params[meter_type].l / 2), params[meter_type].l / 2 - d, cv::Scalar(255), -1, 0);

        cv::Mat dif;  //meter領域のみ残した差分画像
        diff.copyTo(dif, mask_for_dif);
        ///////////////////
        cv::imshow("dif", dif);

        cv::imwrite("./diffjust/" + meter_type_s + "/diff/" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", dif);
        cv::erode(dif, dif, cv::Mat(), cv::Point(-1, -1), ite);
        cv::dilate(dif, dif, cv::Mat(), cv::Point(-1, -1), ite);

        int cnt = cv::countNonZero(dif);


        //        cv::ximgproc::thinning(dif, dif, cv::ximgproc::WMF_EXP);
        //       cv::imshow("thinning", dif);


        cv::Rect roi_onlyPointer(cv::Point(params[meter_type].l / 4, params[meter_type].l / 4), cv::Size(params[meter_type].l / 2, params[meter_type].l / 2));
        cv::Mat pointerImage = dif(roi_onlyPointer);  // 切り出し画像


        cv::ximgproc::thinning(pointerImage, pointerImage, cv::ximgproc::WMF_EXP);
        cv::imshow("thinning", pointerImage);

        std::pair<double, cv::Mat> aa;
        aa.first = 0.;
        aa = Readability::pointerDetection(pointerImage);


        std::cout << it << "-th read value = " << aa.first << std::endl
                  << "white: " << cnt << std::endl
                  << "/////////////////////////////" << std::endl;

        cv::imwrite("./diffjust/" + meter_type_s + "/reading/" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", aa.second);

        if (record) {
            ofs << it << ',' << aa.first << ',' << cnt << std::endl;
        }

        if (argc == 7) {
            cv::waitKey(std::stoi(argv[6]));
        } else {
            cv::waitKey(1);
        }

        continue;

        /////////////移行前のコード
        //
        cv::Mat graydiff;
        //グレースケール化
        cv::cvtColor(dif, graydiff, CV_BGR2GRAY);

        //binarization
        //改善の余地あり

        cv::Mat bin;
        int maxval = 255;
        int thre_type = cv::THRESH_BINARY;

        cv::threshold(graydiff, bin, params[meter_type].thresh, maxval, thre_type);


        cv::imshow("dif", bin);
        cv::imwrite("./diffjust/" + meter_type_s + "/diff/" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", bin);

        //remove noise
        int sigma = 3;
        int k_size = (sigma * 5) | 1;
        cv::GaussianBlur(bin, bin, cv::Size(k_size, k_size), sigma, sigma);

        int iter = 0;
        cv::erode(bin, bin, cv::Mat(), cv::Point(-1, -1), iter);
        cv::dilate(bin, bin, cv::Mat(), cv::Point(-1, -1), iter);

        if (iter) {
            cv::imwrite("./diffjust/" + meter_type_s + "/diff/mor/" + std::to_string(it) + (type ? "pointer" : "normal") + meter_type_s + ".png", bin);
        }
        cv::ximgproc::thinning(bin, bin, cv::ximgproc::WMF_EXP);

        cv::Rect roi_Pointer(cv::Point(220, 220), cv::Size(280, 280));
        cv::Mat pointerImg = bin(roi_Pointer);  // 切り出し画像

        cv::imshow("pointerImage", pointerImg);

        //TODO:Hough Transform
        //PCAにしたい
        std::pair<double, cv::Mat> a;
        a.first = 0.;
        a = Readability::pointerDetection(pointerImg);

        std::cout << it << "-th read value = " << a.first << std::endl
                  << "/////////////////////////////" << std::endl;
        cv::imwrite("./diffjust/" + meter_type_s + "/reading/" + std::to_string(it) + (type ? "pointer" : "normal") + meter_type_s + ".png", a.second);

        if (record) {
            ofs << it << ',' << a.first << std::endl;
        }

        if (argc == 6) {
            cv::waitKey(std::stoi(argv[6]));
        } else {
            cv::waitKey(2);
        }
        continue;
    }

    return 0;
}

int message(int argc, char** argv)
{
    mp["T"] = 0;
    mp["V"] = 1;

    if (argc < 4 || argc > 7) {
        return -1;
    }

    meter_type_s = argv[1];
    std::cout << "type of analog meter:" << (meter_type_s == "T" ? "ThermoMeter" : "Vacuum") << std::endl;
    meter_type = mp[meter_type_s];


    std::string tmp = argv[2];
    type = std::stoi(tmp);
    std::cout << "type of homography: " << type << std::endl;


    tmp = argv[3];
    record = std::stoi(tmp);
    std::cout << "record? :" << (record ? "Yes" : "No") << std::endl;


    return 0;


    /*
    std::cout << "type of analog meter: ThermoMeter -> T or Vacuum -> V" << std::endl;
    std::cin >> meter_type_s;

    meter_type = mp[meter_type_s];

    std::cout << "choose type of homography \n 0:scale-based, 1:pointer based" << std::endl;
    std::cin >> type;

    std::cout << "record in csv file?\n 0:No, 1: Yes" << std::endl;
    std::cin >> record;
    */
}

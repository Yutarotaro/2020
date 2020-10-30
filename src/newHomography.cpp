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

int list[] = {5, 23, 32, 33, 34, 35, 40, 45, 46, 49, 50, 51, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125};

int main(int argc, char** argv)
{


    std::cout << "0:normal homography, 1:pointer_considered homography" << std::endl;
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

    //針以外にマスクをかける処理
    cv::Mat maskp = cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
    cv::ellipse(maskp, cv::Point(2179, 1284), cv::Size(230, 45), 16, 0, 360, cv::Scalar(255), -1, CV_AA);
    cv::Mat onlyPointer;
    Base_calib.copyTo(onlyPointer, maskp);
    ///////////////////////////////


    //基準画像の特徴点を事前に検出しておく
    cv::Ptr<cv::Feature2D> feature;
    feature = cv::AKAZE::create();
    feature->detectAndCompute(Base_clock, cv::Mat(), keypoints1, descriptors1);

    //    cv::Ptr<cv::Feature2D> featureP;
    //    featureP = cv::AKAZE::create();
    //    featureP->detectAndCompute(onlyPointer, cv::Mat(), keypointsP, descriptorsP);


    Module::pose p;
    Calib::calibration(Base_calib, p);  //pに基準とする画像のカメラ視点，位置が入る

    cv::Mat rvec;  //回転ベクトル

    int total = 126;


    int st = (argc == 3 ? std::stoi(argv[2]) : 61);  //start
    int to = 90;                                     //end


    //読み取り結果を記録
    std::ofstream ofs;
    if (record) {
        std::string t = (type ? "pointer" : "normal");
        ofs.open("./reading/reading" + t + ".csv");
    }

    ///////
    double z = 449.35;
    pos.at<double>(0, 2) = z;
    t = R * pos;
    ///////


    //for (int itt = 0; itt < 77; ++itt) {
    for (int itt = 0; itt < 126; ++itt) {
        //for (it = st; it <= to; ++it) {
        //  if (it == 60 || it == 67 || it == 77)
        //    continue;
        //i = -1;

        //it = list[itt];
        it = itt;

        if (it == list[0] && argc == 2) {
            it = std::stoi(argv[1]);
        }

        ////for more accurate Homography
        featurePoint2.clear();          //特徴点ベクトルの初期化
        featurePoint2.shrink_to_fit();  //メモリの開放


        std::cout << std::endl
                  << "picture" << it << std::endl;
        std::string path = "../pictures/meter_experiment/pic" + std::to_string(it) + ".JPG";

        cv::Mat Now_calib = cv::imread(path, 1);    //for calibration
        cv::Mat Now_clock_o = cv::imread(path, 1);  //for matching

        //occlusion
        cv::Mat masko = cv::Mat::zeros(Now_clock_o.rows, Now_clock_o.cols, CV_8UC1);
        cv::circle(masko, cv::Point(2291, 1468), 30, cv::Scalar(255), 1, CV_AA);

        cv::Mat Now_clock;
        //        Now_clock_o.copyTo(Now_clock, masko);
        Now_clock_o.copyTo(Now_clock);


        Module::pose q;
        Calib::calibration(Now_calib, q);  //qに対象画像のカメラ視点，位置が入る

        cv::Mat R_calib = q.orientation * p.orientation.inv();
        cv::Rodrigues(R_calib, rvec);

        cv::Mat calib_pos = p.orientation.inv() * p.position - q.orientation.inv() * q.position;

        H = Module::getHomography(Base_clock, Now_clock);
        Module::pose r = Module::decomposeHomography(H, A);

        //        HP = Module::getHomographyP(onlyPointer, Now_clock);

        cv::Mat R_estimated;
        cv::Rodrigues(r.orientation, R_estimated);

        cv::Mat angle_error_mat = R_estimated * R_calib.inv();
        cv::Mat angle_error;
        cv::Rodrigues(angle_error_mat, angle_error);


        /////////////////////////////////////////////////////


        cv::Mat sub_dst = cv::Mat::zeros(Now_clock.rows, Now_clock.cols, CV_8UC3);
        cv::warpPerspective(Now_clock, sub_dst, (type ? Module::remakeHomography(H).inv() : H.inv()), sub_dst.size());

        cv::Rect roi2(cv::Point(1899, 979), cv::Size(620, 620));  //基準画像におけるメータ文字盤部分の位置
        cv::Mat right = sub_dst(roi2);                            // 切り出し画像

        //cv::imshow("perspective transform", right);

        cv::imwrite("./right/" + std::to_string(it) + ".png", right);


        //cv::Mat new_H = Module::remakeHomography(H);


        //cv::Mat pointer_concerned = cv::Mat::zeros(Now_clock.rows, Now_clock.cols, CV_8UC3);
        //cv::warpPerspective(Now_clock, pointer_concerned, new_H.inv(), pointer_concerned.size());

        //        cv::imshow("pointer_concerned", pointer_concerned);
        //        cv::waitKey();


        //2段階homography
        std::vector<cv::KeyPoint> keypointsR;
        cv::Mat descriptorsR;

        cv::Ptr<cv::Feature2D> featureR;
        featureR = cv::AKAZE::create();
        featureR->detectAndCompute(right, cv::Mat(), keypointsR, descriptorsR);


        cv::Mat HR = Module::getHomography(keypointsR, descriptorsR, right, temp);

        cv::Mat right_modified = cv::Mat::zeros(right.rows, right.cols, CV_8UC3);
        cv::warpPerspective(right, right_modified, HR, right_modified.size());


        cv::Mat dif;
        cv::absdiff(right, right_modified, dif);

        cv::imshow("right_modified", right_modified);
        //        cv::imshow("right_diff", dif);

        //可読性判定part
        //Readability::judge(right, i);
        //judge:(target image, number of trial, read or not)
        //double value = Readability::judge(right_modified, it, 1);
        double value = Readability::judge(right, it, 1);
        if (record)
            ofs << it << ',' << (double)value << ',' << std::endl;
        std::cout << "value " << value << std::endl;

        //Readability::read(subImg);

        if (argc < 2) {
            cv::waitKey();
        }
        cv::Mat gray;


        continue;


#if 1
        cv::Canny(gray, gray, 20, 155);
#else
        int thresh = 140;
        int maxval = 255;
        int type = cv::THRESH_BINARY_INV;
        int method = cv::BORDER_REPLICATE;
        int blocksize = 15;
        double C = 10.0;


        cv::adaptiveThreshold(gray, gray, maxval, method, type, blocksize, C);
#endif


        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(gray,    // 入力画像，8ビット，シングルチャンネル．0以外のピクセルは 1 、0のピクセルは0として扱う。処理結果として image を書き換えることに注意する.
            contours,             // 輪郭を点ベクトルとして取得する
            hierarchy,            // hiararchy ? オプション．画像のトポロジーに関する情報を含む出力ベクトル．
            CV_RETR_EXTERNAL,     // 輪郭抽出モード
            CV_CHAIN_APPROX_NONE  // 輪郭の近似手法
        );

        std::cout << "number of contours" << contours.size() << std::endl;
        for (int i = 0; i < contours.size(); i++) {
            std::cout << contours[i].size() << std::endl;
        }
        cv::imshow("diw", gray);
        cv::imwrite("edge" + std::to_string(it) + ".jpg", gray);
        //cv::waitKey();

        cv::Mat points;

        int tmp = 3;
        int idx = 0;


        cv::Mat dst;
        cv::Mat gray2;
        if (true) {
            //            dst = cv::Mat::zeros(NowWithMask.rows + 200, NowWithMask.cols + 200, CV_8UC3);
            //           cv::warpPerspective(NowWithMask, dst, H.inv(), dst.size());

            dst = cv::Mat::zeros(Now_clock.rows + 200, Now_clock.cols + 200, CV_8UC3);
            cv::warpPerspective(Now_clock, dst, H.inv(), dst.size());


            cv::circle(dst, cv::Point(2214, 1294), 364, cv::Scalar(0, 0, 255), 10, CV_AA);
            cv::imshow("nm", dst);
            cv::waitKey();

            cv::warpPerspective(dst, dst, H, dst.size());

            cv::cvtColor(dst, gray2, CV_BGR2GRAY);

            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(gray2, circles, cv::HOUGH_GRADIENT,
                2, gray2.rows / 4, 300, 120, fmin(gray2.rows, gray2.cols) / 16, fmin(gray2.rows, gray2.cols) / 2);

            for (int i = 0; i < circles.size(); ++i) {
                cv::Point2d center = cv::Point2d(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                if (center.x > dst.cols / 2 - 300 && center.x < dst.cols / 2 + 300 && center.y > dst.rows / 2 - 300 && center.y < dst.rows / 2 + 300) {
                    //            cv::circle(dst, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
                }
            }


            cv::imshow("nm", dst);
        }


        cv::waitKey();
    }

    return 0;
}

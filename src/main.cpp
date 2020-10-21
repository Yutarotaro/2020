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
std::vector<cv::Point> featurePoint2;

cv::Mat H;
cv::Mat HP;

Eigen::Matrix<float, 6, 1> param;


int main(int argc, char** argv)
{

    //parameter
    Init::parseInit();

    cv::Mat Base_calib = cv::imread("../pictures/meter_experiment/pic-1.JPG", 1);
    cv::Mat Base_clock_tmp = cv::imread("../pictures/meter_experiment/pic-3.JPG", 1);
    //    Base_clock_tmp.copyTo(Base_clock_tmp_pointer);

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

    //cv::imshow("pointer", onlyPointer);

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


    for (int i = st; i <= to; ++i) {
        if (i == 60 || i == 67)
            continue;
        //i = -1;

        featurePoint.clear();          //特徴点ベクトルの初期化
        featurePoint.shrink_to_fit();  //メモリの開放

        ////for more accurate Homography
        featurePoint2.clear();          //特徴点ベクトルの初期化
        featurePoint2.shrink_to_fit();  //メモリの開放


        std::cout << std::endl
                  << i + 1 << "回目" << std::endl;
        std::string path = "../pictures/meter_experiment/pic" + std::to_string(i) + ".JPG";

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
        //トリミング

        int x_max = 0, x_min = Now_clock.rows, y_max = 0, y_min = Now_clock.cols;
        for (int i = 0; i < featurePoint.size(); ++i) {
            cv::Point now = featurePoint[i];
            if (now.x < x_min) {
                x_min = now.x;
            }
            if (now.x > x_max) {
                x_max = now.x;
            }
            if (now.y < y_min) {
                y_min = now.y;
            }
            if (now.y > y_max) {
                y_max = now.y;
            }
            //            cv::circle(Now_clock, featurePoint[i], 2, cv::Scalar(0, 0, 255), -1, CV_AA);
            //            マッチング点の描画
        }


        cv::Mat img = Now_clock.clone();
        cv::Rect brect = cv::boundingRect(cv::Mat(featurePoint).reshape(2));
        cv::rectangle(img, cv::Point(brect.tl().x - 100, brect.tl().y - 100), cv::Point(brect.br().x + 100, brect.br().y + 100), cv::Scalar(190, 30, 100), 2, CV_AA);
        //cv::imshow("trimmed", img);


        int margin = (x_max - x_min) / 1.5;
        cv::Mat NowWithMask;
        cv::Mat mask = cv::Mat::zeros(Base_clock_tmp.rows, Base_clock_tmp.cols, CV_8UC1);
        //        cv::rectangle(mask, cv::Point(x_min - margin, y_min - margin), cv::Point(x_max + margin, y_max + margin), cv::Scalar(255), -2, CV_AA);
        cv::rectangle(mask, cv::Point(brect.tl().x - 100, brect.tl().y - 100), cv::Point(brect.br().x + 100, brect.br().y + 100), cv::Scalar(255), -2, CV_AA);  //特徴点の最小外接矩形より少し大きい矩形でトリミング


        Now_clock.copyTo(NowWithMask, mask);

        //    cv::imshow("n", NowWithMask);

        //cv::Rect roi(cv::Point(brect.tl().x - 100, brect.tl().y - 100), cv::Size(brect.br().x - brect.tl().x + 200, brect.br().y - brect.tl().y + 200));
        //cv::Mat subImg = Now_clock(roi);  // 切り出し画像

        cv::Mat sub_dst = cv::Mat::zeros(Now_clock.rows, Now_clock.cols, CV_8UC3);
        cv::warpPerspective(Now_clock, sub_dst, H.inv(), sub_dst.size());

        cv::Rect roi2(cv::Point(1899, 979), cv::Size(620, 620));  //基準画像におけるメータ文字盤部分の位置
        cv::Mat right = sub_dst(roi2);                            // 切り出し画像

        //cv::imshow("perspective transform", right);

        cv::imwrite("./right/" + std::to_string(i) + ".png", right);


        std::cout << Module::remakeHomography() << std::endl;


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
        Readability::judge(right_modified, i, 1);
        //Readability::read(subImg);

        if (argc < 2) {
            cv::waitKey();
        }
        cv::Mat gray;
        cv::cvtColor(NowWithMask, gray, CV_BGR2GRAY);


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
        cv::imwrite("edge" + std::to_string(i) + ".jpg", gray);
        //cv::waitKey();

        cv::Mat points;

        int tmp = 3;
        int idx = 0;

        for (int i = 0; i >= 0; i = hierarchy[i][0]) {
            if (contours[i].size() > 5 /*00*/) {
                // 2 次元の点集合にフィッティングする楕円を取得
                cv::RotatedRect rc = cv::fitEllipseDirect(contours[i]);

                if (rc.size.width * rc.size.height < (x_max - x_min) * (y_max - y_min) / 2) {
                    continue;
                }

                if (!(rc.center.x > x_min && rc.center.x < x_max && rc.center.y > y_min && rc.center.y < y_max)) {
                    continue;
                }
                cv::Point2f vertices[4];
                rc.points(vertices);
                for (int i = 0; i < 4; i++) {
                    //                    cv::line(Now_clock, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 5);
                }

                cv::Point2f po[4];
                for (int i = 0; i < 4; ++i) {
                    po[i] = (vertices[i] + vertices[(i + 1) % 4]) / 2.0;
                }

                cv::Mat I = H.inv();

                for (int i = 0; i < 4; ++i) {
                    po[i].x = (I.at<double>(0, 0) * po[i].x + I.at<double>(0, 1) * po[i].y + I.at<double>(0, 2)) / (I.at<double>(2, 0) * po[i].x + I.at<double>(2, 1) * po[i].y + I.at<double>(2, 2));
                    po[i].y = (I.at<double>(1, 0) * po[i].x + I.at<double>(1, 1) * po[i].y + I.at<double>(1, 2)) / (I.at<double>(2, 0) * po[i].x + I.at<double>(2, 1) * po[i].y + I.at<double>(2, 2));
                }
                double d1 = cv::norm(cv::Mat(po[0]), cv::Mat(po[2]));
                double d2 = cv::norm(cv::Mat(po[1]), cv::Mat(po[3]));

                //std::cout << d1 / d2 << std::endl;

                if (std::abs(1 - d1 / d2) < tmp) {
                    tmp = std::abs(1 - d1 / d2);
                    idx = i;
                }

                if (contours[i].size() > 5000) {
                    for (int j = 0; j < contours[i].size(); j++) {
                        //           cv::circle(Now_clock, contours[i][j], 2, cv::Scalar(0, 255, 255), -1, CV_AA);
                    }
                }

                // 楕円を描画
                //cv::ellipse(Now_clock, rc, cv::Scalar(0, 128, 0), 10 / 2, CV_AA);
            }
        }
        cv::RotatedRect rc = cv::fitEllipseDirect(contours[idx]);
        cv::ellipse(Now_clock, rc, cv::Scalar(128, 128, 0), 10 / 2, CV_AA);
        cv::namedWindow("ellipse", 1);
        cv::imshow("ellipse", Now_clock);


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

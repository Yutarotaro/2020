#include "module.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>


using namespace cv;
using namespace std;

namespace Module
{

Mat getHomography(Mat Src1, Mat Src2)
{
    //キーポイント検出と特徴量記述
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1, descriptors2;
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(Src1, cv::Mat(), keypoints1, descriptors1);
    akaze->detectAndCompute(Src2, cv::Mat(), keypoints2, descriptors2);

    //マッチング(knnマッチング)
    vector<vector<cv::DMatch>> knnmatch_points;
    cv::BFMatcher match(cv::NORM_HAMMING);
    match.knnMatch(descriptors1, descriptors2, knnmatch_points, 2);

    //対応点を絞る
    const double match_par = 0.6;  //候補点を残す場合のしきい値
    vector<cv::DMatch> goodMatch;
    //KeyPoint -> Point2d
    vector<cv::Point2f> match_point1;
    vector<cv::Point2f> match_point2;
    for (size_t i = 0; i < knnmatch_points.size(); ++i) {
        double distance1 = knnmatch_points[i][0].distance;
        double distance2 = knnmatch_points[i][1].distance;

        //第二候補点から距離値が離れている点のみ抽出（いい点だけ残す）
        if (distance1 <= distance2 * match_par) {
            goodMatch.push_back(knnmatch_points[i][0]);
            match_point1.push_back(keypoints1[knnmatch_points[i][0].queryIdx].pt);
            match_point2.push_back(keypoints2[knnmatch_points[i][0].trainIdx].pt);
        }
    }

    //ホモグラフィ行列推定
    cv::Mat masks;
    cv::Mat H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3);

    //RANSACで使われた対応点のみ抽出
    vector<cv::DMatch> inlinerMatch;
    for (size_t i = 0; i < masks.rows; ++i) {
        uchar* inliner = masks.ptr<uchar>(i);
        if (inliner[0] == 1) {
            inlinerMatch.push_back(goodMatch[i]);
        }
    }

    //対応点の表示
    cv::Mat drawmatch;
    cv::drawMatches(Src1, keypoints1, Src2, keypoints2, goodMatch, drawmatch);
    imwrite("../output/match_point.jpg", drawmatch);

    //インライアの対応点のみ表示
    cv::Mat drawMatch_inliner;
    cv::drawMatches(Src1, keypoints1, Src2, keypoints2, inlinerMatch, drawMatch_inliner);
    imwrite("../output/match_inliner.jpg", drawMatch_inliner);

    imshow("DrawMatch", drawmatch);
    imshow("Inliner", drawMatch_inliner);


    return H;
}


std::pair<Point, int> circleDetect(cv::Mat img)
{
    cv::Mat gray;
    //2値化
    cvtColor(img, gray, COLOR_BGR2GRAY);
    //平滑化
    GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
        2, gray.rows / 4, 150, 100);

    // int C_x = gray.rows / 2;
    //int C_y = gray.cols / 2;

    int tmp = 0;
    int index = -1;
    int radius;

    for (size_t i = 0; i < circles.size(); i++) {
        radius = cvRound(circles[i][2]);
        //暫定: 検出された円のうち最も半径の大きいものをメータとみなす
        if (radius > tmp) {
            tmp = radius;
            index = i;
        }
    }

    cv::Point center(cvRound(circles[index][0]), cvRound(circles[index][1]));
    radius = tmp;

    // 円の中心を描画します．
    circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
    // 円を描画します．
    circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);

    namedWindow("circles", 1);
    imshow("circles", img);

    return {center, radius};
}

cv::Mat decompose()
{
    cv::Mat P;
    cv::Mat K(3, 3, cv::DataType<float>::type);  // intrinsic parameter matrix
    cv::Mat R(3, 3, cv::DataType<float>::type);  // rotation matrix
    cv::Mat T(4, 1, cv::DataType<float>::type);  // translation vector

    /*cv::decomposeProjectionMatrix(P, K, R, T);

    std::cout << R << std::endl
              << T << std::endl;
*/

    return P;
}


}  // namespace Module

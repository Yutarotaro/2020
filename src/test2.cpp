#include "include/init.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main(void)
{
    //    Mat Src1 = imread("graf1.png");
    //Mat Src2 = imread("graf3.png");

    int a;

    system("nohup ./start.sh &");

    Mat Src1, Src2;

    if (Init::input_images(1, 2, Src1) || Init::input_images(1, 3, Src2)) {
        cout << "No such file(s)" << endl;
        return 0;
    }

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

    waitKey();

    return 0;
}

#include "module.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

using namespace cv;
using namespace std;


extern cv::Mat A;
extern cv::Mat R;
extern cv::Mat t;

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
    //    imwrite("../output/match_point.jpg", drawmatch);

    //インライアの対応点のみ表示
    cv::Mat drawMatch_inliner;
    cv::drawMatches(Src1, keypoints1, Src2, keypoints2, inlinerMatch, drawMatch_inliner);
    //imwrite("../output/match_inliner.jpg", drawMatch_inliner);

    //    imshow("DrawMatch", drawmatch);
#if 1
    imshow("Inliner", drawMatch_inliner);
    imwrite("./match_inliner.jpg", drawMatch_inliner);
    cv::waitKey();
#else
#endif
    return H;
}

pose decomposeHomography(cv::Mat H, cv::Mat A)
{
    //https://docs.opencv.org/master/de/d45/samples_2cpp_2tutorial_code_2features2D_2Homography_2decompose_homography_8cpp-example.html#a20

    cv::Mat R1 = R;
    cv::Mat rvec1;
    cv::Rodrigues(R1, rvec1);

    //並進ベクトルの初期値
    //TODO:.xmlからの読み取り
    cv::Mat tvec1 = t;


    cv::Mat normal = (cv::Mat_<double>(3, 1) << 1, 1, 1);
    cv::Mat normal1 = R1 * normal;

    Mat origin(3, 1, CV_64F, Scalar(0));
    Mat origin1 = R1 * origin + tvec1;
    double d_inv1 = 1.0 / normal1.dot(origin1);

    std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

    //solutionsの個数が解の個数(それはそう)
    int solutions = cv::decomposeHomographyMat(H, A, Rs_decomp, ts_decomp, normals_decomp);
    std::cout << "Decompose homography matrix estimated by findHomography():" << std::endl
              << std::endl;

    for (int i = 0; i < solutions; i++) {
        double factor_d1 = 1.0 / d_inv1;
        cv::Mat rvec_decomp;
        cv::Rodrigues(Rs_decomp[i], rvec_decomp);
        std::cout << "Solution " << i << ":" << std::endl;
        std::cout << "rvec from homography decomposition: " << rvec_decomp.t() << std::endl;
        std::cout << "tvec from homography decomposition: \n"
                  << ts_decomp[i].t() << " \n scaled by d:\n " << factor_d1 * ts_decomp[i].t() << std::endl
                  << std::endl;
        std::cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << std::endl
                  << std::endl;

        cv::Mat estimated_pose = Rs_decomp[i] * t + ts_decomp[i];

        std::cout << estimated_pose << std::endl;
    }

    return {Rs_decomp[0], ts_decomp[0]};
}
}  // namespace Module

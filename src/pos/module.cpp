#include "module.hpp"
#include "opencv2/calib3d.hpp"
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

    //    imshow("DrawMatch", drawmatch);
    imshow("Inliner", drawMatch_inliner);

    return H;
}

cv::Mat decomposeP(cv::Mat P)
{
    cv::Mat K(3, 3, cv::DataType<float>::type);  // intrinsic parameter matrix
    cv::Mat R(3, 3, cv::DataType<float>::type);  // rotation matrix
    cv::Mat T(4, 1, cv::DataType<float>::type);  // translation vector

    /*cv::decomposeProjectionMatrix(P, K, R, T);

    std::cout << R << std::endl
              << T << std::endl;
*/

    return P;
}

void reconstructH(cv::Mat H, cv::Mat A)
{
    //https://docs.opencv.org/master/de/d45/samples_2cpp_2tutorial_code_2features2D_2Homography_2decompose_homography_8cpp-example.html#a20

    cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    //double d_inv1 = 1.0 / normal1.dot(origin1);

    std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

    //solutionsの個数が解の個数(それはそう)
    int solutions = cv::decomposeHomographyMat(H, A, Rs_decomp, ts_decomp, normals_decomp);
    std::cout << "Decompose homography matrix estimated by findHomography():" << std::endl
              << std::endl;

    for (int i = 0; i < solutions; i++) {
        //        double factor_d1 = 1.0 / d_inv1;
        double factor_d1 = 1.0;
        cv::Mat rvec_decomp;
        cv::Rodrigues(Rs_decomp[i], rvec_decomp);
        std::cout << "Solution " << i << ":" << std::endl;
        std::cout << "rvec from homography decomposition: " << rvec_decomp.t() << std::endl;
        std::cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << std::endl;
        std::cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << std::endl
                  << std::endl;
    }
}


void normalize(cv::Vec3f& vec)
{
    float a = vec[0] * vec[0];
    float b = vec[1] * vec[1];
    float c = vec[2] * vec[2];

    float norm = sqrt(a + b + c);

    vec[0] /= norm;
    vec[1] /= norm;
    vec[2] /= norm;
}

pose decomposeH(cv::Mat H, cv::Mat A)
{
    //画面座標とワールド座標をつなぐhomography


    //Flexible Camera Calibration By Viewing a Plane From Unknown Orientations Zhang に基づく

    cv::Mat RT = A.inv() * H;


    cv::Vec3f rVec1 = Vec3f(RT.at<double>(0, 0),
        RT.at<double>(1, 0),
        RT.at<double>(2, 0));
    cv::Vec3f rVec2 = Vec3f(RT.at<double>(0, 1),
        RT.at<double>(1, 1),
        RT.at<double>(2, 1));


    //TODO: lamdaをかける
    cv::Vec3f t = Vec3f(RT.at<double>(0, 2),
        RT.at<double>(1, 2),
        RT.at<double>(2, 2));

    cv::Vec3f rVec3 = rVec1.cross(rVec2);

    normalize(rVec1);
    normalize(rVec2);
    normalize(rVec3);

    //cv::Mat m = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    cv::Mat R = (cv::Mat_<double>(3, 3) << rVec1[0], rVec1[1], rVec1[2],
        rVec2[0], rVec2[1], rVec2[2],
        rVec3[0], rVec3[1], rVec3[2]);

    std::cout << R << std::endl;
    std::cout << t << std::endl;

    return {R, t};
}


}  // namespace Module

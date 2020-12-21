#include "homography.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "params/pose_params.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

using namespace cv;
using namespace std;

extern Camera_pose camera;

extern std::string meter_type_s;

extern int it; // image index

extern std::vector<cv::KeyPoint> keypoints1;
extern cv::Mat descriptors1;

extern std::vector<cv::KeyPoint> keypointsP;
extern cv::Mat descriptorsP;

extern std::vector<cv::Point> featurePoint;
extern std::vector<cv::Point> featurePoint2;
extern cv::Mat H;
extern cv::Mat HP;

namespace Module {

Mat getHomography(Mat Src1, Mat Src2) {
  //キーポイント検出と特徴量記述
  vector<KeyPoint> keypoints2;
  Mat descriptors2;

  cv::Ptr<cv::Feature2D> feature;
  feature = cv::AKAZE::create();

  feature->detectAndCompute(Src2, cv::Mat(), keypoints2, descriptors2);

  //マッチング(knnマッチング)
  vector<vector<cv::DMatch>> knnmatch_points;
  cv::BFMatcher match(cv::NORM_HAMMING);
  match.knnMatch(descriptors1, descriptors2, knnmatch_points, 2);

  //対応点を絞る
  const double match_par = 0.75; //候補点を残す場合のしきい値originally 0.6
  vector<cv::DMatch> goodMatch;
  // KeyPoint -> Point2d
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
  H = cv::Mat::zeros(3, 3, CV_32F);
  try {
    H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3);
  } catch (cv::Exception &e) {
    std::cerr << e.what() << std::endl;
    return H;
  }

  // RANSACで使われた対応点のみ抽出
  vector<cv::DMatch> inlierMatch;
  for (size_t i = 0; i < masks.rows; ++i) {
    uchar *inlier = masks.ptr<uchar>(i);
    if (inlier[0] == 1) {
      inlierMatch.push_back(goodMatch[i]);
    }
  }

  std::cout << inlierMatch.size() << std::endl;

  for (int i = 0; i < inlierMatch.size(); ++i) {
    int kp2_idx = inlierMatch[i].trainIdx;
    featurePoint.push_back(
        cv::Point(keypoints2[kp2_idx].pt.x, keypoints2[kp2_idx].pt.y));
  }

  cv::Mat drawmatch;
  cv::drawMatches(Src1, keypoints1, Src2, keypoints2, goodMatch, drawmatch);
  //    imwrite("../output/match_point.jpg", drawmatch);

  //インライアの対応点のみ表示
  cv::Mat drawMatch_inlier;
  cv::drawMatches(Src1, keypoints1, Src2, keypoints2, inlierMatch,
                  drawMatch_inlier);
  // imwrite("../output/match_inlier.jpg", drawMatch_inlier);

  imshow("DrawMatch_inlier", drawMatch_inlier);
  imwrite("./diffjust/" + meter_type_s + "/match/match_inlier_" +
              std::to_string(it) + ".png",
          drawMatch_inlier);
  // for masking
  // imwrite("./diffjust/V/match/match_inlier_mask" + std::to_string(it) +
  // ".png", drawMatch_inlier);

  //    cv::waitKey();

  return H;
}

Mat getHomography(vector<KeyPoint> keypoints, Mat descriptors, Mat Src1,
                  Mat Src2) {
  vector<KeyPoint> keypoints2;
  Mat descriptors2;

  cv::Ptr<cv::Feature2D> feature;
  feature = cv::AKAZE::create();

  feature->detectAndCompute(Src2, cv::Mat(), keypoints2, descriptors2);

  //マッチング(knnマッチング)
  vector<vector<cv::DMatch>> knnmatch_points;
  cv::BFMatcher match(cv::NORM_HAMMING);
  match.knnMatch(descriptors, descriptors2, knnmatch_points, 2);

  //対応点を絞る
  const double match_par = 0.70; //候補点を残す場合のしきい値originally 0.6
  vector<cv::DMatch> goodMatch;
  // KeyPoint -> Point2d
  vector<cv::Point2f> match_point;
  vector<cv::Point2f> match_point2;

  for (size_t i = 0; i < knnmatch_points.size(); ++i) {
    double distance1 = knnmatch_points[i][0].distance;
    double distance2 = knnmatch_points[i][1].distance;

    //第二候補点から距離値が離れている点のみ抽出（いい点だけ残す）
    if (distance1 <= distance2 * match_par) {
      goodMatch.push_back(knnmatch_points[i][0]);
      match_point.push_back(keypoints[knnmatch_points[i][0].queryIdx].pt);
      match_point2.push_back(keypoints2[knnmatch_points[i][0].trainIdx].pt);
    }
  }

  //ホモグラフィ行列推定
  cv::Mat masks;
  HP = cv::Mat::zeros(3, 3, CV_32F);
  try {
    HP = cv::findHomography(match_point, match_point2, masks, cv::RANSAC, 3);
  } catch (cv::Exception &e) {
    std::cerr << e.what() << std::endl;
    return HP;
  }

  // RANSACで使われた対応点のみ抽出
  vector<cv::DMatch> inlierMatch;
  for (size_t i = 0; i < masks.rows; ++i) {
    uchar *inlier = masks.ptr<uchar>(i);
    if (inlier[0] == 1) {
      inlierMatch.push_back(goodMatch[i]);
    }
  }

  std::cout << "number of inlier match points" << inlierMatch.size()
            << std::endl;

  for (int i = 0; i < inlierMatch.size(); ++i) {
    int kp2_idx = inlierMatch[i].trainIdx;
    featurePoint.push_back(
        cv::Point(keypoints2[kp2_idx].pt.x, keypoints2[kp2_idx].pt.y));
  }

  // std::cout << "featurePoint" << featurePoint << std::endl;

  cv::Mat drawmatch;
  cv::drawMatches(Src1, keypoints, Src2, keypoints2, goodMatch, drawmatch);
  //    imwrite("../output/match_point.jpg", drawmatch);

  //インライアの対応点のみ表示
  cv::Mat drawMatch_inlier;
  cv::drawMatches(Src1, keypoints, Src2, keypoints2, inlierMatch,
                  drawMatch_inlier);
// imwrite("../output/match_inlier.jpg", drawMatch_inlier);

// imshow("DrawMatch_inlier_detail", drawMatch_inlier);
// imwrite("./output/match_inlier_detail" + std::to_string(it) + ".jpg",
// drawMatch_inlier);
#if 1
  imshow("Inliner", drawMatch_inlier);
  imwrite("./diffjust/" + meter_type_s + "/match/pose/match_inlier_" +
              std::to_string(it) + ".png",
          drawMatch_inlier);
// cv::waitKey(0);
#endif
  return HP;
}

// decomposition of homography
pose decomposeHomography(cv::Mat H, cv::Mat A) {
  // https://docs.opencv.org/master/de/d45/samples_2cpp_2tutorial_code_2features2D_2Homography_2decompose_homography_8cpp-example.html#a20

  cv::Mat R1 = camera.R;
  cv::Mat rvec1;
  cv::Rodrigues(R1, rvec1);

  //並進ベクトルの初期値
  // TODO:.xmlからの読み取り
  cv::Mat tvec1 =
      camera.t; // tvec1はCamera coordinate 12/20 world coordinate じゃね

  //    cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 1, 0);
  //    0, 0, -1 ではない本当は
  cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
  cv::Mat normal1 = R1 * normal;

  Mat origin(3, 1, CV_64F, Scalar(0));
  Mat origin1 = R1 * origin + tvec1;
  double d_inv1 = 1.0 / normal1.dot(origin1);

  std::cout << "Distance to Meter Plane" << normal.dot(origin1) << std::endl;

  std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

  // solutionsの個数が解の個数(それはそう)
  int solutions = cv::decomposeHomographyMat(H, camera.A, Rs_decomp, ts_decomp,
                                             normals_decomp);

  // normals_decompが(0,0,-1)に最も近いものを選択

  double bijiao = -1;
  // cv::Mat z_axis = (cv::Mat_<double>(3, 1) << 0., 0., 1.);
  cv::Mat z_axis = (cv::Mat_<double>(3, 1) << 0., 0., -1.);
  int index = 0;

  double factor_d1 = 1.0 / d_inv1;

  cv::Mat rvec_decomp;

  for (int i = 0; i < solutions; i++) {
    cv::Rodrigues(Rs_decomp[i], rvec_decomp);

    if (true) {
      std::cout << "Solution " << i << ":" << std::endl;
      std::cout << "rvec from homography decomposition: " << rvec_decomp.t()
                << std::endl;

      std::cout << "tvec from homography decomposition: \n"
                << ts_decomp[i].t() << " \n scaled by d:\n "
                << factor_d1 * ts_decomp[i].t() << std::endl
                << std::endl;
      std::cout << "plane normal from homography decomposition: "
                << normals_decomp[i].t() << std::endl
                << std::endl;
    }

    double tmp = normals_decomp[i].dot(z_axis);

    if (tmp > bijiao) {
      // z_axisに一番近いnormals_decompをもつindex
      bijiao = tmp;
      index = i;
    }
  }

  cv::Rodrigues(Rs_decomp[index], rvec_decomp);

  cv::Mat rvec3, tvec3;
  cv::composeRT(rvec1, camera.t, rvec_decomp, ts_decomp[index] * factor_d1,
                rvec3, tvec3);

  cv::Mat R3;
  cv::Rodrigues(rvec3, R3);

  Mat X_w = R3.inv() * (tvec3 - factor_d1 * ts_decomp[index]);
  Mat t_w = R3.inv() * (ts_decomp[index] * factor_d1);

  Mat t_w2 = R3.inv() * tvec3;
  Mat X_w2 = camera.pos + t_w2;

  // t_w World coordinateでの Cam1to2の移動量
  // t_w2 World coordinateでの Oto2の移動量

  /*
  std::cout << "移動後のカメラの位置" << std::endl
            << X_w2 << std::endl
            << std::endl;
            */

  // std::cout << "方向ベクトル" << std::endl
  //          << R3.inv() * ts_decomp[index] << std::endl;
  /*
  std::cout << "移動後のカメラの回転行列" << std::endl
            << Rs_decomp[index] << std::endl
            << std::endl;
  */
  /*    std::cout << "plane normal from homography decomposition: " << std::endl
            << // R.inv()
      normals_decomp[index] << std::endl
                            << std::endl;
  */

  // return {t_w2, Rs_decomp[index]};
  return {-t_w, rvec_decomp};
}

//針の浮きを考慮したhomograhy再構成．ただし，大して読み取り精度には影響しないので，使わない．
// homographyを分解して，基準とする平面までの距離を針の浮きを考慮して変更する
cv::Mat remakeHomography(cv::Mat HG) {

  //針の浮き5mm
  double deviation = 5.5;

  cv::Mat R1 = camera.R;
  cv::Mat rvec1;
  cv::Rodrigues(R1, rvec1);

  //並進ベクトルの初期値
  // TODO:.xmlからの読み取り
  cv::Mat tvec1 = camera.t; // tvec1はCamera coordinate

  //    cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 1, 0);
  cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
  cv::Mat normal1 = R1 * normal;

  Mat origin(3, 1, CV_64F, Scalar(0));
  Mat origin1 = R1 * origin + tvec1;
  double d_inv1 = 1.0 / normal1.dot(origin1);

  std::cout << "距離" << 1. / d_inv1 << std::endl;

  std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

  // solutionsの個数が解の個数(それはそう)
  int solutions = cv::decomposeHomographyMat(HG, camera.A, Rs_decomp, ts_decomp,
                                             normals_decomp);

  // normals_decompが(0,0,1)に最も近いものを選択

  double bijiao = -1;
  // cv::Mat z_axis = (cv::Mat_<double>(3, 1) << 0., 0., 1.);
  cv::Mat z_axis = (cv::Mat_<double>(3, 1) << 0., 0., -1.);
  int index = 0;

  double factor_d1 = 1.0 / d_inv1;

  cv::Mat rvec_decomp;

  for (int i = 0; i < solutions; i++) {
    cv::Rodrigues(Rs_decomp[i], rvec_decomp);

    if (false) {
      std::cout << "Solution " << i << ":" << std::endl;
      std::cout << "rvec from homography decomposition: " << rvec_decomp.t()
                << std::endl;

      std::cout << "tvec from homography decomposition: \n"
                << ts_decomp[i].t() << " \n scaled by d:\n "
                << factor_d1 * ts_decomp[i].t() << std::endl
                << std::endl;
      std::cout << "plane normal from homography decomposition: "
                << normals_decomp[i].t() << std::endl
                << std::endl;
    }

    //        std::cout << i << "th Homography\n"
    //                 << Rs_decomp[i] + ts_decomp[i] * normals_decomp[i].t() <<
    //                 std::endl;

    double tmp = normals_decomp[i].dot(z_axis);

    if (tmp > bijiao) {
      bijiao = tmp;
      index = i;
    }
  }

  // tの更新
  ts_decomp[index] = factor_d1 / (factor_d1 + deviation) * ts_decomp[index];
  cv::Mat new_H =
      Rs_decomp[index] + ts_decomp[index] * normals_decomp[index].t();

  new_H = camera.A * new_H * camera.A.inv();
  new_H /= new_H.at<double>(2, 2); //正規化

  return new_H;
}
} // namespace Module

#include "module.hpp"
#include "opencv2/calib3d.hpp"
#include "params/pose_params.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

using namespace std;


extern Camera_pose camera;

#define PAT_ROW (7)  /* パターンの行数 */
#define PAT_COL (10) /* パターンの列数 */
#define PAT_SIZE (PAT_ROW * PAT_COL)
#define ALL_POINTS (IMAGE_NUM * PAT_SIZE)
//#define CHESS_SIZE (270. * 1.6 / 23.8) /* パターン1マスの1辺サイズ[mm] */
#define CHESS_SIZE (24) /* パターン1マスの1辺サイズ[mm] */


namespace Calib
{
void calibration(cv::Mat& img, Module::pose& p)
{
    int j, k;
    // cv::Mat src_img[IMAGE_NUM];
    cv::Size pattern_size = cv::Size2i(PAT_COL, PAT_ROW);
    vector<cv::Point2f> corners;
    vector<vector<cv::Point2f>> img_points;

    // (2)3次元空間座標の設定

    vector<cv::Point3f> object;
    for (j = 0; j < PAT_ROW; j++) {
        for (k = 0; k < PAT_COL; k++) {
            cv::Point3f p(
                j * CHESS_SIZE,
                k * CHESS_SIZE,
                0.0);
            object.push_back(p);
        }
    }

    vector<vector<cv::Point3f>> obj_points;
    obj_points.push_back(object);

    // ３次元の点を ALL_POINTS * 3 の行列(32Bit浮動小数点数:１チャンネル)に変換する


    // (3)チェスボード（キャリブレーションパターン）のコーナー検出
    cv::namedWindow("Calibration", cv::WINDOW_AUTOSIZE);
    auto found = cv::findChessboardCorners(img, pattern_size, corners);
    if (found) {
        cout << "... ok" << endl;
    } else {
        cerr << "... fail" << endl;
    }

    // (4)コーナー位置をサブピクセル精度に修正，描画
    cv::Mat src_gray = cv::Mat(img.size(), CV_8UC1);
    cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
    cv::find4QuadCornerSubpix(src_gray, corners, cv::Size(3, 3));
    cv::drawChessboardCorners(img, pattern_size, corners, found);

    vector<cv::Point2f> camera_cor;
    vector<cv::Point3f> world;

#if 0
    camera_cor.push_back(corners[0]);
    world.push_back(cv::Point3f(0., 0., 0.));

    camera_cor.push_back(corners[PAT_COL - 1]);
    world.push_back(cv::Point3f((PAT_COL - 1) * CHESS_SIZE + PAT_COL, 0., 0.));

    camera_cor.push_back(corners[(PAT_ROW - 1) * PAT_COL]);
    world.push_back(cv::Point3f(0., 0., -(PAT_ROW - 1) * CHESS_SIZE - PAT_ROW));

    camera_cor.push_back(corners[PAT_ROW * PAT_COL - 1]);
    world.push_back(cv::Point3f((PAT_COL - 1) * CHESS_SIZE + PAT_COL, 0., -(PAT_ROW - 1) * CHESS_SIZE - PAT_ROW));
#else

    camera_cor.push_back(corners[0]);
    world.push_back(cv::Point3f(0., 0., 0.));

    camera_cor.push_back(corners[1]);
    world.push_back(cv::Point3f(0., CHESS_SIZE, 0.));

    camera_cor.push_back(corners[2]);
    world.push_back(cv::Point3f(0., 2. * CHESS_SIZE, 0.));

    camera_cor.push_back(corners[PAT_COL - 1]);
    world.push_back(cv::Point3f(0., (PAT_COL - 1) * CHESS_SIZE, 0.));

    camera_cor.push_back(corners[(PAT_ROW - 1) * PAT_COL]);
    world.push_back(cv::Point3f((PAT_ROW - 1) * CHESS_SIZE, 0., 0.));

    camera_cor.push_back(corners[PAT_ROW * PAT_COL - 1]);
    world.push_back(cv::Point3f((PAT_ROW - 1) * CHESS_SIZE, (PAT_COL - 1) * CHESS_SIZE, 0.));

#endif

    cv::Mat rvec, tvec;

    cv::solvePnP(world, camera_cor, camera.A, camera.distCoeffs, rvec, tvec);


    cv::Mat R_0, t_0;

    cv::Rodrigues(rvec, R_0);

    t_0 = R_0.inv() * (-tvec);

    cout << "worldでの並進" << endl
         << t_0 << endl;

    p.position = tvec;
    p.orientation = R_0;

    //   cv::imshow("Calibration", img);
    //  cv::waitKey(1);
}
}  // namespace Calib

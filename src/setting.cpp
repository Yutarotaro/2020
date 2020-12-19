//ref: https://qiita.com/kazz4423/items/f173f298704bd121043d

#include "params/calibration_params.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#define filepath "/Users/yutaro/research/2020/src"

using namespace std;

int main(int argc, char* argv[])
{
    int i, j, k;
    int corner_count, found;
    int p_count[IMAGE_NUM];
    // cv::Mat src_img[IMAGE_NUM];
    vector<cv::Mat> srcImages;
    cv::Size pattern_size = cv::Size2i(PAT_COL, PAT_ROW);
    vector<cv::Point2f> corners;
    vector<vector<cv::Point2f>> img_points;

    // (1)キャリブレーション画像の読み込み
    for (i = 0; i < IMAGE_NUM; i++) {
        ostringstream ostr;
        //チェスボード写真のpath
        //       ostr << filepath << "/pictures/calib_img/calib" << i << ".png";
        ostr << filepath << "/pictures/dia_experiment_V/calib/pic" << i << ".JPG";
        //ostr << filepath << "/pictures/meter_experiment/calib/pic" << i << ".JPG";
        //        ostr << filepath << "/pictures/calib/pic" << i << ".JPG";
        cv::Mat src = cv::imread(ostr.str(), 1);
        if (src.empty()) {
            cerr << "cannot load image file : " << ostr.str() << endl;
            return 0;
        } else {
            srcImages.push_back(src);
        }
    }


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
    for (i = 0; i < IMAGE_NUM; i++) {
        obj_points.push_back(object);
    }

    // ３次元の点を ALL_POINTS * 3 の行列(32Bit浮動小数点数:１チャンネル)に変換する


    // (3)チェスボード（キャリブレーションパターン）のコーナー検出
    int found_num = 0;
    cv::namedWindow("Calibration", cv::WINDOW_AUTOSIZE);
    for (i = 0; i < IMAGE_NUM; i++) {
        auto found = cv::findChessboardCorners(srcImages[i], pattern_size, corners);
        if (found) {
            cout << setfill('0') << setw(2) << i << "... ok" << endl;
            found_num++;
        } else {
            cerr << setfill('0') << setw(2) << i << "... fail" << endl;
        }

        // (4)コーナー位置をサブピクセル精度に修正，描画
        cv::Mat src_gray = cv::Mat(srcImages[i].size(), CV_8UC1);
        cv::cvtColor(srcImages[i], src_gray, cv::COLOR_BGR2GRAY);
        cv::find4QuadCornerSubpix(src_gray, corners, cv::Size(3, 3));
        cv::drawChessboardCorners(srcImages[i], pattern_size, corners, found);
        img_points.push_back(corners);

        cv::imshow("Calibration", srcImages[i]);
        //        cv::waitKey(1000);
    }
    cv::destroyWindow("Calibration");

    if (found_num != IMAGE_NUM) {
        cerr << "Calibration Images are insufficient." << endl;
        return -1;
    }

    // (5)内部パラメータ，歪み係数の推定
    cv::Mat cam_mat;               // カメラ内部パラメータ行列
    cv::Mat dist_coefs;            // 歪み係数
    vector<cv::Mat> rvecs, tvecs;  // 各ビューの回転ベクトルと並進ベクトル
    cv::calibrateCamera(
        obj_points,
        img_points,
        srcImages[0].size(),
        cam_mat,
        dist_coefs,
        rvecs,
        tvecs);

    cv::Mat R_tmp;
    //後ろから2枚
    for (int i = IMAGE_NUM - 2; i < IMAGE_NUM; i++) {
        std::cout << i << "-th image position" << std::endl;
        cv::Rodrigues(rvecs[i], R_tmp);
        std::cout << R_tmp.inv() * (-tvecs[i]) << std::endl
                  << std::endl;
    }

    // (6)XMLファイルへの書き出し
    cv::FileStorage fs("camera.xml", cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "File can not be opened." << endl;
        return -1;
    }

    cv::Mat t = R_tmp.inv() * (-tvecs[IMAGE_NUM - 1]);

    t.at<double>(0, 2) -= Calibration::offset;

    fs << "intrinsic" << cam_mat;
    fs << "distortion" << dist_coefs;

    //最後の画像の外部パラメータ　いる？
    fs << "R" << R_tmp;
    fs << "t" << t;
    fs.release();


    return 0;
}

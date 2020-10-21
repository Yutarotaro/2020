#include "../common/init.hpp"
#include "../pos/module.hpp"
#include "readability.hpp"
#include <leptonica/allheaders.h>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/layer.details.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <tesseract/baseapi.h>
#include <utility>
#include <vector>

using namespace cv;

extern cv::Mat temp;
extern cv::Mat H;

namespace Readability
{
//tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
//auto ocr = std::make_unique<tesseract::TessBaseAPI>();

int judge(cv::Mat img, int num, int flag)
{

    cv::Mat tmp[3];
    tmp[0] = cv::imread("../pictures/template/0.png", 1);
    tmp[1] = cv::imread("../pictures/template/120_1.png", 1);
    tmp[2] = cv::imread("../pictures/template/center.png", 1);

    cv::Mat result_mat;
    cv::Mat gray_img;
    cv::Mat tmpl;

    /*
    for (int i = 0; i < 3; ++i) {
        temp[i].copyTo(tmpl);

        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY, 0);  //カメラ画像をグレースケールに変換
        cv::cvtColor(tmpl, tmpl, cv::COLOR_BGR2GRAY, 0);     //カメラ画像をグレースケールに変換

        cv::matchTemplate(gray_img, tmpl, result_mat, CV_TM_CCOEFF_NORMED);
        cv::normalize(result_mat, result_mat, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        double minVal;
        double maxVal;
        cv::Point minLoc, maxLoc, matchLoc;
        cv::minMaxLoc(result_mat, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        matchLoc = maxLoc;

        //cv::rectangle(img, matchLoc, cv::Point(matchLoc.x + 0.7 * tmpl.cols, matchLoc.y + 0.7 * tmpl.rows), CV_RGB(255, 0, 0), 3);
    }
    */

    //    cv::Mat out = edgeDetection(img);
    //  cv::imshow("edge", out);

    //    cv::Canny(gray_img, gray_img, 10, 155);

    //  cv::imshow("Canny", gray_img);
    // int niters = 1;
    //    cv::dilate(gray_img, gray_img, cv::Mat(), cv::Point(-1, -1), niters);
    //    cv::erode(gray_img, gray_img, cv::Mat(), cv::Point(-1, -1), niters);

    cv::Mat now_circle;
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    cv::circle(mask, cv::Point(310, 310), 310, cv::Scalar(255), -1, CV_AA);
    img.copyTo(now_circle, mask);

    cv::Mat now_bin, temp_bin;


    int iter = 2;
    cv::Mat res;
    cv::absdiff(now_circle, temp, res);
    //    cv::bitwise_xor(temp, now_circle, res);


    //processing
    cv::erode(res, res, cv::Mat(), cv::Point(-1, -1), iter);
    cv::dilate(res, res, cv::Mat(), cv::Point(-1, -1), iter);

    int n = 3;
    //    cv::bilateralFilter(res, res, n, n * 2, n / 2);

    //    cv::GaussianBlur(res, res, cv::Size(n, n), 0, 0);

    cv::Mat gray;
    cv::cvtColor(res, gray, CV_BGR2GRAY);
    cv::Canny(gray, gray, 0, 1055, 3);


    //    cv::imshow("diff", res);
    //   cv::imshow("edge", gray);
    cv::imwrite("./diff/xor" + std::to_string(num) + (flag ? "HR" : "HR.inv()") + ".png", res);

    if (flag) {
        read(res);
    }

#if 0
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(gray_img,  // 入力画像，8ビット，シングルチャンネル．0以外のピクセルは 1 、0のピクセルは0として扱う。処理結果として image を書き換えることに注意する.
        contours,               // 輪郭を点ベクトルとして取得する
        hierarchy,              // hiararchy ? オプション．画像のトポロジーに関する情報を含む出力ベクトル．
        CV_RETR_EXTERNAL,       // 輪郭抽出モード
        CV_CHAIN_APPROX_NONE    // 輪郭の近似手法
    );

    cv::Mat tmp;
    img.copyTo(tmp);

    int ct = 0;

    //digits detection
    int mar = 5;
    for (int i = 0; i < contours.size(); i++) {
        if (1) {
            cv::Rect brect = cv::boundingRect(cv::Mat(contours[i]).reshape(2));
            cv::rectangle(tmp, brect.tl(), brect.br(), cv::Scalar(190, 30, 100), 2, CV_AA);


            if (brect.tl().x - mar > 0 && brect.tl().y - mar > 0 && brect.br().x + mar < img.cols && brect.br().y + mar < img.rows) {
                //if (brect.tl().x > 0 && brect.tl().y > 0 && brect.br().x - 1 < img.rows && brect.br().y - 1 < img.cols) {
                int brx = brect.br().x + mar;
                int bry = brect.br().y + mar;

                int tlx = brect.tl().x - mar;
                int tly = brect.tl().y - mar;

                int w = brx - tlx;
                int h = bry - tly;


                //if (h * w > 500) {
                if (w * h > 900 && w >= 22 && w <= 40 && h >= 48 && h <= 60) {
                    //                    std::cout << tlx << ' ' << tly << ' ' << h << ' ' << w << std::endl;
                    //cv::Rect roi2(cv::Point(tlx, tly), cv::Size(w, h));
                    cv::rectangle(tmp, brect.tl(), brect.br(), cv::Scalar(190, 30, 100), 2, CV_AA);

                    //    cv::Rect roi2(brect.tl(), brect.br());
                    cv::Rect roi2(cv::Point(tlx, tly), cv::Point(brx, bry));
                    cv::Mat right;
                    right = img(roi2);  // 切り出し画像
                    if (cv::imwrite("./candidate/" + std::to_string(ct) + ".png", right)) {
                        ++ct;
                    }
                }
            }
        } else {
            // CV_*C2型のMatに変換してから，外接矩形（回転あり）を計算
            cv::Point2f center, vtx[4];
            float radius;
            cv::RotatedRect box = cv::minAreaRect(cv::Mat(contours[i]).reshape(2));
            // 外接矩形（回転あり）を描画
            box.points(vtx);
            for (int i = 0; i < 4; ++i) {
                cv::line(img, vtx[i], vtx[i < 3 ? i + 1 : 0], cv::Scalar(100, 100, 200), 2, CV_AA);

                if (vtx[0].x > 0 && vtx[0].y > 0) {
                    //cv::Rect roi2(vtx[0], cv::Size(vtx[1].x - vtx[0].x, vtx[2].y - vtx[0].y) );
                }
            }
        }
    }
    cv::imwrite("./output/minAreaRect.jpg", tmp);


    cv::imshow("result", tmp);

#endif

    return 0;
}

double read(cv::Mat src)
{
    cv::Mat gray;
    //グレースケール化
    cv::cvtColor(src, gray, CV_BGR2GRAY);

    //cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1), 1);

    cv::imshow("grayscale", gray);

    //細線化
    cv::ximgproc::thinning(gray, gray, cv::ximgproc::WMF_EXP);
    //    Thinning(gray);


    //cv::imshow("temp", temp);
    cv::imshow("thinning", gray);
    //    cv::waitKey();
}
}  // namespace Readability

#include "../common/init.hpp"
#include "../pos/homography.hpp"
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

//////////////読み取りのパラメータ
extern int meter_type;
extern Init::Params params[];


extern cv::Mat temp;
extern cv::Mat H;
extern int it;
extern int type;

namespace Readability
{
//tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
//auto ocr = std::make_unique<tesseract::TessBaseAPI>();

int judge(cv::Mat img, int num, int flag)
{
}

std::pair<double, cv::Mat> pointerDetection(cv::Mat src, cv::Mat origin)
{
    cv::imwrite("src.png", src);
    cv::Mat ret;
    src.copyTo(ret);
    cv::cvtColor(ret, ret, CV_GRAY2BGR);


    std::vector<cv::Vec3f> lines;
    int votes = 50;
    cv::HoughLines(src, lines, 1, 0.0001 /*CV_PI / 720.*/, votes, 0, 0);

    std::cout << "Number of detected lines" << lines.size() << std::endl;

    int tmp = 0;
    int index = 0;
    double eps = 0.1;
    double value = 0.;
    double rad = 0.;

    cv::Point pt1, pt2;  //検出された直線の端点が入る


    if (lines.size()) {
        for (size_t i = 0; i < lines.size(); ++i) {
            //投票数が一番多い直線を選ぶ
            if (lines[i][2] > tmp /*&& std::abs(lines[i][1] - CV_PI / 2.) > eps*/) {
                index = i;
                tmp = lines[i][2];
            }
        }

        float rho = lines[index][0], theta = lines[index][1];
        std::cout << index << "-th " << lines[index][2] << " votes" << std::endl
                  << lines[index][1] << " [rad]" << std::endl;
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        cv::line(ret, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        rad = lines[index][1];

        std::cout << "pt1.x: " << pt1.x << "pt1.y: " << pt1.y << "pt2.x: " << pt2.x << "pt2.y: " << pt2.y << std::endl;
    }
    cv::imwrite("hough.png", ret);

    //TODO:針の向き判定
    //line上の点群の重心が上下左右どちらにあるかで場合わけ
    //直線の定式化
    double slope = (double)(pt2.y - pt1.y) / (pt2.x - pt1.x);
    auto y = [=](int x) -> int {
        return slope * (x - pt1.x) + pt1.y;
    };

    cv::Mat bgr_origin;
    cv::cvtColor(origin, bgr_origin, CV_GRAY2BGR);


    std::vector<cv::Point2d> points_on_line;
    cv::Point2d conti_tl = cv::Point(10000, 10000);
    cv::Point2d conti_br = cv::Point(0, 0);
    int conti_thre = 20;
    int conti_count = 0;

    for (int i = 0; i < src.cols; ++i) {
        if (y(i) >= 0 && y(i) <= src.rows) {
            if (origin.at<unsigned char>(cvRound(y(i)), i)) {
                conti_count++;
                points_on_line.push_back(cv::Point(i, cvRound(y(i))));
            } else {
                if (conti_count >= conti_thre) {
                    if (conti_tl.x > i - conti_count) {
                        conti_tl.x = i - conti_count;
                        conti_tl.y = y(i - conti_count);
                    }

                    if (conti_br.x < i - 1) {
                        conti_br.x = i - 1;
                        conti_br.y = y(i - 1);
                    }

                    for (int j = i - conti_count; j <= i; ++j) {
                        cv::circle(bgr_origin, cv::Point(j, cvRound(y(j))), 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
                    }
                }
                conti_count = 0;
            }
        }
    }

    cv::circle(bgr_origin, conti_tl, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    cv::circle(bgr_origin, conti_br, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

    cv::imshow("origin", bgr_origin);
    cv::imwrite("bgr.png", bgr_origin);
    //cv::waitKey();


    //    std::cout << "画像の大きさ: " << origin.rows << ' ' << origin.cols << std::endl;
    std::cout << "tl: " << conti_tl << " br: " << conti_br << std::endl;

    //針の先が第何象限にあるか
    //row と col 自信ない
    cv::Point center = cv::Point(origin.rows / 2, origin.cols / 2);

    //中心からより離れている方が先端
    cv::Point tip = (std::abs(conti_tl.x - center.x) > std::abs(conti_br.x - center.x) ? conti_tl : conti_br);

    if (tip.x < center.x) {
        //第2,3象限ならば，piだけ引く
        rad -= CV_PI;
    }


    //この時点でvalueはrad
    //degに変更

    std::cout << rad << ' ' << params[meter_type].front_rad << std::endl;

    value = params[meter_type].front_value + params[meter_type].k * (rad - params[meter_type].front_rad);

    //    cv::imwrite("./reading/no" + std::to_string(it) + (type ? "pointer" : "normal") + "_" + std::to_string(value) + ".png", ret);

    //可読性xなら0を返す
    ////estimation of readability
    int line_center = y(center.x);
    int readable_thre = 30;
    if (std::abs(center.y - line_center) > readable_thre) {
        value = 0;
        std::cout << "Not Readable!!" << std::endl;
    }


    if (value) {
        std::cout << "Readable!!" << std::endl;
    }


    return {value, ret};
}

//src: 差分とって，erode,dilateした後の画像を入力
double read(cv::Mat src)
{
    cv::Mat gray;
    //グレースケール化
    cv::cvtColor(src, gray, CV_BGR2GRAY);

    //cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1), 1);

    cv::imshow("grayscale", gray);

    cv::Mat bin;
    double thresh = 70;
    double maxval = 255;
    int type = cv::THRESH_BINARY;
    cv::threshold(gray, bin, thresh, maxval, type);


    int iter = 10;
    cv::dilate(bin, bin, cv::Mat(), cv::Point(-1, -1), iter);
    cv::erode(bin, bin, cv::Mat(), cv::Point(-1, -1), iter);


    //細線化
    cv::ximgproc::thinning(bin, bin, cv::ximgproc::WMF_EXP);

    cv::imshow("thinning", bin);
    cv::imwrite("./thinning/" + std::to_string(it) + (type ? "pointer" : "normal") + ".png", bin);

    cv::Rect roi(cv::Point(220, 220), cv::Size(280, 280));
    cv::Mat pointerImage = bin(roi);  // 切り出し画像

    //TODO:Hough Transform
    //PCAにしたい
    std::pair<double, cv::Mat> a;
    //= pointerDetection(pointerImage);

    return a.first;
}
}  // namespace Readability

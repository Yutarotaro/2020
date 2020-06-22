#include "../common/init.hpp"
#include "../pos/module.hpp"
#include "difference.hpp"
#include "template.hpp"
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace Template
{
//cv::Mat temp = Init::input_images2("clock", "temp");
//cv::Mat temp = Init::input_images2("meter", "meter_origin");
cv::Mat temp = cv::imread("/Users/yutaro/research/2020/src/pictures/meter2/pic-1.png", 1);

void equalize(cv::Mat src, cv::Mat& dst)
{
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(src, dst);
}

void tempMatch(cv::Mat img, cv::Mat templ, cv::Mat& subImg)
{
    using namespace cv;
    Mat result;
    const char* image_window = "Source Image";
    const char* result_window = "Result window";

    Mat img_display;
    img.copyTo(img_display);
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    int match_method = TM_SQDIFF_NORMED;
    matchTemplate(img, templ, result, match_method);

    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());


    rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
    rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
    imshow(image_window, img_display);
    //imshow(result_window, result);

    cv::Rect roi(matchLoc, cv::Size(templ.cols, templ.rows));
    subImg = img(roi);

    imshow("subImg", subImg);
    imwrite("./output.png", subImg);
    waitKey();
}


void getDiff(cv::Mat img, cv::Mat& dst)
{
    int thresh = 140;
    int maxval = 255;
    int type = cv::THRESH_BINARY_INV;
    int method = cv::BORDER_REPLICATE;
    int blocksize = 51;
    double C = 10.0;

    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    //cv::cvtColor(templ, templ, cv::COLOR_BGR2GRAY);


    /*    for (C = 0.; C < 2.; C += 0.2) {
        cv::Mat tmp;
        cv::adaptiveThreshold(img, tmp, maxval, method, type, blocksize, C);
        //  cv::adaptiveThreshold(templ, templ, maxval, method, type, blocksize, C);

        std::string s = std::to_string(C);
        cv::imshow(s, tmp);
        // cv::imshow("templ", templ);
    }*/
    //cv::bitwise_and(img, templ, dst);
    //  cv::absdiff(img, templ, dst);

    cv::adaptiveThreshold(img, dst, maxval, method, type, blocksize, C);
    //    cv::threshold(dst, dst, thresh, maxval, type);


    int ite = 1;

    cv::erode(dst, dst, cv::Mat(), cv::Point(-1, -1), ite);
    cv::dilate(dst, dst, cv::Mat(), cv::Point(-1, -1), ite);


    cv::imshow("dst", dst);
    cv::imwrite("./dst.png", dst);

    cv::waitKey();
}

using namespace cv;


void thinningIte(Mat& img, int pattern)
{

    Mat del_marker = Mat::ones(img.size(), CV_8UC1);
    int x, y;

    for (y = 1; y < img.rows - 1; ++y) {

        for (x = 1; x < img.cols - 1; ++x) {

            int v9, v2, v3;
            int v8, v1, v4;
            int v7, v6, v5;

            v1 = img.data[y * img.step + x * img.elemSize()];
            v2 = img.data[(y - 1) * img.step + x * img.elemSize()];
            v3 = img.data[(y - 1) * img.step + (x + 1) * img.elemSize()];
            v4 = img.data[y * img.step + (x + 1) * img.elemSize()];
            v5 = img.data[(y + 1) * img.step + (x + 1) * img.elemSize()];
            v6 = img.data[(y + 1) * img.step + x * img.elemSize()];
            v7 = img.data[(y + 1) * img.step + (x - 1) * img.elemSize()];
            v8 = img.data[y * img.step + (x - 1) * img.elemSize()];
            v9 = img.data[(y - 1) * img.step + (x - 1) * img.elemSize()];

            int S = (v2 == 0 && v3 == 1) + (v3 == 0 && v4 == 1) + (v4 == 0 && v5 == 1) + (v5 == 0 && v6 == 1) + (v6 == 0 && v7 == 1) + (v7 == 0 && v8 == 1) + (v8 == 0 && v9 == 1) + (v9 == 0 && v2 == 1);

            int N = v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9;

            int m1 = 0, m2 = 0;

            if (pattern == 0)
                m1 = (v2 * v4 * v6);
            if (pattern == 1)
                m1 = (v2 * v4 * v8);

            if (pattern == 0)
                m2 = (v4 * v6 * v8);
            if (pattern == 1)
                m2 = (v2 * v6 * v8);

            if (S == 1 && (N >= 2 && N <= 6) && m1 == 0 && m2 == 0)
                del_marker.data[y * del_marker.step + x * del_marker.elemSize()] = 0;
        }
    }

    img &= del_marker;
}

void thinning(const Mat& src, Mat& dst)
{
    dst = src.clone();
    dst /= 255;  // 0は0 , 1以上は1に変換される

    Mat prev = Mat::zeros(dst.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIte(dst, 0);
        thinningIte(dst, 1);
        absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (countNonZero(diff) > 0);

    dst *= 255;
}

void readMeter(cv::Mat src)
{
    cv::Mat tmp;

    //    cv::resize(temp, temp, cv::Size(), 0.5, 0.5);
    //    equalize(temp, tmp);

    auto H = Module::getHomography(temp, src);


    cv::Mat dst = cv::Mat::zeros(src.rows + 100, src.cols + 100, CV_8UC3);
    cv::warpPerspective(src, dst, H.inv(), dst.size());


    cv::Mat subImg;
    tempMatch(dst, temp, subImg);

    auto [p, R] = Difference::circleDetect(subImg);

    cv::Mat target;
    getDiff(subImg, target);


    //    thinning(target, target);

    //    cv::GaussianBlur(target, target, cv::Size(9, 9), 2, 2);

    //確率ハフ変換による直線検出
    int l = 0;
    int r = 1800;

    int maxLineGap = 400;

    std::vector<cv::Vec4i> lines;
    int ct = 0;
    while (true) {
        ct++;
        int toupiao = (l + r) / 2;
        cv::HoughLinesP(target, lines, 1, CV_PI / 180, toupiao, 30, maxLineGap);


        if (lines.size() == 2) {
            for (size_t i = 0; i < lines.size(); i++) {
                cv::line(target, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
            }
            std::cout << lines.size() << std::endl;
            std::cout << ct << ' ' << toupiao << std::endl;
            break;
        } else if (lines.size() > 2) {
            l = toupiao;
        } else {
            r = toupiao;
        }

        std::cout << toupiao << std::endl;

        if (r - l == 1) {
            std::cout << "cannot" << std::endl;
            return;
        }
    }


    cv::imshow("i", target);
    cv::waitKey();
}

}  // namespace Template

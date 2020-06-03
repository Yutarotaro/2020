#include "../common/init.hpp"
#include "difference.hpp"
#include "read.hpp"
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace Difference
{

cv::Mat origin = Init::input_images("pointer");

int sigma = 3;
int ksize = (sigma * 5) | 1;
int thre = 160;


std::pair<cv::Point, int> circleDetect(cv::Mat img)
{
    cv::Mat gray;
    //2値化
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    //平滑化
    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
        2, gray.rows / 4, 150, 100);

    int C_x = gray.rows / 2;
    int C_y = gray.cols / 2;

    double tmp = -1e9;
    int index = -1;
    int radius;


    for (size_t i = 0; i < circles.size(); i++) {
        radius = cvRound(circles[i][2]);
        //暫定: 検出された円のうち最も半径の大きいものをメータとみなす
        int x = cvRound(circles[i][0]);
        int y = cvRound(circles[i][1]);

        double a = 0;

        double bijiao = -(C_x - x) * (C_x - x) * a - (C_y - y) * (C_y - y) * a + radius * radius;

        if (bijiao > tmp) {
            tmp = bijiao;
            index = i;
        }
    }

    cv::Point center(cvRound(circles[index][0]), cvRound(circles[index][1]));
    radius = cvRound(circles[index][2]);

    // 円の中心を描画します．
    circle(img, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    // 円を描画します．
    circle(img, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);

    cv::namedWindow("circles", 1);
    cv::imshow("circles", img);

    return {center, radius};
}

cv::Mat thresh(cv::Mat image)
{
    cv::cvtColor(image, image, CV_BGR2GRAY);

    //cv::GaussianBlur(image, image, cv::Size(ksize, ksize), sigma, sigma);
    //cv::adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 0);
    cv::threshold(image, image, thre /*cv::THRESH_OTSU*/, 255, cv::THRESH_BINARY_INV);
    return image;
}

double dist(cv::Point x, cv::Point y)
{
    return sqrt(pow(x.x - y.x, 2) + pow(y.x - y.y, 2));
}

void Lines(cv::Mat src, std::pair<cv::Point, int> circle, double& m)
{
    cv::Mat dst, color_dst;


    cv::Canny(src, dst, 50, 200, 3);
    cv::cvtColor(dst, color_dst, CV_GRAY2BGR);

    int toupiao = 155;

#if 0
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(dst, lines, 1, CV_PI / 180, toupiao);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)),
            cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)),
            cvRound(y0 - 1000 * (a)));
        cv::line(color_dst, pt1, pt2, cv::Scalar(0, 0, 255), 3, 8);
    }
#else
    //確率ハフ変換による直線検出
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(dst, lines, 1, CV_PI / 180, toupiao, 30, 40);
    std::cout << lines.size() << std::endl;
    for (size_t i = 0; i < lines.size(); i++) {

        cv::line(color_dst, cv::Point(lines[i][0], lines[i][1]),
            cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, cvRound(255 * (i + 1) / double(lines.size()))), 3, 8);
    }

    if (lines.size() != 2) {
        std::cout << "検出された直線が2本ではない" << std::endl;
        return;
    }

    //TODO; 2本の直線の傾きの平均をとってスケールから値に変換
    double a[2];
    for (int i = 0; i < 2; i++) {
        auto p = lines[i];
        a[i] = -(double)(p[1] - p[3]) / (p[0] - p[2]);
        std::cout << a[i] << std::endl;
    }

    m = (a[0] + a[1]) / 2.;

#endif

    imshow("1", color_dst);

    cv::imwrite("output.jpg", color_dst);

}  // namespace Difference

Read::Data readMeter(cv::Mat src)
{
    Read::Data ret = {0, 0};

    //TODO:検出円を囲む正方形領域をcv::Rectでトリミング
    std::pair<cv::Point, int> circle = circleDetect(src);

    //TODO:本番メータでのテンプレート画像のメータの大きさcalibration


    //メーター部分のトリミング
    cv::Rect roi(cv::Point(circle.first.x - circle.second, circle.first.y - circle.second), cv::Size(2 * circle.second, 2 * circle.second));
    cv::Mat subImg = src(roi);


    cv::imshow("trimming", subImg);

    //cv::Mat sub_thre = thresh(subImg);

    double m;
    Lines(subImg, circle, m);
    std::cout << "傾きは" << m << std::endl;


    /*
    double dsize = (double)circle.second / ;
    cv::resize(origin, tmpl, dsize, 0, 0, INTER_LINEAR);
*/

    //2値化
    //cv::Mat src_thre = thresh(src);
    //cv::Mat origin_thre = thresh(origin);


    /*保留
    //針なし画像との差分をとる
    cv::Mat diff;
    cv::absdiff(src, origin, diff);
    */


    return ret;
}

}  // namespace Difference

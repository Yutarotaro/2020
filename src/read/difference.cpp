#include "../common/init.hpp"
#include "difference.hpp"
#include "read.hpp"
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace Difference
{

cv::Mat origin = Init::input_images(0, 2);

int sigma = 3;
int ksize = (sigma * 5) | 1;
int thre = 160;

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

std::vector<cv::Vec2f> Lines(cv::Mat src, std::pair<cv::Point, int> circle)
{
    cv::Mat dst, color_dst;

    cv::Canny(src, dst, 50, 200, 3);
    cv::cvtColor(dst, color_dst, CV_GRAY2BGR);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(dst, lines, 1, CV_PI / 180, 100);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)),
            cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)),
            cvRound(y0 - 1000 * (a)));

        /*if (dist(pt1, circle.first) >= circle.second || dist(pt2, circle.first) >= circle.second) {
            continue;
        }*/

        cv::line(color_dst, pt1, pt2, cv::Scalar(0, 0, 255), 3, 8);
    }

    imshow("1", color_dst);

    cv::imwrite("output.jpg", color_dst);

    return lines;
}

//TODO: originとsrcをきれいに一致させ背景差分から読み取り値を返す
Read::Data readMeter(cv::Mat src)
{
    Read::Data ret = {0, 0};

    //2値化
    cv::Mat src_thre = thresh(src);
    cv::Mat origin_thre = thresh(origin);

    //針なし画像との差分をとる
    cv::Mat diff;
    cv::absdiff(src, origin, diff);
    //cv::bitwise_and(src, origin, diff);

    //cv::imshow("src", diff);

    /*    std::vector<cv::Vec2f> lines = Lines(src);

    float theta[2];

    if (lines.size() == 2) {
        theta[0] = (double)(-lines[0][1] + M_PI / 2.0) * 180.0 / M_PI;
        theta[1] = (-lines[1][1] + M_PI / 2.) * 180. / (double)M_PI;

        ret.value = (theta[0] + theta[1]) / 2.;
    }
*/
    return ret;
}

}  // namespace Difference

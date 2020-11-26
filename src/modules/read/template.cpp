#include "../common/init.hpp"
#include "../pos/homography.hpp"
#include "difference.hpp"
#include "template.hpp"
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

extern cv::Mat temp;
namespace Template
{
//cv::Mat temp = Init::input_images2("clock", "temp");
//cv::Mat temp = Init::input_images2("meter", "meter_origin");
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
    cv::adaptiveThreshold(img, dst, maxval, method, type, blocksize, C);


    int ite = 1;

    //    cv::erode(dst, dst, cv::Mat(), cv::Point(-1, -1), ite);
    //   cv::dilate(dst, dst, cv::Mat(), cv::Point(-1, -1), ite);


    cv::imshow("dst", dst);
    cv::imwrite("./dst.png", dst);

    cv::waitKey();
}

void norm(std::pair<double, double>& x)
{
    double r = std::sqrt(x.first * x.first + x.second * x.second);

    x.first /= r;
    x.second /= r;
}

void minus(double& x)
{
    if (x < 0) {
        x = 2. * CV_PI + x;
    }
}

void readMeter(cv::Mat src)
{
    cv::Mat tmp;

    //    cv::resize(temp, temp, cv::Size(), 0.5, 0.5);
    //    cv::imshow("fi", temp);
    //   cv::waitKey();
    //    equalize(temp, tmp);

    auto H = Module::getHomography(temp, src);


    cv::Mat subImg;
    //cv::Mat dst = cv::Mat::zeros(src.rows + 100, src.cols + 100, CV_8UC3);
    subImg = cv::Mat::zeros(temp.rows, temp.cols, CV_8UC3);
    cv::warpPerspective(src, subImg, H.inv(), subImg.size());
    cv::imwrite("inv.png", subImg);


    //tempMatch(dst, temp, subImg);

    //    subImg = cv::imread("../pictures/prac.png", 1);


    auto [p, R] = Difference::circleDetect(subImg);


    cv::Mat target;
    getDiff(subImg, target);


    //    thinning(target, target);

    //    cv::GaussianBlur(target, target, cv::Size(9, 9), 2, 2);

    //確率ハフ変換による直線検出
    int l = 0;
    int r = 2800;

    int maxLineGap = 0;

    std::vector<cv::Vec4i> lines;
    int ct = 0;
    double fir = 0., sec = 0.;

    double M = -double(1e9);
    double m = double(1e9);

    int M_ind = -1;
    int m_ind = -1;

    std::pair<double, double> max_vec, min_vec;

    while (true) {
        ct++;
        int toupiao = (l + r) / 2;
        cv::HoughLinesP(target, lines, 1, CV_PI / 180, toupiao, 70, maxLineGap);
        std::cout << lines.size() << std::endl;
        std::cout << ct << ' ' << toupiao << std::endl;

        //int num = 9;
        int num = 8;
        if (lines.size() == num) {
            cv::cvtColor(target, target, CV_GRAY2BGR);
            for (size_t i = 0; i < lines.size(); i++) {
                double tmp = -double(lines[i][1] - lines[i][3]) / double(lines[i][0] - lines[i][2]);
                //                cv::line(target, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
                //

                std::cout << i << ' ' << tmp << std::endl;
                if (abs(tmp) > 0.8)
                    continue;
                //if (abs(target.cols / 2 - tmp * (target.rows / 2 - lines[i][0]) - lines[i][1]) > 75) {
                //  continue;
                //}

                if (lines[i][1] > 0.80 * target.cols || lines[i][1] < 0.20 * target.cols || lines[i][3] > 0.80 * target.cols || lines[i][3] < 0.20 * target.cols) {
                    continue;
                }

                if (lines[i][0] > 0.80 * target.rows || lines[i][2] > 0.80 * target.rows || lines[i][0] < 0.20 * target.rows || lines[i][2] < 0.20 * target.rows) {
                    continue;
                }


                if (tmp > M) {
                    M = tmp;
                    max_vec = std::make_pair(double(lines[i][1] - lines[i][3]), double(lines[i][0] - lines[i][2]));
                    M_ind = i;
                }
                if (tmp < m) {
                    m = tmp;
                    min_vec = std::make_pair(double(lines[i][1] - lines[i][3]), double(lines[i][0] - lines[i][2]));
                    m_ind = i;
                }
            }
            break;
        } else if (lines.size() > num) {
            l = toupiao;
        } else {
            r = toupiao;
        }


        if (r - l == 1) {
            std::cout << "cannot" << std::endl;
            return;
        }
    }


    norm(max_vec);
    norm(min_vec);


    std::pair<double, double> vec = {-double(max_vec.first + min_vec.first), double(max_vec.second + min_vec.second)};
    std::cout << vec.first << ' ' << vec.second << std::endl;


    double angle = std::atan2(vec.first, vec.second);

    std::cout << angle * 180. / CV_PI << std::endl;

    //    double min_angle = std::atan2(-175, -150);
    //   double max_angle = std::atan2(-172, 153);
    //angle = std::atan2(35, -195);
    //    angle = std::atan2(1, 0);
    double min_angle = std::atan2(0, -1);
    double max_angle = std::atan2(0, 1);
    //    minus(min_angle);
    //   minus(max_angle);
    //  minus(angle);

    double L = 80.0;
    //double beta = 2. * CV_PI - (max_angle - min_angle);
    double beta = CV_PI - max_angle;
    std::cout << (2. * CV_PI - angle + std::atan2(-175, -150)) * 180 / CV_PI << ' ' << min_angle << ' ' << max_angle << std::endl;

    std::cout << -L * (angle - min_angle) / beta + 20.;

    cv::line(target, cv::Point(lines[M_ind][0], lines[M_ind][1]), cv::Point(lines[M_ind][2], lines[M_ind][3]), cv::Scalar(0, 0, 255), 3, 8);
    cv::line(target, cv::Point(lines[m_ind][0], lines[m_ind][1]), cv::Point(lines[m_ind][2], lines[m_ind][3]), cv::Scalar(0, 0, 255), 3, 8);


    cv::imshow("i", target);
    cv::imwrite("read.png", target);
    cv::waitKey();
}

}  // namespace Template

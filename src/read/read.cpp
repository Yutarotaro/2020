#include "difference.hpp"
#include "read.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace Read
{


void lineDetect(cv::Mat src)
{
    cv::Mat dst, color_dst;

    if (!(src.data)) {
        return;
    }

    cv::Canny(src, dst, 50, 200, 3);
    cv::cvtColor(dst, color_dst, CV_GRAY2BGR);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(dst, lines, 1, CV_PI / 180, 80, 30, 10);

    for (size_t i = 0; i < lines.size(); i++) {
        cv::line(color_dst, cv::Point(lines[i][0], lines[i][1]),
            cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
    }

    cv::namedWindow("Detected Lines", 1);
    cv::imshow("Detected Lines", color_dst);
}


Data readMeter(cv::Mat src)
{

    Data ret;


    ret.value = 1.0;
    ret.percent = 1.0;

    lineDetect(src);


    return ret;
}
}  // namespace Read

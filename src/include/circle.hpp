#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

namespace Circle
{
int circleDetect(cv::Mat img)
{
    cv::Mat gray;
    //2値化
    cvtColor(img, gray, COLOR_BGR2GRAY);
    //平滑化
    GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
        2, gray.rows / 4, 200, 100);
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // 円の中心を描画します．
        circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
        // 円を描画します．
        circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
    }
    namedWindow("circles", 1);
    imshow("circles", img);

    return 0;
}
}  // namespace Circle

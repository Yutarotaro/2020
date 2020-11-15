#pragma once


class Camera_pose
{
public:
    cv::Mat A;           //camera instric param
    cv::Mat distCoeffs;  //camera distortion param
    cv::Mat R;
    cv::Mat pos;
    cv::Mat t;  //Camera1
};

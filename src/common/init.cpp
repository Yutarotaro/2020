#include "init.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

namespace Init
{

int parseA(cv::Mat& A)
{
    std::ostringstream ostr;
    ostr << filepath1 << "/build/camera.xml";

    //[ref] https://qiita.com/wakaba130/items/3ce8d8668d0a698c7e1b

    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "File can not be opened." << std::endl;
    }

    //first method: use operator on FileNode
    int frameCount = (int)fs["frameCount"];


    cv::Mat cameraMatrix, distCoef;
    //    fs["intrinsic"] >> cameraMatrix;
    fs["intrinsic"] >> A;
    fs["distortion"] >> distCoef;

    /*    std::cout << "frameCount: " << frameCount << std::endl
              << "camera matrix: " << cameraMatrix << std::endl
              << "distortion: " << distCoef << std::endl;
*/
    return 0;
}

cv::Mat input_images(std::string s)
{
    std::ostringstream ostr;
    ostr << filepath1 << "/list.xml";
    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "File can not be opened." << std::endl;
    }

    std::ostringstream ostr2;
    ostr2 << filepath2 << std::to_string((int)fs[s][0]) << "/pic" << std::to_string((int)fs[s][1]) << ".jpg";
    cv::Mat image = cv::imread(ostr2.str(), 1);

    return image;
}

}  // namespace Init

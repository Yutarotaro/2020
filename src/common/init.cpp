#include "init.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

extern cv::Mat A;
extern cv::Mat distCoeffs;
extern cv::Mat R;
extern cv::Mat t;
extern cv::Mat pos;

namespace Init
{
//一番マシなメータまでの距離
double z = 449.35;


int parseInit()
{
    std::ostringstream ostr;
    ostr << filepath1 << "/build/camera.xml";

    //[ref] https://qiita.com/wakaba130/items/3ce8d8668d0a698c7e1b

    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "File can not be opened." << std::endl;
    }

    //    fs["intrinsic"] >> cameraMatrix;
    fs["intrinsic"] >> A;
    fs["distortion"] >> distCoeffs;
    fs["R"] >> R;
    fs["t"] >> pos;

    //TODO: .xmlからの配列の読み取り 6/1


    std::ostringstream ostr2;
    ostr2 << filepath1 << "/list.xml";
    //list.xmlは適当に決めた値なので，使わない


    cv::FileStorage fs2(ostr2.str(), cv::FileStorage::READ);
    if (!fs2.isOpened()) {
        std::cerr << "File can not be opened." << std::endl;
    }
    //    fs2["pos"] >> pos;
    //   fs2["R"] >> R;

    //posをCamera1 Coordinateに変換
    //t = R * pos;

    //zのみマシな値に差し替える
    pos.at<double>(0, 2) = z;
    t = R * pos;

    std::cout << "init ok" << std::endl;
    return 0;
}

cv::Mat input_render(std::string s, int num)
{

    std::ostringstream ostr;
    ostr << filepath1 << '/' << s << "/pic" << std::to_string(num) << ".png";
    cv::Mat image = cv::imread(ostr.str(), 1);

    return image;
}

cv::Mat input_images2(std::string s, std::string t)
{
    std::ostringstream ostr;
    ostr << filepath1 << "/images.xml";
    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "images.xml can not be opened." << std::endl;
    }

    std::ostringstream ostr2;
    ostr2 << filepath2 << '/' << s << "/pic" << std::to_string((int)fs[t][0]) << ".png";
    cv::Mat image = cv::imread(ostr2.str(), 1);

    return image;
}
cv::Mat input_images(std::string s)
{
    std::ostringstream ostr;
    ostr << filepath1 << "/images.xml";
    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "images.xml can not be opened." << std::endl;
    }

    //cv::FileNode f = fs[s];
    //std::cout << (std::string)f << std::endl;

    std::ostringstream ostr2;
    ostr2 << filepath2 << "/A" << std::to_string((int)fs[s][0]) << "/pic" << std::to_string((int)fs[s][1]) << ".jpg";
    cv::Mat image = cv::imread(ostr2.str(), 1);

    return image;
}


}  // namespace Init

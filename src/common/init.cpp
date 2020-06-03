#include "init.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

extern cv::Mat A;
extern cv::Vec3f R;
extern cv::Vec3f t;

namespace Init
{

int parseInit()
{
    std::ostringstream ostr;
    ostr << filepath1 << "/build/camera.xml";

    //[ref] https://qiita.com/wakaba130/items/3ce8d8668d0a698c7e1b

    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "File can not be opened." << std::endl;
    }

    cv::Mat cameraMatrix, distCoef;
    //    fs["intrinsic"] >> cameraMatrix;
    fs["intrinsic"] >> A;
    fs["distortion"] >> distCoef;

    /*TODO: .xmlからの配列の読み取り 6/1
    std::ostringstream ostr2;
    ostr2 << filepath1 << "/list.xml";

    //[ref] https://qiita.com/wakaba130/items/3ce8d8668d0a698c7e1b

    cv::FileStorage fs2(ostr.str(), cv::FileStorage::READ);
    if (!fs2.isOpened()) {
        std::cerr << "list.xml can not be opened." << std::endl;
    }

    auto tmp = fs2["R"];

    for (int i = 0; i < 3; i++) {
        R[i] = tmp[i];
    }

    cv::FileNode n = fs2["t"];
    cv::FileNodeIterator it = n.begin(), it_end = n.end();  // Go through the node
    for (; it != it_end; ++it) {
        std::cout << *it << std::endl;
        std::cout << 1 << std::endl;
    }
*/
    R = {90, 0, 0};
    t = {0, -30, 4.9583};

    return 0;
}  // namespace Init

cv::Mat input_images(std::string s)
{
    std::ostringstream ostr;
    ostr << filepath1 << "/images.xml";
    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "images.xml can not be opened." << std::endl;
    }

    std::ostringstream ostr2;
    ostr2 << filepath2 << std::to_string((int)fs[s][0]) << "/pic" << std::to_string((int)fs[s][1]) << ".jpg";
    cv::Mat image = cv::imread(ostr2.str(), 1);

    return image;
}


}  // namespace Init

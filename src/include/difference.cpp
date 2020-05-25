#include "difference.hpp"
#include "init.hpp"
#include "read.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace Difference
{

//TODO: originとsrcをきれいに一致させ背景差分から読み取り値を返す
Read::Data readMeter(cv::Mat src)
{
    cv::Mat origin;

    Init::input_images(0, 1, origin);
    Read::Data ret;


    cv::imshow("origin", origin);


    return ret;
}

}  // namespace Difference

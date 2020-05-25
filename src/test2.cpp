#include "include/difference.hpp"
#include "include/init.hpp"
#include "include/module.hpp"
#include "include/read.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(void)
{

    cv::Mat Src1, Src2;

    if (Init::input_images(1, 5, Src1) || Init::input_images(1, 3, Src2)) {
        cout << "No such file(s)" << endl;
        return 0;
    }

    std::cout << Module::getHomography(Src1, Src2) << std::endl;

    cv::Mat Src3;

    Init::input_images(1, 4, Src3);

    Difference::readMeter(Src3);

    Read::Data result = Read::readMeter(Src3);
    //Read::readMeter(Src3);

    cv::waitKey();

    return 0;
}

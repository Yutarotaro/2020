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

using namespace std;

int main(void)
{

    cv::Mat Src1, Src2;

    if (Init::input_images(1, 2, Src1) || Init::input_images(1, 3, Src2)) {
        cout << "No such file(s)" << endl;
        return 0;
    }

    Module::getHomography(Src1, Src2);

    Read::Data result = Read::readMeter(Src1);
    cv::waitKey();

    return 0;
}

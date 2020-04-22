#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>


//#define filepath "~/research/20200422/src/pictures/A3/pic1.jpg"
#define filepath "/Users/yutaro/research/20200422/src/pictures/A3/pic1.jpg"

int main()
{

    cv::Mat img = cv::imread(filepath, 1);

    if (img.empty()) {
        std::cout << "failed to read pictures" << std::endl;
        return -1;
    }


    cv::namedWindow("picture");
    cv::imshow("picture", img);

    while (true) {
        const int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}

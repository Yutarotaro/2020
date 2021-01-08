#include "include.hpp"

Camera_pose camera;

cv::Mat temp;

std::vector<cv::KeyPoint> keypoints1;
cv::Mat descriptors1;

std::vector<cv::KeyPoint> keypointsP;
cv::Mat descriptorsP;

//対応点
std::vector<cv::Point> featurePoint;
std::vector<cv::Point> featurePoint2;

// H:Template to Test Homography
cv::Mat H;
cv::Mat HP;

// Eigen::Matrix<float, 6, 1> param;

//テスト画像のindex
int it;

int meter_type;
std::string meter_type_s;
// type 0: normal, 1: pointer_considered
int type;
// record 0: no, 1:Yes
int record;
int message(int argc, char** argv);

/*class Params
{
public:
    int thresh;          //2値化の閾値
    int total;           //枚数
    std::string picdir;  //ディレクトリ
    cv::Point tl;        //左上
    int l;               //矩形の大きさ
    double front_value;  //正面から読んだときの値
    double front_rad;    //正面から読み取った針の角度
    double k;            //1[deg]に対する変化率
};
*/

// 0:T, 1:V
Init::Params params[] = {
    {70, 126, "meter_experiment", cv::Point(1899, 979), 620, 25.2, 1.81970,
        80. / CV_PI},
    {100, 92, "meter_experiment_V", cv::Point(1808, 1033), 680, -0.0101,
        1.88299, 33.5 * 0.002 / CV_PI},
    {100, 60, "dia_experiment_V", cv::Point(1808, 1033), 680, -0.0098, 1.8684,
        33.5 * 0.002 / CV_PI},
    {100, 60, "bthesis", cv::Point(1808, 1033), 680, -0.04, 0.4877,
        33.5 * 0.002 / CV_PI},
};

std::map<std::string, int> mp;

int ite = 1;  // erode, dilate

int main(int argc, char** argv)
{
    // cmdline::parser parser;
    //入力が正しいか確認
    if (message(argc, argv)) {
        std::cout << "unexpected inputs" << std::endl;
        return -1;
    }

    // parameter
    Init::parseInit();

    //読み取り結果を記録
    std::string fileName = "./diffjust/" + meter_type_s + "/data/reading/" + (argc >= 4 ? argv[3] : "") + "reading.csv";
    std::ofstream ofs(fileName, std::ios::app);

    cv::Mat Base_clock = cv::imread(
        "../pictures/meter_template/Base_clock" + meter_type_s + ".png", 1);

    // Base_clockの特徴点を保存
    Init::Feature Temp("../pictures/meter_template/temp" + meter_type_s + ".png");
    //Init::Feature Temp("../pictures/meter_template/tempbthesis2.png");
    // index of test image
    it = std::stoi(argv[1]);

    std::cout << "picture " << it << std::endl;
    std::string path = "../pictures/" + params[meter_type].picdir + "/pic" + std::to_string(it) + ".JPG";

    cv::Mat Now_clock = cv::imread(path, 1);  // for matching

    // object detection で 切り取られたメータ領域の画像
    cv::Mat init = cv::imread("../pictures/" + params[meter_type].picdir + "/roi/pic" + std::string(argv[1]) + ".png",
        1);

    //将来的にinitの方もclass管理したい(名前が衝突するからNowにしたい)
    // Init::Feature Now("../pictures/" + params[meter_type].picdir + "/roi/pic" +
    // std::string(argv[1]) + ".png");

    std::ostringstream ostr;
    ostr << "./minimum.xml";

    //[ref] https://qiita.com/wakaba130/items/3ce8d8668d0a698c7e1b

    cv::FileStorage fs(ostr.str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "File can not be opened." << std::endl;
    }

    // template と roi のスケールを合わせる
    double rate = std::max((double)Temp.img.cols / init.cols,
        (double)Temp.img.rows / init.rows);
    cv::resize(init, init, cv::Size(), rate, rate);

    cv::Mat edge;
    cv::Mat edge_temp;
    init.copyTo(edge);
    Temp.img.copyTo(edge_temp);
    // 11/10ここで一回initを先鋭化しておく
    int unsharp = 1;
    if (unsharp) {
        double k = 3.;
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -k, 0, -k, 4 * k + 1., -k, 0, -k, 0);

        cv::filter2D(init, edge, -1, kernel, cv::Point(-1, -1), 0,
            cv::BORDER_DEFAULT);
    }

    Init::Feature Edge(edge);

    //  H = Module::getHomography(Temp.keypoints, Temp.descriptors, edge_temp,
    //  edge);
    H = Temp.getHomography(Edge);

    cv::Mat warped_init = cv::Mat::zeros(init.rows, init.cols, CV_8UC3);
    cv::warpPerspective(init, warped_init, H.inv(), warped_init.size());

    cv::Mat H2 = Module::getHomography(Temp.keypoints, Temp.descriptors, Temp.img,
        warped_init);

    // initをtempに重ねて可読性判定
    cv::Mat warped = cv::Mat::zeros(Temp.img.rows, Temp.img.cols, CV_8UC3);
    // cv::warpPerspective(init, warped, H.inv(), warped.size());
    cv::warpPerspective(warped_init, warped, H2.inv(), warped.size());

    cv::imwrite("./diffjust/" + meter_type_s + "/transformed/pic" + std::to_string(it) + (unsharp ? "unsharp" : "") + ".png",
        warped);

    ////////////////////////////////////////AdaptiveIntegralThresholding

    cv::Mat gray_temp;
    cv::cvtColor(Temp.img, gray_temp, cv::COLOR_BGR2GRAY);
    cv::Mat bwt = cv::Mat::zeros(gray_temp.size(), CV_8UC1);
    Adaptive::thresholdIntegral(gray_temp, bwt);
    // cv::dilate(bwt, bwt, cv::Mat(), cv::Point(-1, -1), 1);

    cv::Mat hsv, mask_img;
    cv::cvtColor(warped, hsv, CV_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, 100), cv::Scalar(179, 255, 255), mask_img);
    mask_img = ~mask_img;

    cv::Mat gray_warped;
    cv::cvtColor(warped, gray_warped, cv::COLOR_BGR2GRAY);
    cv::Mat bwi = cv::Mat::zeros(gray_warped.size(), CV_8UC1);
    Adaptive::thresholdIntegral(gray_warped, bwi);
    // cv::erode(bwi, bwi, cv::Mat(), cv::Point(-1, -1), 1);

    ////////////////////////////////////////

    //cv::imshow("bwi", bwi);
    //cv::imshow("bwt", bwt);

    //cv::imwrite("bwi.png", bwi);
    //cv::imwrite("bwt.png", bwt);

    // cv::Mat diff_tmp = bwi - bwt;
    cv::Mat diff_tmp;
    // substruct image は　排他的論理和
    cv::bitwise_xor(bwi, bwt, diff_tmp);

    int d = 90;
    cv::Mat mask_for_dif = cv::Mat::zeros(diff_tmp.rows, diff_tmp.cols, CV_8UC1);
    cv::circle(mask_for_dif, cv::Point(diff_tmp.cols / 2, diff_tmp.rows / 2), diff_tmp.cols / 2 - d / 2, cv::Scalar(255), -1, 0);
    cv::Mat diff;  // meter領域のみ残した差分画像
    diff_tmp.copyTo(diff, mask_for_dif);

    cv::imwrite("before.png", diff);

    int itr = 3;
    cv::erode(diff, diff, cv::Mat(), cv::Point(-1, -1), itr);
    cv::dilate(diff, diff, cv::Mat(), cv::Point(-1, -1), itr);

    cv::Mat dif;
    cv::imshow("dfe", diff_tmp);
    // diff_tmp.copyTo(diff, mask_img);
    // cv::bitwise_and(diff_tmp, mask_img, diff);
    // for light filter
    // mask_img = ~mask_img;
    // diff = diff_tmp - mask_img;
    dif = diff;


    //ここで背景切り取り


    ///////////////////

    // cv::imwrite("./diffjust/" + meter_type_s + "/diff/" + std::to_string(it) +
    // (type ? "pointer" : "normal") + ".png", dif);
    cv::erode(dif, dif, cv::Mat(), cv::Point(-1, -1), ite);
    cv::dilate(dif, dif, cv::Mat(), cv::Point(-1, -1), ite);

    cv::imshow("diff", dif);
    cv::imwrite("./diffjust/" + meter_type_s + "/diff_min/pic" + std::to_string(it) + (unsharp ? "unsharp" : "") + ".png",
        dif);

    cv::Mat warped_dif;
    cv::Mat dif_24;
    cv::cvtColor(dif, dif_24, CV_GRAY2BGR);
    warped.copyTo(warped_dif, dif_24);
    cv::imshow("warped_dif", warped_dif);
    //    cv::imwrite("./diffjust/" + meter_type_s + "/diff_min/pic" +
    //    std::to_string(it) + ".png", warped_dif);

    cv::Mat thinned_dif;

#if 0
    //moment
    cv::dilate(dif, dif, cv::Mat(), cv::Point(-1, -1), 1);
    using namespace std;
    using namespace cv;
    RNG rng(12345);
    int thresh = 100;
    Mat canny_output;
    Canny(dif, canny_output, thresh, thresh * 2, 3);
    vector<vector<Point>> contours;
    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        mu[i] = moments(contours[i]);
    }
    vector<Point2f> mc(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        //add 1e-5 to avoid division by zero
        mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
            static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
        cout << "mc[" << i << "]=" << mc[i] << endl;
    }
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, 2);
        circle(drawing, mc[i], 4, color, -1);
        //imshow("Contours", drawing);
    }
    cout << "\t Info: Area and Contour Length \n";

    double area_temp = 0.;
    int cont_index = -1;

    for (size_t i = 0; i < contours.size(); i++) {
        cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
             << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength(contours[i], true) << endl;
        if (norm(Point2f(mc[i].x - diff.cols / 2, mc[i].y - diff.rows / 2)) < 100) {
            if (arcLength(contours[i], true) > area_temp) {
                area_temp = arcLength(contours[i], true);
                cont_index = i;
            }
        }
    }

    cout << "cont_index: " << cont_index << endl;

    Mat result_img = Mat::zeros(canny_output.size(), CV_8UC3);
    drawContours(result_img, contours, (int)cont_index, cv::Scalar(255, 0, 0), 2);
    circle(result_img, mc[cont_index], 4, cv::Scalar(255, 0, 0), -1);
    drawContours(result_img, contours, (int)(1 + cont_index), cv::Scalar(255, 0, 0), 2);
    circle(result_img, mc[1 + cont_index], 4, cv::Scalar(255, 0, 0), -1);


    cv::imwrite("./drawing/" + to_string(it) + ".png", result_img);
#endif


    cv::ximgproc::thinning(dif, thinned_dif, cv::ximgproc::WMF_EXP);

    //    コンストラクタ じゃなくて，class内関数でいい
    Readability::result Result = {0, 0, 0, cv::Mat()};
    Result = Readability::pointerDetection(thinned_dif, dif);

    int white_num = cv::countNonZero(dif);

    if (record) {
        ofs << it << ',' << Result.value << ',' << Result.dist << ',' << Result.readability << ',' << white_num << ',' << std::endl;
    }

    std::cout << Result.value << std::endl;
    cv::imwrite("./diffjust/" + meter_type_s + "/reading/" + std::to_string(it) + "min.png", Result.img);


    return 0;
}

int message(int argc, char** argv)
{
    mp["T"] = 0;
    mp["V"] = 1;
    mp["dia_V"] = 2;
    mp["bthesis"] = 3;

    meter_type_s = argv[2];
    std::cout << meter_type_s << std::endl;
    // meter_type_s = "T";
    std::cout << "type of analog meter:" << meter_type_s << std::endl;
    meter_type = mp[meter_type_s];

    // std::string tmp = argv[2];
    std::string tmp = "0";
    type = std::stoi(tmp);
    std::cout << "type of homography: " << type << std::endl;

    // tmp = argv[3];
    tmp = "1";
    record = std::stoi(tmp);
    std::cout << "record? :" << (record ? "Yes" : "No") << std::endl;

    std::cout << std::endl
              << "/////////////////////////////" << std::endl;

    return 0;
}

#include "header.hpp"

class Imgprocess
{
private:
    cv::Mat process_img(cv::Mat img);
    void Tesseract_configure(cv::Mat img, char* type);
    rs2::frameset frame;
    rs2::frameset aligned_set;
    cv::Mat binary_image;
    cv::Mat color_image;
    tesseract::TessBaseAPI api;
    char *outText;
    void make_contours(cv::Mat img, char* type);
    void environment_v();
    rs2::sensor sen;
    int exposure_par = 170;
    cv::Mat PerfectReflectionAlgorithm(cv::Mat src);
    clock_t start, finish;
public:
    void imgCalibration(char *img_locate, char *type);
    void exposure();
    cv::Mat frame_to_mat(const rs2::frame &f);
    void open_rs_camera(char* type);
    bool countDigits();
};
#include "tesseract_OCR.cpp"

bool imgCalibration();


int main()
{
	Imgprocess Imgprocess;
	char* image_locate = "/home/zisheng/icalhomework/auto_exposure/img/calibration.jpg";
	char* tesseract_type = "eng";
	// Imgprocess.imgCalibration(image_locate, tesseract_type);
	// if(Imgprocess.countDigits()!=true)
	// {
	// 	std::cout << "it's worked" << std::endl;
	// }
	Imgprocess.open_rs_camera(tesseract_type);
    return 0;
}

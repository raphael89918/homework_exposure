#include "tesseract_OCR.hpp"

void Imgprocess::imgCalibration(char *img_locate, char *type)
{
	std::string image_name = img_locate;
	cv::Mat imageMat = cv::imread(image_name);
	if (imageMat.data == nullptr)
	{
		printf("No image data \n");
		exit(0);
	}
	make_contours(imageMat, type);
}

cv::Mat Imgprocess::process_img(cv::Mat img)
{
	cv::Mat hsvImg, binaryImg;
	cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);
	inRange(hsvImg, cv::Scalar(80, 95, 0), cv::Scalar(110, 255, 255), binaryImg);
	cv::Mat kernel_3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	cv::Mat kernel_5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	cv::Mat kernel_7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
	cv::erode(binaryImg, binaryImg, kernel_7);
	cv::dilate(binaryImg, binaryImg, kernel_5);
	// GaussianBlur(binaryImg, binaryImg, cv::Size(3, 3), 1);
	return binaryImg;
}

bool Imgprocess::countDigits()
{
	int count = 0;
	for (int i = 0; i < 10; i++)
	{
		if ((int)outText[i] == i + 48)
		{
			count++;
		}
	}
	
	if (count == 10)
	{
		std::cout << "digits corrected" << std::endl;
		return true;
	}
	else
	{
		std::cout << "can't configured 0123456789" << std::endl;
		return false;
	}
}

void Imgprocess::Tesseract_configure(cv::Mat img, char *type)
{
	if (api.Init(NULL, type))
	{
		std::cout << stderr << std::endl;
		exit(0);
	}
	api.SetVariable("tessedit_char_whitelist", "0123456789");
	api.SetVariable("user_defined_dpi", "96");
	api.SetImage((uchar *)img.data, img.cols, img.rows, 1, img.cols);
	// Get OCR result
	this->outText = api.GetUTF8Text();
	// std::cout << outText << std::endl;
}

cv::Mat Imgprocess::frame_to_mat(const rs2::frame &f)
{
	auto vf = f.as<rs2::video_frame>();
	const int w = vf.get_width();
	const int h = vf.get_height();

	if (f.get_profile().format() == RS2_FORMAT_BGR8)
	{
		return cv::Mat(cv::Size(w, h), CV_8UC3, (void *)f.get_data(), cv::Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_RGB8)
	{
		auto r_rgb = cv::Mat(cv::Size(w, h), CV_8UC3, (void *)f.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat r_bgr;
		cv::cvtColor(r_rgb, r_bgr, cv::COLOR_RGB2BGR);
		return r_bgr;
	}
	else if (f.get_profile().format() == RS2_FORMAT_Z16)
	{
		return cv::Mat(cv::Size(w, h), CV_16UC1, (void *)f.get_data(), cv::Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_Y8)
	{
		return cv::Mat(cv::Size(w, h), CV_8UC1, (void *)f.get_data(), cv::Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
	{
		return cv::Mat(cv::Size(w, h), CV_32FC1, (void *)f.get_data(), cv::Mat::AUTO_STEP);
	}

	throw std::runtime_error("Frame format is not supported yet!");
}

void Imgprocess::open_rs_camera(char* type)
{
	rs2::pipeline pipeline;
	rs2::config config;
	rs2::align align_to(RS2_STREAM_COLOR);
	pipeline.start();
	sen = pipeline.get_active_profile().get_device().query_sensors()[1];
	
	while (true)
	{
		frame = pipeline.wait_for_frames(); // Wait for next set of frames from the camera
		aligned_set = align_to.process(frame);
		this->color_image = frame_to_mat(aligned_set.get_color_frame());
		this->binary_image = process_img(color_image);
		// environment_v();
		// color_image = PerfectReflectionAlgorithm(color_image);
		// cv::imshow("test", binary_image);
		make_contours(binary_image, type);
		cv::waitKey(1);
	}
}

void Imgprocess::make_contours(cv::Mat img, char* type)
{
	std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
	findContours( binary_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<cv::Point2f>centers( contours.size() );
    std::vector<float>radius( contours.size() );
    
	for( size_t i = 0; i< contours.size(); i++ )
	{
		approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
	}
	start = clock();
	for( size_t i = 0; i< contours.size(); i++ )
    {
		if(boundRect[i].area() >= 400)
		{
			cv::Rect rect(boundRect[i].x, boundRect[i].y, boundRect[i].width, boundRect[i].height);
			cv::Mat temp = this->color_image(rect);
			Tesseract_configure(temp, type);
			cv::rectangle(color_image, boundRect[i].tl(), boundRect[i].br(), (0, 255, 0), 2);
		}	
    }
	finish = clock();
	double duration = (double)(finish-start)/CLOCKS_PER_SEC;
	std::cout << duration << std::endl;
	
	
	cv::imshow("Contours", color_image);
}

void Imgprocess::environment_v()
{
	cv::Mat hsv;
	cv::cvtColor(color_image, hsv, cv::COLOR_BGR2HSV);
	int totalV = 0;
	int meanv = 0;
	int V;

	for (int i = 0; i < hsv.rows; i++)
	{
		for (int j = 0; j < hsv.cols; j++)
		{
			V = hsv.at<cv::Vec3b>(i, j)[2];
			totalV += V;
		}
	}
	meanv = totalV / (hsv.rows * hsv.cols);
	// std::cout << meanv << std::endl;
	if (meanv >= 130)
	{
		exposure_par -= 50;
		if (exposure_par < 1)
			exposure_par = 1;
	}
	if (meanv <= 80)
	{
		exposure_par += 50;
	}
	sen.set_option(RS2_OPTION_EXPOSURE, exposure_par);
}
cv::Mat Imgprocess::PerfectReflectionAlgorithm(cv::Mat src)
{
	int row = src.rows;
	int col = src.cols;
	cv::Mat dst(row, col, CV_8UC3);
	int HistRGB[767] = {0};
	int MaxVal = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			MaxVal = std::max(MaxVal, (int)src.at<cv::Vec3b>(i, j)[0]);
			MaxVal = std::max(MaxVal, (int)src.at<cv::Vec3b>(i, j)[1]);
			MaxVal = std::max(MaxVal, (int)src.at<cv::Vec3b>(i, j)[2]);
			int sum = src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2];
			HistRGB[sum]++;
		}
	}
	int Threshold = 0;
	int sum = 0;
	for (int i = 766; i >= 0; i--)
	{
		sum += HistRGB[i];
		if (sum > row * col * 0.1)
		{
			Threshold = i;
			break;
		}
	}
	int AvgB = 0;
	int AvgG = 0;
	int AvgR = 0;
	int cnt = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int sumP = src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2];
			if (sumP > Threshold)
			{
				AvgB += src.at<cv::Vec3b>(i, j)[0];
				AvgG += src.at<cv::Vec3b>(i, j)[1];
				AvgR += src.at<cv::Vec3b>(i, j)[2];
				cnt++;
			}
		}
	}
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int Blue = src.at<cv::Vec3b>(i, j)[0] * MaxVal / AvgB;
			int Green = src.at<cv::Vec3b>(i, j)[1] * MaxVal / AvgG;
			int Red = src.at<cv::Vec3b>(i, j)[2] * MaxVal / AvgR;
			if (Red > 255)
			{
				Red = 255;
			}
			else if (Red < 0)
			{
				Red = 0;
			}
			if (Green > 255)
			{
				Green = 255;
			}
			else if (Green < 0)
			{
				Green = 0;
			}
			if (Blue > 255)
			{
				Blue = 255;
			}
			else if (Blue < 0)
			{
				Blue = 0;
			}
			dst.at<cv::Vec3b>(i, j)[0] = Blue;
			dst.at<cv::Vec3b>(i, j)[1] = Green;
			dst.at<cv::Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}
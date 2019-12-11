//#include"main.h"
//
//void main()
//{	
//	/*system("pause");*/
//	Mat grab_frame;
//	Mat src;
//	Mat result;
//	Mat corr_vector;
//	char src_path[50];
//	char dst_path[50];
//
//	double lambda;          //校正增益
//	int pixel_height;        //行间补偿区域高度
//	double kappa;           //行间补偿增益
//	double kappa_frame;     //帧间补偿增益
//	int slice_width;        //裁剪宽度
//	lambda = 0.8;
//	pixel_height = 20;
//	kappa = 0.001;
//	kappa_frame = 0.3;
//	slice_width = 5000;
//
//
//
//	//视频处理
//	//VideoCapture v_capture = VideoCapture(0);
//
//	//v_capture.open(0);
//	//if(!v_capture.isOpened())
//	//	return;
//	//v_capture.read(grab_frame);
//	//cvtColor(grab_frame, grab_frame, CV_BGR2GRAY);
//	//
//	//none_uniformity_corr(grab_frame, result, corr_vector, lambda, pixel_width, kappa, kappa_frame);
//
//	//while (1)
//	//{
//	//	v_capture.read(grab_frame);
//	//	cvtColor(grab_frame, grab_frame,CV_BGR2GRAY);
//	//	none_uniformity_corr(grab_frame, result, corr_vector, lambda, pixel_width, kappa, kappa_frame);
//	//	cv::imshow("result_window", result);
//	//	waitKey(20);
//	//}
//	//v_capture.release();
//
//	double t1 = clock();
//	//图像处理
//	//for (int num = 1; num <= 9; num++)
//	//{
//	//	sprintf(src_path, "D:\\work\\实习\\连续图\\%d.bmp", num);                       
//	//	sprintf(dst_path, "D:\\work\\实习\\连续图\\%d_dst.bmp", num);
//	//	src = imread(src_path, CV_8UC1);
//	//	none_uniformity_corr(src, result, corr_vector, lambda, pixel_height, kappa, kappa_frame,slice_width);
//	//}
//	src = imread("D:\\work\\Rutgers\\Course\\CA\\Project\\白斑\\test2_src.jpg",CV_8UC1);	
//	//var_threshold(src, src, 130, 30, 0.5, 5, 1);
//	none_uniformity_corr(src, result, corr_vector, lambda, pixel_height, kappa, kappa_frame, slice_width);
//	imwrite("D:\\work\\Rutgers\\Course\\CA\\Project\\白斑\\test2_result.bmp", result);
//	double t2 = clock();
//	printf("总时间：%.2f\n", t2 - t1);
//	system("pause");
//}
//
//bool none_uniformity_corr(Mat& input, Mat& output, Mat& corr_vector, double lambda, int pixel_width, double kappa, double kappa_frame,int slice_width)
//{
//	Mat input_sliced;
//	int K = input.rows;
//	int J = input.cols;
//	int iter = K / pixel_width;
//	image_slice(input, input_sliced, slice_width);
//	input_sliced.convertTo(input_sliced, CV_32FC1, 1.0f / 255.0f);
//	correction(input_sliced, output, corr_vector, K, J, lambda, iter, kappa, kappa_frame);
//	output.convertTo(output, CV_8UC1, 255.0);
//	linear_correction(output, output, 25, 230, 0, 255);
//
//	
//	var_threshold(output, output, 30, 30, 0.1, 2.0f/255.0f, 1);
//	return true;
//}
//
//bool correction(Mat& input, Mat& output, Mat& corr_vector, int K, int J, double lambda, int iter, double kappa, double kappa_frame)
//{
//	Mat src_sort;
//	Mat src_sort_roi;
//	Mat column_avg;
//	Mat patch_sum;
//	Mat corr_start;
//	Mat result;
//	Mat avg_min_single;
//	output = input;
//
//	int P = K / iter;
//	double avg_min = 0;
//	Scalar patch_mean = Scalar(0);
//
//	//sort(input, src_sort, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
//	//reduce(input(Rect(0, (int)floor((double)K * 0.2), input.cols, (int)ceil((double)K * 0.8))), column_avg, 0, CV_REDUCE_AVG);
//	reduce(input, column_avg, 0, CV_REDUCE_AVG);
//	minMaxIdx(column_avg, &avg_min, NULL, NULL, NULL);
//	corr_start = column_avg - avg_min;
//	if (!corr_vector.empty())
//		corr_start = corr_start - (corr_start - corr_vector) * kappa_frame;
//	
//	for (int i = 0; i <= iter - 1; i++)
//	{
//		reduce(input(Rect(0, i * P, input.cols, P)), patch_sum, 0, CV_REDUCE_AVG);
//		patch_mean = mean(patch_sum);
//		column_avg = column_avg - (column_avg - patch_mean(0)) * kappa;
//		minMaxIdx(column_avg, &avg_min, NULL, NULL, NULL);
//		corr_start = column_avg - avg_min;
//		for (int j = 0; j <= P - 1; j++)
//		{
//			output.row(i * P + j) = input.row(i * P + j) - corr_start * lambda;
//		}
//	}
//
//	//for (int i = 0; i <= K - 1; i++)
//	//{
//	//	column_avg = column_avg - (column_avg - input.row(i)) * kappa;
//	//	minMaxIdx(column_avg, &avg_min, NULL, NULL, NULL);
//	//	corr_start = column_avg - avg_min;
//	//	output.row(i) = input.row(i) - corr_start * lambda;
//	//} 
//	corr_vector = corr_start;
//	return true;
//}
//
//inline bool linear_correction(Mat& input, Mat& output, double input_low, double input_high, double output_low, double output_high)
//{
//	output = input * (output_high - output_low) / (input_high - input_low);
//	output.setTo(0, (input <= input_low));
//	output.setTo(255, (input >= input_high));
//	return true;
//}
//
//inline bool image_slice(Mat& input, Mat& output, int width)
//{
//	Mat mask;
//	Rect roi_rect;
//	int contour_num = 0;
//	std::vector<std::vector<Point>> rect_contours;
//	threshold(input, mask, 175, 255, THRESH_BINARY_INV);
//	findContours(mask, rect_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	while (1)
//	{
//		if (contourArea(rect_contours[contour_num]) > 3000000)
//		{
//			roi_rect = boundingRect(rect_contours[contour_num]);
//			break;
//		}
//		contour_num++;
//	}
//	output = input(Rect(roi_rect.x + 15, 0, width, 1000));
//	return true;
//}
//
//bool var_threshold(Mat& input, Mat& output, int mask_width, int mask_height, double std_dev_scale, double abs_threshold, int Lightdark)
//{
//	Mat conv_mean;
//	Mat conv_square;
//	Mat mean_kernal;
//	Mat input_result;
//	Mat input_square;
//	Mat std_mask = Mat(Size(input.cols, input.rows), CV_32FC1);
//	Mat var_result = Mat(Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0));
//	Mat conv_mean_square;
//	Mat result = Mat(Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0));
//	std::vector<std::vector<Point>> contours;
//
//	double num = mask_width * mask_height;
//	//input.convertTo(input, CV_32FC1, 1.0f / 255.0f);
//	//abs_threshold = abs_threshold / 255.0f;
//
//	input.convertTo(output, CV_32FC1, 1.0f / 255.0f);
//	mean_kernal = Mat(Size(mask_width, mask_height), CV_8UC1, cv::Scalar(1));
//	blur(input, conv_mean, Size(mask_width, mask_height));
//	//filter2D(input, conv_mean, CV_32F, mean_kernal);
//	//conv_mean = conv_mean / num;
//	cv::pow(input, 2, input_square);
//	blur(input_square, conv_square, Size(mask_width, mask_height));
//	//filter2D(input_square, conv_square, CV_32F, mean_kernal);
//
//	pow(conv_mean, 2, conv_mean_square);
//	std_mask = (conv_square / num - conv_mean_square);
//	cv::sqrt(cv::abs(std_mask), std_mask);
//	std_mask = std_mask * std_dev_scale;
//	std_mask.setTo(abs_threshold, (std_mask < abs_threshold));
//	var_result.setTo(255, (conv_mean - std_mask > input));
//	var_result.setTo(255, (conv_mean + std_mask < input));
//
//	morphologyEx(var_result, var_result, MORPH_OPEN, cv::getStructuringElement(MORPH_RECT, Size(3, 3)));
//	morphologyEx(var_result, var_result, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT, Size(17, 17)),Point(-1, -1), 2);
//
//	findContours(var_result, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//	std::vector <std::vector<Point>>::iterator iter = contours.begin();
//	for (; iter != contours.end();)
//	{
//		double g_dConArea = contourArea(*iter);
//		if (g_dConArea < 1)
//		{
//			iter = contours.erase(iter);
//		}
//		else
//		{
//			++iter;
//		}
//	}
//
//	drawContours(result, contours, -1, Scalar(255), CV_FILLED);
//	dilate(result, result, getStructuringElement(MORPH_RECT, Size(5, 5)));
//	output = result;
//	return true;
//}
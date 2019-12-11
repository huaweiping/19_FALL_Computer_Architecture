#include"main_GPU.h"
#include <string.h>


#define CHECK_ERROR(call){\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}

void main()
{
	/*system("pause");*/
	Mat grab_frame;
	Mat input;
	Mat result;
	Mat corr_vector;
	Mat Buff;
	Mat hist;
	char src_path[50];
	char dst_path[50];

	double lambda;          //校正增益
	int pixel_height;        //行间补偿区域高度
	double kappa;           //行间补偿增益
	double kappa_frame;     //帧间补偿增益
	int slice_width;        //裁剪宽度
	int K;
	int J;
	int iter;


	lambda = 0.8;
	pixel_height = 20;
	kappa = 0.001;
	kappa_frame = 0.3;
	slice_width = 5500;

	Buffer_cuda_var_threshold buff_cuda_var_threshold;


	//视频处理
	//VideoCapture v_capture = VideoCapture(0);

	//v_capture.open(0);
	//if(!v_capture.isOpened())
	//	return;
	//v_capture.read(grab_frame);
	//cvtColor(grab_frame, grab_frame, CV_BGR2GRAY);
	//
	//none_uniformity_corr(grab_frame, result, corr_vector, lambda, pixel_width, kappa, kappa_frame);

	//while (1)
	//{
	//	v_capture.read(grab_frame);
	//	cvtColor(grab_frame, grab_frame,CV_BGR2GRAY);
	//	none_uniformity_corr(grab_frame, result, corr_vector, lambda, pixel_width, kappa, kappa_frame);
	//	cv::imshow("result_window", result);
	//	waitKey(20);
	//}
	//v_capture.release();
	double t1, t2;
	String path_read = "D:\\work\\Rutgers\\Course\\CA\\Project\\白斑\\report\\test11_src.jpg";
	String path_save = "D:\\work\\Rutgers\\Course\\CA\\Project\\白斑\\report\\test11_result.jpg";
	input = imread(path_read, CV_8UC1);
	input.convertTo(input, CV_32FC1, 1.0f / 255.0f);
	cuda_var_threshold(input, result, 130, 30, 0.05, 5, 1, buff_cuda_var_threshold);
	t1 = clock();
	for (int num = 0; num <= 0; num++)
	{
		
		//sprintf_s(src_path, "D:\\work\\实习\\连续图\\%d.bmp", num);                       
		//sprintf_s(dst_path, "D:\\work\\实习\\连续图\\%d_dst.bmp", num);
		input = imread(path_read, CV_8UC1);
		
		//input = imread(src_path, CV_8UC1);
		K = input.rows;
		J = input.cols;
		iter = K / pixel_height;
		//image_slice(input, input, slice_width);
		input.convertTo(input, CV_32FC1, 1.0f / 255.0f);
		correction(input, input, corr_vector, K, J, lambda, iter, kappa, kappa_frame);
		//input.convertTo(Buff, CV_8UC1, 255.0f);

		//imwrite("D:\\work\\实习\\白斑\\test3_step1.jpg", Buff);
		//input.convertTo(input, CV_8UC1, 255.0);
		linear_correction(input, input, 25, 230, 0, 255);
		//input.convertTo(Buff, CV_8UC1, 255.0f);
		//imwrite("D:\\work\\实习\\白斑\\test3_step2.jpg", Buff);



		var_threshold(input, result, 130, 30, 0.05, 5.0, 1);
		//cuda_var_threshold(input, result, 130, 30, 0.05, 5.0, 1, buff_cuda_var_threshold);
		result.convertTo(Buff, CV_8UC1, 255.0f);
		imwrite(path_save, Buff);

		//result.convertTo(result, CV_8UC1, 255);
		//
		//std::vector<std::vector<Point>> contours;
		//cv::findContours(result, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		//std::vector <std::vector<Point>>::iterator contour_num = contours.begin();
		//for (; contour_num != contours.end();)
		//{
		//	double g_dConArea = contourArea(*contour_num);
		//	if (g_dConArea < 1)
		//	{
		//		contour_num = contours.erase(contour_num);
		//	}
		//	else
		//	{
		//		++contour_num;
		//	}
		//}

		//drawContours(result, contours, -1, Scalar(255), CV_FILLED);
		//cv::dilate(result, result, getStructuringElement(MORPH_RECT, Size(5, 5)));

	}
	t2 = clock();
	std::printf("总时间：%.2f\n", t2 - t1);
	
	
	//cv::waitKey();
	//system("pause");
}

bool correction(Mat& input, Mat& output, Mat& corr_vector, int K, int J, double lambda, int iter, double kappa, double kappa_frame)
{
	Mat src_sort;
	Mat src_sort_roi;
	Mat column_avg;
	Mat patch_sum;
	Mat corr_start;
	Mat result;
	Mat avg_min_single;
	output = input;

	int P = K / iter;
	double avg_min = 0;
	Scalar patch_mean = Scalar(0);

	//sort(input, src_sort, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	//reduce(input(Rect(0, (int)floor((double)K * 0.2), input.cols, (int)ceil((double)K * 0.8))), column_avg, 0, CV_REDUCE_AVG);
	reduce(input, column_avg, 0, CV_REDUCE_AVG);
	/*minMaxIdx(column_avg, &avg_min, NULL, NULL, NULL);*/
	//corr_start = column_avg - avg_min;
	if (!corr_vector.empty())
		column_avg = column_avg - (column_avg - corr_vector) * kappa_frame;
	corr_vector = column_avg;

	for (int i = 0; i <= iter - 1; i++)
	{
		reduce(input(Rect(0, i * P, input.cols, P)), patch_sum, 0, CV_REDUCE_AVG);
		patch_mean = mean(patch_sum);
		column_avg = column_avg - (column_avg - patch_mean(0)) * kappa;
		minMaxIdx(column_avg, &avg_min, NULL, NULL, NULL);
		corr_start = column_avg - avg_min;
		for (int j = 0; j <= P - 1; j++)
		{
			output.row(i * P + j) = input.row(i * P + j) - corr_start * lambda;
		}
	}

	//for (int i = 0; i <= K - 1; i++)
	//{
	//	column_avg = column_avg - (column_avg - input.row(i)) * kappa;
	//	minMaxIdx(column_avg, &avg_min, NULL, NULL, NULL);
	//	corr_start = column_avg - avg_min;
	//	output.row(i) = input.row(i) - corr_start * lambda;
	//} 

	return true;
}

inline bool linear_correction(Mat& input, Mat& output, double input_low, double input_high, double output_low, double output_high)
{
	output = input * (output_high / 255.0f - output_low / 255.0f) / (input_high / 255.0f - input_low / 255.0f);
	output.setTo(0, (input <= input_low / 255.0f));
	output.setTo(1, (input >= input_high / 255.0f));
	return true;
}

inline bool image_slice(Mat& input, Mat& output, int width)
{
	Mat mask;
	Rect roi_rect;
	int contour_num = 0;
	std::vector<std::vector<Point>> rect_contours;
	threshold(input, mask, 175, 255, THRESH_BINARY_INV);
	cv::findContours(mask, rect_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	while (1)
	{
		if (contourArea(rect_contours[contour_num]) > 3000000)
		{
			roi_rect = boundingRect(rect_contours[contour_num]);
			break;
		}
		contour_num++;
	}
	output = input(Rect(roi_rect.x + 15, 0, width, 1000));
	return true;
}

bool var_threshold(Mat& input, Mat& output, int mask_width, int mask_height, double std_dev_scale, double abs_threshold, int Lightdark)
{
	Mat conv_mean = Mat(Size(input.cols, input.rows), CV_32FC1);
	Mat conv_square = Mat(Size(input.cols, input.rows), CV_32FC1);
	Mat mean_kernal;
	Mat input_result = Mat(Size(input.cols, input.rows), CV_32FC1);;
	Mat input_square = Mat(Size(input.cols, input.rows), CV_32FC1);;
	Mat std_mask = Mat(Size(input.cols, input.rows), CV_32FC1);
	Mat var_result = Mat(Size(input.cols, input.rows), CV_32FC1, cv::Scalar(0));
	Mat conv_mean_square = Mat(Size(input.cols, input.rows), CV_32FC1);;
	Mat result = Mat(Size(input.cols, input.rows), CV_8UC1, cv::Scalar(0));
	std::vector<std::vector<Point>> contours;

	double num = mask_width * mask_height;
	//input.convertTo(input, CV_32FC1, 1.0f / 255.0f);
	//abs_threshold = abs_threshold / 255.0f;


	mean_kernal = Mat(Size(mask_width, mask_height), CV_8UC1, cv::Scalar(1));
	blur(input, conv_mean, Size(mask_width, mask_height));
	//filter2D(input, conv_mean, CV_32F, mean_kernal);
	//conv_mean = conv_mean / num;
	cv::pow(input, 2, input_square);
	blur(input_square, conv_square, Size(mask_width, mask_height));
	//filter2D(input_square, conv_square, CV_32F, mean_kernal);

	pow(conv_mean, 2, conv_mean_square);
	std_mask = (conv_square / num - conv_mean_square);
	cv::sqrt(cv::abs(std_mask), std_mask);
	std_mask = std_mask * std_dev_scale;
	std_mask.setTo(abs_threshold/255.0f, (std_mask < (abs_threshold/255.0f)));
	var_result.setTo(1, (conv_mean - std_mask > input));
	var_result.setTo(1, (conv_mean + std_mask < input));

	morphologyEx(var_result, var_result, MORPH_OPEN, cv::getStructuringElement(MORPH_RECT, Size(3, 3)));
	morphologyEx(var_result, var_result, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT, Size(17, 17)), Point(-1, -1), 2);

	//findContours(var_result, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//std::vector <std::vector<Point>>::iterator iter = contours.begin();
	//for (; iter != contours.end();)
	//{
	//	double g_dConArea = contourArea(*iter);
	//	if (g_dConArea < 1)
	//	{
	//		iter = contours.erase(iter);
	//	}
	//	else
	//	{
	//		++iter;
	//	}
	//}

	//drawContours(result, contours, -1, Scalar(255), CV_FILLED);
	//dilate(result, result, getStructuringElement(MORPH_RECT, Size(5, 5)));
	output = var_result;
	return true;
}

bool cuda_var_threshold(Mat& input, Mat& output, int mask_width, int mask_height, float std_dev_scale, float abs_threshold, int Lightdark, Buffer_cuda_var_threshold buffer)
{
	
	cuda::Stream stream;
	buffer.input.upload(input, stream);
	buffer.output.upload(input, stream);

	buffer.output.setTo(Scalar(0), stream);
	//buffer.kernal_row = cv::cuda::GpuMat(1, 30, CV_32F, cv::Scalar(1));
	//buffer.kernal_col = cv::cuda::GpuMat(30, 1, CV_32F, cv::Scalar(1));
	//Mat kernal;
	//gen(kernal, 60, 20, CV_32F, 0, 1);
	
	Ptr<cuda::Filter> blur_filter = cv::cuda::createBoxFilter(buffer.input.type(), -1, Size(mask_width, mask_height));    //mask_width, mask_height
																										 //Ptr<cuda::Filter> seperatable_filter = cv::cuda::createSeparableLinearFilter(buffer.input.type(), buffer.input.type(),buffer.kernal_row, buffer.kernal_col);
																										 //Ptr<cuda::Filter> filter2D_filter = cv::cuda::createLinearFilter(buffer.input.type(), -1, kernal);
																									 //Ptr<cuda::Convolution> convonlution_box = cv::cuda::createConvolution();
	Ptr<cuda::Filter> morph_open = cv::cuda::createMorphologyFilter(MORPH_OPEN, buffer.input.type(), cv::getStructuringElement(MORPH_RECT, Size(3, 3)));
	Ptr<cuda::Filter> morph_close = cv::cuda::createMorphologyFilter(MORPH_CLOSE, buffer.input.type(), cv::getStructuringElement(MORPH_RECT, Size(17, 17)), Point(-1, -1), 2);

	abs_threshold = abs_threshold / 255.0f;

	cv::cuda::sqr(buffer.input, buffer.conv_square, stream);
	//seperatable_filter->apply(buffer.input, buffer.conv_mean);
	//seperatable_filter->apply(buffer.conv_square, buffer.conv_square);

	blur_filter->apply(buffer.input, buffer.conv_mean, stream);
	blur_filter->apply(buffer.conv_square, buffer.conv_square, stream);
	//filter2D_filter->apply(buffer.input, buffer.conv_mean);
	//filter2D_filter->apply(buffer.conv_square, buffer.conv_square);
	
	cv::cuda::sqr(buffer.conv_mean, buffer.buff, stream);
	cv::cuda::divide(buffer.buff, mask_width * mask_height, buffer.buff, 1, -1, stream);
	cv::cuda::subtract(buffer.conv_square, buffer.buff, buffer.conv_square, cuda::GpuMat(), -1, stream);
	cv::cuda::abs(buffer.conv_square, buffer.conv_square, stream);
	cv::cuda::sqrt(buffer.conv_square, buffer.conv_square, stream);

	cv::cuda::multiply(buffer.conv_square, std_dev_scale, buffer.conv_square, 1, -1, stream);

	// CV_8U可以直接转换到CV_32F,但是CV_32F转换到CV_8U的时候会出现问题
	//buffer.conv_mean.convertTo(buffer.conv_mean,CV_8U,stream);
	//buffer.conv_square.convertTo(buffer.conv_square, CV_8U, stream);
	//buffer.conv_mean.convertTo(buffer.buff, CV_8U, 255, stream);
	//buffer.conv_square.convertTo(buffer.conv_mean, CV_8U, 255, stream);
	//buffer.buff.copyTo(buffer.conv_square,stream);
	
	cv::cuda::threshold(buffer.conv_square, buffer.buff, abs_threshold, 1, THRESH_BINARY_INV, stream);

	buffer.buff.convertTo(buffer.buff2, CV_8U, 255, stream);

	buffer.conv_square.setTo(abs_threshold, buffer.buff2, stream);

	cv::cuda::subtract(buffer.conv_mean, buffer.conv_square, buffer.buff, cuda::GpuMat(), -1, stream);

	cv::cuda::compare(buffer.buff, buffer.input, buffer.buff2, CMP_GE, stream);

	buffer.output.setTo(1, buffer.buff2, stream);
	
	//cv::cuda::subtract(buffer.conv_mean, buffer.conv_square, buffer.buff, cuda::GpuMat(), -1, stream);
	//buffer.buff.convertTo(buffer.buff2, CV_8U, 255, stream);
	//buffer.input.convertTo(buffer.buff3, CV_8U, 255, stream);
	//cv::cuda::compare(buffer.buff2, buffer.buff3, buffer.buff2, CMP_GE, stream);
	//buffer.output.setTo(1, buffer.buff2, stream);

	cv::cuda::add(buffer.conv_mean, buffer.conv_square, buffer.buff, cuda::GpuMat(), -1, stream);
	cv::cuda::compare(buffer.buff, buffer.input, buffer.buff2, CMP_LE, stream);
	buffer.output.setTo(1, buffer.buff2, stream);
	
	//cv::cuda::add(buffer.conv_mean, buffer.conv_square, buffer.buff, cuda::GpuMat(), -1, stream);
	//buffer.buff.convertTo(buffer.buff3, CV_8U, 255, stream);
	//cv::cuda::compare(buffer.buff2, buffer.buff3, buffer.buff2, CMP_LE, stream);
	//buffer.output.setTo(1, buffer.buff2, stream);
	
	morph_open->apply(buffer.output, buffer.output,stream);
	morph_close->apply(buffer.output, buffer.output,stream);
	
	
	
	
	

	buffer.output.download(output, stream);

	return true;
}


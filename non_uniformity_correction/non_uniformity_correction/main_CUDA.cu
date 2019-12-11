//
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <windows.h>
//#include <cuda_runtime.h>
//#include <nppi.h>
//#include <npp.h>
//
//#include<iostream>
////#include <Exceptions.h>
//#include <nppi_morphological_operations.h>
//
//#include <string.h>
//#include <fstream>
//#include <helper_string.h>
//#include <helper_cuda.h>
//
//
//#include <ImageIO.h>
//#include <ImagesNPP.h>
//#include <ImagesCPU.h>
//#include <string.h>
//
//#include<opencv2\highgui.hpp>
//
//using namespace cv;
//
//inline int cudaDeviceInit(int argc, const char **argv)
//{
//	int deviceCount;
//	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
//
//	if (deviceCount == 0)
//	{
//		std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
//
//		exit(EXIT_FAILURE);
//	}
//
//	int dev = findCudaDevice(argc, argv);
//
//	cudaDeviceProp deviceProp;
//	cudaGetDeviceProperties(&deviceProp, dev);
//	std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;
//
//	checkCudaErrors(cudaSetDevice(dev));
//
//	return dev;
//}
//
//void main()
//{
//	try
//	{
//		//-----------------------------CUDA_declaration-----------------------------------//
//
//
//		//read image from disk to Cpu
//		//std::string file_src("D:\\work\\ÊµÏ°\\°×°ß\\test3_corr.jpg");
//		//npp::ImageCPU_8u_C1 Host_Src;
//				//npp::loadImage(file_src, Host_Src);
//		//int image_width = Host_Src.width();
//		//int image_height = Host_Src.height();
//		String path = "D:\\work\\ÊµÏ°\\°×°ß\\test1_corr.jpg";
//		Mat Img_src = cv::imread(path, CV_8UC1);
//		int image_width = Img_src.cols;
//		int image_height = Img_src.rows;
//		npp::ImageNPP_8u_C1 Device_Src_8u(image_width, image_height);
//		auto err_cuda = cudaMemcpy2D
//		(
//			Device_Src_8u.data(), Device_Src_8u.pitch(),
//			Img_src.ptr<uchar>(0),
//			sizeof(unsigned char) * image_width, sizeof(unsigned char) * image_width, image_height, cudaMemcpyHostToDevice
//		);
//
//
//		npp::ImageCPU_8u_C1 Host_Dst(image_width, image_height);
//		npp::ImageNPP_8u_C1 Device_Dst_8u(Device_Src_8u.size());
//		//printf("%d\n", err_cuda);
//		//-----------------------------CUDA_declare-----------------------------------//
//
//
//
//		//-----------------------------Nonuniformity_Correction_declare-----------------------------------//
//		int sub_height = 50;
//		int iter = image_height / sub_height;
//		npp::ImageNPP_32f_C1 Device_column_mean(image_width, 1);
//		npp::ImageNPP_32f_C1 Device_column_mean_sub(image_width, 1);
//		npp::ImageNPP_32f_C1 Device_column_start(image_width, 1);
//		NppiSize column_range_size = { image_width, 1 };
//		NppiSize sub_column_range_size = { sub_height, 1 };
//		NppStatus err;
//		int min_buffer_size;
//		int mean_buffer_size;
//		float kappa = 0.01;
//		float lambda = 0.8;
//		Npp8u* min_buffer = 0;
//		Npp8u* mean_buffer = 0;
//		float* Device_min;
//		double* Device_mean;
//		Npp32f min_val = 0;
//		Npp64f mean_val = 0;
//
//		int blur_mask_width = 130;         //It is the half width. No more than 98 allowed as the copyborder is only 100 including the extra column for integral image
//		int blur_mask_height = 30;        //It is the half height.No more than 98 allowed as the copyborder is only 100 including the extra row for integral image
//		int copy_border_top = 100;
//		int copy_border_left = 100;
//
//		NppiSize sub_roi = { image_width, sub_height };
//
//		//-----------------------------Nonuniformity_Correction_declare-----------------------------------//
//
//
//		//-----------------------------Var_Threshold_declare-----------------------------------//
//		NppiSize roi_size = { (int)Device_Src_8u.width(), (int)Device_Src_8u.height() };
//
//		npp::ImageNPP_16u_C1 Device_Src_16u(image_width, image_height);
//		npp::ImageNPP_32f_C1 Device_Src_32f(image_width, image_height);
//		npp::ImageNPP_32f_C1 Device_uniform_32f(image_width, image_height);
//		npp::ImageNPP_16u_C1 Device_Conv_Mean_16u(image_width, image_height);
//		npp::ImageNPP_16u_C1 Device_Conv_Square_16u(image_width, image_height);
//		npp::ImageNPP_32f_C1 Device_Buff_32f(image_width, image_height);
//		npp::ImageNPP_16u_C1 Device_Buff_1_16u(image_width, image_height);
//		npp::ImageNPP_8u_C1 Device_Buff_2_8u(image_width, image_height);
//
//		npp::ImageNPP_32f_C1 Device_test_32f(image_width, image_height);
//		npp::ImageNPP_16u_C1 Device_test_16u(image_width, image_height);
//		npp::ImageNPP_8u_C1 Device_test_8u(image_width, image_height);
//		npp::ImageNPP_8u_C1 Close_mask(25, 25);
//		npp::ImageNPP_32s_C1 mean_mask(131, 31);
//		npp::ImageNPP_8u_C1 Device_Dst_copyborder_8u(image_width + 50, image_height + 50);
//		//npp::ImageCPU_8u_C1 mask_tes(Close_mask.size());
//
//		NppiSize blur_mask_size = { blur_mask_width * 2 + 1, blur_mask_height * 2 + 1 };
//		NppiPoint offset = { 0, 0 };
//		NppiPoint anchor = { 0, 0 };
//		NppiPoint mask_anchor = { (blur_mask_size.width - 1) / 2, (blur_mask_size.height - 1) / 2 };
//		const double std_dev_scale = 0.5;
//		const int val_threshold = 5;
//		NppiSize close_mask_size = { (int)Close_mask.width(), (int)Close_mask.height() };
//		NppiPoint close_mask_anchor = { (close_mask_size.width - 1) / 2, (close_mask_size.height - 1) / 2 };
//
//
//		int hpbuffersize;
//		Npp8u* pBuffer;
//		//-----------------------------Var_Threshold_declare-----------------------------------//
//
//		//-----------------------------Integral_Image_declare-----------------------------------//
//
//
//		npp::ImageNPP_8u_C1 Device_Integral_src(image_width, image_height);
//
//		npp::ImageNPP_32f_C1 Integral((Device_Integral_src.width() + 2 * blur_mask_width + 1), (Device_Integral_src.height() + +2 * blur_mask_height + 1));
//		npp::ImageNPP_32f_C1 Integral_point_right_bottom(Integral.size());
//		npp::ImageNPP_32f_C1 Integral_point_left_bottom(Integral.size());
//		npp::ImageNPP_32f_C1 Integral_point_right_top(Integral.size());
//		npp::ImageNPP_32f_C1 Device_Integral_Std_32f(image_width, image_height);
//		npp::ImageNPP_32f_C1 Device_Integral_mean_32f(image_width, image_height);
//		npp::ImageNPP_8u_C1 Device_Integral_copyborder_8u(Device_Integral_src.width() + 2 * copy_border_top, Device_Integral_src.height() + 2 * copy_border_left);
//
//		size_t pitch_64;
//		Npp64f* SqrIntegral;
//		cudaMallocPitch((void**)&SqrIntegral, &pitch_64, sizeof(Npp64f) * Integral.width(), Integral.height());
//
//		NppiSize integral_size = { Device_Integral_src.width() + 2 * blur_mask_width, Device_Integral_src.height() + +2 * blur_mask_height };
//		NppiSize copyborder_size = { (int)Device_Integral_copyborder_8u.width(), (int)Device_Integral_copyborder_8u.height() };
//		NppiRect mask_rect = { 0, 0, blur_mask_width * 2 + 1, blur_mask_height * 2 + 1 };
//		//-----------------------------Integral_Image_declare-----------------------------------//
//
//
//
//
//		//------------------------------Declare Over-------------------------//
//
//
//
//
//
//		//-----------------------------Nonuniformity_Correction-----------------------------------//
//
//
//		nppiMinIndxGetBufferHostSize_32f_C1R(column_range_size, &min_buffer_size);
//		nppiMeanGetBufferHostSize_32f_C1R(column_range_size, &mean_buffer_size);
//		cudaMalloc((void**)(&min_buffer), min_buffer_size);
//		cudaMalloc((void**)(&mean_buffer), mean_buffer_size);
//		cudaMalloc(&Device_min, sizeof(float));
//		cudaMalloc(&Device_mean, sizeof(double));
//		int t1;
//		int t2, t3, t4, t5, t6, t7, t8;
//		int t_avg;
//		cudaEvent_t time1, time2;
//		float time;
//		cudaEventCreate(&time1);
//		cudaEventCreate(&time2);
//
//		t1 = clock();
//		for (int ppp = 0; ppp < 9; ppp++) {
//			Img_src = cv::imread(path, CV_8UC1);
//			err_cuda = cudaMemcpy2D
//			(
//				Device_Src_8u.data(), Device_Src_8u.pitch(),
//				Img_src.ptr<uchar>(0),
//				sizeof(unsigned char) * image_width, sizeof(unsigned char) * image_width, image_height, cudaMemcpyHostToDevice
//			);
//			nppiConvert_8u32f_C1R(Device_Src_8u.data(), Device_Src_8u.pitch(), Device_Src_32f.data(), Device_Src_32f.pitch(), roi_size);
//			//npp::loadImage(file_src, Host_Src);
//			//npp::ImageNPP_8u_C1 Device_Src_8u(Host_Src);
//			nppiSumWindowColumn_8u32f_C1R(
//				Device_Src_8u.data(), Device_Src_8u.pitch(),
//				Device_column_mean.data(), Device_column_mean.pitch(),
//				column_range_size, Device_Src_8u.height(), 0);
//			nppiDivC_32f_C1IR(
//				Device_Src_8u.height(),
//				Device_column_mean.data(), Device_column_mean.pitch(),
//				column_range_size
//			);
//
//			//nppiMinGetBufferHostSize_32f_C1R(column_range_size, &min_buffer_size);
//
//
//			//nppiSet_32f_C1R(1, Device_column_mean.data(), Device_column_mean.pitch(), column_range_size);
//
//
//			//err = nppiMin_32f_C1R(
//			//	Device_column_mean.data(), Device_column_mean.pitch(),
//			//	column_range_size, min_buffer, Device_min);
//			//cudaMemcpy(&min_val, Device_min, sizeof(float), cudaMemcpyDeviceToHost);
//			//printf("%.2f\n", min_val);
//			//printf("%d\n", sizeof(column_range_size));
//			//system("pause");
//			//
//			//err = nppiSubC_32f_C1R(
//			//	Device_column_mean.data(), Device_column_mean.pitch(),
//			//	min_val,
//			//	Device_column_start.data(), Device_column_start.pitch(),
//			//	{ (int)Device_column_mean.width(), 1}
//			//);
//			int i;
//			int j;
//
//
//			for (i = 0; i <= iter - 1; ++i)
//			{
//
//				nppiSumWindowColumn_8u32f_C1R(
//					Device_Src_8u.data() + i * sub_height * Device_Src_8u.pitch(), Device_Src_8u.pitch(),
//					Device_column_mean_sub.data(), Device_column_mean_sub.pitch(),
//					column_range_size, sub_height, 0);
//
//				nppiDivC_32f_C1IR(
//					sub_height,
//					Device_column_mean_sub.data(), Device_column_mean_sub.pitch(),
//					column_range_size
//				);
//
//				nppiMean_32f_C1R(
//					Device_column_mean_sub.data(), Device_column_mean_sub.pitch(),
//					column_range_size, mean_buffer, Device_mean
//				);
//
//
//				cudaMemcpy(&mean_val, Device_mean, sizeof(double), cudaMemcpyDeviceToHost);
//				nppiSubC_32f_C1R(
//					Device_column_mean.data(), Device_column_mean.pitch(),
//					(float)mean_val,
//					Device_Buff_32f.data(), Device_Buff_32f.pitch(),
//					column_range_size
//				);
//
//
//				nppiMulC_32f_C1IR(kappa, Device_Buff_32f.data(), Device_Buff_32f.pitch(), column_range_size);
//
//
//
//				nppiSub_32f_C1R(
//					Device_Buff_32f.data(), Device_Buff_32f.pitch(),
//					Device_column_mean.data(), Device_column_mean.pitch(),
//					Device_column_mean.data(), Device_column_mean.pitch(),
//					column_range_size
//				);
//
//				//Save image
//
//
//				nppiMin_32f_C1R(
//					Device_column_mean.data(), Device_column_mean.pitch(),
//					column_range_size, min_buffer, Device_min);
//				cudaMemcpy(&min_val, Device_min, sizeof(float), cudaMemcpyDeviceToHost);
//
//				nppiSubC_32f_C1R(
//					Device_column_mean.data(), Device_column_mean.pitch(),
//					min_val,
//					Device_column_start.data(), Device_column_start.pitch(),
//					column_range_size
//				);
//				nppiMulC_32f_C1IR(
//					lambda, Device_column_start.data(), Device_column_start.pitch(), column_range_size
//				);
//
//				//Device_column_start.copyTo(Device_uniform_32f.data(), Device_uniform_32f.pitch());
//				for (j = 0; j <= sub_height - 1; ++j)
//				{
//					nppiSub_32f_C1R(
//						Device_column_start.data(), Device_column_start.pitch(),
//						Device_Src_32f.data() + (i * sub_height + j) * Device_Src_32f.pitch() / 4, Device_Src_32f.pitch(),
//						Device_uniform_32f.data() + (i * sub_height + j) * Device_uniform_32f.pitch() / 4, Device_uniform_32f.pitch(),
//						column_range_size
//					);
//				}
//			}
//
//
//			//printf("nppiSumWindowColumn error: %d\n", (int)err);
//
//			//-----------------------------Nonuniformity_Correction-----------------------------------//
//
//			//-----------------------------linear_Correction-----------------------------------//
//			nppiSubC_32f_C1IR(25, Device_uniform_32f.data(), Device_uniform_32f.pitch(), roi_size);
//
//			nppiMulC_32f_C1IR(255.0f / (230.0f - 25.0f), Device_uniform_32f.data(), Device_uniform_32f.pitch(), roi_size);
//
//
//
//
//
//			//Device_uniform_32f.copyTo(Device_test_32f.data(), Device_test_32f.pitch());
//			//nppiConvert_32f8u_C1R(
//			//	Device_test_32f.data(), Device_test_32f.pitch(),
//			//	Device_test_8u.data(), Device_test_8u.pitch(),
//			//	roi_size, NPP_RND_FINANCIAL);
//			//Device_test_8u.copyTo(Host_Dst.data(), Host_Dst.pitch());
//			//npp::saveImage("D:\\work\\ÊµÏ°\\°×°ß\\test3_test.pgm", Host_Dst);
//
//			//-----------------------------linear_Correction-----------------------------------//
//			//system("pause");
//
//
//			//-----------------------------Var_Threshold-----------------------------------//
//			nppiConvert_32f16u_C1R(Device_uniform_32f.data(), Device_uniform_32f.pitch(), Device_Src_16u.data(), Device_Src_16u.pitch(), roi_size, NPP_RND_FINANCIAL);
//			nppiConvert_32f8u_C1R(Device_uniform_32f.data(), Device_uniform_32f.pitch(), Device_Integral_src.data(), Device_Integral_src.pitch(), roi_size, NPP_RND_FINANCIAL);
//			nppiSet_8u_C1R(0, Device_Dst_8u.data(), Device_Dst_8u.pitch(), roi_size);
//			nppiSet_8u_C1R(1, Close_mask.data(), Close_mask.pitch(), close_mask_size);
//			nppiSet_32s_C1R(1, mean_mask.data(), mean_mask.pitch(), blur_mask_size);
//
//
//
//			////mean_filter_part ---------- with different methods, the fastest method is Integral image, which is used eventually.
//			////box filter part
//			nppiSqr_16u_C1RSfs(
//				Device_Src_16u.data(), Device_Src_16u.pitch(),
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//				roi_size, 0
//			);
//
//			//nppiFilterBoxBorder_16u_C1R(
//			//	Device_Src_16u.data(), Device_Src_16u.pitch(),
//			//	roi_size, offset,
//			//	Device_Conv_Mean_16u.data(), Device_Conv_Mean_16u.pitch(),
//			//	roi_size, blur_mask_size, mask_anchor, NPP_BORDER_REPLICATE
//			//);
//
//			//nppiFilterBoxBorder_16u_C1R(
//			//	Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//			//	roi_size, offset,
//			//	Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//			//	roi_size, blur_mask_size, mask_anchor, NPP_BORDER_REPLICATE
//			//);
//
//			// convolution part
//			nppiFilterBorder_16u_C1R(
//				Device_Src_16u.data(), Device_Src_16u.pitch(),
//				roi_size, offset,
//				Device_Conv_Mean_16u.data(), Device_Conv_Mean_16u.pitch(),
//				roi_size, mean_mask.data(), blur_mask_size, mask_anchor, 4061, NPP_BORDER_REPLICATE
//			);
//
//
//
//
//			nppiFilterBorder_16u_C1R(
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//				roi_size, offset,
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//				roi_size, mean_mask.data(), blur_mask_size, mask_anchor, 4061, NPP_BORDER_REPLICATE
//			);
//			
//			nppiSqr_16u_C1RSfs(
//				Device_Conv_Mean_16u.data(), Device_Conv_Mean_16u.pitch(),
//				Device_Buff_1_16u.data(), Device_Buff_1_16u.pitch(),
//				roi_size, 0
//			); 
//
//			nppiSub_16u_C1RSfs(
//				Device_Buff_1_16u.data(), Device_Buff_1_16u.pitch(), Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//				roi_size, 0
//			);
//
//			nppiSqrt_16u_C1IRSfs(Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(), roi_size, 0);
//
//
//
//			//Integral_Image_part
//
//
//			//nppiCopyReplicateBorder_8u_C1R(
//			//	Device_Integral_src.data(), Device_Integral_src.pitch(),
//			//	roi_size,
//			//	Device_Integral_copyborder_8u.data(), Device_Integral_copyborder_8u.pitch(),
//			//	copyborder_size,
//			//	copy_border_top, copy_border_left
//			//);
//
//			//nppiSqrIntegral_8u32f64f_C1R(
//			//	Device_Integral_copyborder_8u.data() + (copy_border_top - blur_mask_height) * Device_Integral_copyborder_8u.pitch() + (copy_border_left - blur_mask_width) * sizeof(Npp8u), Device_Integral_copyborder_8u.pitch(),
//			//	Integral.data(), Integral.pitch(),
//			//	SqrIntegral, pitch_64,
//			//	integral_size, 0, 0
//			//);
//
//			//nppiRectStdDev_32f_C1R(
//			//	Integral.data(), Integral.pitch(),
//			//	SqrIntegral, pitch_64,
//			//	Device_Integral_Std_32f.data(), Device_Integral_Std_32f.pitch(),
//			//	roi_size, mask_rect
//			//);
//
//
//
//			//Integral.copyTo(Integral_point_right_bottom.data(), Integral_point_right_bottom.pitch());
//			//Integral.copyTo(Integral_point_left_bottom.data(), Integral_point_left_bottom.pitch());
//			//Integral.copyTo(Integral_point_right_top.data(), Integral_point_right_top.pitch());
//
//			//nppiAdd_32f_C1IR(
//			//	Integral_point_right_bottom.data() + mask_rect.height * Integral_point_right_bottom.pitch() / 4 + mask_rect.width * sizeof(Npp32f) / 4, Integral_point_right_bottom.pitch(),
//			//	Integral.data(), Integral.pitch(),
//			//	roi_size
//			//);
//
//			//nppiAdd_32f_C1IR(
//			//	Integral_point_left_bottom.data() + mask_rect.height * Integral_point_right_bottom.pitch() / 4, Integral_point_left_bottom.pitch(),
//			//	Integral_point_right_top.data() + mask_rect.width * sizeof(Npp32f) / 4, Integral_point_right_top.pitch(),
//			//	roi_size
//			//);
//
//			//nppiSub_32f_C1IR(
//			//	Integral_point_right_top.data() + mask_rect.width * sizeof(Npp32f) / 4, Integral_point_right_top.pitch(),
//			//	Integral.data(), Integral.pitch(),
//			//	roi_size
//			//);
//
//			//nppiDivC_32f_C1R(
//			//	Integral.data(), Integral.pitch(),
//			//	mask_rect.height * mask_rect.width,
//			//	Device_Integral_mean_32f.data(), Device_Integral_mean_32f.pitch(),
//			//	roi_size
//			//);
//
//
//			//nppiConvert_32f16u_C1R(Device_Integral_Std_32f.data(), Device_Integral_Std_32f.pitch(), Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(), roi_size, NPP_RND_FINANCIAL);
//			//nppiConvert_32f16u_C1R(Device_Integral_mean_32f.data(), Device_Integral_mean_32f.pitch(), Device_Conv_Mean_16u.data(), Device_Conv_Mean_16u.pitch(), roi_size, NPP_RND_FINANCIAL);
//			//mean_filter_part_over
//
//			//nppiConvert_16u8u_C1R(
//			//	Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//			//	Device_test_8u.data(), Device_test_8u.pitch(),
//			//	roi_size);
//			//Device_test_8u.copyTo(Host_Dst.data(), Host_Dst.pitch());
//			//npp::saveImage("D:\\work\\ÊµÏ°\\°×°ß\\test3_test1.pgm", Host_Dst);
//
//
//
//
//			nppiMulC_16u_C1IRSfs(
//				std_dev_scale, Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(), roi_size, 0);
//
//			nppiThreshold_LT_16u_C1IR(
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(), roi_size, val_threshold
//			);
//
//
//
//
//
//
//
//			nppiSub_16u_C1RSfs(
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(), Device_Conv_Mean_16u.data(), Device_Conv_Mean_16u.pitch(),
//				Device_Buff_1_16u.data(), Device_Buff_1_16u.pitch(),
//				roi_size, 0
//			);
//
//
//
//			nppiCompare_16u_C1R(
//				Device_Buff_1_16u.data(), Device_Buff_1_16u.pitch(),
//				Device_Src_16u.data(), Device_Src_16u.pitch(),
//				Device_Buff_2_8u.data(), Device_Buff_2_8u.pitch(),
//				roi_size, NPP_CMP_GREATER_EQ
//			);
//
//
//
//			nppiSet_8u_C1MR(
//				255,
//				Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//				roi_size,
//				Device_Buff_2_8u.data(), Device_Buff_2_8u.pitch()
//			);
//
//
//
//
//
//			nppiAdd_16u_C1RSfs(
//				Device_Conv_Mean_16u.data(), Device_Conv_Mean_16u.pitch(),
//				Device_Conv_Square_16u.data(), Device_Conv_Square_16u.pitch(),
//				Device_Buff_1_16u.data(), Device_Buff_1_16u.pitch(),
//				roi_size, 0
//			);
//
//			nppiCompare_16u_C1R(
//				Device_Buff_1_16u.data(), Device_Buff_1_16u.pitch(),
//				Device_Src_16u.data(), Device_Src_16u.pitch(),
//				Device_Buff_2_8u.data(), Device_Buff_2_8u.pitch(),
//				roi_size, NPP_CMP_LESS
//			);
//
//			nppiSet_8u_C1MR(
//				255,
//				Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//				roi_size,
//				Device_Buff_2_8u.data(), Device_Buff_2_8u.pitch()
//			);
//
//
//
//
//			nppiErode3x3Border_8u_C1R(
//				Device_Dst_8u.data(),
//				Device_Dst_8u.pitch(),
//				roi_size, offset,
//				Device_Buff_2_8u.data(),
//				Device_Buff_2_8u.pitch(),
//				roi_size, NPP_BORDER_REPLICATE
//			);
//
//
//
//			nppiDilate3x3Border_8u_C1R(
//				Device_Buff_2_8u.data(),
//				Device_Buff_2_8u.pitch(),
//				roi_size, offset,
//				Device_Dst_8u.data(),
//				Device_Dst_8u.pitch(),
//				roi_size, NPP_BORDER_REPLICATE
//			);
//
//		}
//		cudaDeviceSynchronize();
//		t8 = clock();
//
//		printf("t8 - t1:%d\n", t8 - t1);
//		cudaDeviceSynchronize();
//		//Device_Dst_8u.copyTo(Host_Dst.data(), Host_Dst.pitch());
//
//		//npp::saveImage("D:\\work\\ÊµÏ°\\°×°ß\\test2_test.pgm", Host_Dst);
//
//
//		cudaEventSynchronize(time2);
//		cudaEventElapsedTime(&time, time1, time2);
//		printf("GPU_time:%.2f\n", time);
//		system("pause");
//
//
//
//		//nppiDilateBorder_8u_C1R(
//		//	Device_Dst_8u.data(),
//		//	Device_Dst_8u.pitch(),
//		//	roi_size, offset,
//		//	Device_test_8u.data(),
//		//	Device_test_8u.pitch(),
//		//	roi_size,
//		//	Close_mask.data(),
//		//	close_mask_size, close_mask_anchor,
//		//	NPP_BORDER_REPLICATE
//		//);
//
//
//		//nppiCopy_8u_C1R(
//		//	Device_Dst_8u.data(),
//		//	Device_Dst_8u.pitch(),
//		//	Device_test_8u.data(), Device_test_8u.pitch(),
//		//	roi_size
//		//);
//		//system("pause");
//
//		//nppiCopyConstBorder_8u_C1R(
//		//	Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//		//	roi_size,
//		//	Device_Dst_copyborder_8u.data(), Device_Dst_copyborder_8u.pitch(),
//		//	copybordered_size,
//		//	25, 25, 0
//		//);
//
//
//		//nppiErode3x3Border_8u_C1R(
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	copybordered_size, offset,
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	roi_size, NPP_BORDER_REPLICATE
//		//);
//
//
//
//		//nppiDilate3x3Border_8u_C1R(
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	copybordered_size, offset,
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	roi_size, NPP_BORDER_REPLICATE
//		//);
//
//
//
//		////nppiCopyReplicateBorder_8u_C1R
//
//		//nppiDilateBorder_8u_C1R(
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	copybordered_size, offset,
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	roi_size,
//		//	Close_mask.data(),
//		//	close_mask_size, close_mask_anchor,
//		//	NPP_BORDER_REPLICATE
//		//);
//
//
//
//
//
//		//auto kkk = nppiDilate_8u_C1R(
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	Device_Dst_copyborder_8u.data() + 25 * Device_Dst_copyborder_8u.pitch() + 25 * sizeof(Npp8u),
//		//	Device_Dst_copyborder_8u.pitch(),
//		//	roi_size,
//		//	Close_mask.data(),
//		//	close_mask_size, close_mask_anchor
//		//	);
//
//
//
//
//		//kkk = nppiDilate_8u_C1R(
//		//	Device_Dst_8u.data(),Device_Dst_8u.pitch(),
//		//	Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//		//	roi_size,
//		//	Close_mask.data(),
//		//	{ 11, 11 }, {-1, -1}
//		//);
//
//
//
//
//
//		//nppiErode_8u_C1R(
//		//	Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//		//	Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//		//	copybordered_size,
//		//	Close_mask.data(),
//		//	{ 3 , 3 }, close_mask_anchor
//		//);
//
//		//nppiMorphGetBufferSize_8u_C1R(roi_size, &hpbuffersize);
//		//cudaMalloc((void**)(&pBuffer), hpbuffersize);
//		//err = nppiMorphCloseBorder_8u_C1R(
//		//	Device_Dst_8u.data(), Device_Dst_8u.pitch(),
//		//	roi_size, offset,
//		//	Device_Buff_2_8u.data(), Device_Buff_2_8u.pitch(),
//		//	roi_size, Close_mask.data(), close_mask_size, close_mask_anchor,
//		//	pBuffer, NPP_BORDER_REPLICATE
//		//);
//
//
//		//printf("%d\n", err);
//
//		//-----------------------------Var_Threshold-----------------------------------//
//
//		//printf("%s", kkk);
//		//system("pause");
//		//);
//		//Cop image from GPU to CPU
//
//		//Device_Dst_8u.copyTo(Host_Dst.data(), Host_Dst.pitch());
//		//nppiConvert_32f8u_C1R(
//		//	Device_test_32f.data(), Device_test_32f.pitch(),
//		//	Device_test_8u.data(), Device_test_8u.pitch(),
//		//	roi_size, NPP_RND_FINANCIAL);
//		//Device_test_8u.copyTo(Host_Dst.data(), Host_Dst.pitch());
//		//Save image
//		//npp::saveImage("D:\\work\\ÊµÏ°\\°×°ß\\test3_test.pgm", Host_Dst);
//
//		//Close_mask.copyTo(mask_tes.data(), mask_tes.pitch());
//		//npp::saveImage("D:\\work\\ÊµÏ°\\°×°ß\\test3_mask.pgm", mask_tes);
//
//
//		nppiFree(Device_Dst_8u.data());
//		nppiFree(Device_Src_8u.data());
//		nppiFree(Device_Conv_Mean_16u.data());
//		nppiFree(Device_Conv_Square_16u.data());
//		nppiFree(Device_Buff_1_16u.data());
//		nppiFree(Device_Buff_2_8u.data());
//		nppiFree(Device_test_8u.data());
//		nppiFree(Device_test_16u.data());
//		cudaFree(min_buffer);
//		//cudaFree(pBuffer);
//
//
//		exit(EXIT_SUCCESS);
//	}
//	catch (npp::Exception &rException)
//	{
//		std::cerr << "Program error! The following exception occurred: \n";
//		std::cerr << rException << std::endl;
//		std::cerr << "Aborting." << std::endl;
//		system("pause");
//		exit(EXIT_FAILURE);
//	}
//	catch (...)
//	{
//		std::cerr << "Program error! An unknow type of exception occurred. \n";
//		std::cerr << "Aborting." << std::endl;
//		system("pause");
//		exit(EXIT_FAILURE);
//		//return -1;
//	}
//}
#pragma once
#include<opencv2\imgproc.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\core.hpp>

#include<cuda.h>
#include<iostream>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>
#include<sstream>


#include<opencv2\cudaimgproc.hpp>

#include<opencv2\cudafilters.hpp>
#include <opencv2\cudaarithm.hpp>



using namespace cv;
using namespace std;

struct Buffer_cuda_var_threshold
{
	cuda::GpuMat input, output;
	cuda::GpuMat conv_mean, conv_square;
	cuda::GpuMat buff, buff2;
	//cuda::GpuMat kernal_row, kernal_col;
	//cuda::GpuMat kernal;
};
double t1;
double t2;
void main();
bool correction(Mat& input, Mat& output, Mat& corr_vector, const int K, const int J, const double lambda, const int pixel_width, const double kappa, const double kappa_frame);
inline bool linear_correction(Mat& input, Mat& output, double input_low, double input_high, double output_low, double output_high);
bool image_slice(Mat& input, Mat& output, int width);
bool var_threshold(Mat& input, Mat& output, int mask_width, int mask_height, double std_dev_scale, double abs_threshold, int Lightdark);
bool cuda_var_threshold(Mat& input, Mat& output, int mask_width, int mask_height, float std_dev_scale, float abs_threshold, int Lightdark, Buffer_cuda_var_threshold buffer);


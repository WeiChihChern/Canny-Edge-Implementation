#include "Utils.h"
#include <algorithm>

//#include "opencv2/opencv.hpp"
//
//using namespace std;

Utils::Utils()
{
}


Utils::~Utils()
{
}


//template <typename T1>
void Utils::conv2(const Mat &src, Mat &dst, const vector< vector<float>> &kernel) {
	if (src.empty()) { cout << "conv2() error. Empty input!\n"; return; }
	if (src.channels() == 3) { cvtColor(src, src, COLOR_BGR2GRAY); }
	// src is now in grayscale
	dst = Mat(src.rows, src.cols, CV_32FC1, cv::Scalar(0));

	int k_rows = kernel.size(),
		k_cols = kernel[0].size(),
		src_rows = src.rows,
		src_cols = src.cols;

	int offset_row = k_rows / 2,
		offset_col = k_cols / 2;
		
	for (int i = offset_row; i < src_rows - offset_row; i++) {
		//uchar* src_ptr = src.ptr<uchar>(i);
		float* dst_ptr = dst.ptr<float>(i);
		
		for (int j = offset_col; j < src_cols - offset_col; j++) {


			float sum = 0; //Kernel could be in float sometimes
			for (int k = -k_rows/2; k < k_rows/2+1; k++) {
				const uchar* src_ptr = src.ptr<uchar>(i+k) + j;
				for (int l = -k_cols/2; l < k_cols/2+1; l++) {
					sum += (src_ptr[l] * kernel[k+k_rows/2][l+k_cols/2]);
				}
			}
			
			dst_ptr[j] = sum;
		}
	}

	return;
}






// change name to conv2_h
void Utils::conv2_h(const Mat& src, Mat& dst, const vector<float> kernel) {
	
	int k_size = kernel.size(),
		src_rows = src.rows,
		src_cols = src.cols;

	int offset_col = k_size / 2;
	int k_idx = -offset_col;

	for (int i = 0; i < src_rows; i++) {
		const uchar* src_ptr = src.ptr<uchar>(i);
		float* dst_ptr = dst.ptr<float>(i);

		for (int j = offset_col; j < (src_cols - offset_col); j++) {

			
			const uchar* src_temp = src_ptr+j;
			float sum = 0; 

			std::for_each(kernel.begin(), kernel.end(), 
				[src_temp, &sum, k_idx](const float &k_val) mutable
				{
					sum += (src_temp[k_idx++] * k_val);
				}
			);
			
			dst_ptr[j] = sum;
		}
	}
}



void Utils::conv2_v(const Mat& src, Mat& dst, const vector<float> kernel) {

	int k_size = kernel.size(),
		src_rows = src.rows,
		src_cols = src.cols;

	int offset_row = k_size / 2;
	int k_idx = -offset_row;

	for (int i = offset_row; i < (src.rows - offset_row); i++) {
		float* dst_ptr = dst.ptr<float>(i);

		for (int j = 0; j < src_cols; j++) {
			const float* src_ptr = src.ptr<float>(i) + j;
			float sum = 0;
			
			std::for_each(kernel.begin(), kernel.end(),
				[src_ptr, &sum, k_idx, &src_cols] (const float& k_val) mutable
				{
					const float *val = src_ptr - (k_idx++ * src_cols);
					sum += (*val * k_val);
				}
			);

			dst_ptr[j] = sum;
		}
	}
}
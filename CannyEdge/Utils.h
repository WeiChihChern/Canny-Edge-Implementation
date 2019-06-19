#pragma once

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


// This class is for some general utilities' functions:
// All the convolution processes here is designed for grayscale image only for now

// Conv2:
//
class Utils
{

public:
	Utils() {};
	~Utils() {};
	
	template<typename src_type, typename kernel_type>
	void conv2(const Mat& src, Mat& dst, const vector<vector<kernel_type>>& kernel) {
		if (src.empty()) {
			cout << "conv2() input error!\n"; 
			return;
		}
		if (src.channels() == 3) { 
			cvtColor(src, src, COLOR_BGR2GRAY); }


		dst = Mat(src.rows, src.cols, CV_32FC1, cv::Scalar(0));

		int k_rows = kernel.size(),
			k_cols = kernel[0].size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_row = k_rows / 2,
			offset_col = k_cols / 2;

		for (int i = offset_row; i < src_rows - offset_row; i++) {
			float* dst_ptr = dst.ptr<float>(i);

			for (int j = offset_col; j < src_cols - offset_col; j++) {

				float sum = 0;
				for (int k = -offset_row; k < offset_row + 1; k++) {
					const src_type* src_ptr = src.ptr<src_type>(i + k) + j;

					for (int l = -offset_col; l < offset_col + 1; l++) {
						sum += (src_ptr[l] * kernel[k + offset_row][l + offset_col]);
					}
				}

				dst_ptr[j] = sum;
			}
		}

		return;
	};
	



	/* 
		Added for separable kernel to enhance conv2's speed
		conv2_v() is for 1-D kernel like 3x1, 5x1, Nx1 kernel
		conv2_h() is for 1-D kernel like 1x3, 1x5, 1xN kernel
	*/
	
	template<typename src_type, typename kernel_type>
	void conv2_v(const Mat& src, Mat& dst, const vector<kernel_type>& kernel) {
		if (src.empty() || src.channels() == 3) {
			cout << "conv2_v() input error.\n";
			return;
		}
		if (dst.empty()) dst = Mat(src.rows, src.cols, CV_32FC1);

		int k_size = kernel.size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_row = k_size / 2;
		int k_idx = -offset_row;

		for (int i = offset_row; i < (src.rows - offset_row); i++) {


			for (int j = 0; j < src_cols; j++) {
				const src_type* src_ptr = src.ptr<src_type>(i) + j;
				         float* dst_ptr = dst.ptr<float>(i) + j;
				         float  sum     = 0;

				//std::for_each(kernel.begin(), kernel.end(),
				//	[src_ptr, &sum, k_idx, &src_cols] (const float& k_val) mutable
				//	{
				//		const float *val = src_ptr - (k_idx++ * src_cols);
				//		sum += (*val * k_val);
				//	}
				//);

				// For loop proven to be faster than std::for_each in this case
				for (int k = 0; k < kernel.size(); k++)
				{
					sum += src_ptr[(k - offset_row) * src_cols] * kernel[k];
				}


				*dst_ptr = sum;
			}
		}
	};


	template<typename src_type, typename kernel_type>
	void conv2_h(const Mat& src, Mat& dst, const vector<kernel_type>& kernel) {
		if (src.empty() || src.channels() == 3) {
			cout << "conv2_h() input error.\n";
			return;
		}
		if (dst.empty()) dst = Mat(src.rows, src.cols, CV_32FC1);


		int k_size = kernel.size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_col = k_size / 2;
		int k_idx = -offset_col;

		for (int i = 0; i < src_rows; i++) {
			const src_type* src_ptr = src.ptr<src_type>(i);
			         float* dst_ptr = dst.ptr<float>(i);

			for (int j = offset_col; j < (src_cols - offset_col); j++) {


				const src_type* src_temp = src_ptr + j;
				float sum = 0;

				//std::for_each(kernel.begin(), kernel.end(), 
				//	[src_temp, &sum, k_idx](const float &k_val) mutable
				//	{
				//		sum += (src_temp[k_idx++] * k_val);
				//	}
				//);

				// For loop proven to be faster than std::for_each in this case
				for (int k = 0; k < kernel.size(); k++)
				{
					sum += src_temp[k - offset_col] * kernel[k];
				}


				dst_ptr[j] = sum;
			}
		}
	
	};


};


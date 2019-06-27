#pragma once

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


#ifdef _OPENMP
	#include <omp.h>
	#define numThreads 4		

	#ifdef __GNUC__
		#define OMP_FOR(n)  _Pragma("omp parallel for if (n>300000)")
	#elif _MSC_VER
		#define OMP_FOR(n)  __pragma(omp parallel for if (n>100) num_threads(numThreads)) 
	#endif	
#else
	#define omp_get_thread_num() 0
	#define OMP_FOR(n)
#endif // _OPENMP




// This class is for some general utilities' functions:
// All the convolution processes here is designed for grayscale image only for now

class Utils
{

public:

	Utils() {};
	~Utils() {};
	










	// User will have to specifiy what type the input 'src' is, 
	//     Ex: uchar -> CV_8UC1, 
	//         float -> CV_32FC1,
	//         double-> CV_64FC1
	template<typename src_type, typename dst_type, typename kernel_type>
	void conv2(const Mat& src, Mat& dst, const vector<vector<kernel_type>>& kernel) {
		
		// Params check
		if (src.empty()) {
			cout << "conv2() input error!\n"; 
			return;
		}
		if (kernel.size() % 2 == 0 || kernel[0].size() % 2 == 0) {
			cout << "kernel's rows & cols should be a odd number, error!\n";
			return;
		}
		if (src.channels() == 3) {
			cvtColor(src, src, COLOR_BGR2GRAY);
		}

		
		// Inits
		if(dst.empty())	dst = Mat(src.rows, src.cols, CV_32FC1, cv::Scalar(0));

		int k_rows = kernel.size(),
			k_cols = kernel[0].size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_row = k_rows / 2,
			offset_col = k_cols / 2;

		OMP_FOR( (src_rows - offset_row) * (src_cols - offset_col) ) // Automatically ignored if no openmp support
		for (int i = offset_row; i < src_rows - offset_row; i++) { // Start looping process
			dst_type* dst_ptr = dst.ptr<dst_type>(i);

			for (int j = offset_col; j < src_cols - offset_col; j++) {

				float sum = 0;

				// Loop through each element in kernel with corresponding pixel value from src
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
	
	template<typename src_type, typename dst_type, typename kernel_type>
	void conv2_v(const Mat& src, Mat& dst, const vector<kernel_type>& kernel) {
		
		// Params check
		if (src.empty() || src.channels() == 3) {
			cout << "conv2_v() input error.\n";
			return;
		}
		if (kernel.size() % 2 == 0) {
			cout << "conv2_v() kernel size error. \n";
			return;
		}


		// Inits
		if (dst.empty()) dst = Mat(src.rows, src.cols, CV_32FC1);

		int k_size = kernel.size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_row = k_size / 2;
		int k_idx = -offset_row;


		OMP_FOR((src_rows - offset_row) * (src_cols - offset_col)) // Automatically ignored if no openmp support
		for (int i = offset_row; i < (src.rows - offset_row); i++) { // Start looping


			for (int j = 0; j < src_cols; j++) {
				const src_type* src_ptr = src.ptr<src_type>(i) + j;
				      dst_type* dst_ptr = dst.ptr<dst_type>(i) + j;
				         float  sum     = 0;

				//std::for_each(kernel.begin(), kernel.end(),
				//	[src_ptr, &sum, k_idx, &src_cols] (const float& k_val) mutable
				//	{
				//		const float *val = src_ptr - (k_idx++ * src_cols);
				//		sum += (*val * k_val);
				//	}
				//);


				// Profiling result shown simple for loop is faster than std::for_each
				for (int k = 0; k < kernel.size(); k++)
				{
					sum += src_ptr[(k - offset_row) * src_cols] * kernel[k];
				}


				*dst_ptr = sum;
			}
		}
	};












	template<typename src_type, typename dst_type, typename kernel_type>
	inline void conv2_v_sobel(const Mat& src, Mat& dst, const vector<kernel_type>& kernel) {

		// Params check
		if (src.empty() || src.channels() == 3) {
			cout << "conv2_vsobel() input error.\n";
			return;
		}
		if (kernel.size() % 2 == 0) {
			cout << "conv2_v_sobel() kernel size error. \n";
			return;
		}


		// Inits
		if (dst.empty()) dst = Mat(src.rows, src.cols, CV_32FC1);

		int k_size = kernel.size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_row = k_size / 2;
		int k_idx = -offset_row;

		
		OMP_FOR( (src_rows - offset_row) * src_cols ) // Automatically ignored if no openmp support
		for (int i = offset_row; i < (src.rows - offset_row); i++) { // Start looping
			const src_type* src_ptr = src.ptr<src_type>(i);
				  dst_type* dst_ptr = dst.ptr<dst_type>(i);

			for (int j = 0; j < src_cols; j++) {

				float sum  = *(src_ptr + j - src_cols)       * kernel[0];
				      sum += *(src_ptr + j)                  * kernel[1];
				*(dst_ptr + j) = sum + *(src_ptr + j + src_cols) * kernel[2];
			}
		}
	};












	template<typename src_type, typename dst_type, typename kernel_type>
	void conv2_h(const Mat& src, Mat& dst, const vector<kernel_type>& kernel) {
		
		// Params check
		if (src.empty() || src.channels() == 3) {
			cout << "conv2_h() input error.\n";
			return;
		}
		if (kernel.size() % 2 == 0) {
			cout << "conv2_h() kernel size error. \n";
			return;
		}

		
		// Inits
		if (dst.empty()) dst = Mat(src.rows, src.cols, CV_32FC1);

		int k_size = kernel.size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_col = k_size / 2;
		int k_idx = -offset_col;

		OMP_FOR( src_rows * (src_cols - offset_col) ) // Automatically ignored if no openmp support
		for (int i = 0; i < src_rows; i++) {
			const src_type* src_ptr = src.ptr<src_type>(i);
			      dst_type* dst_ptr = dst.ptr<dst_type>(i);

			for (int j = offset_col; j < (src_cols - offset_col); j++) {


				const src_type* src_temp = src_ptr + j;
				float sum = 0;

				//std::for_each(kernel.begin(), kernel.end(), 
				//	[src_temp, &sum, k_idx](const float &k_val) mutable
				//	{
				//		sum += (src_temp[k_idx++] * k_val);
				//	}
				//);


				// Profiling result shown simple for loop is faster than std::for_each
				for (int k = 0; k < kernel.size(); k++)
				{
					sum += src_temp[k - offset_col] * kernel[k];
				}


				dst_ptr[j] = sum;
			}
		}

	
	};












	template<typename src_type, typename dst_type, typename kernel_type>
	inline void conv2_h_sobel(const Mat& src, Mat& dst, const vector<kernel_type>& kernel) {

		// Params check
		if (src.empty() || src.channels() == 3) {
			cout << "conv2_h_sobel() input error.\n";
			return;
		}
		if (kernel.size() % 2 == 0) {
			cout << "conv2_h_sobel() kernel size error. \n";
			return;
		}


		// Inits
		if (dst.empty()) dst = Mat(src.rows, src.cols, CV_32FC1);

		int k_size = kernel.size(),
			src_rows = src.rows,
			src_cols = src.cols;

		int offset_col = k_size / 2;
		int k_idx = -offset_col;

		OMP_FOR( src_rows * src_cols ) // Automatically ignored if no openmp support
		for (int i = 0; i < src_rows; i++) {
			const src_type* src_ptr = src.ptr<src_type>(i);
				  dst_type* dst_ptr = dst.ptr<dst_type>(i);

			for (int j = offset_col; j < (src_cols - offset_col); j++) {
				
				float sum  = *(src_ptr+1)        * kernel[0];
				      sum += *(src_ptr+j)            * kernel[1];
			    *(dst_ptr+j) = sum + *(src_ptr+j+1)  * kernel[2];
				
			}
		}

	};



};


#pragma once

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


#ifdef _OPENMP
	#include <omp.h>
	#define maxThreads omp_get_max_threads()
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


#pragma omp parallel for 
		for (size_t i = offset_row; i < src_rows - offset_row; i++) { // Start looping process
			dst_type* dst_ptr = dst.ptr<dst_type>(i);

			for (size_t j = offset_col; j < src_cols - offset_col; j++) {

				float sum = 0;

				// Loop through each element in kernel with corresponding pixel value from src
				for (size_t k = -offset_row; k < offset_row + 1; k++) {
					const src_type* src_ptr = src.ptr<src_type>(i + k) + j;

					for (size_t l = -offset_col; l < offset_col + 1; l++) {
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


#pragma omp parallel for
		for (size_t i = offset_row; i < (src.rows - offset_row); ++i) { // Start looping
			for (size_t j = 0; j < src_cols; j++) {
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
				for (size_t k = 0; k < kernel.size(); ++k)
				{
					sum += src_ptr[(k - offset_row) * src_cols] * kernel[k];
				}


				*dst_ptr = sum;
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

#pragma omp parallel for
		for (size_t i = 0; i < src_rows; ++i) {
			const src_type* src_ptr = src.ptr<src_type>(i);
			      dst_type* dst_ptr = dst.ptr<dst_type>(i);

			for (size_t j = offset_col; j < (src_cols - offset_col); ++j) {


				const src_type* src_temp = src_ptr + j;
				float sum = 0;

				//std::for_each(kernel.begin(), kernel.end(), 
				//	[src_temp, &sum, k_idx](const float &k_val) mutable
				//	{
				//		sum += (src_temp[k_idx++] * k_val);
				//	}
				//);


				// Profiling result shown simple for loop is faster than std::for_each
				for (size_t k = 0; k < kernel.size(); k++)
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

		int k_size = kernel.size();

		int offset_col = k_size / 2;
		int rows = src.rows;
		int cols = src.cols;


		#pragma omp parallel for
		for (size_t i = 0; i < rows; ++i) {
			const src_type* src_ptr = src.ptr<src_type>(i);
				  dst_type* dst_ptr = dst.ptr<dst_type>(i);
			
#ifdef __GNUC__
			// Vectorized confirmed
			#pragma omp simd // for -O2 optimization
#endif
			for (size_t j = offset_col; j < (cols - offset_col); ++j) 
				dst_ptr[j] = src_ptr[j - 1] * kernel[0] + src_ptr[j] * kernel[1] + src_ptr[j + 1] * kernel[2];

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

		int k_size = kernel.size();

		int offset_row = k_size / 2;
		int rows = src.rows;
		int cols = src.cols;


#pragma omp parallel for 
		for (size_t i = offset_row; i < (rows - offset_row); ++i) { // Start looping
				const src_type* src_ptr = src.ptr<src_type>(i);
				      dst_type* dst_ptr = dst.ptr<dst_type>(i);

#ifdef __GNUC__
			// Vectorized confirmed
			#pragma omp simd // for -O2 optimization
#endif
			for (size_t j = 0; j < cols; ++j)
				dst_ptr[j] = src_ptr[j - cols] * kernel[0] + src_ptr[j] * kernel[1] + src_ptr[j + cols] * kernel[2];
				
			}
	};






	// Make the edge pixels to zero (not the edge detection's edge)
	// Edge pixles referring here is the pixels in first row and col, 
	// and in last row and col
	template <typename inputType>
	void edge2zero(Mat &src)
	{
		int r = src.rows,
			c = src.cols;

		inputType *src_f   = src.ptr<inputType>(0);          // first row
		inputType *src_l   = src.ptr<inputType>(src.rows-1); // last row
		inputType *src_f_l = src.ptr<inputType>(0) + c - 1 ; // first row last element

		
#ifdef __GNUC__
		#pragma omp simd // for -O2 optimization
#endif		
		for (size_t j = 0; j < c; ++j) // vectorized confirm
		{
			src_f[j] = 0;
			src_l[j] = 0;
		}
			
#ifdef __GNUC__
		#pragma omp simd // for -O2 optimization
#endif		
		for (size_t i = 0; i < r; ++i) // vectorized confirm
		{
			src_f[i * c] = 0;
			src_f_l[i*c] = 0;
		}
		
	}











#ifdef _OPENMP
	// Prevent overhead for allcating threads
	int threadControl(const int size){
		if(size >= 180000 && size < 1000000)
			return (maxThreads >= 4) ? 4 : maxThreads;
		else if (size >= 1000000 && size < 2073600)
			return (maxThreads >= 6) ? 6 : maxThreads;
		else if (size >= 2073600)
			return (maxThreads >= 8) ? 8 : maxThreads;
		else 
			return (maxThreads >= 2) ? 2 : maxThreads;
	}
#else
	int threadControl(const int size) {
		return 1;
	}

#endif

};

